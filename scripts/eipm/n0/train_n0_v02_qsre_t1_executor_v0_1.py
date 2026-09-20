from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

from alice_personality.n0.qsre_t1_executor import (
    QSRET1Config,
    QSRET1Executor,
    QSRET1OracleOperator,
)


CONTROL_RELATIONAL = 1

TENSOR_KEYS = (
    "field_state",
    "field_metadata",
    "field_valid_mask",
    "edge_index",
    "edge_relation_id",
    "edge_metadata",
    "edge_valid_mask",
    "field_support_weight",
    "edge_support_weight",
    "relation_sequence_id",
    "relation_sequence_mask",
    "role_id",
    "operation_id",
    "focus_field_weight",
    "operator_context",
    "applicability",
    "target_distribution",
    "expected_control",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def to_device_batch(
    split: dict,
    indices: torch.Tensor,
    device: torch.device,
) -> tuple[
    dict,
    QSRET1OracleOperator,
    torch.Tensor,
    torch.Tensor,
]:
    batch = {
        key: split[key][indices].to(device)
        for key in TENSOR_KEYS
    }

    operator = QSRET1OracleOperator(
        relation_sequence_id=batch.pop(
            "relation_sequence_id"
        ),
        relation_sequence_mask=batch.pop(
            "relation_sequence_mask"
        ),
        role_id=batch.pop("role_id"),
        operation_id=batch.pop(
            "operation_id"
        ),
        focus_field_weight=batch.pop(
            "focus_field_weight"
        ),
        context=batch.pop(
            "operator_context"
        ),
        applicability=batch.pop(
            "applicability"
        ),
    )

    expected_control = batch.pop(
        "expected_control"
    )
    target = batch.pop(
        "target_distribution"
    )

    return (
        batch,
        operator,
        target,
        expected_control,
    )


def family_metrics(
    families: list[str],
    success: torch.Tensor,
) -> dict[str, float]:
    totals: dict[
        str,
        list[int],
    ] = defaultdict(lambda: [0, 0])

    for family, ok in zip(
        families,
        success.tolist(),
    ):
        totals[family][0] += int(ok)
        totals[family][1] += 1

    return {
        family: passed / count
        for family, (passed, count)
        in sorted(totals.items())
    }


@torch.inference_mode()
def evaluate(
    model: QSRET1Executor,
    split: dict,
    *,
    device: torch.device,
    batch_size: int,
    plural_l1_threshold: float,
) -> dict:
    model.eval()
    row_count = split[
        "field_state"
    ].size(0)

    probabilities: list[torch.Tensor] = []
    controls: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []

    for start in range(
        0,
        row_count,
        batch_size,
    ):
        indices = torch.arange(
            start,
            min(
                start + batch_size,
                row_count,
            ),
        )
        (
            batch,
            operator,
            _target,
            _expected,
        ) = to_device_batch(
            split,
            indices,
            device,
        )

        output = model(
            **batch,
            operator=operator,
        )
        probabilities.append(
            output[
                "relational_probability"
            ].cpu()
        )
        controls.append(
            output["control_state"].cpu()
        )
        masks.append(
            output[
                "readout_field_mask"
            ].cpu()
        )

    probability = torch.cat(
        probabilities,
        dim=0,
    )
    control = torch.cat(
        controls,
        dim=0,
    )
    readout_mask = torch.cat(
        masks,
        dim=0,
    )

    target = split[
        "target_distribution"
    ].float()
    expected = split[
        "expected_control"
    ].long()

    target_size = (
        target > 0
    ).sum(dim=-1)
    relational = expected.eq(
        CONTROL_RELATIONAL
    )
    single = (
        relational
        & target_size.eq(1)
    )
    plural = (
        relational
        & target_size.gt(1)
    )
    nonrelational = ~relational

    control_ok = control.eq(expected)
    row_success = control_ok.clone()

    if bool(single.any()):
        row_success[single] = (
            probability[single]
            .argmax(dim=-1)
            .eq(
                target[single]
                .argmax(dim=-1)
            )
        )

    plural_l1_row = torch.zeros(
        row_count
    )
    if bool(plural.any()):
        plural_l1_row[plural] = (
            probability[plural]
            - target[plural]
        ).abs().sum(dim=-1)
        row_success[plural] = (
            plural_l1_row[plural]
            .le(plural_l1_threshold)
        )

    if bool(nonrelational.any()):
        row_success[nonrelational] &= (
            probability[nonrelational]
            .abs()
            .sum(dim=-1)
            .eq(0)
        )

    if bool((~readout_mask).any()):
        outside_mass = float(
            probability.masked_select(
                ~readout_mask
            )
            .abs()
            .max()
            .item()
        )
    else:
        outside_mass = 0.0

    family = family_metrics(
        split["families"],
        row_success,
    )

    pair_indices: dict[
        str,
        list[int],
    ] = defaultdict(list)

    for index, group in enumerate(
        split["causal_groups"]
    ):
        pair_indices[group].append(index)

    pair_success: dict[str, bool] = {}
    outside_delta = 0.0

    for group, indices in (
        pair_indices.items()
    ):
        if len(indices) != 2:
            raise RuntimeError(
                f"causal group {group} "
                "count drift"
            )

        pair_success[group] = bool(
            row_success[indices].all()
        )

        family_name = split[
            "families"
        ][indices[0]]

        if (
            family_name
            == "outside_support_distractor"
        ):
            outside_delta = max(
                outside_delta,
                float(
                    (
                        probability[indices[0]]
                        - probability[indices[1]]
                    )
                    .abs()
                    .max()
                    .item()
                ),
            )

    if bool(relational.any()):
        relational_probability = (
            probability[relational]
            .clamp_min(1e-12)
        )
        relational_target = target[
            relational
        ]

        cross_entropy = float(
            (
                -(
                    relational_target
                    * relational_probability.log()
                ).sum(dim=-1)
            )
            .mean()
            .item()
        )

        target_mass = float(
            (
                probability[relational]
                * relational_target.gt(0)
            )
            .sum(dim=-1)
            .mean()
            .item()
        )
    else:
        cross_entropy = 0.0
        target_mass = 0.0

    if bool(plural.any()):
        plural_l1 = float(
            plural_l1_row[plural]
            .mean()
            .item()
        )
    else:
        plural_l1 = 0.0

    return {
        "control_accuracy": float(
            control_ok.float().mean().item()
        ),
        "single_target_top1_accuracy": (
            float(
                row_success[single]
                .float()
                .mean()
                .item()
            )
            if bool(single.any())
            else 1.0
        ),
        "plural_l1": plural_l1,
        "relational_cross_entropy": (
            cross_entropy
        ),
        "mean_target_support_mass": (
            target_mass
        ),
        "outside_support_mass_max": (
            outside_mass
        ),
        "outside_support_invariance_max_delta": (
            outside_delta
        ),
        "causal_pair_completion": (
            sum(pair_success.values())
            / max(len(pair_success), 1)
        ),
        "family_success": family,
        "family_min_success": (
            min(family.values())
            if family
            else 1.0
        ),
        "row_success_accuracy": float(
            row_success.float().mean().item()
        ),
        "rows": row_count,
    }


def is_eligible(
    metrics: dict,
    contract: dict,
) -> bool:
    eligible = contract["eligibility"]

    return (
        metrics["control_accuracy"]
        >= eligible[
            "control_accuracy_min"
        ]
        and metrics[
            "single_target_top1_accuracy"
        ]
        >= eligible[
            "single_target_top1_accuracy_min"
        ]
        and metrics[
            "causal_pair_completion"
        ]
        >= eligible[
            "causal_pair_completion_min"
        ]
        and metrics[
            "family_min_success"
        ]
        >= eligible[
            "family_min_success_min"
        ]
        and metrics["plural_l1"]
        <= eligible["plural_l1_max"]
        and metrics[
            "outside_support_mass_max"
        ]
        <= eligible[
            "outside_support_mass_max"
        ]
        and metrics[
            "outside_support_invariance_max_delta"
        ]
        <= eligible[
            "outside_support_invariance_max_delta"
        ]
    )


def save_checkpoint(
    model: QSRET1Executor,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    torch.save(
        model.state_dict(),
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        required=True,
    )
    parser.add_argument(
        "--prepared-cache",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    args = parser.parse_args()

    contract_path = Path(args.contract)
    prepared_path = Path(
        args.prepared_cache
    )
    output_dir = Path(
        args.output_dir
    )

    contract = json.loads(
        contract_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        contract.get("schema")
        != "alice.eipm.n0.qsre-t1-training-contract.v0.2"
    ):
        raise SystemExit(
            "T1 training contract drift"
        )

    if (
        contract.get(
            "gpu_training_authorized"
        )
        is not True
    ):
        raise SystemExit(
            "training contract does not "
            "authorize GPU execution"
        )

    if (
        int(
            contract.get(
                "max_gpu_runs",
                0,
            )
        )
        != 1
        or contract.get(
            "automatic_rerun"
        )
        is not False
    ):
        raise SystemExit(
            "one-shot governance drift"
        )

    if (
        contract.get(
            "causal_test_open"
        )
        is not False
        or contract.get(
            "frozen_challenge_open"
        )
        is not False
    ):
        raise SystemExit(
            "heldout boundary drift"
        )

    if not torch.cuda.is_available():
        raise SystemExit(
            "T1 governed run requires CUDA"
        )

    expected_prepared_sha = contract.get(
        "expected_prepared_cache_sha256"
    )
    actual_prepared_sha = sha256(prepared_path)
    if (
        not expected_prepared_sha
        or actual_prepared_sha
        != expected_prepared_sha
    ):
        raise SystemExit(
            "prepared cache file hash drift: "
            f"{actual_prepared_sha}"
        )

    device = torch.device("cuda")

    prepared = torch.load(
        prepared_path,
        map_location="cpu",
    )

    if (
        prepared.get("schema")
        != contract[
            "prepared_cache_schema"
        ]
    ):
        raise SystemExit(
            "prepared cache schema drift"
        )

    if (
        prepared.get(
            "source_cache_sha256"
        )
        != contract[
            "source_cache_sha256"
        ]
    ):
        raise SystemExit(
            "source cache lineage drift"
        )

    if (
        prepared.get(
            "curriculum_sha256"
        )
        != contract[
            "expected_curriculum_sha256"
        ]
    ):
        raise SystemExit(
            "curriculum lineage drift"
        )

    if (
        prepared.get(
            "test_present"
        )
        is not False
        or prepared.get(
            "private_identity_data"
        )
        is not False
    ):
        raise SystemExit(
            "prepared boundary drift"
        )

    if output_dir.exists():
        raise SystemExit(
            "refusing to overwrite "
            f"training output: {output_dir}"
        )
    output_dir.mkdir(
        parents=True
    )

    seed = int(
        contract["training"]["seed"]
    )
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_contract = contract["model"]

    if (
        int(
            prepared["field_state_dim"]
        )
        != int(
            model_contract[
                "field_state_dim"
            ]
        )
    ):
        raise SystemExit(
            "field state width drift"
        )

    model = QSRET1Executor(
        QSRET1Config(
            **model_contract
        )
    ).to(device)

    optimizer_contract = contract[
        "optimizer"
    ]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(
            optimizer_contract[
                "learning_rate"
            ]
        ),
        weight_decay=float(
            optimizer_contract[
                "weight_decay"
            ]
        ),
    )

    train = prepared["train"]
    dev = prepared["dev"]

    batch_size = int(
        contract["training"][
            "batch_size"
        ]
    )
    max_steps = int(
        contract["training"][
            "max_steps"
        ]
    )
    eval_every = int(
        contract["training"][
            "eval_every_steps"
        ]
    )
    plural_threshold = float(
        contract["eligibility"][
            "plural_l1_max"
        ]
    )

    checkpoints: dict[
        str,
        dict,
    ] = {}
    selected_step: int | None = None

    initial = evaluate(
        model,
        dev,
        device=device,
        batch_size=batch_size,
        plural_l1_threshold=(
            plural_threshold
        ),
    )
    checkpoints[
        "step-00000000"
    ] = {
        "dev": initial,
        "eligible": is_eligible(
            initial,
            contract,
        ),
    }

    print(
        "step=0 dev="
        + json.dumps(
            initial,
            sort_keys=True,
        ),
        flush=True,
    )

    if checkpoints[
        "step-00000000"
    ]["eligible"]:
        selected_step = 0

    generator = (
        torch.Generator()
        .manual_seed(seed + 1)
    )
    order = torch.randperm(
        train["field_state"].size(0),
        generator=generator,
    )
    cursor = 0
    step = 0

    model.train()

    while (
        selected_step is None
        and step < max_steps
    ):
        if (
            cursor + batch_size
            > order.numel()
        ):
            order = torch.randperm(
                train[
                    "field_state"
                ].size(0),
                generator=generator,
            )
            cursor = 0

        indices = order[
            cursor : cursor + batch_size
        ]
        cursor += batch_size

        (
            batch,
            operator,
            target,
            expected_control,
        ) = to_device_batch(
            train,
            indices,
            device,
        )

        relational = (
            expected_control
            .eq(CONTROL_RELATIONAL)
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            **batch,
            operator=operator,
        )

        probability = output[
            "relational_probability"
        ].clamp_min(1e-12)

        if bool(relational.any()):
            loss = (
                -(
                    target[relational]
                    * probability[
                        relational
                    ].log()
                )
                .sum(dim=-1)
                .mean()
            )

            if not torch.isfinite(loss):
                raise SystemExit(
                    "non-finite T1 loss"
                )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                float(
                    optimizer_contract[
                        "gradient_clip_norm"
                    ]
                ),
            )

            optimizer.step()

        step += 1

        if (
            step % eval_every == 0
            or step == max_steps
        ):
            metrics = evaluate(
                model,
                dev,
                device=device,
                batch_size=batch_size,
                plural_l1_threshold=(
                    plural_threshold
                ),
            )

            eligible = is_eligible(
                metrics,
                contract,
            )

            key = f"step-{step:08d}"

            checkpoints[key] = {
                "dev": metrics,
                "eligible": eligible,
            }

            checkpoint_path = (
                output_dir
                / key
                / "qsre_t1_executor.pt"
            )

            save_checkpoint(
                model,
                checkpoint_path,
            )

            checkpoints[key][
                "checkpoint_sha256"
            ] = sha256(checkpoint_path)

            print(
                f"step={step} "
                f"eligible={str(eligible).lower()} "
                "dev="
                + json.dumps(
                    metrics,
                    sort_keys=True,
                ),
                flush=True,
            )

            if eligible:
                selected_step = step
                break

            model.train()

    result = {
        "schema": (
            "alice.eipm.n0."
            "qsre-t1-training-result.v0.1"
        ),
        "status": (
            "PASS_QSRE_T1_EXECUTOR_DEV_CONTRACT"
            if selected_step is not None
            else (
                "FAIL_QSRE_T1_EXECUTOR_DEV_CONTRACT_"
                "STOP_NO_AUTOMATIC_RERUN"
            )
        ),
        "training_contract_sha256": (
            sha256(contract_path)
        ),
        "prepared_cache_sha256": (
            actual_prepared_sha
        ),
        "prepared_cache_sha256_expected": (
            expected_prepared_sha
        ),
        "source_cache_sha256": (
            prepared[
                "source_cache_sha256"
            ]
        ),
        "curriculum_sha256": (
            prepared[
                "curriculum_sha256"
            ]
        ),
        "selected_checkpoint_step": (
            selected_step
        ),
        "checkpoints": checkpoints,
        "optimizer": "AdamW",
        "gradient_performed": True,
        "gpu_training": True,
        "semantic_backbone_gradient": False,
        "graph_parent_gradient": False,
        "learned_operator": False,
        "learned_support": False,
        "causal_test_opened": False,
        "frozen_challenge_opened": False,
        "private_identity_gradient": False,
        "automatic_rerun_authorized": False,
        "next_action": (
            "interpret T1 executor competence "
            "before any T2/T3 work"
        ),
    }

    result_path = (
        output_dir / "result.json"
    )
    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "result_sha256="
        + sha256(result_path),
        flush=True,
    )
    print(
        "status="
        + result["status"],
        flush=True,
    )


if __name__ == "__main__":
    main()
