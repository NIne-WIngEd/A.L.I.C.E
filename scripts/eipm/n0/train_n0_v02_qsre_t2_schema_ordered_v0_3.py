from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from alice_personality.n0.qsre_t1_executor import (
    QSRE_T1_OPERATION_PATH_FOLLOW,
    QSRE_T1_OPERATION_ROLE_SELECT,
    QSRET1Config,
    QSRET1Executor,
    QSRET1OracleOperator,
)
from alice_personality.n0.qsre_t2_schema_ordered_operator import (
    QSRET2SchemaOrderedConfig,
    QSRET2SchemaOrderedOperatorEncoder,
)

CONTROL_RELATIONAL = 1

STRUCTURAL_KEYS = (
    "field_state",
    "field_metadata",
    "field_valid_mask",
    "edge_index",
    "edge_relation_id",
    "edge_metadata",
    "edge_valid_mask",
    "field_support_weight",
    "edge_support_weight",
)

EXPECTED_PREPARED_SCHEMA = "alice.eipm.n0.qsre-t2-tokenized-preparation.v0.1"
EXPECTED_HIDDEN_SCHEMA = "alice.eipm.n0.qsre-t2-hidden-cache.v0.2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def family_metrics(
    families: list[str],
    success: torch.Tensor,
) -> dict[str, float]:
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for family, ok in zip(families, success.tolist()):
        totals[family][0] += int(ok)
        totals[family][1] += 1
    return {
        family: passed / count
        for family, (passed, count) in sorted(totals.items())
    }


def _special_token_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_ids: set[int],
) -> torch.Tensor:
    mask = attention_mask.bool().clone()
    for token_id in special_ids:
        mask &= input_ids.ne(int(token_id))
    empty = mask.sum(dim=-1).eq(0)
    if bool(empty.any()):
        mask[empty] = attention_mask[empty].bool()
    return mask


@torch.inference_mode()
def materialize_hidden_cache(
    *,
    prepared: dict,
    semantic_config_path: Path,
    semantic_checkpoint: Path,
    tokenizer_dir: Path,
    relation_schema_path: Path,
    output_path: Path,
    device: torch.device,
    batch_size: int,
) -> dict:
    from safetensors.torch import load_file

    from alice_personality.n0.config import load_n0_config
    from alice_personality.n0.curriculum_data import load_tokenizer
    from alice_personality.n0.v02_model import AliceN0V02Model

    cfg = load_n0_config(semantic_config_path)
    model = AliceN0V02Model(cfg)
    missing, unexpected = model.load_state_dict(
        load_file(str(semantic_checkpoint), device="cpu"),
        strict=False,
    )
    if missing or unexpected:
        raise RuntimeError(
            "semantic checkpoint mismatch "
            f"missing={list(missing)} unexpected={list(unexpected)}"
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.to(device).eval()

    tokenizer = load_tokenizer(tokenizer_dir)
    special_ids = set(
        int(value)
        for value in (getattr(tokenizer, "all_special_ids", []) or [])
    )

    relation_schema = json.loads(
        relation_schema_path.read_text(encoding="utf-8")
    )
    if (
        relation_schema.get("schema")
        != "alice.eipm.n0.qsre-t2-relation-schema.v0.1"
    ):
        raise RuntimeError("relation schema contract drift")
    relation_rows = sorted(
        relation_schema["relations"],
        key=lambda row: int(row["id"]),
    )
    if [int(row["id"]) for row in relation_rows] != list(range(6)):
        raise RuntimeError("relation schema id/order drift")

    gloss_text: list[str] = []
    gloss_owner: list[int] = []
    for row in relation_rows:
        for gloss in row["glosses"]:
            gloss_text.append(str(gloss))
            gloss_owner.append(int(row["id"]))

    schema_tokens = tokenizer(
        gloss_text,
        padding=True,
        truncation=True,
        max_length=96,
        return_tensors="pt",
    )
    schema_ids = schema_tokens["input_ids"].to(device)
    schema_attention = schema_tokens["attention_mask"].to(device)
    schema_outputs = model.backbone(
        input_ids=schema_ids,
        attention_mask=schema_attention,
        output_hidden_states=True,
        return_dict=True,
    )
    schema_layers = schema_outputs.hidden_states
    if schema_layers is None or len(schema_layers) != 17:
        raise RuntimeError("relation schema hidden-state depth drift")
    schema_stack = torch.stack(schema_layers, dim=1)
    schema_mask = _special_token_mask(
        schema_tokens["input_ids"],
        schema_tokens["attention_mask"],
        special_ids,
    ).to(device)
    weights = schema_mask[:, None, :, None].to(schema_stack.dtype)
    gloss_layer_state = (
        (schema_stack * weights).sum(dim=2)
        / weights.sum(dim=2).clamp_min(1.0)
    )

    relation_schema_states = []
    owner_tensor = torch.tensor(gloss_owner, device=device)
    for relation_id in range(6):
        selected = owner_tensor.eq(relation_id)
        if not bool(selected.any()):
            raise RuntimeError(
                f"relation schema has no glosses for id {relation_id}"
            )
        relation_schema_states.append(
            gloss_layer_state[selected].mean(dim=0)
        )
    relation_schema_hidden_states = torch.stack(
        relation_schema_states,
        dim=0,
    ).detach().float().cpu()

    hidden_payload: dict[str, object] = {
        "schema": EXPECTED_HIDDEN_SCHEMA,
        "semantic_checkpoint_sha256": sha256(semantic_checkpoint),
        "relation_schema_sha256": sha256(relation_schema_path),
        "relation_schema_hidden_states": relation_schema_hidden_states,
        "num_hidden_states": 17,
        "semantic_dim": 640,
        "dtype": "float16_cache_float32_training",
        "semantic_backbone_gradient": False,
        "private_identity_data": False,
        "test_present": False,
    }

    for split_name in ("train", "dev"):
        split = prepared[split_name]
        input_ids = split["input_ids"].long()
        attention_mask = split["attention_mask"].bool()
        hidden_chunks: list[torch.Tensor] = []
        mask_chunks: list[torch.Tensor] = []

        for start in range(0, input_ids.size(0), batch_size):
            stop = min(start + batch_size, input_ids.size(0))
            ids = input_ids[start:stop].to(device)
            attention = attention_mask[start:stop].to(device)
            outputs = model.backbone(
                input_ids=ids,
                attention_mask=attention,
                output_hidden_states=True,
                return_dict=True,
            )
            states = outputs.hidden_states
            if states is None or len(states) != 17:
                raise RuntimeError(
                    "semantic hidden-state depth drift: "
                    f"{0 if states is None else len(states)}"
                )
            stacked = torch.stack(states, dim=1)
            if stacked.size(-1) != 640:
                raise RuntimeError("semantic hidden-state width drift")

            hidden_chunks.append(
                stacked.detach().to(dtype=torch.float16).cpu()
            )
            mask_chunks.append(
                _special_token_mask(
                    input_ids[start:stop],
                    attention_mask[start:stop],
                    special_ids,
                ).cpu()
            )

        hidden_payload[split_name] = {
            "query_hidden_states": torch.cat(hidden_chunks, dim=0),
            "query_token_mask": torch.cat(mask_chunks, dim=0),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(hidden_payload, output_path)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return hidden_payload


def _paired_batches(
    source_ids: list[str],
    *,
    batch_size: int,
    seed: int,
) -> list[torch.Tensor]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, source_id in enumerate(source_ids):
        groups[str(source_id)].append(index)

    pairs = []
    for source_id, indices in groups.items():
        if len(indices) != 2:
            raise RuntimeError(
                f"query-view pair count drift for {source_id}: {len(indices)}"
            )
        pairs.append(tuple(indices))

    rng = random.Random(seed)
    rng.shuffle(pairs)
    pairs_per_batch = max(batch_size // 2, 1)

    return [
        torch.tensor(
            [
                index
                for pair in pairs[start : start + pairs_per_batch]
                for index in pair
            ],
            dtype=torch.long,
        )
        for start in range(0, len(pairs), pairs_per_batch)
    ]


def _supervised_contrastive_operator_loss(
    continuous: torch.Tensor,
    relation_target: torch.Tensor,
    role_target: torch.Tensor,
    operation_target: torch.Tensor,
    control_target: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    z = F.normalize(continuous.float(), dim=-1)
    relational = control_target.eq(CONTROL_RELATIONAL)
    op_signature = torch.where(
        relational,
        operation_target,
        torch.full_like(operation_target, -1),
    )
    signature = torch.cat(
        [
            relation_target.long(),
            role_target.long().unsqueeze(-1),
            op_signature.long().unsqueeze(-1),
            control_target.long().unsqueeze(-1),
        ],
        dim=-1,
    )
    same = signature[:, None, :].eq(signature[None, :, :]).all(dim=-1)
    eye = torch.eye(z.size(0), device=z.device, dtype=torch.bool)
    positive = same & ~eye

    logits = (z @ z.transpose(0, 1)) / float(temperature)
    logits = logits - logits.max(dim=-1, keepdim=True).values.detach()
    exp_logits = logits.exp() * (~eye).to(logits.dtype)
    denominator = exp_logits.sum(dim=-1).clamp_min(1.0e-12)
    numerator = (
        exp_logits * positive.to(logits.dtype)
    ).sum(dim=-1)
    valid = numerator.gt(0)

    if not bool(valid.any()):
        return continuous.sum() * 0.0

    return -(
        numerator[valid].clamp_min(1.0e-12).log()
        - denominator[valid].log()
    ).mean()


def operator_loss(
    output: dict[str, torch.Tensor],
    split: dict,
    indices: torch.Tensor,
    *,
    weights: dict,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = output["relation_logits"].device
    relation_target = split["relation_target"][indices].to(device)
    role_target = split["role_target"][indices].to(device)
    operation_target = split["operation_target"][indices].to(device)
    control_target = split["control_target"][indices].to(device)

    relation_ce = F.cross_entropy(
        output["relation_logits"].reshape(
            -1,
            output["relation_logits"].size(-1),
        ),
        relation_target.reshape(-1),
    )
    role_ce = F.cross_entropy(
        output["role_logits"],
        role_target,
    )
    control_ce = F.cross_entropy(
        output["control_logits"],
        control_target,
    )

    relational = control_target.eq(CONTROL_RELATIONAL)
    if bool(relational.any()):
        operation_ce = F.cross_entropy(
            output["operation_logits"][relational],
            operation_target[relational],
        )
    else:
        operation_ce = output["operation_logits"].sum() * 0.0

    continuous_supcon = _supervised_contrastive_operator_loss(
        output["continuous_state"],
        relation_target,
        role_target,
        operation_target,
        control_target,
        temperature=temperature,
    )

    total = (
        float(weights["relation"]) * relation_ce
        + float(weights["role"]) * role_ce
        + float(weights["operation"]) * operation_ce
        + float(weights["control"]) * control_ce
        + float(weights["continuous_supcon"]) * continuous_supcon
    )

    return total, {
        "relation_ce": float(relation_ce.detach().item()),
        "role_ce": float(role_ce.detach().item()),
        "operation_ce": float(operation_ce.detach().item()),
        "control_ce": float(control_ce.detach().item()),
        "continuous_supcon": float(continuous_supcon.detach().item()),
        "total": float(total.detach().item()),
    }


def _safe_operator_for_frozen_t1_eval(
    operator: QSRET1OracleOperator,
    *,
    predicted_control: torch.Tensor,
) -> tuple[QSRET1OracleOperator, torch.Tensor]:
    """Make evaluator execution total without forgiving invalid operators.

    PATH_FOLLOW requires a non-empty focus frontier in the frozen T1
    executor. During learning, T2 may temporarily predict PATH_FOLLOW on
    rows where the oracle focus is empty. That is a model error when the
    predicted control is RELATIONAL, but it must be scored as a failure
    rather than crashing the evaluator.

    For execution only, impossible PATH_FOLLOW values are canonicalized to
    ROLE_SELECT. Relational rows that required this canonicalization are
    returned in invalid_relational and are later forced to downstream
    failure. For non-relational control, operation is semantically inactive,
    so canonicalization only satisfies the frozen executor input contract.
    """
    if predicted_control.shape != operator.operation_id.shape:
        raise ValueError("predicted_control shape drift")

    empty_focus = operator.focus_field_weight.sum(dim=-1).le(0)
    path_without_focus = (
        operator.operation_id.eq(QSRE_T1_OPERATION_PATH_FOLLOW)
        & empty_focus
    )
    invalid_relational = (
        path_without_focus
        & predicted_control.eq(CONTROL_RELATIONAL)
    )

    if not bool(path_without_focus.any()):
        return operator, invalid_relational

    safe_operation = operator.operation_id.clone()
    safe_operation[path_without_focus] = QSRE_T1_OPERATION_ROLE_SELECT

    return (
        replace(
            operator,
            operation_id=safe_operation,
        ),
        invalid_relational,
    )


def _operator_tuple(
    relation: torch.Tensor,
    role: torch.Tensor,
    operation: torch.Tensor,
    control: torch.Tensor,
    index: int,
) -> tuple:
    rel = tuple(int(value) for value in relation[index].tolist())
    ctl = int(control[index])
    op = int(operation[index]) if ctl == CONTROL_RELATIONAL else -1
    return rel, int(role[index]), op, ctl


@torch.inference_mode()
def evaluate(
    *,
    operator_model: QSRET2SchemaOrderedOperatorEncoder,
    executor: QSRET1Executor,
    prepared_split: dict,
    hidden_split: dict,
    device: torch.device,
    batch_size: int,
    plural_l1_threshold: float,
) -> dict:
    operator_model.eval()
    executor.eval()

    row_count = prepared_split["input_ids"].size(0)

    predicted_relation: list[torch.Tensor] = []
    predicted_role: list[torch.Tensor] = []
    predicted_operation: list[torch.Tensor] = []
    predicted_control: list[torch.Tensor] = []
    uncertainties: list[torch.Tensor] = []

    downstream_probability: list[torch.Tensor] = []
    downstream_control: list[torch.Tensor] = []
    downstream_masks: list[torch.Tensor] = []
    downstream_invalid_relational: list[torch.Tensor] = []

    for start in range(0, row_count, batch_size):
        stop = min(start + batch_size, row_count)
        indices = torch.arange(start, stop)

        output = operator_model(
            query_hidden_states=hidden_split[
                "query_hidden_states"
            ][indices].float().to(device),
            query_token_mask=hidden_split[
                "query_token_mask"
            ][indices].to(device),
        )

        predicted_relation.append(
            output["relation_logits"].argmax(dim=-1).cpu()
        )
        predicted_role.append(
            output["role_logits"].argmax(dim=-1).cpu()
        )
        predicted_operation.append(
            output["operation_logits"].argmax(dim=-1).cpu()
        )
        predicted_control.append(
            output["control_logits"].argmax(dim=-1).cpu()
        )
        uncertainties.append(
            output["uncertainty"].cpu()
        )

        decoded = operator_model.decode_for_frozen_t1(
            output,
            focus_field_weight=prepared_split[
                "focus_field_weight"
            ][indices].to(device),
        )
        decoded, invalid_relational = _safe_operator_for_frozen_t1_eval(
            decoded,
            predicted_control=output[
                "control_logits"
            ].argmax(dim=-1),
        )

        structural = {
            key: prepared_split[key][indices].to(device)
            for key in STRUCTURAL_KEYS
        }
        downstream = executor(
            **structural,
            operator=decoded,
        )

        downstream_probability.append(
            downstream["relational_probability"].cpu()
        )
        downstream_control.append(
            downstream["control_state"].cpu()
        )
        downstream_masks.append(
            downstream["readout_field_mask"].cpu()
        )
        downstream_invalid_relational.append(
            invalid_relational.cpu()
        )

    relation = torch.cat(predicted_relation, dim=0)
    role = torch.cat(predicted_role, dim=0)
    operation = torch.cat(predicted_operation, dim=0)
    control = torch.cat(predicted_control, dim=0)
    uncertainty = torch.cat(uncertainties, dim=0)

    relation_target = prepared_split["relation_target"].long()
    role_target = prepared_split["role_target"].long()
    operation_target = prepared_split["operation_target"].long()
    control_target = prepared_split["control_target"].long()

    relation_exact = relation.eq(relation_target).all(dim=-1)
    role_ok = role.eq(role_target)
    operation_ok = operation.eq(operation_target)
    control_ok = control.eq(control_target)
    relational = control_target.eq(CONTROL_RELATIONAL)

    full_operator = control_ok & relation_exact & role_ok
    if bool(relational.any()):
        full_operator[relational] &= operation_ok[relational]

    operator_family = family_metrics(
        prepared_split["families"],
        full_operator,
    )

    pair_groups: dict[str, list[int]] = defaultdict(list)
    for index, source_id in enumerate(
        prepared_split["source_t1_row_ids"]
    ):
        pair_groups[str(source_id)].append(index)

    pair_consistency = []
    for source_id, indices in pair_groups.items():
        if len(indices) != 2:
            raise RuntimeError(
                f"query-view pair count drift for {source_id}: "
                f"{len(indices)}"
            )
        first, second = indices
        pair_consistency.append(
            _operator_tuple(
                relation,
                role,
                operation,
                control,
                first,
            )
            == _operator_tuple(
                relation,
                role,
                operation,
                control,
                second,
            )
        )

    probability = torch.cat(
        downstream_probability,
        dim=0,
    )
    downstream_ctl = torch.cat(
        downstream_control,
        dim=0,
    )
    readout_mask = torch.cat(
        downstream_masks,
        dim=0,
    )
    invalid_relational_bridge = torch.cat(
        downstream_invalid_relational,
        dim=0,
    )

    if bool(invalid_relational_bridge.any()):
        probability = probability.clone()
        probability[invalid_relational_bridge] = 0

    target = prepared_split["target_distribution"].float()
    expected = prepared_split["expected_control"].long()
    target_size = (target > 0).sum(dim=-1)

    downstream_relational = expected.eq(CONTROL_RELATIONAL)
    single = downstream_relational & target_size.eq(1)
    plural = downstream_relational & target_size.gt(1)
    nonrelational = ~downstream_relational

    downstream_control_ok = downstream_ctl.eq(expected)
    row_success = downstream_control_ok.clone()

    if bool(single.any()):
        row_success[single] = (
            probability[single]
            .argmax(dim=-1)
            .eq(
                target[single].argmax(dim=-1)
            )
        )

    plural_l1_row = torch.zeros(row_count)
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

    if bool(invalid_relational_bridge.any()):
        row_success[invalid_relational_bridge] = False

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

    downstream_family = family_metrics(
        prepared_split["families"],
        row_success,
    )

    causal_groups: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(
        prepared_split["causal_groups"]
    ):
        causal_groups[str(group)].append(index)

    causal_pair_ok = []
    outside_delta = 0.0

    for group, indices in causal_groups.items():
        if len(indices) != 2:
            raise RuntimeError(
                f"causal group {group} count drift: {len(indices)}"
            )

        causal_pair_ok.append(
            bool(row_success[indices].all())
        )

        if (
            prepared_split["families"][indices[0]]
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

    if bool(downstream_relational.any()):
        relational_probability = (
            probability[downstream_relational]
            .clamp_min(1.0e-12)
        )
        relational_target = target[
            downstream_relational
        ]

        downstream_ce = float(
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
                probability[downstream_relational]
                * relational_target.gt(0)
            )
            .sum(dim=-1)
            .mean()
            .item()
        )
    else:
        downstream_ce = 0.0
        target_mass = 0.0

    plural_l1 = (
        float(
            plural_l1_row[plural]
            .mean()
            .item()
        )
        if bool(plural.any())
        else 0.0
    )

    return {
        "operator": {
            "relation_sequence_exact_accuracy": float(
                relation_exact
                .float()
                .mean()
                .item()
            ),
            "role_accuracy": float(
                role_ok.float().mean().item()
            ),
            "role_accuracy_relational": (
                float(
                    role_ok[relational]
                    .float()
                    .mean()
                    .item()
                )
                if bool(relational.any())
                else 1.0
            ),
            "operation_accuracy_relational": (
                float(
                    operation_ok[relational]
                    .float()
                    .mean()
                    .item()
                )
                if bool(relational.any())
                else 1.0
            ),
            "control_accuracy": float(
                control_ok.float().mean().item()
            ),
            "full_operator_exact_accuracy": float(
                full_operator
                .float()
                .mean()
                .item()
            ),
            "query_view_pair_consistency": (
                sum(pair_consistency)
                / max(len(pair_consistency), 1)
            ),
            "family_success": operator_family,
            "family_min_success": (
                min(operator_family.values())
                if operator_family
                else 1.0
            ),
            "mean_uncertainty": float(
                uncertainty.mean().item()
            ),
            "invalid_relational_path_without_focus_count": int(
                invalid_relational_bridge.sum().item()
            ),
            "invalid_relational_path_without_focus_rate": float(
                invalid_relational_bridge.float().mean().item()
            ),
        },
        "downstream": {
            "control_accuracy": float(
                downstream_control_ok
                .float()
                .mean()
                .item()
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
            "relational_cross_entropy": downstream_ce,
            "mean_target_support_mass": target_mass,
            "outside_support_mass_max": outside_mass,
            "outside_support_invariance_max_delta": (
                outside_delta
            ),
            "causal_pair_completion": (
                sum(causal_pair_ok)
                / max(len(causal_pair_ok), 1)
            ),
            "family_success": downstream_family,
            "family_min_success": (
                min(downstream_family.values())
                if downstream_family
                else 1.0
            ),
            "row_success_accuracy": float(
                row_success
                .float()
                .mean()
                .item()
            ),
        },
        "rows": row_count,
    }


def is_eligible(
    metrics: dict,
    contract: dict,
) -> bool:
    gate = contract["eligibility"]
    operator = metrics["operator"]
    downstream = metrics["downstream"]

    return (
        operator["relation_sequence_exact_accuracy"]
        >= gate["relation_sequence_exact_accuracy_min"]
        and operator["role_accuracy_relational"]
        >= gate["role_accuracy_relational_min"]
        and operator["operation_accuracy_relational"]
        >= gate["operation_accuracy_relational_min"]
        and operator["control_accuracy"]
        >= gate["operator_control_accuracy_min"]
        and operator["full_operator_exact_accuracy"]
        >= gate["full_operator_exact_accuracy_min"]
        and operator["query_view_pair_consistency"]
        >= gate["query_view_pair_consistency_min"]
        and operator["family_min_success"]
        >= gate["operator_family_min_success_min"]
        and downstream["control_accuracy"]
        >= gate["downstream_control_accuracy_min"]
        and downstream["single_target_top1_accuracy"]
        >= gate["downstream_single_target_top1_accuracy_min"]
        and downstream["causal_pair_completion"]
        >= gate["downstream_causal_pair_completion_min"]
        and downstream["family_min_success"]
        >= gate["downstream_family_min_success_min"]
        and downstream["plural_l1"]
        <= gate["downstream_plural_l1_max"]
        and downstream["outside_support_mass_max"]
        <= gate["downstream_outside_support_mass_max"]
        and downstream["outside_support_invariance_max_delta"]
        <= gate[
            "downstream_outside_support_invariance_max_delta"
        ]
    )


def _score(
    metrics: dict,
) -> tuple[float, ...]:
    operator = metrics["operator"]
    downstream = metrics["downstream"]

    return (
        downstream["family_min_success"],
        downstream["row_success_accuracy"],
        operator["full_operator_exact_accuracy"],
        operator["family_min_success"],
        operator["query_view_pair_consistency"],
        -downstream["plural_l1"],
    )


def save_checkpoint(
    model: QSRET2SchemaOrderedOperatorEncoder,
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


def _load_executor(
    checkpoint: Path,
    model_contract: dict,
    device: torch.device,
) -> QSRET1Executor:
    model = QSRET1Executor(
        QSRET1Config(
            **model_contract
        )
    )
    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location="cpu",
        ),
        strict=True,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model.to(device).eval()


def self_test() -> None:
    torch.manual_seed(11)

    operator_model = QSRET2SchemaOrderedOperatorEncoder(
        QSRET2SchemaOrderedConfig(
            semantic_dim=32,
            model_dim=64,
            num_hidden_states=5,
            num_attention_heads=4,
            refinement_layers=1,
            num_relations=6,
            num_roles=4,
            num_operations=5,
            num_controls=3,
            max_relation_steps=2,
            executor_context_dim=4,
            dropout=0.0,
        ),
        relation_schema_hidden_states=torch.randn(6, 5, 32),
    )

    batch = 4
    output = operator_model(
        query_hidden_states=torch.randn(
            batch,
            5,
            9,
            32,
        ),
        query_token_mask=torch.ones(
            batch,
            9,
            dtype=torch.bool,
        ),
    )

    split = {
        "relation_target": torch.tensor(
            [
                [0, 6],
                [0, 6],
                [4, 5],
                [4, 5],
            ]
        ),
        "role_target": torch.tensor(
            [0, 0, 1, 1]
        ),
        "operation_target": torch.tensor(
            [0, 0, 1, 1]
        ),
        "control_target": torch.tensor(
            [1, 1, 1, 1]
        ),
    }

    loss, parts = operator_loss(
        output,
        split,
        torch.arange(batch),
        weights={
            "relation": 1.0,
            "role": 1.0,
            "operation": 1.0,
            "control": 1.0,
            "continuous_supcon": 0.1,
        },
        temperature=0.1,
    )
    loss.backward()

    if not torch.isfinite(loss):
        raise RuntimeError(
            "self-test loss is not finite"
        )
    if parts["continuous_supcon"] <= 0:
        raise RuntimeError(
            "continuous operator loss did not activate"
        )
    if (
        operator_model
        .continuous_projection
        .weight
        .grad
        is None
    ):
        raise RuntimeError(
            "continuous residual projection received no gradient"
        )

    bridge_operator = QSRET1OracleOperator(
        relation_sequence_id=torch.zeros(2, 2, dtype=torch.long),
        relation_sequence_mask=torch.ones(2, 2, dtype=torch.bool),
        role_id=torch.zeros(2, dtype=torch.long),
        operation_id=torch.full(
            (2,),
            QSRE_T1_OPERATION_PATH_FOLLOW,
            dtype=torch.long,
        ),
        focus_field_weight=torch.zeros(2, 4),
        context=torch.zeros(2, 4),
        applicability=torch.tensor([0.9, 0.1]),
    )
    safe_operator, invalid = _safe_operator_for_frozen_t1_eval(
        bridge_operator,
        predicted_control=torch.tensor(
            [CONTROL_RELATIONAL, 0],
            dtype=torch.long,
        ),
    )
    if safe_operator.operation_id.tolist() != [
        QSRE_T1_OPERATION_ROLE_SELECT,
        QSRE_T1_OPERATION_ROLE_SELECT,
    ]:
        raise RuntimeError(
            "evaluator bridge did not canonicalize impossible path operations"
        )
    if invalid.tolist() != [True, False]:
        raise RuntimeError(
            "evaluator bridge invalid-relational classification drift"
        )

    print(
        "PASS_QSRE_T2_TRAINER_SELF_TEST"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-test",
        action="store_true",
    )
    parser.add_argument("--contract")
    parser.add_argument("--prepared-cache")
    parser.add_argument("--semantic-config")
    parser.add_argument("--semantic-checkpoint")
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--t1-checkpoint")
    parser.add_argument("--relation-schema")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    required = {
        "contract": args.contract,
        "prepared_cache": args.prepared_cache,
        "semantic_config": args.semantic_config,
        "semantic_checkpoint": args.semantic_checkpoint,
        "tokenizer_dir": args.tokenizer_dir,
        "t1_checkpoint": args.t1_checkpoint,
        "relation_schema": args.relation_schema,
        "output_dir": args.output_dir,
    }
    missing_args = [
        name
        for name, value in required.items()
        if not value
    ]
    if missing_args:
        raise SystemExit(
            f"missing required arguments: {missing_args}"
        )

    contract_path = Path(
        args.contract
    ).resolve()
    prepared_path = Path(
        args.prepared_cache
    ).resolve()
    semantic_config_path = Path(
        args.semantic_config
    ).resolve()
    semantic_checkpoint = Path(
        args.semantic_checkpoint
    ).resolve()
    tokenizer_dir = Path(
        args.tokenizer_dir
    ).resolve()
    t1_checkpoint = Path(
        args.t1_checkpoint
    ).resolve()
    relation_schema_path = Path(
        args.relation_schema
    ).resolve()
    output_dir = Path(
        args.output_dir
    ).resolve()

    contract = json.loads(
        contract_path.read_text(
            encoding="utf-8"
        )
    )

    allowed_contract_schemas = {
        "alice.eipm.n0.qsre-t2-training-contract.v0.3",
    }
    if contract.get("schema") not in allowed_contract_schemas:
        raise SystemExit(
            "T2 training contract drift"
        )
    if (
        sha256(relation_schema_path)
        != contract["relation_schema_sha256"]
    ):
        raise SystemExit(
            "T2 relation-schema hash drift"
        )

    if (
        contract.get(
            "operator_architecture"
        )
        != "schema_grounded_ordered_relation_v0.3"
    ):
        raise SystemExit(
            "T2 v0.3 operator architecture drift"
        )

    if (
        contract.get(
            "gpu_training_authorized"
        )
        is not True
    ):
        raise SystemExit(
            "T2 GPU authorization missing"
        )
    if (
        contract.get("max_gpu_runs")
        != 1
    ):
        raise SystemExit(
            "T2 one-shot run-count drift"
        )

    for key in (
        "automatic_rerun",
        "automatic_hotfix",
        "causal_test_open",
        "frozen_challenge_open",
        "semantic_backbone_gradient",
        "t1_executor_gradient",
        "learned_support",
        "private_identity_gradient",
    ):
        if contract.get(key) is not False:
            raise SystemExit(
                f"T2 boundary drift: {key}"
            )

    if not torch.cuda.is_available():
        raise SystemExit(
            "governed T2 run requires CUDA"
        )

    if output_dir.exists():
        raise SystemExit(
            "refusing to overwrite "
            f"T2 training output: {output_dir}"
        )

    if (
        sha256(prepared_path)
        != contract[
            "expected_prepared_cache_sha256"
        ]
    ):
        raise SystemExit(
            "T2 prepared-cache hash drift"
        )

    if (
        sha256(semantic_checkpoint)
        != contract[
            "semantic_checkpoint_sha256"
        ]
    ):
        raise SystemExit(
            "semantic checkpoint hash drift"
        )

    if (
        sha256(t1_checkpoint)
        != contract[
            "t1_checkpoint_sha256"
        ]
    ):
        raise SystemExit(
            "T1 checkpoint hash drift"
        )

    prepared = torch.load(
        prepared_path,
        map_location="cpu",
    )

    if (
        prepared.get("schema")
        != EXPECTED_PREPARED_SCHEMA
    ):
        raise SystemExit(
            "T2 prepared cache schema drift"
        )

    if (
        prepared.get("curriculum_sha256")
        != contract["curriculum_sha256"]
    ):
        raise SystemExit(
            "T2 prepared curriculum lineage drift"
        )

    if (
        prepared.get("test_present")
        is not False
    ):
        raise SystemExit(
            "T2 prepared cache contains TEST"
        )

    if (
        prepared.get(
            "private_identity_data"
        )
        is not False
    ):
        raise SystemExit(
            "T2 prepared cache contains private identity data"
        )

    device = torch.device("cuda")
    output_dir.mkdir(
        parents=True
    )

    hidden_cache_path = (
        output_dir
        / "hidden-cache-v0.1.pt"
    )

    seed = int(
        contract["training"]["seed"]
    )
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    hidden_cache = materialize_hidden_cache(
        prepared=prepared,
        semantic_config_path=semantic_config_path,
        semantic_checkpoint=semantic_checkpoint,
        tokenizer_dir=tokenizer_dir,
        relation_schema_path=relation_schema_path,
        output_path=hidden_cache_path,
        device=device,
        batch_size=int(
            contract["training"][
                "semantic_cache_batch_size"
            ]
        ),
    )

    hidden_cache_sha = sha256(
        hidden_cache_path
    )

    if (
        hidden_cache["schema"]
        != EXPECTED_HIDDEN_SCHEMA
    ):
        raise SystemExit(
            "hidden-cache schema drift"
        )

    if (
        hidden_cache[
            "semantic_checkpoint_sha256"
        ]
        != contract[
            "semantic_checkpoint_sha256"
        ]
    ):
        raise SystemExit(
            "hidden-cache semantic lineage drift"
        )

    if (
        hidden_cache["relation_schema_sha256"]
        != contract["relation_schema_sha256"]
    ):
        raise SystemExit(
            "hidden-cache relation-schema lineage drift"
        )

    if (
        hidden_cache["test_present"]
        is not False
    ):
        raise SystemExit(
            "hidden cache contains TEST"
        )

    if (
        hidden_cache[
            "private_identity_data"
        ]
        is not False
    ):
        raise SystemExit(
            "hidden cache contains private identity data"
        )

    executor = _load_executor(
        t1_checkpoint,
        contract[
            "t1_executor_model"
        ],
        device,
    )

    operator_model = QSRET2SchemaOrderedOperatorEncoder(
        QSRET2SchemaOrderedConfig(
            **contract[
                "operator_model"
            ]
        ),
        relation_schema_hidden_states=hidden_cache[
            "relation_schema_hidden_states"
        ],
    ).to(device)

    optimizer = torch.optim.AdamW(
        operator_model.parameters(),
        lr=float(
            contract[
                "optimizer"
            ]["learning_rate"]
        ),
        weight_decay=float(
            contract[
                "optimizer"
            ]["weight_decay"]
        ),
    )

    training = contract["training"]
    batch_size = int(
        training["batch_size"]
    )
    max_steps = int(
        training["max_steps"]
    )
    eval_every = int(
        training[
            "eval_every_steps"
        ]
    )
    gradient_clip = float(
        contract[
            "optimizer"
        ][
            "gradient_clip_norm"
        ]
    )
    loss_weights = contract[
        "loss_weights"
    ]
    supcon_temperature = float(
        contract[
            "continuous_operator_objective"
        ][
            "temperature"
        ]
    )
    plural_threshold = float(
        contract[
            "eligibility"
        ][
            "downstream_plural_l1_max"
        ]
    )

    train_split = prepared["train"]
    dev_split = prepared["dev"]
    train_hidden = hidden_cache["train"]
    dev_hidden = hidden_cache["dev"]

    checkpoints: dict[str, dict] = {}

    step0 = evaluate(
        operator_model=operator_model,
        executor=executor,
        prepared_split=dev_split,
        hidden_split=dev_hidden,
        device=device,
        batch_size=batch_size,
        plural_l1_threshold=plural_threshold,
    )

    checkpoints[
        "step-00000000"
    ] = {
        "dev": step0,
        "eligible": False,
    }

    print(
        "step=0 dev="
        + json.dumps(
            step0,
            sort_keys=True,
        )
    )

    selected_step: int | None = None
    best_score = _score(step0)
    best_step = 0

    best_path = (
        output_dir
        / "best-observed"
        / "qsre_t2_operator.pt"
    )

    save_checkpoint(
        operator_model,
        best_path,
    )

    step = 0
    epoch = 0
    last_loss: dict[
        str,
        float,
    ] | None = None

    while (
        step < max_steps
        and selected_step is None
    ):
        batches = _paired_batches(
            train_split[
                "source_t1_row_ids"
            ],
            batch_size=batch_size,
            seed=seed + epoch,
        )
        epoch += 1

        for indices in batches:
            if (
                step >= max_steps
                or selected_step
                is not None
            ):
                break

            operator_model.train()
            optimizer.zero_grad(
                set_to_none=True
            )

            output = operator_model(
                query_hidden_states=train_hidden[
                    "query_hidden_states"
                ][indices]
                .float()
                .to(device),
                query_token_mask=train_hidden[
                    "query_token_mask"
                ][indices].to(device),
            )

            loss, last_loss = operator_loss(
                output,
                train_split,
                indices,
                weights=loss_weights,
                temperature=supcon_temperature,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                operator_model.parameters(),
                max_norm=gradient_clip,
            )

            optimizer.step()
            step += 1

            if (
                step % eval_every == 0
                or step == max_steps
            ):
                metrics = evaluate(
                    operator_model=operator_model,
                    executor=executor,
                    prepared_split=dev_split,
                    hidden_split=dev_hidden,
                    device=device,
                    batch_size=batch_size,
                    plural_l1_threshold=plural_threshold,
                )

                eligible = is_eligible(
                    metrics,
                    contract,
                )

                key = (
                    f"step-{step:08d}"
                )

                checkpoints[key] = {
                    "dev": metrics,
                    "eligible": eligible,
                    "last_train_loss": (
                        last_loss
                    ),
                }

                print(
                    f"step={step} "
                    f"eligible={str(eligible).lower()} "
                    + "dev="
                    + json.dumps(
                        metrics,
                        sort_keys=True,
                    )
                )

                score = _score(
                    metrics
                )

                if score > best_score:
                    best_score = score
                    best_step = step

                    save_checkpoint(
                        operator_model,
                        best_path,
                    )

                if eligible:
                    selected_step = step

                    save_checkpoint(
                        operator_model,
                        output_dir
                        / key
                        / "qsre_t2_operator.pt",
                    )
                    break

    if selected_step is None:
        final_key = (
            f"step-{step:08d}"
        )

        save_checkpoint(
            operator_model,
            output_dir
            / final_key
            / "qsre_t2_operator.pt",
        )

    selected_path = (
        output_dir
        / f"step-{selected_step:08d}"
        / "qsre_t2_operator.pt"
        if selected_step
        is not None
        else None
    )

    result = {
        "schema": (
            "alice.eipm.n0."
            "qsre-t2-training-result.v0.3"
        ),
        "status": (
            "PASS_QSRE_T2_OPERATOR_DEV_CONTRACT"
            if selected_step is not None
            else "FAIL_QSRE_T2_OPERATOR_DEV_CONTRACT"
        ),
        "prepared_cache_sha256": (
            sha256(prepared_path)
        ),
        "curriculum_sha256": (
            prepared[
                "curriculum_sha256"
            ]
        ),
        "semantic_checkpoint_sha256": (
            sha256(
                semantic_checkpoint
            )
        ),
        "relation_schema_sha256": (
            sha256(
                relation_schema_path
            )
        ),
        "t1_checkpoint_sha256": (
            sha256(
                t1_checkpoint
            )
        ),
        "hidden_cache_sha256": (
            hidden_cache_sha
        ),
        "selected_checkpoint_step": (
            selected_step
        ),
        "selected_checkpoint_sha256": (
            sha256(
                selected_path
            )
            if selected_path
            is not None
            else None
        ),
        "best_observed_step": (
            best_step
        ),
        "best_observed_checkpoint_sha256": (
            sha256(
                best_path
            )
        ),
        "checkpoints": checkpoints,
        "causal_test_opened": False,
        "frozen_challenge_opened": False,
        "support_learning_opened": False,
        "semantic_backbone_gradient": False,
        "t1_executor_gradient": False,
        "private_identity_gradient": False,
        "automatic_rerun_authorized": False,
        "automatic_hotfix_authorized": False,
        "n0_complete": False,
    }

    result_path = (
        output_dir
        / "result.json"
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
        + sha256(
            result_path
        )
    )

    print(
        "status="
        + result["status"]
    )


if __name__ == "__main__":
    main()
