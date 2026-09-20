from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from alice_personality.n0.qsre_t2_schema_ordered_operator import (
    QSRET2SchemaOrderedConfig,
    QSRET2SchemaOrderedOperatorEncoder,
)

from train_n0_v02_qsre_t2_schema_ordered_v0_3 import (
    STRUCTURAL_KEYS,
    _load_executor,
    _safe_operator_for_frozen_t1_eval,
    materialize_hidden_cache,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _slice_split(split: dict, count: int) -> dict:
    out = {}
    for key, value in split.items():
        if isinstance(value, torch.Tensor):
            out[key] = value[:count].clone()
        elif isinstance(value, list):
            out[key] = list(value[:count])
        else:
            out[key] = value
    return out


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--contract", required=True)
    p.add_argument("--prepared-cache", required=True)
    p.add_argument("--semantic-config", required=True)
    p.add_argument("--semantic-checkpoint", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--t1-checkpoint", required=True)
    p.add_argument("--relation-schema", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    contract_path = Path(args.contract).resolve()
    prepared_path = Path(args.prepared_cache).resolve()
    semantic_config = Path(args.semantic_config).resolve()
    semantic_checkpoint = Path(args.semantic_checkpoint).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve()
    t1_checkpoint = Path(args.t1_checkpoint).resolve()
    relation_schema = Path(args.relation_schema).resolve()
    output_dir = Path(args.output_dir).resolve()

    if output_dir.exists():
        raise SystemExit(
            f"refusing to overwrite T2 v0.3 runtime evidence: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != "alice.eipm.n0.qsre-t2-training-contract.v0.3":
        raise SystemExit("T2 v0.3 contract drift")
    if contract.get("operator_architecture") != "schema_grounded_ordered_relation_v0.3":
        raise SystemExit("T2 v0.3 architecture drift")

    checks = {
        "prepared_cache_sha256": (prepared_path, contract["expected_prepared_cache_sha256"]),
        "semantic_checkpoint_sha256": (semantic_checkpoint, contract["semantic_checkpoint_sha256"]),
        "t1_checkpoint_sha256": (t1_checkpoint, contract["t1_checkpoint_sha256"]),
        "relation_schema_sha256": (relation_schema, contract["relation_schema_sha256"]),
    }
    actual_hashes = {}
    for label, (path, expected) in checks.items():
        actual = sha256(path)
        actual_hashes[label] = actual
        if actual != expected:
            raise SystemExit(f"{label} drift: {actual} != {expected}")

    prepared = torch.load(prepared_path, map_location="cpu")
    if prepared.get("schema") != "alice.eipm.n0.qsre-t2-tokenized-preparation.v0.1":
        raise SystemExit("prepared cache schema drift")
    if prepared.get("curriculum_sha256") != contract["curriculum_sha256"]:
        raise SystemExit("prepared curriculum lineage drift")
    if prepared.get("test_present") is not False:
        raise SystemExit("TEST unexpectedly present")
    if prepared.get("private_identity_data") is not False:
        raise SystemExit("private identity data unexpectedly present")

    mini = dict(prepared)
    mini["train"] = _slice_split(prepared["train"], 4)
    mini["dev"] = _slice_split(prepared["dev"], 4)

    hidden_path = output_dir / "real-artifact-hidden-smoke.pt"
    hidden = materialize_hidden_cache(
        prepared=mini,
        semantic_config_path=semantic_config,
        semantic_checkpoint=semantic_checkpoint,
        tokenizer_dir=tokenizer_dir,
        relation_schema_path=relation_schema,
        output_path=hidden_path,
        device=torch.device("cpu"),
        batch_size=4,
    )

    if hidden.get("relation_schema_sha256") != contract["relation_schema_sha256"]:
        raise RuntimeError("runtime relation-schema lineage drift")
    schema_states = hidden["relation_schema_hidden_states"]
    if tuple(schema_states.shape) != (6, 17, 640):
        raise RuntimeError(
            f"runtime relation schema shape drift: {tuple(schema_states.shape)}"
        )

    operator = QSRET2SchemaOrderedOperatorEncoder(
        QSRET2SchemaOrderedConfig(**contract["operator_model"]),
        relation_schema_hidden_states=schema_states,
    ).eval()
    for parameter in operator.parameters():
        parameter.requires_grad = False

    executor = _load_executor(
        t1_checkpoint,
        contract["t1_executor_model"],
        torch.device("cpu"),
    )

    split = mini["dev"]
    hidden_split = hidden["dev"]
    output = operator(
        query_hidden_states=hidden_split["query_hidden_states"].float(),
        query_token_mask=hidden_split["query_token_mask"],
    )

    if tuple(output["relation_logits"].shape) != (4, 2, 7):
        raise RuntimeError("relation-logit shape drift")
    if not bool(torch.isfinite(output["relation_logits"]).all()):
        raise RuntimeError("nonfinite relation logits")

    decoded = operator.decode_for_frozen_t1(
        output,
        focus_field_weight=split["focus_field_weight"],
    )
    decoded, invalid = _safe_operator_for_frozen_t1_eval(
        decoded,
        predicted_control=output["control_logits"].argmax(dim=-1),
    )

    structural = {key: split[key] for key in STRUCTURAL_KEYS}
    downstream = executor(**structural, operator=decoded)
    for key in ("relational_probability", "control_state"):
        value = downstream[key]
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"nonfinite downstream tensor: {key}")

    report = operator.parameter_report()
    if report["learned_relation_class_anchor"] is not False:
        raise RuntimeError("opaque relation class anchor reintroduced")
    if report["schema_grounded_relation_readout"] is not True:
        raise RuntimeError("schema grounding missing")
    if report["ordered_relation_transition"] is not True:
        raise RuntimeError("ordered transition missing")
    if report["hard_layer_mask"] is not False:
        raise RuntimeError("hard layer restriction introduced")

    receipt = {
        "schema": "alice.eipm.n0.qsre-t2-v03-real-artifact-runtime-qualification.v0.1",
        "status": "PASS_QSRE_T2_V03_REAL_ARTIFACT_RUNTIME_QUALIFICATION",
        "contract_sha256": sha256(contract_path),
        **actual_hashes,
        "hidden_smoke_sha256": sha256(hidden_path),
        "rows_executed": {"train_semantic": 4, "dev_semantic_and_t1": 4},
        "relation_schema_shape": list(schema_states.shape),
        "relation_logits_shape": list(output["relation_logits"].shape),
        "invalid_relational_path_without_focus_count": int(invalid.sum().item()),
        "parameter_report": report,
        "optimizer": False,
        "gradient": False,
        "gpu": False,
        "semantic_backbone_gradient": False,
        "t1_executor_gradient": False,
        "learned_support": False,
        "test_open": False,
        "private_identity_data": False,
        "training_authorized_by_this_receipt": True,
        "max_gpu_runs": 1,
        "automatic_rerun": False,
        "automatic_hotfix": False,
    }
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
