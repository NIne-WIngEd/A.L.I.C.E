from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from alice_personality.n0.qsre_t2_schema_ordered_operator import (
    QSRET2SchemaOrderedConfig,
    QSRET2SchemaOrderedOperatorEncoder,
)

CONTROL_RELATIONAL = 1
NONE_RELATION = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def family_metrics(families: list[str], success: torch.Tensor) -> dict[str, float]:
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for family, ok in zip(families, success.tolist()):
        totals[str(family)][0] += int(ok)
        totals[str(family)][1] += 1
    return {
        family: passed / total
        for family, (passed, total) in sorted(totals.items())
    }


def special_token_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, special_ids: set[int]) -> torch.Tensor:
    mask = attention_mask.bool().clone()
    for token_id in special_ids:
        mask &= input_ids.ne(int(token_id))
    empty = mask.sum(dim=-1).eq(0)
    if bool(empty.any()):
        mask[empty] = attention_mask[empty].bool()
    return mask


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--contract", required=True)
    p.add_argument("--challenge", required=True)
    p.add_argument("--semantic-config", required=True)
    p.add_argument("--semantic-checkpoint", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--operator-checkpoint", required=True)
    p.add_argument("--relation-schema", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=128)
    args = p.parse_args()

    from safetensors.torch import load_file
    from alice_personality.n0.config import load_n0_config
    from alice_personality.n0.curriculum_data import load_tokenizer
    from alice_personality.n0.v02_model import AliceN0V02Model

    contract_path = Path(args.contract)
    challenge_path = Path(args.challenge)
    semantic_checkpoint = Path(args.semantic_checkpoint)
    operator_checkpoint = Path(args.operator_checkpoint)
    relation_schema_path = Path(args.relation_schema)
    output = Path(args.output)

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != "alice.eipm.n0.qsre-t2-training-contract.v0.3":
        raise SystemExit("T2 v0.3 contract drift")
    if contract.get("operator_architecture") != "schema_grounded_ordered_relation_v0.3":
        raise SystemExit("T2 v0.3 operator architecture drift")
    if sha256(relation_schema_path) != contract["relation_schema_sha256"]:
        raise SystemExit("T2 v0.3 relation-schema hash drift")

    challenge_spec = contract["locked_operator_challenge"]
    if sha256(challenge_path) != challenge_spec["expected_sha256"]:
        raise SystemExit("locked challenge hash drift")

    rows = read_jsonl(challenge_path)
    if len(rows) != int(challenge_spec["rows"]):
        raise SystemExit("locked challenge row-count drift")
    if any(r.get("training_allowed") is not False for r in rows):
        raise SystemExit("challenge training boundary drift")
    if any(r.get("private_identity_data") is not False for r in rows):
        raise SystemExit("private identity data present")

    if sha256(semantic_checkpoint) != contract["semantic_checkpoint_sha256"]:
        raise SystemExit("semantic checkpoint hash drift")

    cfg = load_n0_config(Path(args.semantic_config))
    semantic = AliceN0V02Model(cfg)
    missing, unexpected = semantic.load_state_dict(
        load_file(str(semantic_checkpoint), device="cpu"),
        strict=False,
    )
    if missing or unexpected:
        raise SystemExit(
            f"semantic checkpoint mismatch missing={list(missing)} unexpected={list(unexpected)}"
        )
    for parameter in semantic.parameters():
        parameter.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    semantic.to(device).eval()

    operator_state = torch.load(operator_checkpoint, map_location="cpu")
    schema_buffer = operator_state.get("relation_schema_hidden_states")
    if schema_buffer is None:
        raise SystemExit("operator checkpoint missing relation schema buffer")
    operator = QSRET2SchemaOrderedOperatorEncoder(
        QSRET2SchemaOrderedConfig(**contract["operator_model"]),
        relation_schema_hidden_states=schema_buffer,
    )
    operator.load_state_dict(
        operator_state,
        strict=True,
    )
    for parameter in operator.parameters():
        parameter.requires_grad = False
    operator.to(device).eval()

    tokenizer = load_tokenizer(Path(args.tokenizer_dir))
    special_ids = set(
        int(x) for x in (getattr(tokenizer, "all_special_ids", []) or [])
    )

    relation_pred = []
    role_pred = []
    operation_pred = []
    control_pred = []
    uncertainty = []

    texts = [str(r["query_text"]) for r in rows]
    for start in range(0, len(rows), int(args.batch_size)):
        batch_text = texts[start : start + int(args.batch_size)]
        encoded = tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            max_length=int(args.max_length),
            return_tensors="pt",
        )
        ids = encoded["input_ids"].to(device)
        attn = encoded["attention_mask"].to(device)
        outputs = semantic.backbone(
            input_ids=ids,
            attention_mask=attn,
            output_hidden_states=True,
            return_dict=True,
        )
        states = outputs.hidden_states
        if states is None or len(states) != int(contract["operator_model"]["num_hidden_states"]):
            raise RuntimeError("semantic hidden-state depth drift")
        stacked = torch.stack(states, dim=1)
        token_mask = special_token_mask(
            encoded["input_ids"],
            encoded["attention_mask"],
            special_ids,
        ).to(device)

        pred = operator(
            query_hidden_states=stacked.float(),
            query_token_mask=token_mask,
        )
        relation_pred.append(pred["relation_logits"].argmax(dim=-1).cpu())
        role_pred.append(pred["role_logits"].argmax(dim=-1).cpu())
        operation_pred.append(pred["operation_logits"].argmax(dim=-1).cpu())
        control_pred.append(pred["control_logits"].argmax(dim=-1).cpu())
        uncertainty.append(pred["uncertainty"].cpu())

    relation = torch.cat(relation_pred, dim=0)
    role = torch.cat(role_pred, dim=0)
    operation = torch.cat(operation_pred, dim=0)
    control = torch.cat(control_pred, dim=0)
    uncertainty_tensor = torch.cat(uncertainty, dim=0)

    relation_target = torch.full(
        (len(rows), int(contract["operator_model"]["max_relation_steps"])),
        NONE_RELATION,
        dtype=torch.long,
    )
    role_target = torch.zeros(len(rows), dtype=torch.long)
    operation_target = torch.zeros(len(rows), dtype=torch.long)
    control_target = torch.zeros(len(rows), dtype=torch.long)

    for i, row in enumerate(rows):
        target = row["operator_target"]
        seq = [int(x) for x in target["relation_sequence_id"]]
        relation_target[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        role_target[i] = int(target["role_id"])
        operation_target[i] = int(target["operation_id"])
        control_target[i] = int(target["control_id"])

    relation_exact = relation.eq(relation_target).all(dim=-1)
    role_ok = role.eq(role_target)
    operation_ok = operation.eq(operation_target)
    control_ok = control.eq(control_target)
    relational = control_target.eq(CONTROL_RELATIONAL)

    full_operator = control_ok & relation_exact & role_ok
    if bool(relational.any()):
        full_operator[relational] &= operation_ok[relational]

    families = [str(r["family"]) for r in rows]
    fam = family_metrics(families, full_operator)

    payload = {
        "schema": "alice.eipm.n0.qsre-t2-locked-operator-challenge-result.v0.3",
        "status": "COMPLETE_QSRE_T2_V02_LOCKED_OPERATOR_CHALLENGE",
        "challenge_sha256": sha256(challenge_path),
        "operator_checkpoint_sha256": sha256(operator_checkpoint),
        "semantic_checkpoint_sha256": sha256(semantic_checkpoint),
        "relation_schema_sha256": sha256(relation_schema_path),
        "rows": len(rows),
        "relation_sequence_exact_accuracy": float(relation_exact.float().mean().item()),
        "role_accuracy": float(role_ok.float().mean().item()),
        "role_accuracy_relational": (
            float(role_ok[relational].float().mean().item())
            if bool(relational.any())
            else 1.0
        ),
        "operation_accuracy_relational": (
            float(operation_ok[relational].float().mean().item())
            if bool(relational.any())
            else 1.0
        ),
        "control_accuracy": float(control_ok.float().mean().item()),
        "full_operator_exact_accuracy": float(full_operator.float().mean().item()),
        "family_success": fam,
        "family_min_success": min(fam.values()) if fam else 1.0,
        "mean_uncertainty": float(uncertainty_tensor.mean().item()),
        "optimizer": False,
        "gradient": False,
        "semantic_backbone_gradient": False,
        "private_identity_data": False,
        "training_allowed": False,
        "downstream_t1_scored": False,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
