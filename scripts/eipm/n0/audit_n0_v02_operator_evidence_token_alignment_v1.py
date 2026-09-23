#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.semantic_operator_evidence_targets_v1 import (
    compile_operator_evidence_targets,
)
from alice_personality.n0.v02_training import verify_tokenizer_v021


PASS = "PASS_N0_OPERATOR_EVIDENCE_TOKEN_ALIGNMENT_V1"
FAIL = "FAIL_N0_OPERATOR_EVIDENCE_TOKEN_ALIGNMENT_V1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--rows",required=True)
    p.add_argument("--tokenizer-dir",required=True)
    p.add_argument("--source-revision",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--max-length",type=int,default=512)
    args=p.parse_args()

    source_revision=str(args.source_revision).strip().lower()
    if len(source_revision)!=40 or any(
        ch not in "0123456789abcdef" for ch in source_revision
    ):
        raise SystemExit("source revision must be exact 40-hex git commit")

    rows_path=Path(args.rows).resolve()
    tokenizer_dir=Path(args.tokenizer_dir).resolve()
    output=Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    if args.max_length <= 0:
        raise SystemExit("max-length must be positive")

    receipt=verify_tokenizer_v021(tokenizer_dir)
    tokenizer=load_tokenizer(tokenizer_dir)
    rows=read_jsonl(rows_path)
    if not rows:
        raise SystemExit("operator evidence audit received no rows")

    errors=[]
    relation_steps=0
    positive_query_tokens=0
    positive_relation_schema_tokens=0
    positive_factor_schema_tokens=0
    positive_step_factor_schema_tokens=0
    max_query_tokens=0
    max_relation_schema_tokens=0
    max_factor_schema_tokens=0

    for row in rows:
        rid=str(row.get("id","<missing>"))
        try:
            compiled=compile_operator_evidence_targets(
                row,
                tokenizer,
                max_length=args.max_length,
            )
            relation_steps += int(compiled["relation_steps"])
            positive_query_tokens += int(
                compiled["relation_query_evidence_target"].sum().item()
            )
            positive_relation_schema_tokens += int(
                compiled["relation_schema_evidence_target"].sum().item()
            )
            positive_factor_schema_tokens += sum(
                int(value.sum().item())
                for value in compiled["factor_schema_evidence_target"].values()
            )
            positive_step_factor_schema_tokens += sum(
                int(value.sum().item())
                for value in compiled[
                    "step_factor_schema_evidence_target"
                ].values()
            )
            max_query_tokens=max(
                max_query_tokens,
                int(compiled["query"]["attention_mask"].sum(dim=-1).max().item()),
            )
            max_relation_schema_tokens=max(
                max_relation_schema_tokens,
                int(
                    compiled["relation_schema"]["attention_mask"]
                    .sum(dim=-1)
                    .max()
                    .item()
                ),
            )
            for encoded in compiled["factor_schemas"].values():
                max_factor_schema_tokens=max(
                    max_factor_schema_tokens,
                    int(encoded["attention_mask"].sum(dim=-1).max().item()),
                )
        except Exception as exc:
            errors.append(f"{rid}: {type(exc).__name__}: {exc}")
            if len(errors) >= 25:
                break

    if relation_steps > 0:
        if positive_query_tokens <= 0:
            errors.append("relation programs produced no positive query evidence tokens")
        if positive_relation_schema_tokens <= 0:
            errors.append("relation programs produced no positive schema evidence tokens")
        if positive_step_factor_schema_tokens <= 0:
            errors.append("relation programs produced no positive step-factor evidence tokens")
    if positive_factor_schema_tokens <= 0:
        errors.append("factor supervision produced no positive schema evidence tokens")

    result={
        "schema":"alice.eipm.n0.operator-evidence-token-alignment-audit.v1",
        "status":PASS if not errors else FAIL,
        "source_revision":source_revision,
        "rows_sha256":sha256(rows_path),
        "tokenizer_json_sha256":sha256(tokenizer_dir/"tokenizer.json"),
        "rows":len(rows),
        "relation_steps":relation_steps,
        "positive_query_tokens":positive_query_tokens,
        "positive_relation_schema_tokens":positive_relation_schema_tokens,
        "positive_factor_schema_tokens":positive_factor_schema_tokens,
        "positive_step_factor_schema_tokens":positive_step_factor_schema_tokens,
        "max_query_tokens":max_query_tokens,
        "max_relation_schema_tokens":max_relation_schema_tokens,
        "max_factor_schema_tokens":max_factor_schema_tokens,
        "max_length_operating_point":int(args.max_length),
        "tokenizer_model_id":receipt.get("model_id"),
        "tokenizer_vocab_size":int(receipt.get("vocab_size_observed",0)),
        "errors":errors,
        "private_identity_data":False,
        "gradient":False,
        "optimizer":False,
        "training_authorized_by_audit":False,
        "max_length_is_product_ceiling":False,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__=="__main__":
    main()
