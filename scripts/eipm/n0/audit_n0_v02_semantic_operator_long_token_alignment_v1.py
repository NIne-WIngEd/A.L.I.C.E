#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.source_authority_v1 import require_clean_exact_revision
from alice_personality.n0.semantic_operator_evidence_targets_v1 import (
    compile_operator_evidence_targets,
)
from alice_personality.n0.v02_training import verify_tokenizer_v021


PASS="PASS_N0_SEMANTIC_OPERATOR_LONG_TOKEN_ALIGNMENT_V1"
FAIL="FAIL_N0_SEMANTIC_OPERATOR_LONG_TOKEN_ALIGNMENT_V1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str,Any]]:
    rows=[
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit("semantic long-context token audit received no rows")
    return rows


def positive_positions(tensor: Any) -> list[int]:
    import torch
    value=torch.as_tensor(tensor)
    if value.numel()==0:
        return []
    nz=value.gt(0).nonzero(as_tuple=False)
    if nz.numel()==0:
        return []
    return [int(x) for x in nz[:,-1].tolist()]


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--rows",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--contract",required=True)
    p.add_argument("--tokenizer-dir",required=True)
    p.add_argument("--source-revision",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()

    source_revision=str(args.source_revision).strip().lower()
    if len(source_revision)!=40 or any(
        ch not in "0123456789abcdef" for ch in source_revision
    ):
        raise SystemExit("source revision must be exact 40-hex git commit")
    require_clean_exact_revision(
        expected_revision=source_revision,label="P40C semantic long-token audit"
    )

    rows_path=Path(args.rows).resolve()
    manifest_path=Path(args.manifest).resolve()
    contract_path=Path(args.contract).resolve()
    tokenizer_dir=Path(args.tokenizer_dir).resolve()
    output=Path(args.output).resolve()
    if output.exists():
        raise SystemExit("refusing to overwrite semantic long token receipt")

    rows=read_jsonl(rows_path)
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    contract=json.loads(contract_path.read_text(encoding="utf-8"))
    if manifest.get("schema")!="alice.eipm.n0.semantic-operator-long-context-manifest.v1":
        raise SystemExit("semantic long-context manifest schema drift")
    if manifest.get("sha256")!=sha256(rows_path):
        raise SystemExit("semantic long-context row hash drift")
    if contract.get("schema")!="alice.eipm.n0.semantic-operator-long-context-contract.v1":
        raise SystemExit("semantic long-context contract schema drift")

    tokenizer_receipt=verify_tokenizer_v021(tokenizer_dir)
    tokenizer=load_tokenizer(tokenizer_dir)
    native=int(contract["operating_point"]["native_window_tokens"])
    required=set(map(str,contract["required_surfaces"]))
    coverage={"train":set(),"dev":set()}
    errors=[]
    row_receipts=[]

    for row in rows:
        rid=str(row.get("id","<missing>"))
        split=str(row.get("split",""))
        surface=str(row.get("long_context_surface",""))
        if split not in coverage:
            errors.append(f"{rid}: non TRAIN/DEV split")
            continue
        coverage[split].add(surface)
        if surface not in required:
            errors.append(f"{rid}: unexpected long semantic surface {surface!r}")
            continue
        if row.get("private_identity_data") is not False:
            errors.append(f"{rid}: private identity data")
            continue
        try:
            compiled=compile_operator_evidence_targets(
                row,tokenizer,max_length=None
            )
            if surface=="query":
                tensor=compiled["relation_query_evidence_target"]
                total_tokens=int(compiled["query"]["input_ids"].size(1))
            elif surface=="relation_schema":
                tensor=compiled["relation_schema_evidence_target"]
                locator=dict(row["long_context_locator"])
                indices=[
                    int(x)
                    for x in locator.get("candidate_indices",[])
                ]
                expected_indices=sorted({
                    int(span["candidate_index"])
                    for span in row["relation_schema_evidence_char_spans"]
                })
                if sorted(indices) != expected_indices:
                    raise ValueError(
                        "relation-schema locator does not cover every supervised candidate"
                    )
                if not indices:
                    raise ValueError("relation-schema long surface has no supervised candidates")
                total_tokens=min(
                    int(compiled["relation_schema"]["input_ids"][index].numel())
                    for index in indices
                )
            elif surface=="factor_schema":
                locator=dict(row["long_context_locator"])
                bank=str(locator["bank"])
                index=int(locator["candidate_index"])
                tensor=compiled["factor_schema_evidence_target"][bank][index]
                total_tokens=int(
                    compiled["factor_schemas"][bank]["input_ids"][index].numel()
                )
                step_tensor=compiled[
                    "step_factor_schema_evidence_target"
                ].get(bank)
                step_positions=(
                    [] if step_tensor is None
                    else positive_positions(step_tensor)
                )
                if not step_positions:
                    raise ValueError(
                        "long factor surface produced no positive step-factor evidence tokens"
                    )
                if min(step_positions)<=native:
                    raise ValueError(
                        "decisive step-factor evidence did not remain beyond native "
                        f"window: {min(step_positions)} <= {native}"
                    )
            else:
                raise ValueError(f"unsupported long semantic surface: {surface}")
            positions=positive_positions(tensor)
            if not positions:
                raise ValueError("long semantic surface produced no positive evidence tokens")
            if total_tokens<=native:
                raise ValueError(
                    f"long semantic surface did not cross native window: "
                    f"{total_tokens} <= {native}"
                )
            if min(positions)<=native:
                raise ValueError(
                    f"decisive evidence did not remain beyond native window: "
                    f"{min(positions)} <= {native}"
                )
            row_receipts.append({
                "id":rid,
                "split":split,
                "surface":surface,
                "total_tokens":total_tokens,
                "positive_evidence_tokens":len(positions),
                "first_positive_evidence_token":min(positions),
                "native_window_tokens":native,
            })
        except Exception as exc:
            errors.append(f"{rid}: {type(exc).__name__}: {exc}")

    for split in ("train","dev"):
        if coverage[split]!=required:
            errors.append(
                f"{split}: semantic long-token surface coverage drift "
                f"{sorted(coverage[split])}"
            )

    result={
        "schema":"alice.eipm.n0.semantic-operator-long-token-alignment-audit.v1",
        "status":PASS if not errors else FAIL,
        "source_revision":source_revision,
        "auditor_sha256":sha256(Path(__file__).resolve()),
        "rows_sha256":sha256(rows_path),
        "manifest_sha256":sha256(manifest_path),
        "contract_sha256":sha256(contract_path),
        "tokenizer_json_sha256":sha256(tokenizer_dir/"tokenizer.json"),
        "tokenizer_model_id":tokenizer_receipt.get("model_id"),
        "tokenizer_vocab_size":int(tokenizer_receipt.get("vocab_size_observed",0)),
        "native_window_tokens":native,
        "required_surfaces":sorted(required),
        "surface_coverage":{
            split:sorted(values) for split,values in coverage.items()
        },
        "row_receipts":row_receipts,
        "errors":errors,
        "private_identity_data":False,
        "gradient":False,
        "optimizer":False,
        "gpu_training_authorized":False,
        "training_authorized_by_audit":False,
        "final_results_observed":False,
        "final_opening_authorized":False,
        "n0_complete":False,
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
