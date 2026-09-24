#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.source_authority_v1 import require_clean_exact_revision

PASS="PASS_N0_FULL_ENVELOPE_LONG_CONTEXT_TOKEN_BOUNDARY_ALIGNMENT_V1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str,Any]]:
    rows=[
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit("long-context rows are empty")
    return rows


def semantic_text(row: dict[str,Any]) -> str:
    locator=dict(row["long_context_locator"])
    kind=str(locator["kind"])
    index=int(locator.get("index",0))
    if kind=="query":
        return str(row["query"])
    if kind=="relation_schema":
        return str(row["relation_candidates"][index]["text"])
    if kind=="factor_schema":
        return str(row["factor_schemas"][str(locator["bank"])][index]["text"])
    if kind=="type_schema":
        return str(row["type_schema"][index]["text"])
    if kind=="field_text":
        return str(row["fields"][index]["text"])
    if kind=="field_descriptor":
        return str(row["fields"][index]["descriptors"][str(locator["descriptor"])])
    if kind=="candidate_text":
        return str(row["candidate_answers"][index])
    if kind=="internal_view_descriptor":
        return str(row["internal_view_descriptions"][index])
    if kind=="additional_view_descriptor":
        return str(row["additional_views"][index]["descriptor_text"])
    if kind=="additional_view_source":
        return str(row["additional_views"][index]["source_text"])
    raise ValueError(f"unsupported long-context locator kind: {kind}")


def word_span_to_char_span(text: str, start_word: int, end_word: int) -> tuple[int,int]:
    words=text.split()
    if not (0 <= start_word < end_word <= len(words)):
        raise ValueError("declared decisive word span outside semantic text")
    prefix=" ".join(words[:start_word])
    selected=" ".join(words[start_word:end_word])
    char_start=len(prefix)+(1 if prefix else 0)
    char_end=char_start+len(selected)
    if text[char_start:char_end] != selected:
        raise ValueError("word-to-character span reconstruction drift")
    return char_start,char_end


def decisive_token_span(tokenizer: Any, text: str, char_start: int, char_end: int) -> tuple[int,int,int]:
    encoded=tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=True,
    )
    offsets=encoded["offset_mapping"]
    if offsets and isinstance(offsets[0],list):
        offsets=offsets[0]
    owned=[
        i for i,(start,end) in enumerate(offsets)
        if int(end)>char_start and int(start)<char_end and int(end)>int(start)
    ]
    if not owned:
        raise ValueError("governed tokenizer produced no decisive tokens")
    return min(owned),max(owned)+1,len(encoded["input_ids"])


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
        expected_revision=source_revision,label="P40B long-context boundary audit"
    )

    rows_path=Path(args.rows)
    manifest_path=Path(args.manifest)
    contract_path=Path(args.contract)
    output=Path(args.output)
    if output.exists():
        raise SystemExit("refusing to overwrite tokenizer boundary receipt")

    rows=read_jsonl(rows_path)
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    contract=json.loads(contract_path.read_text(encoding="utf-8"))
    tokenizer_dir=Path(args.tokenizer_dir).resolve()
    tokenizer=load_tokenizer(tokenizer_dir)

    if manifest.get("sha256")!=sha256(rows_path):
        raise SystemExit("long-context row hash drift")
    boundary=contract["boundary_shift_robustness"]
    native=int(contract["operating_point"]["native_window_tokens"])
    overlap=int(contract["operating_point"]["overlap_tokens"])
    if not (0 <= overlap < native):
        raise SystemExit("invalid governed overlap operating point")
    first_stride=native-overlap
    first_ownership_boundary=(first_stride+native)//2

    grouped: dict[tuple[str,str,str],dict[str,dict[str,Any]]] = defaultdict(dict)
    errors=[]
    token_rows=[]
    for row in rows:
        variant=str(row.get("long_context_placement_variant",""))
        if variant not in {"boundary_early","boundary_late"}:
            continue
        split=str(row.get("split",""))
        surface=str(row.get("long_context_surface",""))
        pair_id=str(row.get("boundary_shift_pair_id") or "")
        text=semantic_text(row)
        start_word=row.get("decisive_start_word")
        end_word=row.get("decisive_end_word")
        if not isinstance(start_word,int) or not isinstance(end_word,int):
            errors.append(f"{row.get('id')}: decisive word span missing")
            continue
        try:
            char_start,char_end=word_span_to_char_span(
                text,start_word,end_word
            )
            token_start,token_end,total_tokens=decisive_token_span(
                tokenizer,text,char_start,char_end
            )
        except Exception as exc:
            errors.append(f"{row.get('id')}: {exc}")
            continue
        record={
            "id":str(row.get("id")),
            "split":split,
            "surface":surface,
            "pair_id":pair_id,
            "variant":variant,
            "decisive_token_start":token_start,
            "decisive_token_end":token_end,
            "total_tokens":total_tokens,
        }
        token_rows.append(record)
        grouped[(split,surface,pair_id)][variant]=record

    required=set(map(str,boundary["surfaces"]))
    verified=0
    expected=2*len(required)
    pair_receipts=[]
    for split in ("train","dev"):
        for surface in sorted(required):
            matches=[
                value for (s,ss,_),value in grouped.items()
                if s==split and ss==surface
            ]
            if len(matches)!=1:
                errors.append(f"{split}/{surface}: expected one boundary pair")
                continue
            variants=matches[0]
            if set(variants)!={"boundary_early","boundary_late"}:
                errors.append(f"{split}/{surface}: incomplete tokenizer boundary pair")
                continue
            early=variants["boundary_early"]
            late=variants["boundary_late"]
            if early["total_tokens"] <= native or late["total_tokens"] <= native:
                errors.append(f"{split}/{surface}: pair did not cross native-window length")
                continue
            if early["decisive_token_end"] >= first_ownership_boundary:
                errors.append(
                    f"{split}/{surface}: early decisive span reaches first ownership boundary "
                    f"{early['decisive_token_end']} >= {first_ownership_boundary}"
                )
                continue
            if late["decisive_token_start"] <= first_ownership_boundary:
                errors.append(
                    f"{split}/{surface}: late decisive span did not move beyond first ownership boundary "
                    f"{late['decisive_token_start']} <= {first_ownership_boundary}"
                )
                continue
            verified+=1
            pair_receipts.append({
                "split":split,
                "surface":surface,
                "first_ownership_boundary_token":first_ownership_boundary,
                "early_decisive_token_span":[
                    early["decisive_token_start"],early["decisive_token_end"]
                ],
                "late_decisive_token_span":[
                    late["decisive_token_start"],late["decisive_token_end"]
                ],
                "early_total_tokens":early["total_tokens"],
                "late_total_tokens":late["total_tokens"],
            })

    result={
        "schema":"alice.eipm.n0.full-envelope-long-context-token-boundary-audit.v1",
        "status":PASS if not errors and verified==expected else "FAIL_N0_FULL_ENVELOPE_LONG_CONTEXT_TOKEN_BOUNDARY_ALIGNMENT_V1",
        "source_revision":source_revision,
        "errors":errors,
        "rows_sha256":sha256(rows_path),
        "manifest_sha256":sha256(manifest_path),
        "contract_sha256":sha256(contract_path),
        "tokenizer_json_sha256":sha256(tokenizer_dir/"tokenizer.json"),
        "tokenizer_vocab_size":len(tokenizer),
        "native_window_tokens":native,
        "overlap_tokens":overlap,
        "first_ownership_boundary_token":first_ownership_boundary,
        "required_surfaces":sorted(required),
        "verified_pairs":verified,
        "expected_pairs":expected,
        "pair_receipts":pair_receipts,
        "exact_untrained_output_equality_claimed":False,
        "training_authorized_by_audit":False,
        "gradient":False,
        "optimizer":False,
        "gpu_training_authorized":False,
        "final_opening_authorized":False,
        "private_identity_data":False,
        "n0_complete":False,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    if result["status"]!=PASS:
        raise SystemExit(2)


if __name__=="__main__":
    main()
