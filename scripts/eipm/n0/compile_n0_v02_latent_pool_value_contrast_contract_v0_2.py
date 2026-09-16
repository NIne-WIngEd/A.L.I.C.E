#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import build_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_2 as builder


EXPECTED_COUNTERFACTUAL_FAMILIES = {
    "novel_semantic_authority": 0,
    "novel_structured_authority": 1,
    "missing_structured_evidence": 2,
    "semantic_reliability_reversal": 0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _source_text(row: dict[str, Any], view: int) -> str:
    if view == 0:
        return str(row["raw_text"])
    if view in {1, 2}:
        return "\n".join(str(field["text"]) for field in row["fields"])
    raise ValueError(f"unsupported source view {view}")


def _domain_values(row: dict[str, Any]) -> tuple[str, str, str, str]:
    family = str(row["family"])
    row_index = int(str(row["id"]).rsplit("-", 1)[1]) - 1
    family_index = builder.prior.base.FAMILIES.index(family)
    builder.prior.base.SUBJECTS = builder.NEW_SUBJECTS
    builder.prior.base.ATTRIBUTES = builder.NEW_ATTRIBUTES
    return builder.prior.base.parts(row_index, family_index)


def compile_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    required_ids = {str(row["id"]) for row in rows if bool(row.get("counterfactual_required"))}

    for row in rows:
        if not bool(row.get("counterfactual_required")):
            continue
        row_id = str(row["id"])
        family = str(row["family"])
        if family not in EXPECTED_COUNTERFACTUAL_FAMILIES:
            raise ValueError(f"unexpected counterfactual family in contrast contract: {family}")
        expected_view = EXPECTED_COUNTERFACTUAL_FAMILIES[family]
        actual_view = int(row["counterfactual_view"])
        if actual_view != expected_view:
            raise ValueError(
                f"counterfactual view drift for {row_id}: expected {expected_view}, got {actual_view}"
            )
        available = [bool(value) for value in row["view_available"]]
        if not available[actual_view]:
            raise ValueError(f"required contrast source is unavailable before intervention for {row_id}")

        subject, attribute, previous, current = _domain_values(row)
        target_text = str(row["target_summary_text"])

        # These four families all ask for the authoritative current value by
        # construction. This is an explicit family contract, not a global rule
        # applied to historical/current/uncertainty families.
        if target_text.count(current) != 1:
            raise ValueError(
                f"eligible target must contain its explicit current value exactly once: {row_id} value={current!r} target={target_text!r}"
            )
        foil_text = target_text.replace(current, previous, 1)
        if foil_text == target_text or previous not in foil_text:
            raise ValueError(f"failed to materialize matched foil for {row_id}")

        required_source_text = _source_text(row, actual_view)
        if current not in required_source_text:
            raise ValueError(
                f"required source does not explicitly contain target value for {row_id}: {current!r}"
            )
        all_input_text = str(row["raw_text"]) + "\n" + "\n".join(
            str(field["text"]) for field in row["fields"]
        )
        foil_present = previous in all_input_text
        foil_grounding = (
            "observed_competing_value_in_input"
            if foil_present
            else "controlled_same_attribute_domain_countervalue"
        )

        cases.append(
            {
                "row_id": row_id,
                "family": family,
                "counterfactual_view": actual_view,
                "query_answer_role": "authoritative_current_value",
                "target_value": current,
                "target_value_role": "authoritative_current",
                "foil_value": previous,
                "foil_value_role": "matched_previous_or_lower_authority_alternative",
                "target_text": target_text,
                "foil_text": foil_text,
                "subject": subject,
                "attribute": attribute,
                "foil_grounding": foil_grounding,
                "foil_present_in_input": foil_present,
                "required_source_contains_target_value": True,
                "contrast_scope": "diagnostic_only_no_ratification_effect",
            }
        )

    case_ids = {case["row_id"] for case in cases}
    if case_ids != required_ids:
        missing = sorted(required_ids - case_ids)
        extra = sorted(case_ids - required_ids)
        raise ValueError(f"contrast contract does not exactly cover required rows: missing={missing[:5]} extra={extra[:5]}")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    challenge = Path(args.challenge).resolve()
    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    rows = read_jsonl(challenge)
    if len(rows) != 160:
        raise SystemExit(f"expected frozen v0.2 challenge with 160 rows, got {len(rows)}")

    cases = compile_cases(rows)
    counts = Counter(case["family"] for case in cases)
    if len(cases) != 32 or set(counts) != set(EXPECTED_COUNTERFACTUAL_FAMILIES):
        raise SystemExit(f"contrast case coverage drift: rows={len(cases)} families={sorted(counts)}")
    if any(value != 8 for value in counts.values()):
        raise SystemExit(f"expected 8 contrast rows per family, got {dict(counts)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(case, sort_keys=True) + "\n" for case in cases), encoding="utf-8")
    manifest = {
        "schema": "alice.eipm.n0.v02-latent-pool-value-contrast-contract.v0.2",
        "status": "COMPILED_DIAGNOSTIC_CONTRACT_NOT_EVALUATED",
        "challenge_sha256": sha256_file(challenge),
        "contrast_contract_sha256": sha256_file(output),
        "rows": len(cases),
        "families": dict(sorted(counts.items())),
        "selection": "only_rows_with_counterfactual_required_true",
        "query_conditioned_answer_semantics": True,
        "global_current_value_assumption": False,
        "historical_query_rows_included": False,
        "uncertainty_rows_included": False,
        "target_and_foil_materialized_before_gpu": True,
        "challenge_rows_used_for_training": False,
        "gradient_performed": False,
        "ratification_effect": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
