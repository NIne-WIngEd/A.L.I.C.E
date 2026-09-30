"""Compare public MFM role outputs against the pinned Gemma source.

This diagnostic accepts recorded outputs from three *actual* executions on the
same frozen, public synthetic cases: the untouched source, the assembled MFM,
and that same MFM with its formation component disabled. It never executes a
model or treats a constructed JSONL row as evidence of a trained component.

The seven-case fixture is public diagnostic material, not independent FINAL.
Even a green result here cannot qualify MFM or prove the absence of inherited
behavior. The complete source and each component still need separate runtime,
lineage, privacy, multimodal and downstream qualification receipts.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

if __package__:
    from .evaluate_base_behavior import (
        DiagnosticError, _input_bytes, _rows, load_diagnostic, score_case,
    )
    from .run_gemma4_base_behavior import _source_receipt
    from .verify_gemma4_pretrained import FILES, REPO, REVISION
else:
    from evaluate_base_behavior import (
        DiagnosticError, _input_bytes, _rows, load_diagnostic, score_case,
    )
    from run_gemma4_base_behavior import _source_receipt
    from verify_gemma4_pretrained import FILES, REPO, REVISION


SCHEMA = "mfm-v1-role-boundary-diagnostic-v1"
SOURCE_SHA256 = FILES["model.safetensors"][1]
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
ROLES = ("untouched", "assembled", "ablated")


def _controls(cases: dict, runs: dict[str, dict[str, dict]],
              source_receipt_sha256: str) -> dict[str, dict]:
    if not isinstance(source_receipt_sha256, str) or not _HEX64.fullmatch(source_receipt_sha256):
        raise DiagnosticError("source receipt digest is missing or malformed")
    expected_input = sha256(_input_bytes(cases)).hexdigest()
    controls = {}
    for role in ROLES:
        rows = runs[role]
        if set(rows) != set(cases):
            raise DiagnosticError(f"{role}: missing or extra diagnostic cases")
        fixed = {}
        for case_id, row in rows.items():
            if row.get("case_id") != case_id:
                raise DiagnosticError(f"{role}: case ID differs from row key")
            if row.get("prompt_set_sha256") != expected_input:
                raise DiagnosticError(f"{role}: prompt digest differs")
            if row.get("context_digest") != cases[case_id][0].context.content_digest():
                raise DiagnosticError(f"{role}: context digest differs")
            if row.get("source_repository") != REPO or row.get("source_revision") != REVISION:
                raise DiagnosticError(f"{role}: source identity differs")
            if row.get("base_source_sha256", row.get("model_artifact_digest") if role == "untouched" else None) != SOURCE_SHA256:
                raise DiagnosticError(f"{role}: source weight lineage differs")
            if role == "untouched":
                if (row.get("model_artifact_digest") != SOURCE_SHA256 or
                        row.get("model_receipt") != source_receipt_sha256):
                    raise DiagnosticError("untouched run is not bound to the verified source")
            else:
                component = row.get("formation_component_sha256")
                if not isinstance(component, str) or not _HEX64.fullmatch(component):
                    raise DiagnosticError(f"{role}: missing formation component digest")
                if row.get("formation_component_active") is not (role == "assembled"):
                    raise DiagnosticError(f"{role}: formation component state differs")
                prepared = row.get("prepared_base_sha256")
                if (not isinstance(prepared, str) or not _HEX64.fullmatch(prepared) or
                        prepared == SOURCE_SHA256 or
                        row.get("prepared_base_parent_sha256") != SOURCE_SHA256):
                    raise DiagnosticError(f"{role}: missing distinct prepared base lineage")
                base_receipt = row.get("prepared_base_receipt_sha256")
                if not isinstance(base_receipt, str) or not _HEX64.fullmatch(base_receipt):
                    raise DiagnosticError(f"{role}: missing prepared base receipt digest")
                if not isinstance(row.get("model_artifact_digest"), str) or not _HEX64.fullmatch(row["model_artifact_digest"]):
                    raise DiagnosticError(f"{role}: missing model artifact digest")
                if not isinstance(row.get("model_receipt"), str) or not _HEX64.fullmatch(row["model_receipt"]):
                    raise DiagnosticError(f"{role}: missing model receipt digest")
                for field, value in (("formation_component_sha256", component),
                                     ("prepared_base_sha256", prepared),
                                     ("prepared_base_receipt_sha256", base_receipt)):
                    fixed.setdefault(field, value)
                    if fixed[field] != value:
                        raise DiagnosticError(f"{role}: {field} changed between cases")
            generation = row.get("generation")
            if not isinstance(generation, dict) or not generation or generation.get("do_sample") is not False:
                raise DiagnosticError(f"{role}: decoding must be recorded and deterministic")
            for field in ("generation", "model_artifact_digest", "model_receipt"):
                value = generation if field == "generation" else row[field]
                fixed.setdefault(field, value)
                if fixed[field] != value:
                    raise DiagnosticError(f"{role}: {field} changed between cases")
        controls[role] = fixed
    if any(controls[role]["generation"] != controls["untouched"]["generation"]
           for role in ROLES[1:]):
        raise DiagnosticError("decoding differs between role controls")
    if controls["assembled"]["formation_component_sha256"] != controls["ablated"]["formation_component_sha256"]:
        raise DiagnosticError("ablated run references a different component")
    for field in ("prepared_base_sha256", "prepared_base_receipt_sha256"):
        if controls["assembled"][field] != controls["ablated"][field]:
            raise DiagnosticError(f"ablated run changed {field}")
    return controls


def assess(cases: dict, runs: dict[str, dict[str, dict]],
           *, source_receipt_sha256: str) -> dict:
    """Inspect recorded outputs; never infer that JSONL alone authenticates a run."""
    controls = _controls(cases, runs, source_receipt_sha256)
    results = []
    contribution_cases = []
    assembled_failures = []
    uncovered = []
    for case_id, (gold, opened, _) in cases.items():
        scores = {role: score_case(gold, opened, runs[role][case_id])
                  for role in ROLES}
        statuses = {score["status"] for score in scores.values()}
        if len(statuses) != 1:
            uncovered.append(f"{case_id}:incomparable_modality_coverage")
        elif "unexercised" in statuses:
            uncovered.append(f"{case_id}:unexercised_modality")
        else:
            assembled = scores["assembled"]
            ablated = scores["ablated"]
            for failure in assembled["critical_failures"]:
                assembled_failures.append(f"{case_id}:{failure}")
            for field in ("false_positives", "false_negatives", "disposition_false_negatives"):
                if assembled[field]:
                    assembled_failures.append(f"{case_id}:{field}:{assembled[field]}")
            # A component has a positive diagnostic contribution only if its
            # removal worsens the same source-bound, exactly matched case.
            if (len(assembled["critical_failures"]) < len(ablated["critical_failures"]) or
                    any(assembled[field] < ablated[field] for field in
                        ("false_positives", "false_negatives", "disposition_false_negatives"))):
                contribution_cases.append(case_id)
        results.append({"case_id": case_id, **scores})
    return {
        "schema": SCHEMA, "diagnostic_only": True, "qualification_claim": False,
        "source_repository": REPO, "source_revision": REVISION,
        "source_weight_sha256": SOURCE_SHA256,
        "source_receipt_sha256": source_receipt_sha256,
        "prompt_set_sha256": sha256(_input_bytes(cases)).hexdigest(),
        "fixture_sha256": sha256(Path(__file__).resolve().parents[2].joinpath(
            "tests/fixtures/mfm/base_behavior_diagnostic_v1.json").read_bytes()).hexdigest(),
        "controls": controls, "cases": results,
        "formation_contribution_cases": contribution_cases,
        "assembled_failures": assembled_failures, "coverage_gaps": uncovered,
        "role_boundary_diagnostic_pass": bool(contribution_cases) and
            not assembled_failures and not uncovered,
        "unmeasured": ["independent_FINAL", "cross_person_swap", "long_history",
                       "full_multimodal", "private_runtime_no_egress",
                       "MFM_to_Claim_to_retrieval_to_judgment", "artifact_authenticity"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path,
                        help="complete pinned eight-file publisher snapshot")
    parser.add_argument("--source-receipt", required=True, type=Path)
    for role in ROLES:
        parser.add_argument(f"--{role}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    snapshot = args.snapshot.expanduser().resolve()
    if (output.exists() or snapshot == output or snapshot in output.parents or
            output in {args.untouched.resolve(), args.assembled.resolve(),
                       args.ablated.resolve(), args.source_receipt.resolve()}):
        parser.error("output exists or overlaps an input")
    try:
        receipt = _source_receipt(args.snapshot, args.source_receipt)
        cases = load_diagnostic()
        report = assess(cases, {role: _rows(getattr(args, role), cases)
                                for role in ROLES},
                        source_receipt_sha256=receipt["receipt_sha256"])
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    except (DiagnosticError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"role-boundary diagnostic failed: {exc}\n")
    print(json.dumps({key: report[key] for key in
                      ("role_boundary_diagnostic_pass", "formation_contribution_cases",
                       "assembled_failures", "coverage_gaps", "qualification_claim")}))


if __name__ == "__main__":
    main()
