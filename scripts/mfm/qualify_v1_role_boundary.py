"""Score actual trained and seeded-untrained V1 formation decoder outputs.

The publisher Gemma language head is an optional descriptive reference. Its
decoder and 1024-token cap differ from the first-party formation specialist,
so only the trained and seeded-untrained runs form a matched role comparison.
Rows alone cannot authenticate model execution. A diagnostic pass requires
rehashing the local, sealed specialist and prepared-base artifacts as well.
Even then, this seven-case public fixture is not independent qualification.
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


SCHEMA = "mfm-v1-role-boundary-diagnostic-v2"
SOURCE_SHA256 = FILES["model.safetensors"][1]
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
PAIRED_ROLES = ("trained", "seeded-untrained")
REFERENCE_ROLE = "untouched"


def verify_local_lineage(component_dir: Path, prepared_base_dir: Path,
                         prepared_base_receipt: Path, preflight_receipt: Path) -> dict:
    """Rehash real trained/seed weight files and their common sealed run.

    This proves artifact custody and a shared run manifest, not that a JSONL
    completion came from executing those artifacts. The latter needs an
    independently observed inference run.
    """
    if __package__:
        from . import run_v1_formation_specialist as inference
        from . import train_v1_formation_specialist as training
    else:
        import run_v1_formation_specialist as inference
        import train_v1_formation_specialist as training

    prepared, preflight, component, trained_file = inference.verify_artifacts(
        component_dir, prepared_base_dir, prepared_base_receipt, preflight_receipt,
        control="trained")
    seed_prepared, seed_preflight, seed_component, seed_file = inference.verify_artifacts(
        component_dir, prepared_base_dir, prepared_base_receipt, preflight_receipt,
        control="seeded-untrained")
    if (prepared != seed_prepared or preflight != seed_preflight or
            component != seed_component):
        raise DiagnosticError("specialist controls do not share a prepared base and run")
    seed = training._read_sealed(component_dir / "seed-control.json",
                                 training.SEED_CONTROL_SCHEMA)
    trained_digest, seed_digest = training._digest(trained_file), training._digest(seed_file)
    if trained_digest == seed_digest:
        raise DiagnosticError("trained and seed control weight files are identical")
    if component["run_manifest_sha256"] != seed["run_manifest_sha256"]:
        raise DiagnosticError("specialist controls do not share a sealed run manifest")
    return {
        "run_manifest_sha256": component["run_manifest_sha256"],
        "prepared_base_sha256": component["prepared_base_sha256"],
        "prepared_base_receipt_sha256": prepared["receipt_sha256"],
        "prepared_base_parent_sha256": component["prepared_base_parent_sha256"],
        "max_target_tokens": preflight["max_target_tokens"],
        "trained": {"weight_sha256": trained_digest,
                    "receipt_sha256": component["record_sha256"]},
        "seeded-untrained": {"weight_sha256": seed_digest,
                             "receipt_sha256": seed["record_sha256"]},
    }


def _controls(cases: dict, runs: dict[str, dict[str, dict]],
              source_receipt_sha256: str, verified_lineage: dict | None) -> dict[str, dict]:
    if not isinstance(source_receipt_sha256, str) or not _HEX64.fullmatch(source_receipt_sha256):
        raise DiagnosticError("source receipt digest is missing or malformed")
    if not set(PAIRED_ROLES) <= set(runs) or set(runs) - {*PAIRED_ROLES, REFERENCE_ROLE}:
        raise DiagnosticError("need trained and seeded-untrained specialist runs")
    if verified_lineage is not None:
        for key in ("prepared_base_sha256", "prepared_base_receipt_sha256",
                    "prepared_base_parent_sha256", "run_manifest_sha256"):
            if not isinstance(verified_lineage.get(key), str) or not _HEX64.fullmatch(
                    verified_lineage[key]):
                raise DiagnosticError(f"verified lineage lacks {key}")
        if (type(verified_lineage.get("max_target_tokens")) is not int or
                verified_lineage["max_target_tokens"] < 1):
            raise DiagnosticError("verified lineage lacks target budget")
        for role in PAIRED_ROLES:
            binding = verified_lineage.get(role)
            if not isinstance(binding, dict) or any(
                    not isinstance(binding.get(key), str) or not _HEX64.fullmatch(binding[key])
                    for key in ("weight_sha256", "receipt_sha256")):
                raise DiagnosticError(f"verified lineage lacks {role} artifacts")
    expected_input = sha256(_input_bytes(cases)).hexdigest()
    controls: dict[str, dict] = {}
    for role, rows in runs.items():
        if set(rows) != set(cases):
            raise DiagnosticError(f"{role}: missing or extra diagnostic cases")
        fixed: dict = {}
        for case_id, row in rows.items():
            gold = cases[case_id][0]
            if row.get("case_id") != case_id:
                raise DiagnosticError(f"{role}: case ID differs from row key")
            if row.get("prompt_set_sha256") != expected_input:
                raise DiagnosticError(f"{role}: prompt digest differs")
            if row.get("context_digest") != gold.context.content_digest():
                raise DiagnosticError(f"{role}: context digest differs")
            if row.get("source_repository") != REPO or row.get("source_revision") != REVISION:
                raise DiagnosticError(f"{role}: source identity differs")
            if role == REFERENCE_ROLE:
                if (row.get("model_artifact_digest") != SOURCE_SHA256 or
                        row.get("model_receipt") != source_receipt_sha256):
                    raise DiagnosticError("untouched run is not bound to the verified source")
                if row.get("base_source_sha256", SOURCE_SHA256) != SOURCE_SHA256:
                    raise DiagnosticError("untouched source weight lineage differs")
            else:
                if row.get("base_source_sha256") != SOURCE_SHA256 or \
                        row.get("prepared_base_parent_sha256") != SOURCE_SHA256:
                    raise DiagnosticError(f"{role}: source weight lineage differs")
                for field in ("formation_component_sha256", "prepared_base_sha256",
                              "prepared_base_receipt_sha256", "model_artifact_digest",
                              "model_receipt"):
                    value = row.get(field)
                    if not isinstance(value, str) or not _HEX64.fullmatch(value):
                        raise DiagnosticError(f"{role}: missing {field}")
                    fixed.setdefault(field, value)
                    if fixed[field] != value:
                        raise DiagnosticError(f"{role}: {field} changed between cases")
                if (row["model_artifact_digest"] != row["formation_component_sha256"] or
                        row["prepared_base_receipt_sha256"] == source_receipt_sha256):
                    raise DiagnosticError(f"{role}: component or prepared-base receipt differs")
                if row.get("control_kind") != role:
                    raise DiagnosticError(f"{role}: runner control kind differs")
                if row.get("status") != "generated":
                    raise DiagnosticError(f"{role}: specialist runner must record generated output")
                if row.get("processed_modalities") != sorted({
                        ref.modality for ref in gold.context.evidence}):
                    raise DiagnosticError(f"{role}: specialist modality coverage differs")
                if (not isinstance(row.get("output_text"), str) or
                        row.get("raw_output_text") != row["output_text"] or
                        not isinstance(row.get("generated_token_ids"), list) or
                        any(type(token) is not int or token < 0
                            for token in row["generated_token_ids"]) or
                        type(row.get("eos_observed")) is not bool):
                    raise DiagnosticError(f"{role}: specialist generation record differs")
                valid = row.get("validation_status") == "grounded_proposal_only"
                invalid = (row.get("validation_status") == "invalid" and
                           isinstance(row.get("validation_error"), str) and
                           bool(row["validation_error"]))
                if not (valid or invalid) or (valid and (not row["eos_observed"] or
                                                row.get("validation_error") is not None)):
                    raise DiagnosticError(f"{role}: specialist validation record differs")
                if verified_lineage is not None:
                    binding = verified_lineage[role]
                    if (row["model_artifact_digest"] != binding["weight_sha256"] or
                            row["model_receipt"] != binding["receipt_sha256"] or
                            row["prepared_base_sha256"] != verified_lineage["prepared_base_sha256"] or
                            row["prepared_base_receipt_sha256"] != verified_lineage["prepared_base_receipt_sha256"] or
                            row["prepared_base_parent_sha256"] != verified_lineage["prepared_base_parent_sha256"]):
                        raise DiagnosticError(f"{role}: output row differs from rehashed local artifacts")
            generation = row.get("generation")
            if not isinstance(generation, dict) or generation.get("do_sample") is not False:
                raise DiagnosticError(f"{role}: decoding must be recorded and deterministic")
            if role == REFERENCE_ROLE:
                if (generation.get("num_beams") != 1 or
                        type(generation.get("max_new_tokens")) is not int or
                        not 1 <= generation["max_new_tokens"] <= 1024 or
                        generation.get("dtype") != "bfloat16"):
                    raise DiagnosticError("untouched: publisher baseline decoding differs")
            elif (generation.get("decoder") != "specialist-greedy-bos-eos-v1" or
                    type(generation.get("max_new_tokens")) is not int or
                    not 1 <= generation["max_new_tokens"] <= (
                        verified_lineage["max_target_tokens"] if verified_lineage else 1_000_000)):
                raise DiagnosticError(f"{role}: specialist decoding differs")
            for field in ("generation", "model_artifact_digest", "model_receipt"):
                value = generation if field == "generation" else row[field]
                fixed.setdefault(field, value)
                if fixed[field] != value:
                    raise DiagnosticError(f"{role}: {field} changed between cases")
        controls[role] = fixed
    if controls["trained"]["generation"] != controls["seeded-untrained"]["generation"]:
        raise DiagnosticError("decoding differs between specialist controls")
    for field in ("prepared_base_sha256", "prepared_base_receipt_sha256"):
        if controls["trained"][field] != controls["seeded-untrained"][field]:
            raise DiagnosticError(f"specialist controls changed {field}")
    if (controls["trained"]["formation_component_sha256"] ==
            controls["seeded-untrained"]["formation_component_sha256"] or
            controls["trained"]["model_receipt"] ==
            controls["seeded-untrained"]["model_receipt"]):
        raise DiagnosticError("trained and seeded controls must have distinct artifacts")
    if verified_lineage is not None and verified_lineage["prepared_base_parent_sha256"] != SOURCE_SHA256:
        raise DiagnosticError("verified prepared-base parent differs from pinned source")
    return controls


def assess(cases: dict, runs: dict[str, dict[str, dict]], *,
           source_receipt_sha256: str,
           verified_lineage: dict | None = None) -> dict:
    """Score recorded outputs; only local artifact verification can enable pass."""
    controls = _controls(cases, runs, source_receipt_sha256, verified_lineage)
    results = []
    contribution_cases = []
    trained_failures = []
    uncovered = []
    reference_gaps = []
    for case_id, (gold, opened, _) in cases.items():
        scores = {role: score_case(gold, opened, runs[role][case_id]) for role in runs}
        trained, control = scores["trained"], scores["seeded-untrained"]
        if (trained["status"] != control["status"] or
                trained["status"] != "generated"):
            uncovered.append(f"{case_id}:incomparable_specialist_coverage")
        else:
            if runs["trained"][case_id]["validation_status"] != "grounded_proposal_only":
                trained_failures.append(f"{case_id}:invalid_specialist_completion")
            for failure in trained["critical_failures"]:
                trained_failures.append(f"{case_id}:{failure}")
            for field in ("false_positives", "false_negatives", "disposition_false_negatives"):
                if trained[field]:
                    trained_failures.append(f"{case_id}:{field}:{trained[field]}")
            if (len(trained["critical_failures"]) < len(control["critical_failures"]) or
                    any(trained[field] < control[field] for field in
                        ("false_positives", "false_negatives", "disposition_false_negatives"))):
                contribution_cases.append(case_id)
        if REFERENCE_ROLE in scores and scores[REFERENCE_ROLE]["status"] == "unexercised":
            reference_gaps.append(f"{case_id}:publisher_reference_unexercised")
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
        "verified_lineage": verified_lineage,
        "artifact_binding_verified": verified_lineage is not None,
        "formation_contribution_cases": contribution_cases,
        "trained_failures": trained_failures, "coverage_gaps": uncovered,
        "reference_coverage_gaps": reference_gaps,
        "role_boundary_diagnostic_pass": verified_lineage is not None and
            bool(contribution_cases) and not trained_failures and not uncovered,
        "unmeasured": ["independent_FINAL", "cross_person_swap", "long_history",
                       "full_multimodal", "private_runtime_no_egress",
                       "MFM_to_Claim_to_retrieval_to_judgment",
                       "inference_execution_authenticity"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path,
                        help="complete pinned eight-file publisher snapshot")
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--component-dir", required=True, type=Path)
    parser.add_argument("--prepared-base-dir", required=True, type=Path)
    parser.add_argument("--prepared-base-receipt", required=True, type=Path)
    parser.add_argument("--preflight-receipt", required=True, type=Path)
    parser.add_argument("--trained", required=True, type=Path)
    parser.add_argument("--seeded-untrained", required=True, type=Path)
    parser.add_argument("--untouched", type=Path,
                        help="optional, differently decoded publisher reference")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    snapshot = args.snapshot.expanduser().resolve()
    inputs = [args.source_receipt, args.component_dir, args.prepared_base_dir,
              args.prepared_base_receipt, args.preflight_receipt, args.trained,
              args.seeded_untrained]
    if args.untouched:
        inputs.append(args.untouched)
    if output.exists() or snapshot == output or snapshot in output.parents or any(
            output == path.resolve() or path.resolve() in output.parents for path in inputs):
        parser.error("output exists or overlaps an input")
    try:
        receipt = _source_receipt(args.snapshot, args.source_receipt)
        lineage = verify_local_lineage(args.component_dir, args.prepared_base_dir,
                                       args.prepared_base_receipt, args.preflight_receipt)
        cases = load_diagnostic()
        rows = {"trained": _rows(args.trained, cases),
                "seeded-untrained": _rows(args.seeded_untrained, cases)}
        if args.untouched:
            rows[REFERENCE_ROLE] = _rows(args.untouched, cases)
        report = assess(cases, rows, source_receipt_sha256=receipt["receipt_sha256"],
                        verified_lineage=lineage)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    except (DiagnosticError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"role-boundary diagnostic failed: {exc}\n")
    print(json.dumps({key: report[key] for key in
                      ("role_boundary_diagnostic_pass", "formation_contribution_cases",
                       "trained_failures", "coverage_gaps", "reference_coverage_gaps",
                       "qualification_claim")}))


if __name__ == "__main__":
    main()
