"""Custody-bound v1.6 specialist assessment against independently reviewed cases.

This standalone command may open FINAL payloads only when explicitly invoked by
its custodian. The trainer and development selector never import this module.
It compares a bounded one-step probe or a signed full-fit artifact to its
matched seeded control. The report does not establish model execution
authenticity, downstream memory-gate behavior, or product qualification.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from uuid import uuid4

from cognitive_kernel.canonical import (
    CognitiveKernelContractError, canonical_sha256, require_sha256,
)
from cognitive_kernel.formation_adjudication_v16 import verify_adjudicated_corpus_v16
from cognitive_kernel.formation_dataset_admission import admit_formation_corpus
from cognitive_kernel.formation_learning_v16 import (
    FULL_ROLE_DIMENSIONS, TARGET_SCHEMA_V16, _positive_dimensions,
)
from cognitive_kernel.formation_semantics_v16 import (
    FormationGoldCaseV16, OUTPUT_SCHEMA_V16, assess_formation_v16,
    bundle_v16_from_output, context_v16_from_record,
    validate_formation_grounding_v16,
)

from . import run_v16_formation_specialist as inference
from . import train_v16_formation_specialist as training


SCHEMA = "mfm-v16-specialist-custody-assessment-v1"
CONTROL_ROLES = ("trained", "seeded-untrained")


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CognitiveKernelContractError("duplicate JSON key in v1.6 assessment input")
        result[key] = value
    return result


def _bound_bytes(path: Path, expected: str, name: str) -> bytes:
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != require_sha256(expected, name):
        raise CognitiveKernelContractError(f"{name} changed from external digest")
    return raw


def _rows(raw: bytes, name: str) -> list[dict]:
    result = []
    for line in raw.splitlines():
        if not line.strip():
            raise CognitiveKernelContractError(f"{name} contains an empty line")
        row = json.loads(line, object_pairs_hook=_unique_json)
        if not isinstance(row, dict):
            raise CognitiveKernelContractError(f"{name} contains a non-object row")
        result.append(row)
    if not result:
        raise CognitiveKernelContractError(f"{name} has no cases")
    ids = [row.get("case_id") for row in result]
    if any(not isinstance(case_id, str) for case_id in ids) or len(ids) != len(set(ids)):
        raise CognitiveKernelContractError(f"{name} has missing or repeated case IDs")
    return result


def _gold_cases(admission, manifest: dict, split: str) -> dict[str, FormationGoldCaseV16]:
    """Read signed target and exact source bytes after split admission.

    FINAL is opened exclusively here, by the explicit custodian invocation.
    No training example or optimizer data structure is constructed from it.
    """
    selected = admission.development if split == "development" else admission.final_metadata
    if split == "development":
        admission.audit_handoff(
            gradient_paths=(), development_paths=tuple(
                ref.path for case in selected for ref in
                (*case.source_payloads, case.target_payload)))
    rows = {row["case_id"]: row for row in manifest["cases"]}
    result = {}
    for case in selected:
        entry = rows[case.case_id]
        source_rows = entry["sources"]
        opened = tuple((source["source_id"], _bound_bytes(
            admission.root / ref.path, ref.sha256, "reviewed source"))
            for source, ref in zip(source_rows, case.source_payloads, strict=True))
        target = json.loads(_bound_bytes(admission.root / case.target_payload.path,
                                         case.target_payload.sha256, "reviewed target"),
                            object_pairs_hook=_unique_json)
        if not isinstance(target, dict) or set(target) != {
                "schema", "context", "source_ids", "proposals", "dispositions",
                "adjudications"} or target["schema"] != TARGET_SCHEMA_V16 or \
                target["source_ids"] != [ref for ref, _ in opened]:
            raise CognitiveKernelContractError("reviewed v1.6 target fields or sources differ")
        context = context_v16_from_record(target["context"])
        if context.base.scope.host_instance_id != entry["host_family"]:
            raise CognitiveKernelContractError("reviewed target host differs from admitted case")
        bundle = bundle_v16_from_output(
            context, {"schema": OUTPUT_SCHEMA_V16,
                      "proposals": target["proposals"],
                      "dispositions": target["dispositions"]},
            artifact_sha256="0" * 64, inference_run_id="reviewed-gold")
        adjudications = target["adjudications"]
        if not isinstance(adjudications, dict) or set(adjudications) != FULL_ROLE_DIMENSIONS:
            raise CognitiveKernelContractError("reviewed target lacks ten full-role adjudications")
        positive = _positive_dimensions(bundle)
        if any(state not in {"present", "negative"} or
               (state == "present") != (name in positive)
               for name, state in adjudications.items()):
            raise CognitiveKernelContractError("reviewed target has unknown or inconsistent label")
        reviewers = tuple(sorted(review["reviewer_id"] for review in entry["reviews"]))
        gold = FormationGoldCaseV16(
            case_id=case.case_id, context=context,
            expected=bundle.proposals,
            expected_dispositions=bundle.base.dispositions,
            critical_forbidden=(),
            adjudicated_dimensions=frozenset(adjudications),
            reviewer_refs=reviewers, opened_sources=opened)
        gold.validate()
        result[case.case_id] = gold
    if not result:
        raise CognitiveKernelContractError("reviewed split is empty")
    return result


def _assert_input(rows: list[dict], gold: dict[str, FormationGoldCaseV16]) -> None:
    if {row["case_id"] for row in rows} != set(gold):
        raise CognitiveKernelContractError("inference inputs differ from reviewed case roster")
    for row in rows:
        case_id, context, opened = inference.read_input(row)
        expected = gold[case_id]
        if context.record() != expected.context.record() or opened != expected.opened_sources:
            raise CognitiveKernelContractError("inference input differs from exact reviewed evidence")


def verify_local_controls(component_dir: Path, prepared_base_dir: Path,
                          prepared_base_receipt: Path, preflight_receipt: Path, *,
                          manifest_sha256: str | None = None,
                          roster_sha256: str | None = None,
                          review_sha256: str | None = None) -> dict:
    verified = {role: inference.verify_artifacts(
        component_dir, prepared_base_dir, prepared_base_receipt,
        preflight_receipt, control=role,
        expected_manifest_sha256=manifest_sha256,
        expected_roster_sha256=roster_sha256,
        expected_review_sha256=review_sha256) for role in CONTROL_ROLES}
    (prepared, preflight, component, trained_file) = verified["trained"]
    other_prepared, other_preflight, other_component, seed_file = verified["seeded-untrained"]
    if (prepared != other_prepared or preflight != other_preflight or
            component != other_component):
        raise CognitiveKernelContractError("paired controls use different v1.6 lineages")
    seed = training.shared._read_sealed(component_dir / "seed-control.json",
                                         training.SEED_SCHEMA)
    if seed["run_manifest_sha256"] != component["run_manifest_sha256"]:
        raise CognitiveKernelContractError("paired controls use different training runs")
    weights = {"trained": training.shared._digest(trained_file),
               "seeded-untrained": training.shared._digest(seed_file)}
    if weights["trained"] == weights["seeded-untrained"]:
        raise CognitiveKernelContractError("paired specialist weights are identical")
    return {
        "component_sha256": component["record_sha256"],
        "run_manifest_sha256": component["run_manifest_sha256"],
        "prepared_base_sha256": component["prepared_base_sha256"],
        "prepared_base_parent_sha256": component["prepared_base_parent_sha256"],
        "prepared_base_receipt_sha256": prepared["receipt_sha256"],
        "source_repository": prepared["repository"],
        "source_revision": prepared["revision"],
        "max_target_tokens": preflight["max_target_tokens"],
        "preflight_sha256": preflight["record_sha256"],
        "training_input_sha256": preflight["corpus_sha256"],
        "trust_roster_sha256": component.get("trust_roster_sha256"),
        "signed_review_receipt_sha256": component.get("signed_review_receipt_sha256"),
        "weights": weights,
        "receipts": {"trained": component["record_sha256"],
                     "seeded-untrained": seed["record_sha256"]},
        "probe_only": component["probe_only"],
        "full_fit": component.get("full_fit", False),
        "optimizer_steps": component["optimizer_steps"],
    }


def _check_output_row(row: dict, case: FormationGoldCaseV16, role: str,
                      input_sha: str, lineage: dict) -> None:
    expected = {
        "case_id": case.case_id,
        "status": "fit_generated" if lineage["full_fit"] else "probe_generated",
        "context_digest": case.context.content_digest(),
        "processed_modalities": sorted({ref.modality for ref in case.context.base.evidence}),
        "model_artifact_digest": lineage["weights"][role],
        "model_receipt": lineage["receipts"][role],
        "formation_component_sha256": lineage["weights"][role],
        "control_kind": role,
        "full_fit": lineage["full_fit"],
        "probe_only": lineage["probe_only"],
        "optimizer_steps": lineage["optimizer_steps"],
        "qualified_for_product": False,
        "run_manifest_sha256": lineage["run_manifest_sha256"],
        "preflight_sha256": lineage["preflight_sha256"],
        "training_input_sha256": lineage["training_input_sha256"],
        "trust_roster_sha256": lineage["trust_roster_sha256"],
        "signed_review_receipt_sha256": lineage["signed_review_receipt_sha256"],
        "source_repository": lineage["source_repository"],
        "source_revision": lineage["source_revision"],
        "base_source_sha256": training.shared.SOURCE_WEIGHT_SHA256,
        "prepared_base_sha256": lineage["prepared_base_sha256"],
        "prepared_base_parent_sha256": lineage["prepared_base_parent_sha256"],
        "prepared_base_receipt_sha256": lineage["prepared_base_receipt_sha256"],
        "prompt_set_sha256": input_sha,
        "inference_runner_sha256": training.shared._digest(Path(inference.__file__)),
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise CognitiveKernelContractError(f"{role}: output run or artifact receipt differs")
    generation = row.get("generation")
    if not isinstance(generation, dict) or set(generation) != {
            "decoder", "do_sample", "max_new_tokens"} or generation.get("do_sample") is not False or \
            generation.get("decoder") != "v1.6-specialist-greedy-bos-eos" or \
            type(generation.get("max_new_tokens")) is not int or \
            not 1 <= generation["max_new_tokens"] <= lineage["max_target_tokens"]:
        raise CognitiveKernelContractError(f"{role}: generation settings differ")
    if not isinstance(row.get("output_text"), str) or \
            row.get("raw_output_text") != row["output_text"] or \
            not isinstance(row.get("generated_token_ids"), list) or \
            any(type(token) is not int or token < 0 for token in row["generated_token_ids"]) or \
            type(row.get("eos_observed")) is not bool:
        raise CognitiveKernelContractError(f"{role}: raw decoder output differs")
    valid_status = ("grounded_fit_proposal_only" if lineage["full_fit"] else
                    "grounded_probe_proposal_only")
    valid = row.get("validation_status") == valid_status
    invalid = row.get("validation_status") == "invalid" and \
        isinstance(row.get("validation_error"), str) and bool(row["validation_error"])
    if not (valid or invalid) or (valid and (row.get("validation_error") is not None or
                                              row["eos_observed"] is not True)):
        raise CognitiveKernelContractError(f"{role}: runner validation state differs")
    if invalid and row["eos_observed"]:
        try:
            parsed = json.loads(row["output_text"], object_pairs_hook=_unique_json)
            parsed_bundle = bundle_v16_from_output(
                case.context, parsed, artifact_sha256=lineage["weights"][role],
                inference_run_id=f"assessment-invalid-check-{case.case_id}")
            validate_formation_grounding_v16(
                case.context, parsed_bundle, case.opened_sources)
        except (ValueError, TypeError, KeyError, CognitiveKernelContractError):
            pass
        else:
            raise CognitiveKernelContractError(
                f"{role}: runner marked a grounded EOS completion invalid")


def _positive_gold_dimensions(case: FormationGoldCaseV16) -> set[str]:
    positive = set()
    for proposal in case.expected:
        if proposal.sensitivity_hint is not None:
            positive.add("sensitivity")
        if proposal.episode is not None:
            positive.add("episode")
        if proposal.relationship_counterpart_ref is not None:
            positive.add("relationship")
        if proposal.mission_target_refs:
            positive.add("mission")
        if proposal.workspace_target_refs:
            positive.add("workspace")
        base = proposal.base
        if base.kind in {"correction_request", "deletion_request", "revocation_request"}:
            positive.add("correction")
        if base.domain == "source_person" or base.kind == "source_person_evidence" or \
                base.epistemic_status == "source_person_attestation":
            positive.add("source_person")
        if base.kind == "contradiction" or base.contradicts:
            positive.add("contradiction")
        if base.kind == "outcome":
            positive.add("outcome")
    if any(d.action == "abstain" for d in case.expected_dispositions):
        positive.add("abstention")
    return positive


def assess_pair(gold: dict[str, FormationGoldCaseV16], input_sha: str,
                runs: dict[str, list[dict]], lineage: dict) -> dict:
    """Assess two observed run files; never infer execution from JSON alone."""
    if set(runs) != set(CONTROL_ROLES) or any(
            {row["case_id"] for row in runs[role]} != set(gold)
            for role in CONTROL_ROLES):
        raise CognitiveKernelContractError("paired runs have missing or extra reviewed cases")
    full_fit = lineage.get("full_fit", False)
    if type(full_fit) is not bool or (
            full_fit and (
                lineage.get("probe_only") is not False or
                type(lineage.get("optimizer_steps")) is not int or
                lineage["optimizer_steps"] < 1 or
                any(not isinstance(lineage.get(field), str) or
                    len(lineage[field]) != 64 for field in (
                        "trust_roster_sha256", "signed_review_receipt_sha256")))
            ) or (
                not full_fit and (lineage.get("probe_only") is not True or
                                  lineage.get("optimizer_steps") != 1)):
        raise CognitiveKernelContractError("assessment requires signed full fit or one-step probe")
    indexed = {role: {row["case_id"]: row for row in runs[role]}
               for role in CONTROL_ROLES}
    report = []
    improvements = []
    failures = []
    coverage = Counter({name: 0 for name in FULL_ROLE_DIMENSIONS})
    for case_id, case in gold.items():
        coverage.update(_positive_gold_dimensions(case))
        scores = {}
        for role in CONTROL_ROLES:
            row = indexed[role][case_id]
            _check_output_row(row, case, role, input_sha, lineage)
            if row["validation_status"] == "invalid":
                scores[role] = {"status": "invalid", "true_positives": 0,
                                "false_positives": 0,
                                "false_negatives": len(case.expected),
                                "disposition_false_negatives": len(case.expected_dispositions),
                                "critical_failures": ["invalid_completion"],
                                "validation_error": row["validation_error"]}
                continue
            try:
                parsed = json.loads(row["output_text"], object_pairs_hook=_unique_json)
                bundle = bundle_v16_from_output(
                    case.context, parsed, artifact_sha256=lineage["weights"][role],
                    inference_run_id=f"assessment-{role}-{case_id}")
                scored = assess_formation_v16(case, bundle)
            except (ValueError, TypeError, KeyError, CognitiveKernelContractError) as exc:
                raise CognitiveKernelContractError(
                    f"{role}: output marked grounded but cannot be assessed") from exc
            wrong_subject = any(p.base.subject_ref not in {
                item.base.subject_ref for item in case.expected}
                for p in bundle.proposals)
            missing_abstention = (any(d.action == "abstain" for d in case.expected_dispositions)
                                  and not any(d.action == "abstain" for d in bundle.base.dispositions))
            hint_mismatch = any(p.sensitivity_hint not in {
                e.sensitivity_hint for e in case.expected} for p in bundle.proposals)
            critical = [*scored.critical_failures]
            if scored.false_positives:
                critical.append("false_memory")
            if wrong_subject:
                critical.append("wrong_subject")
            if missing_abstention:
                critical.append("missing_abstention")
            if hint_mismatch:
                critical.append("sensitivity_mismatch")
            scores[role] = {
                "status": "grounded_fit_proposal_only" if full_fit else
                          "grounded_probe_proposal_only",
                "true_positives": scored.true_positives,
                "false_positives": scored.false_positives,
                "false_negatives": scored.false_negatives,
                "disposition_false_negatives": scored.disposition_false_negatives,
                "critical_failures": sorted(set(critical)),
            }
        trained, control = (scores[role] for role in CONTROL_ROLES)
        def burden(item):
            return (len(item["critical_failures"]), item["false_positives"],
                    item["false_negatives"], item["disposition_false_negatives"])
        if burden(trained) < burden(control):
            improvements.append(case_id)
        if trained["critical_failures"] or trained["false_negatives"] or \
                trained["disposition_false_negatives"]:
            failures.append(case_id)
        report.append({"case_id": case_id, **scores})
    if any(indexed["trained"][case_id]["generation"] !=
           indexed["seeded-untrained"][case_id]["generation"] for case_id in gold):
        raise CognitiveKernelContractError("matched controls used different decoder settings")
    if any(len({json.dumps(indexed[role][case_id]["generation"], sort_keys=True)
                for case_id in gold}) != 1 for role in CONTROL_ROLES):
        raise CognitiveKernelContractError("specialist decoding changed between reviewed cases")
    return {"cases": report, "trained_improvement_cases": improvements,
            "trained_failure_cases": failures,
            "observed_gold_coverage": dict(sorted(coverage.items())),
            "missing_positive_gold_dimensions": sorted(
                name for name, count in coverage.items() if count == 0),
            "assessment_mode": "signed_full_fit" if full_fit else "one_step_probe",
            "full_fit_artifact_not_supported": not full_fit,
            "inference_execution_authenticity_verified": False,
            "qualification_claim": False}


def run(args: argparse.Namespace) -> dict:
    # This command is a separate evaluation process. The trainer never opens
    # FINAL. Require explicit custodian designation before even loading it.
    if args.split not in {"development", "final"}:
        raise CognitiveKernelContractError("assessment needs development or custodian FINAL")
    if args.split == "final" and not args.final_custodian:
        raise CognitiveKernelContractError("FINAL requires explicit separate custodian mode")
    if args.split == "development" and args.final_custodian:
        raise CognitiveKernelContractError("custodian mode must name FINAL")
    # Admission hashes private train/development evidence and a FINAL custodian
    # may also open sealed payloads. Enforce the same no-egress precondition as
    # training and inference before reading either split.
    training.shared._require_private_network_isolation()
    paths = [args.manifest, args.roster, args.input_jsonl, args.trained,
             args.seeded_untrained, args.component_dir, args.prepared_base_dir,
             args.prepared_base_receipt, args.preflight_receipt]
    output = args.output.resolve()
    if output.exists() or any(output == path.resolve() or output in path.resolve().parents
                              for path in paths):
        raise CognitiveKernelContractError("assessment report overlaps an input")
    admission = admit_formation_corpus(args.manifest,
                                       expected_sha256=args.manifest_sha256)
    review = verify_adjudicated_corpus_v16(
        admission, args.manifest, args.roster,
        expected_roster_sha256=args.roster_sha256)
    if (review.get("corpus_manifest_sha256") != admission.manifest_sha256 or
            review.get("externally_pinned_roster_sha256") != args.roster_sha256 or
            review.get("rights_and_review_signatures_verified") is not True or
            review.get("final_payloads_opened") is not False):
        raise CognitiveKernelContractError("custodian signed review receipt differs")
    # Check all full-fit bindings before the custodian opens any FINAL payload.
    local_run = training.shared._read_sealed(
        args.component_dir / "run.json", training.RUN_SCHEMA)
    full_fit = local_run.get("full_fit", False)
    lineage = verify_local_controls(
        args.component_dir, args.prepared_base_dir,
        args.prepared_base_receipt, args.preflight_receipt,
        manifest_sha256=admission.manifest_sha256 if full_fit else None,
        roster_sha256=args.roster_sha256 if full_fit else None,
        review_sha256=canonical_sha256(review) if full_fit else None)
    manifest = json.loads(_bound_bytes(args.manifest, args.manifest_sha256,
                                       "corpus manifest"), object_pairs_hook=_unique_json)
    gold = _gold_cases(admission, manifest, args.split)
    input_raw = _bound_bytes(args.input_jsonl, args.input_sha256, "inference input")
    input_rows = _rows(input_raw, "inference input")
    _assert_input(input_rows, gold)
    runs = {"trained": _rows(_bound_bytes(args.trained, args.trained_sha256,
                                          "trained output"), "trained output"),
            "seeded-untrained": _rows(_bound_bytes(
                args.seeded_untrained, args.seeded_untrained_sha256,
                "seeded output"), "seeded output")}
    report = {
        "schema": SCHEMA, "split": args.split,
        "corpus_manifest_sha256": admission.manifest_sha256,
        "review_receipt": review,
        "inference_input_sha256": args.input_sha256,
        "output_run_sha256": {"trained": args.trained_sha256,
                              "seeded-untrained": args.seeded_untrained_sha256},
        "local_artifacts": lineage,
        "gold_status": "caller-pinned-signed-roster; identity-and-semantic-truth-need-steward",
        "final_custodian_invocation": args.split == "final",
        "synthetic_authoring_seed_qualifies": False,
        "limits": (["signed_full_fit_behavior_diagnostic_only"] if full_fit else
                   ["one_step_probe_only"]) + ["runner_output_does_not_attest_execution",
                   "external_reviewer_identity_and_semantic_truth_unverified",
                   "no_memory_gate_or_downstream_judgment_result"],
        **assess_pair(gold, sha256(input_raw).hexdigest(), runs, lineage),
    }
    temporary = output.with_name(f".{output.name}.partial-{uuid4().hex}")
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("development", "final"))
    parser.add_argument("--final-custodian", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--roster-sha256", required=True)
    parser.add_argument("--component-dir", type=Path, required=True)
    parser.add_argument("--prepared-base-dir", type=Path, required=True)
    parser.add_argument("--prepared-base-receipt", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--trained-sha256", required=True)
    parser.add_argument("--seeded-untrained", type=Path, required=True)
    parser.add_argument("--seeded-untrained-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run(args)
    except (CognitiveKernelContractError, OSError, ValueError, KeyError,
            json.JSONDecodeError) as exc:
        parser.exit(2, f"v1.6 assessment failed: {exc}\n")
    print(json.dumps({key: report[key] for key in
                      ("split", "trained_improvement_cases", "trained_failure_cases",
                       "qualification_claim")}, sort_keys=True))


if __name__ == "__main__":
    main()
