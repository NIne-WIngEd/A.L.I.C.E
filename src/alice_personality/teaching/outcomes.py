"""Root/owner explicit review of actual construction outcomes for N3.

This does not produce gold automatically, certify independence, or replace
the pinned review/authorization and allocation checks in the actual trainer.
"""
from hashlib import sha256
import re

from .cases import NamedTarget, OUTCOME_CASE_SCHEMA, ReviewedTeachingCase
from .contracts import TeachingError, canonical

_SHA = re.compile(r"[0-9a-f]{64}")


def validate_outcome_binding(binding, document):
    fields = {"response_receipt_sha256", "model_state_sha256", "document_sha256", "checkpoint_manifest_sha256",
              "response_text_sha256", "actual_candidate_ids", "rendering_status", "review_scope",
              "owner_fidelity_approved", "spoken_fidelity_verified"}
    if type(binding) is not dict or set(binding) != fields:
        raise TeachingError("actual outcome needs complete response/model/document binding")
    for key in ("response_receipt_sha256", "model_state_sha256", "document_sha256"):
        if not isinstance(binding[key], str) or not _SHA.fullmatch(binding[key]):
            raise TeachingError("actual outcome needs exact response/model/document hashes")
    if binding["document_sha256"] != document.sha256:
        raise TeachingError("actual outcome document differs")
    if (type(binding["checkpoint_manifest_sha256"]) is not list or not binding["checkpoint_manifest_sha256"]
            or any(not isinstance(x, str) or not _SHA.fullmatch(x) for x in binding["checkpoint_manifest_sha256"])
            or type(binding["actual_candidate_ids"]) is not list or not binding["actual_candidate_ids"]
            or any(x not in {e.candidate_id for e in document.candidates} for x in binding["actual_candidate_ids"])):
        raise TeachingError("actual response candidates/checkpoints must remain explicit")
    if binding["response_text_sha256"] is not None and (not isinstance(binding["response_text_sha256"], str)
            or not _SHA.fullmatch(binding["response_text_sha256"])):
        raise TeachingError("actual response text hash differs")
    if (binding["review_scope"] != "PRIMARY_DEVELOPMENT_TEXT" or binding["owner_fidelity_approved"] is not False
            or binding["spoken_fidelity_verified"] is not False):
        raise TeachingError("primary development review cannot manufacture owner/spoken approval")
    if binding["rendering_status"] not in {"NOT_RENDERED", "TEXT_REALIZED_DELIVERY_UNVERIFIED", "UNRESOLVED_RENDERER"}:
        raise TeachingError("primary review must preserve actual renderer uncertainty")


def calibration_case_from_review(*, response_receipt, expected_response_sha256, document, case_id, split,
                                source_class, lineage, targets, artifact_sha256, review_id, annotation_note):
    if not isinstance(response_receipt, dict):
        raise TeachingError("review needs the actual response receipt")
    body = {k: v for k, v in response_receipt.items() if k != "receipt_sha256"}
    if (response_receipt.get("schema") != "alice-personality-response-construction-v1"
            or response_receipt.get("receipt_sha256") != expected_response_sha256
            or sha256(canonical(body)).hexdigest() != expected_response_sha256):
        raise TeachingError("reviewed actual response differs from its external pin")
    if (response_receipt.get("acceptance_authority") is not False
            or response_receipt.get("owner_fidelity_approved") is not False
            or response_receipt.get("spoken_fidelity_verified") is not False
            or response_receipt.get("qualification") != "UNQUALIFIED"):
        raise TeachingError("construction response cannot manufacture fidelity approval")
    if response_receipt.get("document_sha256") != document.sha256:
        raise TeachingError("reviewed actual response document differs")
    if type(targets) is not tuple or not targets or any(type(t) is not NamedTarget for t in targets):
        raise TeachingError("N3 needs explicit reviewed outcome facts; confidence is not gold")
    actual = response_receipt.get("selected_candidate_ids", [])
    intents = {item["candidate_id"]: item for item in response_receipt.get("intents", [])}
    candidates = {e.candidate_id: e.text for e in document.candidates}
    if (not actual or any(p not in intents or intents[p]["text"] != candidates.get(p) for p in actual)
            or (response_receipt.get("response_text") is not None and
                (len(actual) != 1 or response_receipt["response_text"] != candidates[actual[0]]))):
        raise TeachingError("review needs an actual response or explicitly unresolved actual alternatives")
    for target in targets:
        for fact in target.facts:
            if fact.available and (not fact.pointers or fact.pointers[0] not in actual):
                raise TeachingError("outcome facts must address the actual response candidates")
    text = response_receipt.get("response_text")
    binding = {"response_receipt_sha256": expected_response_sha256,
        "model_state_sha256": response_receipt["model_state_sha256"], "document_sha256": document.sha256,
        "checkpoint_manifest_sha256": list(response_receipt["checkpoint_manifest_sha256"]),
        "response_text_sha256": sha256(text.encode("utf-8")).hexdigest() if text is not None else None,
        "actual_candidate_ids": list(actual), "rendering_status": response_receipt["rendering"]["status"],
        "review_scope": "PRIMARY_DEVELOPMENT_TEXT", "owner_fidelity_approved": False, "spoken_fidelity_verified": False}
    origin = "mechanical_fixture" if source_class == "public_mechanical_fixture" else "reviewed_source_case"
    case = ReviewedTeachingCase(case_id, "calibration", split, source_class, document,
        tuple(sorted({e.record.source_family_id for e in document.sources})), artifact_sha256,
        origin, review_id, annotation_note, targets, OUTCOME_CASE_SCHEMA, lineage, binding)
    case.validate()
    return case
