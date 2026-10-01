"""Scoped source quotes become exact anchors, never authenticated targets."""

from __future__ import annotations

from base64 import b64encode
from hashlib import sha256
import unittest

from cognitive_kernel.formation_learning_v16 import FULL_ROLE_DIMENSIONS
from scripts.mfm.prepare_v16_role_excerpt_teacher_cases import (
    SOURCE_SCHEMA, RESPONSE_SCHEMA, _quote_anchor, _raw, convert, rendered_prompt,
    source_view,
)


def source_example() -> dict:
    raw = b'At 10:00 the assistant logged an unsupported answer.'
    return {"schema": SOURCE_SCHEMA, "case_id": "synthetic-self-case",
            "host_id": "fictional-host", "self_id": "fictional-assistant-self",
            "authority_id": "fictional-authority", "rights_status": "unverified",
            "training_admitted": False, "targets": [],
            "sources": [{"ref_id": "self-log-1", "role": "assistant_self_event",
                         "modality": "text", "subject_ref": "fictional-assistant-self",
                         "speaker_ref": "fictional-assistant-self",
                         "source_item_ref": "fictional-log-1",
                         "observed_at": "2026-01-01T10:00:00.000000Z",
                         "recorded_at": "2026-01-01T10:00:00.000000Z",
                         "temporal_granularity": "instant",
                         "minimum_sensitivity": "private",
                         "content_b64": b64encode(raw).decode("ascii")}]}


def response_example(source: dict) -> dict:
    view = _raw(source_view(source))
    return {"schema": RESPONSE_SCHEMA, "case_id": source["case_id"],
            "source_view_sha256": sha256(view).hexdigest(),
            "rendered_prompt_sha256": sha256(rendered_prompt(view)).hexdigest(),
            "proposals": [{"proposal": {
                "proposal_id": "self-proposal-1", "kind": "self_observation",
                "domain": "self", "subject_ref": source["self_id"],
                "value_ref": "self-value-1", "evidence_refs": ["self-log-1"],
                "value_text": "The assistant log recorded an unsupported answer.",
                "anchors": [{"ref_id": "self-log-1",
                             "quote": "the assistant logged an unsupported answer",
                             "locator": None}],
                "epistemic_status": "observation",
                "valid_from": "2026-01-01T10:00:00.000000Z",
                "valid_to": "2026-01-01T10:00:00.000000Z",
                "temporal_granularity": "instant", "confidence": None,
                "uncertainty_ref": None, "contradicts": [], "target_refs": [],
                "disposition_scope_ref": "self_log_scope"},
                "sensitivity_hint": "private", "episode": None,
                "relationship_counterpart_ref": None,
                "mission_target_refs": [], "workspace_target_refs": []}],
            "dispositions": [{"scope_ref": "self_log_scope", "action": "propose",
                              "evidence_refs": ["self-log-1"], "target_refs": []}],
            "adjudications": {key: ("present" if key == "sensitivity" else "negative")
                              for key in FULL_ROLE_DIMENSIONS}}


class RoleExcerptTeacherTests(unittest.TestCase):
    def test_private_self_source_converts_to_full_v16_candidate(self) -> None:
        source = source_example()
        response = response_example(source)
        case, provenance = convert(source, response,
                                   source_sha256=sha256(_raw(source)).hexdigest(),
                                   response_sha256=sha256(_raw(response)).hexdigest(),
                                   prompt_sha256=response["rendered_prompt_sha256"])
        anchor = case["target"]["proposals"][0]["proposal"]["anchors"][0]
        raw = b"At 10:00 the assistant logged an unsupported answer."
        self.assertEqual(raw[anchor["start_byte"]:anchor["end_byte"]],
                         b"the assistant logged an unsupported answer")
        self.assertEqual(case["target"]["proposals"][0]["proposal"]["domain"], "self")
        self.assertFalse(case["status"]["training_admitted"])
        self.assertFalse(provenance["semantic_entailment_reviewed_independently"])

    def test_quote_must_be_unique_exact_bytes(self) -> None:
        with self.assertRaisesRegex(ValueError, "absent or ambiguous"):
            _quote_anchor("self-log-1", b"twice twice",
                          {"ref_id": "self-log-1", "quote": "twice",
                           "locator": None}, "text")
        with self.assertRaisesRegex(ValueError, "absent or ambiguous"):
            _quote_anchor("self-log-1", "café".encode(),
                          {"ref_id": "self-log-1", "quote": "cafe",
                           "locator": None}, "text")

    def test_nontext_requires_locator_and_wrong_sensitivity_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "modality-aware verifier"):
            _quote_anchor("media-1", b"\x00\x01",
                          {"ref_id": "media-1", "quote": "heard laughter",
                           "locator": None}, "audio")
        with self.assertRaisesRegex(ValueError, "modality-aware verifier"):
            _quote_anchor("media-1", b"\x00\x01",
                          {"ref_id": "media-1", "quote": None,
                           "locator": "seconds:0-1"}, "audio")
        source = source_example()
        response = response_example(source)
        response["proposals"][0]["sensitivity_hint"] = "public"
        with self.assertRaisesRegex(Exception, "lowers source policy"):
            convert(source, response, source_sha256=sha256(_raw(source)).hexdigest(),
                    response_sha256=sha256(_raw(response)).hexdigest(),
                    prompt_sha256=response["rendered_prompt_sha256"])

    def test_response_binding_rejects_other_source_view(self) -> None:
        source = source_example()
        response = response_example(source)
        source["sources"][0]["content_b64"] = b64encode(b"A different event").decode()
        with self.assertRaisesRegex(ValueError, "bind to the exact source"):
            convert(source, response, source_sha256=sha256(_raw(source)).hexdigest(),
                    response_sha256=sha256(_raw(response)).hexdigest(),
                    prompt_sha256=response["rendered_prompt_sha256"])

    def test_fictional_source_cannot_claim_owner_authentication_or_swap_self(self) -> None:
        source = source_example()
        source["sources"][0]["role"] = "authenticated_owner_statement"
        with self.assertRaisesRegex(ValueError, "authenticated role"):
            source_view(source)
        source = source_example()
        source["sources"][0]["subject_ref"] = source["host_id"]
        with self.assertRaisesRegex(ValueError, "registered self"):
            source_view(source)


if __name__ == "__main__":
    unittest.main()
