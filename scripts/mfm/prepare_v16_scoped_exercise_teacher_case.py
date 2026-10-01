#!/usr/bin/env python3
"""One source-grounded v1.6 exercise teacher candidate; no admission or rights.

`prepare` gives a teacher only exact field values from an isolated packet.
`convert` verifies a raw teacher answer and the private source/index mapping,
then emits one source-grounded v1.6 *candidate* outside Git. It does not prove
semantic entailment or complete review beyond the three selected fields.
"""

from __future__ import annotations

import argparse
from base64 import b64decode, b64encode
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cognitive_kernel.canonical import canonical_json_bytes
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.formation_contracts import (
    FormationContextPacket, FormationDisposition, FormationEvidenceAnchor,
    FormationEvidenceRef, FormationProposal, MemoryProposalBundle,
)
from cognitive_kernel.formation_learning_v16 import (
    FULL_ROLE_DIMENSIONS, TARGET_SCHEMA_V16, CURRICULUM_SCHEMA_V16,
    learning_example_v16_from_record, supervised_output_record_v16,
)
from cognitive_kernel.formation_semantics_v16 import (
    FormationContextV16, FormationProposalV16, MemoryProposalBundleV16,
    RegisteredSourceSensitivity, validate_formation_grounding_v16,
)
from scripts.mfm.build_v16_source_review_packets import _load
from scripts.mfm.materialize_v16_source_slices import (
    _opaque_id, _space, _string_end, isolated_packet,
)
from scripts.mfm.prepare_v16_annotation_drafts import write_private_new


PROMPT = ROOT / "scripts/mfm/prompts/v16_scoped_exercise_teacher_v1.txt"
SCOPE = "selected-exercise-fields-only"
SCHEMA = "mfm-v16-scoped-exercise-teacher-input-v1"
RESPONSE_SCHEMA = "mfm-v16-scoped-exercise-teacher-response-v1"
AUTHORIZATION = "owner-directed-mfm-v16-synthetic"
FIELDS = (("planner", "exercise_target"),
          ("daily_self_report", "exercise"),
          ("device_log", "activity_tracker"))
CLAIM_PROPERTIES = {
    "planner": ("intended", "type", "duration_min"),
    "daily_self_report": ("did_exercise", "type", "duration_min"),
    "device_log": ("active_minutes", "workout_detected"),
}


def _digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _raw(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def rendered_prompt(scoped_raw: bytes) -> bytes:
    """Bind the reusable instruction to the exact teacher-visible input bytes."""
    return PROMPT.read_bytes().rstrip() + b"\n\nSOURCE_VIEW_JSONL:\n" + scoped_raw


def exact_field_value(raw: bytes, key: str) -> tuple[bytes, int, int]:
    """Select one original JSON value without reserializing or exposing siblings."""
    document = _load(raw)
    expected = document.get(key, document.get("signals", {}).get(key))
    if not isinstance(expected, dict):
        raise ValueError("selected source field is not an object")
    matches = list(re.finditer(rb'"' + key.encode() + rb'"\s*:\s*', raw))
    if len(matches) != 1:
        raise ValueError("selected source field is missing or ambiguous")
    start = matches[0].end()
    text = raw[start:].decode("utf-8")
    parsed, characters = json.JSONDecoder().raw_decode(text)
    end = start + len(text[:characters].encode("utf-8"))
    selected = raw[start:end]
    if parsed != expected or _load(selected) != expected:
        raise ValueError("selected field differs from exact source")
    return selected, start, end


def exact_property_spans(raw: bytes, names: tuple[str, ...]) -> tuple[tuple[str, int, int], ...]:
    """Locate original UTF-8 `key: value` pairs within one selected object."""
    document = _load(raw)
    spans = {}
    cursor = _space(raw, 0)
    if cursor >= len(raw) or raw[cursor] != 123:
        raise ValueError("claim source is not a JSON object")
    cursor = _space(raw, cursor + 1)
    while cursor < len(raw) and raw[cursor] != 125:
        start = cursor
        key_end = _string_end(raw, cursor)
        key = json.loads(raw[cursor:key_end])
        cursor = _space(raw, key_end)
        if cursor >= len(raw) or raw[cursor] != 58:
            raise ValueError("claim property lacks colon")
        cursor = _space(raw, cursor + 1)
        text = raw[cursor:].decode("utf-8")
        value, characters = json.JSONDecoder().raw_decode(text)
        end = cursor + len(text[:characters].encode("utf-8"))
        if key in spans or key not in document or value != document[key]:
            raise ValueError("claim property differs from parsed source")
        spans[key] = (start, end)
        cursor = _space(raw, end)
        if cursor < len(raw) and raw[cursor] == 44:
            cursor = _space(raw, cursor + 1)
        elif cursor < len(raw) and raw[cursor] != 125:
            raise ValueError("claim property lacks comma or closing brace")
    if cursor >= len(raw) or raw[cursor] != 125 or _space(raw, cursor + 1) != len(raw):
        raise ValueError("claim source has trailing or incomplete JSON")
    if set(names) - set(spans) or len(names) != len(set(names)):
        raise ValueError("claim evidence property is missing or repeated")
    for name in names:
        start, end = spans[name]
        if _load(b"{" + raw[start:end] + b"}") != {name: document[name]}:
            raise ValueError("claim evidence bytes do not match property")
    return tuple((name, *spans[name]) for name in names)


def scoped_teacher_input(isolated: dict) -> dict:
    if (isolated.get("schema") != "mfm-v16-isolated-source-packet-v1" or
            isolated.get("training_admitted") is not False or
            isolated.get("rights_status") != "unverified"):
        raise ValueError("teacher packet is not an unadmitted isolated source view")
    source_by_type = {row["source_type"]: row for row in isolated["sources"]}
    if len(source_by_type) != len(isolated["sources"]) or \
            not set(dict(FIELDS)).issubset(source_by_type):
        raise ValueError("one source per required exercise stream is needed")
    selected = []
    for kind, key in FIELDS:
        item = source_by_type[kind]
        if len(item["fragments"]) != 1:
            raise ValueError("exercise source needs one exact daily record")
        encoded = item["fragments"][0]["payload_base64"]
        raw = b64decode(encoded, validate=True)
        if b64encode(raw).decode("ascii") != encoded or \
                _digest(raw) != item["fragments"][0]["sha256"]:
            raise ValueError("isolated source fragment differs")
        field, _, _ = exact_field_value(raw, key)
        selected.append({"source_id": item["source_id"], "source_type": kind,
                         "field_sha256": _digest(field),
                         "payload_base64": b64encode(field).decode("ascii")})
    return {"schema": SCHEMA, "status": "source-only-unreviewed-unadmitted",
            "training_admitted": False, "rights_status": "unverified",
            "review_scope": SCOPE, "packet_id": isolated["packet_id"],
            "host_id": isolated["host_id"], "window": isolated["window"],
            "sources": selected}


def _canonical_id(original: str) -> str:
    return original.replace("_", "-").replace("/", "-")


def convert(*, scoped: dict, response: dict, isolated: dict,
            private_index_row: dict, source_root: Path,
            receipt_sha256: str, scoped_sha256: str,
            response_sha256: str, isolated_sha256: str,
            rendered_prompt_sha256: str) -> tuple[dict, dict]:
    if scoped != scoped_teacher_input(isolated):
        raise ValueError("teacher source subset differs from isolated exact bytes")
    if (response.get("schema") != RESPONSE_SCHEMA or
            response.get("source_view_sha256") != scoped_sha256 or
            response.get("review_scope") != SCOPE or
            len(response.get("proposals", ())) != 3):
        raise ValueError("raw teacher response has wrong source binding or scope")
    dimension_states = response.get("adjudications")
    if not isinstance(dimension_states, dict) or set(dimension_states) != FULL_ROLE_DIMENSIONS or \
            set(dimension_states.values()) - {"present", "negative"}:
        raise ValueError("teacher has not explicitly adjudicated all ten dimensions")
    if dimension_states != {name: ("present" if name in {"outcome", "abstention"}
                                   else "negative")
                            for name in FULL_ROLE_DIMENSIONS}:
        raise ValueError("this scoped candidate has reported outcome and abstention positive")
    if private_index_row["schema"] != "mfm-v16-source-slice-index-v1" or \
            private_index_row["training_admitted"] is not False or \
            private_index_row["rights_status"] != "unverified":
        raise ValueError("private source index is not an unadmitted source candidate")
    host = _canonical_id(private_index_row["lineage"]["host_family"])
    if isolated["host_id"] != _opaque_id(
            receipt_sha256, "host", private_index_row["lineage"]["host_family"]):
        raise ValueError("opaque host handle differs from private source index")
    scope = ProductHostScope.create(product_id="alice", host_instance_id=host,
                                    schema_version="1.0.0",
                                    encryption_domain="mfm-teacher-synthetic")
    authority = f"{host}-teacher-exercise-authority"
    index_by_type = {item["source_type"]: item for item in private_index_row["sources"]}
    isolated_by_type = {item["source_type"]: item for item in isolated["sources"]}
    opened = []
    evidence = []
    sensitive = []
    private_offsets = []
    dates = set()
    by_type = {}
    for selected, (kind, key) in zip(scoped["sources"], FIELDS, strict=True):
        original = index_by_type[kind]
        visible = isolated_by_type[kind]
        if selected["source_type"] != kind or \
                selected["source_id"] != visible["source_id"] or \
                visible["source_id"] != _opaque_id(receipt_sha256, "source",
                                                    original["source_id"]):
            raise ValueError("teacher source handle differs from private index")
        parent = source_root / original["parent_source_path"]
        raw_parent = parent.read_bytes()
        if _digest(raw_parent) != original["parent_source_file_sha256"]:
            raise ValueError("original parent source changed")
        span = original["fragments"][0]["original_file_byte_anchor"]
        unit = raw_parent[span["start"]:span["end"]]
        if _digest(unit) != original["fragments"][0]["sha256"] or \
                b64encode(unit).decode("ascii") != visible["fragments"][0]["payload_base64"]:
            raise ValueError("private original-unit span differs from teacher-visible bytes")
        field, start, end = exact_field_value(unit, key)
        if b64encode(field).decode("ascii") != selected["payload_base64"] or \
                _digest(field) != selected["field_sha256"]:
            raise ValueError("teacher field is not the original byte slice")
        original_record = _load(unit)
        dates.add(original_record["date"])
        ref_id = f"{_canonical_id(original['source_id'])}-{key.replace('_', '-')}-field"
        day_stamp = f"{original_record['date']}T00:00:00.000000Z"
        role = "historical_experience" if kind == "daily_self_report" else \
            "tool_or_action_observation"
        speaker = host if kind == "daily_self_report" else f"{kind.replace('_', '-')}-system"
        evidence.append(FormationEvidenceRef(
            ref_id=ref_id, scope=scope, authority_namespace_id=authority,
            content_digest=_digest(field), role=role, modality="structured",
            subject_ref=host, speaker_ref=speaker,
            source_item_ref=_canonical_id(original["source_id"]),
            observed_at=day_stamp, recorded_at=None,
            temporal_granularity="day"))
        opened.append((ref_id, field))
        sensitive.append(RegisteredSourceSensitivity(ref_id, "public"))
        by_type[kind] = ref_id
        private_offsets.append({"source_id": original["source_id"],
                                "teacher_source_handle": selected["source_id"],
                                "source_type": kind, "field": key,
                                "parent_file_sha256": original["parent_source_file_sha256"],
                                "parent_byte_anchor": {
                                    "start": span["start"] + start,
                                    "end": span["start"] + end},
                                "field_sha256": _digest(field)})
    if len(dates) != 1 or private_index_row["window"]["end_day_index"] != 1:
        raise ValueError("this first candidate must be one calendar day")
    date = next(iter(dates))
    base = FormationContextPacket(scope, authority, tuple(by_type[k] for k, _ in FIELDS),
                                  tuple(evidence))
    context = FormationContextV16(base, tuple(sensitive))
    proposals = []
    for index, (teacher, (kind, _)) in enumerate(
            zip(response["proposals"], FIELDS, strict=True), 1):
        if (set(teacher) != {"source_type", "kind", "value_text", "scope_ref",
                             "evidence_properties"} or
                teacher["source_type"] != kind or
                teacher["kind"] != {"planner": "goal",
                                    "daily_self_report": "outcome",
                                    "device_log": "host_observation"}[kind] or
                teacher["evidence_properties"] != list(CLAIM_PROPERTIES[kind])):
            raise ValueError("teacher proposal has wrong field or kind")
        ref = by_type[kind]
        payload = dict(opened)[ref]
        property_spans = exact_property_spans(payload, CLAIM_PROPERTIES[kind])
        anchors = tuple(FormationEvidenceAnchor(ref, start, end)
                        for _, start, end in property_spans)
        proposal = FormationProposal(
            proposal_id=f"{host}-day-01-exercise-proposal-{index}",
            kind=teacher["kind"], domain="host", subject_ref=host,
            value_ref=f"{host}-day-01-exercise-value-{index}",
            evidence_refs=(ref,), value_text=teacher["value_text"],
            anchors=anchors,
            epistemic_status="observation",
            valid_from=f"{date}T00:00:00.000000Z",
            valid_to=f"{date}T00:00:00.000000Z", temporal_granularity="day",
            confidence=None, disposition_scope_ref=teacher["scope_ref"])
        proposals.append(FormationProposalV16(proposal))
        private_offsets[index - 1]["claim_property_original_byte_anchors"] = [
            {"property": name,
             "start": private_offsets[index - 1]["parent_byte_anchor"]["start"] + start,
             "end": private_offsets[index - 1]["parent_byte_anchor"]["start"] + end}
            for name, start, end in property_spans]
    dispositions = []
    for decision in response.get("dispositions", ()):
        if set(decision) != {"scope_ref", "action", "source_types"} or \
                not isinstance(decision["source_types"], list):
            raise ValueError("teacher disposition has wrong fields")
        types = decision["source_types"]
        if not types or len(types) != len(set(types)) or set(types) - set(by_type):
            raise ValueError("teacher disposition source type differs")
        dispositions.append(FormationDisposition(
            scope_ref=decision["scope_ref"], action=decision["action"],
            evidence_refs=tuple(by_type[k] for k in types)))
    expected_proposes = {p.base.disposition_scope_ref for p in proposals}
    if ({d.scope_ref for d in dispositions if d.action == "propose"} !=
            expected_proposes or
            not any(d.scope_ref == "completed_planned_run" and d.action == "abstain" and
                    set(d.evidence_refs) == set(by_type.values()) for d in dispositions)):
        raise ValueError("teacher omitted propose or planned-run abstain decision")
    case_id = f"{host}-day-01-exercise-v16"
    bundle = MemoryProposalBundle(scope, authority, f"{case_id}-bundle",
                                  base.experience_refs, base.content_digest(),
                                  "0" * 64, f"teacher-{case_id}",
                                  tuple(p.base for p in proposals), tuple(dispositions))
    versioned = MemoryProposalBundleV16(bundle, context.content_digest(), tuple(proposals))
    validate_formation_grounding_v16(context, versioned, tuple(opened))
    output = versioned.output_record()
    case = {"schema": CURRICULUM_SCHEMA_V16, "case_id": case_id,
            "split": "train", "authorization_id": AUTHORIZATION,
            "context": context.record(),
            "sources": [{"ref_id": ref, "content_b64": b64encode(raw).decode("ascii")}
                        for ref, raw in opened],
            "target": {"schema": TARGET_SCHEMA_V16,
                       "proposals": output["proposals"],
                       "dispositions": output["dispositions"],
                       "adjudications": dimension_states},
            "lineage": private_index_row["lineage"],
            "status": {"training_admitted": False,
                       "rights": "unverified",
                       "target_author": "openai-codex-synthetic-teacher",
                       "review": "single-teacher-unreviewed",
                       "review_scope": SCOPE,
                       "qualification": "source-grounded-candidate-only"}}
    example = learning_example_v16_from_record(case)
    supervised_output_record_v16(example)
    provenance = {"schema": "mfm-v16-scoped-teacher-candidate-provenance-v1",
                  "training_admitted": False, "rights_status": "unverified",
                  "independent_review": False,
                  "case_id": case_id, "source_slice_receipt_sha256": receipt_sha256,
                  "teacher_view_sha256": scoped_sha256,
                  "isolated_packet_sha256": isolated_sha256,
                  "raw_teacher_response_sha256": response_sha256,
                  "prompt_template_sha256": _digest(PROMPT.read_bytes()),
                  "rendered_prompt_sha256": rendered_prompt_sha256,
                  "converter_sha256": _digest(Path(__file__).read_bytes()),
                  "selected_original_field_offsets": private_offsets,
                  "semantic_entailment_reviewed_independently": False,
                  "case_sha256": _digest(_raw(case))}
    return case, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prepare = modes.add_parser("prepare")
    prepare.add_argument("--isolated-packet", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--output-prompt", type=Path, required=True)
    convert_mode = modes.add_parser("convert")
    for name in ("isolated-packet", "scoped-input", "raw-response", "rendered-prompt",
                 "source-slices-dir", "source-root", "output-case",
                 "output-provenance"):
        convert_mode.add_argument("--" + name, type=Path, required=True)
    for name in ("isolated-sha256", "scoped-sha256", "response-sha256",
                 "rendered-prompt-sha256", "receipt-sha256", "original-packet-id"):
        convert_mode.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        isolated = _load(args.isolated_packet.read_bytes())
        scoped = scoped_teacher_input(isolated)
        raw = _raw(scoped)
        write_private_new(args.output, raw)
        prompt_raw = rendered_prompt(raw)
        write_private_new(args.output_prompt, prompt_raw)
        print(json.dumps({"schema": SCHEMA, "output_sha256": _digest(raw),
                          "rendered_prompt_sha256": _digest(prompt_raw),
                          "sources": len(scoped["sources"]),
                          "training_admitted": False}, sort_keys=True))
        return
    inputs = ((args.isolated_packet, args.isolated_sha256),
              (args.scoped_input, args.scoped_sha256),
              (args.raw_response, args.response_sha256),
              (args.rendered_prompt, args.rendered_prompt_sha256))
    for path, expected in inputs:
        if _digest(path.read_bytes()) != expected:
            raise ValueError("private teacher input or response differs from SHA-256 pin")
    isolated = _load(args.isolated_packet.read_bytes())
    scoped = _load(args.scoped_input.read_bytes())
    response = _load(args.raw_response.read_bytes())
    if args.rendered_prompt.read_bytes() != rendered_prompt(args.scoped_input.read_bytes()):
        raise ValueError("rendered teacher prompt differs from template and source input")
    rebuilt = isolated_packet(args.source_slices_dir,
                              packet_id=args.original_packet_id,
                              receipt_sha256=args.receipt_sha256)
    if rebuilt != isolated:
        raise ValueError("teacher isolated packet differs from private index replay")
    index_rows = [_load(line) for line in
                  (args.source_slices_dir / "index.jsonl").read_bytes().splitlines()]
    selected = [row for row in index_rows if row["packet_id"] == args.original_packet_id]
    if len(selected) != 1:
        raise ValueError("private index lacks one exact original packet")
    case, provenance = convert(
        scoped=scoped, response=response, isolated=isolated,
        private_index_row=selected[0], source_root=args.source_root,
        receipt_sha256=args.receipt_sha256,
        scoped_sha256=args.scoped_sha256,
        response_sha256=args.response_sha256,
        isolated_sha256=args.isolated_sha256,
        rendered_prompt_sha256=args.rendered_prompt_sha256)
    write_private_new(args.output_case, _raw(case))
    write_private_new(args.output_provenance, _raw(provenance))
    print(json.dumps({"case_sha256": provenance["case_sha256"],
                      "provenance_sha256": _digest(_raw(provenance)),
                      "training_admitted": False,
                      "v16_parser_and_grounding": True}, sort_keys=True))


if __name__ == "__main__":
    main()
