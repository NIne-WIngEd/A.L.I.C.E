#!/usr/bin/env python3
"""Deterministic source and label candidate for v1.6 diagnostic development.

This emits invented histories and a single author's source-bound targets. It
does not create a rights receipt, independent review, training admission,
FINAL set, or model result. The source-only packet is separate from targets.
"""

from __future__ import annotations

import argparse
from base64 import b64decode, b64encode
from hashlib import sha256
import json
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.canonical import canonical_json_bytes, require_identifier
from cognitive_kernel.formation_contracts import (
    FormationContextPacket, FormationDisposition, FormationEvidenceAnchor,
    FormationEvidenceRef, FormationProposal, MemoryProposalBundle,
)
from cognitive_kernel.formation_learning_v16 import (
    CURRICULUM_SCHEMA_V16, FULL_ROLE_DIMENSIONS, TARGET_SCHEMA_V16,
    learning_example_v16_from_record, supervised_output_record_v16,
)
from cognitive_kernel.formation_semantics_v16 import (
    EpisodeSemantics, FormationContextV16, FormationProposalV16,
    MemoryProposalBundleV16, RegisteredFormationTarget,
    RegisteredSourceSensitivity, validate_formation_grounding_v16,
)
from scripts.mfm.v16_development_history_spec import HISTORIES

FAMILY = "deterministic-fictional-workbench-garden-20261001"
SOURCE_SCHEMA = "mfm-v16-deterministic-source-window-v1"
RECEIPT_SCHEMA = "mfm-v16-deterministic-development-candidate-v1"
PRODUCER_ID = "mfm-v16-deterministic-history-author"


def _raw(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _day(value: str) -> str:
    return value + "T00:00:00.000000Z"


def _window(history: dict, window: tuple, index: int,
            authorization_id: str) -> tuple[dict, dict]:
    name, as_of, visible, selected, decisions, positives, negatives = window[:7]
    parent_name = window[7] if len(window) > 7 else (
        history["windows"][index - 1][0] if index else None)
    pair_id = window[8] if len(window) > 8 else None
    case_id = f"det-dev-{history['id']}-{name}"
    event_map = {item["name"]: item for item in history["events"]}
    if len(event_map) != len(history["events"]) or len(set(visible)) != len(visible):
        raise ValueError("duplicate event or visible source")
    if not set(visible).issubset(event_map) or not set(selected).issubset(history["claims"]):
        raise ValueError("unregistered event or claim")
    if any(event_map[item]["day"] > as_of for item in visible):
        raise ValueError("future event exposed to an as-of window")
    if ("outside" in visible and "withdrawal" in visible) or \
            ("withdrawal" in visible and "tombstone" not in visible):
        raise ValueError("withdrawn source content visible or tombstone missing")
    if len(set(selected)) != len(selected) or len(decisions) < len(selected) or \
            any(decisions[i][1] != "propose" for i in range(len(selected))):
        raise ValueError("selected claim needs one ordered proposal disposition")
    if (set(positives) & set(negatives) or len(set(positives)) != len(positives) or
            len(set(negatives)) != len(negatives) or
            set(positives) | set(negatives) != FULL_ROLE_DIMENSIONS):
        raise ValueError("every dimension needs one explicit scoped judgment")
    scope = ProductHostScope.create(
        product_id="alice", host_instance_id=f"det-{history['id']}-{history['host']}",
        schema_version="1.0.0", encryption_domain=f"det-{history['id']}-fictional")
    authority = f"det-{history['id']}-authority"
    # Each window is a distinct training example. Its source ref and payload
    # path must be unique even when it selects an earlier original item again.
    refs = {key: f"det-{history['id']}-{name}-{key}" for key in visible}
    evidence, opened, sensitivity = [], [], []
    for key in visible:
        item = event_map[key]
        raw = item["text"].encode("utf-8")
        ref = refs[key]
        evidence.append(FormationEvidenceRef(
            ref_id=ref, scope=scope, authority_namespace_id=authority,
            content_digest=sha256(raw).hexdigest(), role=item["role"],
            modality="text", subject_ref=f"det-{history['id']}-{item['subject']}",
            speaker_ref=f"det-{history['id']}-{item['speaker']}",
            source_item_ref=f"det-{history['id']}-{key}-original",
            observed_at=_day(item["day"]), recorded_at=_day(item["day"])))
        opened.append((ref, raw))
        sensitivity.append(RegisteredSourceSensitivity(ref, item["minimum"]))
    base = FormationContextPacket(scope, authority, tuple(refs.values()), tuple(evidence))
    targets = tuple(RegisteredFormationTarget(
        f"det-{history['id']}-{target_id}", kind, scope, authority,
        tuple(refs[key] for key in sources if key in refs))
        for target_id, kind, sources in history["targets"]
        if any(key in refs for key in sources))
    context = FormationContextV16(base, tuple(sensitivity), targets)
    proposals = []
    for number, key in enumerate(selected, 1):
        p = history["claims"][key]
        if not set(p["quotes"]).issubset(refs):
            raise ValueError(f"{case_id}: claim depends on later/withdrawn source")
        anchors = []
        for source_name, quote in p["quotes"].items():
            raw, needle = event_map[source_name]["text"].encode(), quote.encode()
            if not needle or raw.count(needle) != 1:
                raise ValueError(f"{case_id}: quote missing or ambiguous: {source_name}")
            start = raw.index(needle)
            anchors.append(FormationEvidenceAnchor(refs[source_name], start,
                                                   start + len(needle)))
        decision = decisions[number - 1]
        if tuple(decision[2]) != tuple(p["quotes"]):
            raise ValueError(f"{case_id}: disposition sources differ from claim")
        if tuple(decision[3]) != tuple(p["target_refs"]):
            raise ValueError(f"{case_id}: disposition target refs differ from claim")
        latest = max(event_map[source_name]["day"] for source_name in p["quotes"])
        basic = FormationProposal(
            proposal_id=f"{case_id}-{key}-proposal", kind=p["kind"],
            domain=p["domain"], subject_ref=f"det-{history['id']}-{p['subject']}",
            value_ref=f"{case_id}-{key}-value", evidence_refs=tuple(refs[n] for n in p["quotes"]),
            value_text=p["value"], anchors=tuple(anchors),
            epistemic_status=p["epistemic"], valid_from=_day(latest), valid_to=None,
            confidence=None, contradicts=tuple(p["contradicts"]),
            target_refs=tuple(p["target_refs"]), disposition_scope_ref=decision[0])
        episode = p["episode"]
        proposals.append(FormationProposalV16(
            basic, sensitivity_hint=p["sensitivity"],
            episode=EpisodeSemantics(episode[0], episode[1],
                tuple(f"det-{history['id']}-{n}" for n in episode[2]),
                tuple(f"det-{history['id']}-{n}" for n in episode[3])) if episode else None,
            relationship_counterpart_ref=(f"det-{history['id']}-{p['counterpart']}"
                                          if p["counterpart"] else None),
            mission_target_refs=tuple(f"det-{history['id']}-{n}" for n in p["mission"]),
            workspace_target_refs=tuple(f"det-{history['id']}-{n}" for n in p["workspace"])))
    dispositions = tuple(FormationDisposition(
        scope_ref=scope_ref, action=action,
        evidence_refs=tuple(refs[n] for n in cited), target_refs=tuple(target_refs))
        for scope_ref, action, cited, target_refs in decisions)
    bundle = MemoryProposalBundle(scope, authority, f"{case_id}-bundle",
                                  base.experience_refs, base.content_digest(),
                                  "0" * 64, f"{case_id}-deterministic-generator",
                                  tuple(p.base for p in proposals), dispositions)
    result = MemoryProposalBundleV16(bundle, context.content_digest(), tuple(proposals))
    validate_formation_grounding_v16(context, result, tuple(opened))
    output = result.output_record()
    row = {
        "schema": CURRICULUM_SCHEMA_V16, "case_id": case_id,
        "split": "development", "authorization_id": authorization_id,
        "context": context.record(),
        "sources": [{"ref_id": ref, "content_b64": b64encode(raw).decode()}
                    for ref, raw in opened],
        "target": {"schema": TARGET_SCHEMA_V16,
                   "proposals": output["proposals"],
                   "dispositions": output["dispositions"],
                   "adjudications": {dimension: "present" for dimension in positives} |
                                    {dimension: "negative" for dimension in negatives}},
        "lineage": {"source_family": f"det-fictional-{history['id']}",
                    "host_family": scope.host_instance_id,
                    "generator_family": FAMILY, "scenario_family": FAMILY,
                    "parent_case_ids": [f"det-dev-{history['id']}-{parent_name}"
                                        ] if parent_name else [],
                    "counterfactual_pair": pair_id,
                    "history_id": history["id"],
                    "source_generator_input": "scripts/mfm/v16_development_history_spec.py",
                    "target_converter": "scripts/mfm/build_v16_development_history.py"},
        "status": {"training_admitted": False, "rights": "unverified",
                   "review": "single-author-unreviewed",
                   "qualification": "deterministic-synthetic-diagnostic-candidate-only"},
    }
    example = learning_example_v16_from_record(row, split="development")
    supervised_output_record_v16(example)
    source_packet = {
        "schema": SOURCE_SCHEMA, "case_id": case_id, "as_of": _day(as_of),
        "history_id": history["id"], "host_id": scope.host_instance_id,
        "source_family": f"det-fictional-{history['id']}",
        "generator_family": FAMILY, "parent_case_ids": row["lineage"]["parent_case_ids"],
        "counterfactual_pair": pair_id,
        "sources": [{"ref_id": ref, "source_item_ref": evidence[i].source_item_ref,
                     "observed_at": evidence[i].observed_at,
                     "recorded_at": evidence[i].recorded_at,
                     "role": evidence[i].role, "subject_ref": evidence[i].subject_ref,
                     "speaker_ref": evidence[i].speaker_ref,
                     "minimum_sensitivity": sensitivity[i].minimum,
                     "sha256": sha256(raw).hexdigest(), "content_b64": b64encode(raw).decode()}
                    for i, (ref, raw) in enumerate(opened)],
        "rights_status": "unverified", "training_admitted": False,
    }
    return source_packet, row


def render(authorization_id: str) -> tuple[bytes, bytes, bytes]:
    if require_identifier(authorization_id, "authorization_id") != authorization_id:
        raise ValueError("authorization_id must be canonical")
    packets, targets = [], []
    for history in HISTORIES:
        if len({e["name"] for e in history["events"]}) != len(history["events"]):
            raise ValueError("history duplicates event names")
        for index, window in enumerate(history["windows"]):
            source, target = _window(history, window, index, authorization_id)
            packets.append(_raw(source))
            targets.append(_raw(target))
    source_raw, target_raw = b"".join(packets), b"".join(targets)
    spec = ROOT / "scripts/mfm/v16_development_history_spec.py"
    converter = Path(__file__).resolve()
    receipt = _raw({
        "schema": RECEIPT_SCHEMA, "status": "unreviewed-unadmitted-synthetic",
        "cases": len(targets), "source_histories": len(HISTORIES),
        "authorization_id": authorization_id,
        "generator_family": FAMILY,
        "authorship": "single-ai-assisted-author; not independent gold",
        "spec_sha256": sha256(spec.read_bytes()).hexdigest(),
        "converter_sha256": sha256(converter.read_bytes()).hexdigest(),
        "source_packets_sha256": sha256(source_raw).hexdigest(),
        "candidate_targets_sha256": sha256(target_raw).hexdigest(),
    })
    return source_raw, target_raw, receipt


def _materialized_files(source_raw: bytes, target_raw: bytes,
                        receipt: bytes) -> dict[str, bytes]:
    """Separate exact generator input, source, candidate and target bytes.

    The source-only packet is not a teacher response. The target JSON is the
    deterministic generator's actual output for the assembler's response ref.
    Rights records remain deliberately absent.
    """
    spec_raw = (ROOT / "scripts/mfm/v16_development_history_spec.py").read_bytes()
    converter_raw = Path(__file__).resolve().read_bytes()
    files = {"source_windows.jsonl": source_raw, "v16_targets.jsonl": target_raw,
             "receipt.json": receipt, "generator-input.py": spec_raw,
             "converter.py": converter_raw}
    index = []
    packets = [json.loads(line) for line in source_raw.splitlines()]
    candidates = [json.loads(line) for line in target_raw.splitlines()]
    if len(packets) != len(candidates):
        raise ValueError("source and target counts differ")
    spec_sha, converter_sha = sha256(spec_raw).hexdigest(), sha256(converter_raw).hexdigest()
    for packet, candidate in zip(packets, candidates, strict=True):
        case_id = candidate["case_id"]
        if packet["case_id"] != case_id:
            raise ValueError("source packet differs from target case")
        base = f"cases/{case_id}"
        candidate_raw = _raw(candidate)
        target_record = {
            "schema": TARGET_SCHEMA_V16, "context": candidate["context"],
            "source_ids": [item["ref_id"] for item in candidate["sources"]],
            **{key: candidate["target"][key] for key in
               ("proposals", "dispositions", "adjudications")},
        }
        target_bytes = canonical_json_bytes(target_record) + b"\n"
        target_path = f"{base}/target.json"
        files[target_path] = target_bytes
        files[f"{base}/candidate.json"] = candidate_raw
        files[f"{base}/source_packet.json"] = _raw(packet)
        provenance = {
            "schema": "mfm-v16-deterministic-conversion-provenance-v1",
            "case_id": case_id, "case_sha256": sha256(candidate_raw).hexdigest(),
            "rendered_prompt_sha256": spec_sha,
            "generator_output_sha256": sha256(target_bytes).hexdigest(),
            "converter_sha256": converter_sha,
            "producer_id": PRODUCER_ID, "producer_version": FAMILY,
            "authorship": "single-ai-assisted-author; not independent gold",
            "rights_status": "unverified", "training_admitted": False,
        }
        files[f"{base}/provenance.json"] = _raw(provenance)
        sources = []
        for item in candidate["sources"]:
            ref = item["ref_id"]
            raw = b64decode(item["content_b64"], validate=True)
            path = f"{base}/sources/{ref}.txt"
            files[path] = raw
            sources.append({"source_id": ref, "path": path,
                            "sha256": sha256(raw).hexdigest(),
                            "parent_source_ids": [],
                            "rights_status": "unverified"})
        index.append({
            "case_id": case_id, "split": "development",
            "authorization_id": candidate["authorization_id"],
            "host_family": candidate["lineage"]["host_family"],
            "source_family": candidate["lineage"]["source_family"],
            "generator_family": FAMILY, "scenario_family": FAMILY,
            "duplicate_group": candidate["lineage"]["source_family"],
            "parent_case_ids": candidate["lineage"]["parent_case_ids"],
            "target_origin_candidate": "licensed-deterministic-generator",
            "author_id_candidate": PRODUCER_ID,
            "candidate": {"path": f"{base}/candidate.json",
                          "sha256": sha256(candidate_raw).hexdigest()},
            "prompt": {"path": "generator-input.py", "sha256": spec_sha},
            "response": {"path": target_path,
                         "sha256": sha256(target_bytes).hexdigest()},
            "converter": {"path": "converter.py", "sha256": converter_sha},
            "converter_provenance": {
                "path": f"{base}/provenance.json",
                "sha256": sha256(files[f"{base}/provenance.json"]).hexdigest()},
            "sources": sources,
        })
    files["bundle-index.json"] = _raw({
        "schema": "mfm-v16-deterministic-development-bundle-index-v1",
        "status": "unreviewed-unadmitted-source-rights-absent",
        "authorization_id": candidates[0]["authorization_id"],
        "producer_id_candidate": PRODUCER_ID,
        "generator_family": FAMILY, "cases": index,
    })
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source, target, receipt = render(args.authorization_id)
    expected = _materialized_files(source, target, receipt)
    if args.check:
        if any((args.output_dir / name).read_bytes() != raw
               for name, raw in expected.items()):
            raise SystemExit("deterministic development bytes differ")
    else:
        if args.output_dir.exists():
            raise ValueError("output directory already exists")
        args.output_dir.mkdir(mode=0o700, parents=True)
        for name, raw in expected.items():
            path = args.output_dir / name
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(raw)
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    print(json.dumps({"cases": len(target.splitlines()),
                      "source_sha256": sha256(source).hexdigest(),
                      "target_sha256": sha256(target).hexdigest(),
                      "receipt_sha256": sha256(receipt).hexdigest(),
                      "status": "synthetic-unreviewed-unadmitted"}, sort_keys=True))


if __name__ == "__main__":
    main()
