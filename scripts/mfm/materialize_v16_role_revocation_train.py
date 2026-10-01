#!/usr/bin/env python3
"""Freeze the exact fictional Mira revocation training case, unadmitted.

The original six-case generator and selected source/target bytes are pinned.
No revoked archive payload is opened. Permission drafts remain unissued.
"""

from __future__ import annotations

import argparse
from base64 import b64decode
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cognitive_kernel.formation_learning_v16 import (
    TARGET_SCHEMA_V16, learning_example_v16_from_record,
    supervised_output_record_v16,
)
from scripts.mfm import build_v16_role_teacher_candidates as original
from scripts.mfm.materialize_v16_authoring_seed_train import (
    _private_output, _raw, _rights_draft, _sha, _write,
)


CASE_ID = "role-source-person-revocation"
ORIGINAL_CORPUS_SHA256 = "2096cb9a148678e9a2a4a2fea9a3c09d44e9f45db448713998667f3f4219d08e"
ORIGINAL_CODE_SHA256 = "a5d6f1f35d809d30f3fbc21d4344874fbe057be6c1bf734b582f5a5fcb600d5a"


def materialize(output_dir: Path) -> dict:
    _private_output(output_dir)
    code = Path(original.__file__).read_bytes()
    corpus = original.render()
    if (_sha(code) != ORIGINAL_CODE_SHA256 or
            _sha(corpus) != ORIGINAL_CORPUS_SHA256 or
            original.GENERATOR_FAMILY != "codex-fictional-role-chronicle-20261001" or
            original.AUTHORIZATION != "owner-directed-mfm-v16-synthetic"):
        raise ValueError("original role generator or six-case corpus differs from pin")
    selected = [(spec, line, json.loads(line)) for spec, line in zip(
        original.SPECS, corpus.splitlines(keepends=True), strict=True)
        if spec["case_id"] == CASE_ID]
    if len(selected) != 1:
        raise ValueError("original revocation case missing or duplicated")
    spec, line, candidate = selected[0]
    if (candidate["case_id"] != CASE_ID or candidate["split"] != "train" or
            candidate["authorization_id"] != original.AUTHORIZATION or
            candidate["lineage"]["generator_family"] != original.GENERATOR_FAMILY or
            candidate["status"]["qualification"] != "source-grounded-candidate-only"):
        raise ValueError("original role candidate metadata differs")
    example = learning_example_v16_from_record(candidate, split="train")
    supervised_output_record_v16(example)
    kinds = {p.base.kind for p in example.target.proposals}
    if not {"revocation_request", "deletion_request"}.issubset(kinds):
        raise ValueError("original candidate lost revocation/deletion proposal")
    output_dir.mkdir(mode=0o700)
    generator_ref = _write(output_dir, "role-generator-original.py", code)
    converter_ref = _write(output_dir, "converter-implementation.py", Path(__file__).read_bytes())
    prefix = f"cases/{CASE_ID}"
    candidate_ref = _write(output_dir, f"{prefix}/candidate.json", line)
    input_ref = _write(output_dir, f"{prefix}/generator-input.json", _raw({
        "schema": "mfm-v16-role-generator-input-v1", "case_id": CASE_ID,
        "generator_family": original.GENERATOR_FAMILY,
        "generator_code_sha256": generator_ref["sha256"], "original_spec": spec,
    }))
    sources = []
    drafts = []
    host_family = candidate["context"]["base_context"]["scope"]["host_instance_id"]
    for index, source in enumerate(candidate["sources"]):
        raw = b64decode(source["content_b64"], validate=True)
        ref = _write(output_dir, f"{prefix}/source-{index:03d}.txt", raw)
        sources.append({"source_id": source["ref_id"], **ref})
        drafts.append(_rights_draft(output_dir, prefix, index, sources[-1], host_family))
    target = {"schema": TARGET_SCHEMA_V16, "context": candidate["context"],
              "source_ids": [source["ref_id"] for source in candidate["sources"]],
              **{field: candidate["target"][field] for field in (
                  "proposals", "dispositions", "adjudications")}}
    target_ref = _write(output_dir, f"{prefix}/target.json", _raw(target))
    provenance_ref = _write(output_dir, f"{prefix}/provenance.json", _raw({
        "schema": "mfm-v16-role-revocation-train-provenance-v1",
        "case_id": CASE_ID, "case_sha256": candidate_ref["sha256"],
        "rendered_prompt_sha256": input_ref["sha256"],
        "generator_output_sha256": target_ref["sha256"],
        "converter_sha256": converter_ref["sha256"],
        "generator_source_code_sha256": generator_ref["sha256"],
        "original_six_case_corpus_sha256": ORIGINAL_CORPUS_SHA256,
        "training_admitted": False, "rights_status": "unverified",
    }))
    receipt_ref = _write(output_dir, "receipt.json", _raw({
        "schema": "mfm-v16-role-revocation-materialization-v1",
        "status": "unadmitted-pending-rights", "case_id": CASE_ID,
        "authorization_id": candidate["authorization_id"],
        "generator_family": candidate["lineage"]["generator_family"],
        "host_family": host_family,
        "source_family": candidate["lineage"]["host_family"],
        "original_corpus_sha256": ORIGINAL_CORPUS_SHA256,
        "original_generator": generator_ref, "target_converter": converter_ref,
        "candidate": candidate_ref, "generator_input": input_ref,
        "generator_output": target_ref, "sources": sources,
        "rights_drafts": drafts, "converter_provenance": provenance_ref,
        "source_rights_created": False, "qualified_for_product": False,
    }))
    return {"receipt_path": str(output_dir / "receipt.json"),
            "receipt_sha256": receipt_ref["sha256"],
            "case_id": CASE_ID, "status": "unadmitted-pending-rights"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
