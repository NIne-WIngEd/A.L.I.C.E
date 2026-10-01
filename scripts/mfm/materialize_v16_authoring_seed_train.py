#!/usr/bin/env python3
"""Materialize only the original 15 fictional v1.6 train rows, unadmitted.

This preserves the frozen 21-row seed and original generator bytes. It creates
no rights receipts, teacher service responses, independent reviews, FINAL or
owner-teacher manifest. The source and target author are one generator family.
"""

from __future__ import annotations

import argparse
from base64 import b64decode
from hashlib import sha256
import json
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cognitive_kernel.canonical import canonical_json_bytes
from cognitive_kernel.formation_learning_v16 import (
    TARGET_SCHEMA_V16, learning_example_v16_from_record,
    supervised_output_record_v16,
)
from scripts.mfm import build_v16_authoring_seed as original


CORPUS_SHA256 = "8e967ee6494f2c5ec47ab602bad213df8c07c53b1abe9a9aff950f19c0e78e22"
ORIGINAL_CODE_SHA256 = "2b6815139e33acd92abf3e215a04ef9bdf57fb2868e6567dde2d734518c057ce"
AUTHORIZATION = "owner-directed-mfm-v16-synthetic"
RECEIPT_SCHEMA = "mfm-v16-authoring-seed-train-materialization-v1"
SOURCE_REF_REMAP = {
    "v16-seed-03": {"v16-pair-temperament-evidence-flip-later":
                    "v16-seed-03-temperament-later"},
}


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _raw(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _private_output(root: Path) -> None:
    resolved = root.resolve()
    if (root.exists() or resolved.is_relative_to(ROOT.resolve()) or
            not root.parent.exists()):
        raise ValueError("new private output directory outside Git is required")


def _write(root: Path, relative: str, raw: bytes) -> dict[str, str]:
    path = root / relative
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return {"path": relative, "sha256": _sha(raw)}


def _rights_draft(root: Path, prefix: str, index: int, source: dict,
                  host_family: str) -> dict[str, str]:
    """Bind exact source bytes for later review without issuing permission."""
    return _write(root, f"{prefix}/rights-draft-{index:03d}.json", _raw({
        "schema": "mfm-source-rights-v1", "draft_unissued": True,
        "issuer_id": None, "authority_ref": None,
        "source_id": source["source_id"], "source_sha256": source["sha256"],
        "host_family": host_family,
        "formation_training": None, "formation_evaluation": None,
        "model_distribution": None, "revoked": None,
    }))


def _remap_refs(value: object, names: dict[str, str]) -> object:
    if isinstance(value, str):
        return names.get(value, value)
    if isinstance(value, list):
        return [_remap_refs(item, names) for item in value]
    if isinstance(value, dict):
        return {key: _remap_refs(item, names) for key, item in value.items()}
    return value


def materialize(output_dir: Path) -> dict:
    """Reproduce frozen bytes and publish only source/target candidates."""
    _private_output(output_dir)
    corpus_path = ROOT / "benchmarks/mfm/v16_authoring_seed.jsonl"
    corpus = corpus_path.read_bytes()
    code = Path(original.__file__).read_bytes()
    converter = Path(__file__).read_bytes()
    if (_sha(corpus) != CORPUS_SHA256 or _sha(code) != ORIGINAL_CODE_SHA256 or
            original.render() != corpus or
            original.GENERATOR_FAMILY != "assistant-authored-v16-seed-20260930"):
        raise ValueError("original v1.6 corpus or generator differs from pinned version")
    lines = corpus.splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    if (len(rows) != 21 or sum(row["split"] == "train" for row in rows) != 15 or
            any(row.get("authorization_id") != AUTHORIZATION for row in rows) or
            [row["case_id"] for row in rows] != [spec["case_id"] for spec in original.CASES]):
        raise ValueError("frozen v1.6 seed shape differs")
    for row in rows:
        learning_example_v16_from_record(row, split=row["split"])
    output_dir.mkdir(mode=0o700)
    code_ref = _write(output_dir, "generator_original.py", code)
    conversion_ref = _write(output_dir, "converter_implementation.py", converter)
    cases = []
    source_refs: dict[str, dict[str, str]] = {}
    for row, line, spec in zip(rows, lines, original.CASES, strict=True):
        if row["split"] != "train":
            continue
        case_id = row["case_id"]
        if (row["status"]["qualification"] != "diagnostic-only-unadmitted" or
                row["lineage"]["generator_family"] != original.GENERATOR_FAMILY):
            raise ValueError("original status or generator lineage differs")
        original_line = line
        remap = SOURCE_REF_REMAP.get(case_id, {})
        if remap:
            if not all(any(s["ref_id"] == old for s in row["sources"])
                       for old in remap):
                raise ValueError("known counterfactual source ref differs")
            row = _remap_refs(row, remap)
            line = _raw(row)
        example = learning_example_v16_from_record(row, split="train")
        supervised_output_record_v16(example)
        prefix = f"cases/{case_id}"
        original_candidate_ref = _write(output_dir,
                                        f"{prefix}/candidate-original.json", original_line)
        candidate_ref = _write(output_dir, f"{prefix}/candidate.json", line)
        generator_input_ref = _write(output_dir, f"{prefix}/generator-input.json",
                                     _raw({"schema": "mfm-v16-authoring-seed-generator-input-v1",
                                           "case_id": case_id,
                                           "generator_family": original.GENERATOR_FAMILY,
                                           "generator_code_sha256": code_ref["sha256"],
                                           "original_spec": spec}))
        sources = []
        rights_drafts = []
        host_family = row["context"]["base_context"]["scope"]["host_instance_id"]
        for index, source in enumerate(row["sources"]):
            raw = b64decode(source["content_b64"], validate=True)
            source_id = source["ref_id"]
            if source_id in source_refs:
                ref = source_refs[source_id]
                if ref["sha256"] != _sha(raw):
                    raise ValueError("source ID has different bytes after counterfactual remap")
            else:
                ref = _write(output_dir, f"sources/{source_id}.txt", raw)
                source_refs[source_id] = ref
            sources.append({"source_id": source["ref_id"], **ref})
            rights_drafts.append(_rights_draft(output_dir, prefix, index,
                                               sources[-1], host_family))
        source_only_ref = _write(output_dir, f"{prefix}/source-only.json", _raw({
            "schema": "mfm-v16-authoring-seed-source-only-v1",
            "case_id": case_id, "training_admitted": False,
            "generator_family": original.GENERATOR_FAMILY,
            "sources": sources,
        }))
        target = {"schema": TARGET_SCHEMA_V16, "context": row["context"],
                  "source_ids": [source["ref_id"] for source in row["sources"]],
                  **{field: row["target"][field] for field in (
                      "proposals", "dispositions", "adjudications")}}
        target_ref = _write(output_dir, f"{prefix}/target.json", _raw(target))
        provenance_ref = _write(output_dir, f"{prefix}/provenance.json", _raw({
            "schema": "mfm-v16-authoring-seed-train-provenance-v1",
            "case_id": case_id, "case_sha256": candidate_ref["sha256"],
            "rendered_prompt_sha256": generator_input_ref["sha256"],
            "generator_output_sha256": target_ref["sha256"],
            "converter_sha256": conversion_ref["sha256"],
            "generator_source_code_sha256": code_ref["sha256"],
            "frozen_corpus_sha256": CORPUS_SHA256,
            "original_candidate_sha256": original_candidate_ref["sha256"],
            "source_ref_remapping": remap,
            "source_only_sha256": source_only_ref["sha256"],
            "rights_status": "unverified", "training_admitted": False,
            "independent_review": False,
        }))
        cases.append({"case_id": case_id,
                      "authorization_id": AUTHORIZATION,
                      "host_family": host_family,
                      "source_family": row["lineage"]["host_family"],
                      "generator_family": original.GENERATOR_FAMILY,
                      "parent_case_ids": row["lineage"]["parent_case_ids"],
                      "candidate": candidate_ref,
                      "original_candidate": original_candidate_ref,
                      "source_ref_remapping": remap,
                      "source_only": source_only_ref,
                      "sources": sources, "rights_drafts": rights_drafts,
                      "generator_input": generator_input_ref,
                      "generator_output": target_ref, "converter_provenance": provenance_ref})
    if len(cases) != 15:
        raise ValueError("materialization lost a train case")
    receipt = {"schema": RECEIPT_SCHEMA, "status": "unadmitted-pending-rights",
               "authorization_id": AUTHORIZATION,
               "generator_family": original.GENERATOR_FAMILY,
               "frozen_corpus_sha256": CORPUS_SHA256,
               "original_generator": code_ref, "target_converter": conversion_ref,
               "train_cases": len(cases), "cases": cases,
               "source_rights_created": False,
               "pending_rights_drafts_created": True,
               "qualified_for_product": False}
    receipt_ref = _write(output_dir, "receipt.json", _raw(receipt))
    return {"receipt_path": str(output_dir / receipt_ref["path"]),
            "receipt_sha256": receipt_ref["sha256"],
            "train_cases": len(cases), "status": receipt["status"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
