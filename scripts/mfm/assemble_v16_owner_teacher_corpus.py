#!/usr/bin/env python3
"""Freeze exact, already-authored v1.6 teacher candidates into a private corpus.

This tool creates targets and a manifest, never source permissions or teacher
answers. The owner/steward supplies and authenticates rights, provenance and
lineage independently. Neither a successful assembly nor admission is a gold
review, a GPU authorization, or a capability result.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cognitive_kernel.canonical import canonical_json_bytes
from cognitive_kernel.formation_dataset_admission import (
    TEACHER_SCHEMA, _identifier, _path, _unique_json, admit_formation_corpus,
)
from cognitive_kernel.formation_learning_v16 import (
    CURRICULUM_SCHEMA_V16, TARGET_SCHEMA_V16, admitted_rows_v16,
    learning_example_v16_from_record, supervised_output_record_v16,
)


INTAKE_SCHEMA = "mfm-v16-owner-teacher-assembly-intake-v1"
STATUS = "owner-authorized-teacher-training-only-unqualified"
FAMILY_FIELDS = ("host_family", "source_family", "generator_family",
                 "scenario_family", "duplicate_group")


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _read_ref(root: Path, ref: dict, name: str) -> bytes:
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError(f"{name} needs exact path and SHA-256")
    _, path = _path(root, ref["path"], name)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{name} is missing or unreadable") from exc
    if _sha(raw) != ref["sha256"]:
        raise ValueError(f"{name} differs from externally pinned bytes")
    return raw


def _ref(root: Path, value: object, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{name} needs exact path and SHA-256")
    _read_ref(root, value, name)
    return {"path": value["path"], "sha256": value["sha256"]}


def _source(root: Path, value: object, expected_id: str, expected_raw: bytes) -> dict:
    if not isinstance(value, dict) or set(value) != {
            "source_id", "path", "sha256", "rights_path", "rights_sha256",
            "parent_source_ids"}:
        raise ValueError("source needs exact bytes, existing rights and ancestry")
    if _identifier(value["source_id"], "source_id") != expected_id:
        raise ValueError("source ID differs from the candidate evidence ref")
    source_ref = {key: value[key] for key in ("path", "sha256")}
    if _read_ref(root, source_ref, "source") != expected_raw:
        raise ValueError("source bytes differ from the candidate evidence")
    _read_ref(root, {"path": value["rights_path"],
                     "sha256": value["rights_sha256"]}, "source rights")
    parents = value["parent_source_ids"]
    if not isinstance(parents, list) or len(parents) != len(set(parents)):
        raise ValueError("source parent IDs must be a unique list")
    for parent in parents:
        _identifier(parent, "parent_source_id")
    return value


def _provenance(root: Path, value: object, *, candidate: dict, case_raw: bytes,
                prompt: dict, response: dict, converter: dict, origin: str) -> None:
    raw = _read_ref(root, value, "converter provenance")
    record = _unique_json(raw, "converter provenance")
    bindings = {"case_id": candidate["case_id"], "case_sha256": _sha(case_raw),
                "rendered_prompt_sha256": prompt["sha256"],
                "converter_sha256": converter["sha256"]}
    bindings[("teacher_response_sha256" if origin == "owner-authorized-service-teacher"
              else "generator_output_sha256")] = response["sha256"]
    for name, expected in bindings.items():
        if record.get(name) != expected:
            raise ValueError(f"converter provenance has wrong {name}")


def _candidate(root: Path, row: dict, authorization: str) -> tuple[dict, bytes]:
    case_raw = _read_ref(root, row["candidate"], "candidate")
    candidate = _unique_json(case_raw, "candidate")
    if (candidate.get("schema") != CURRICULUM_SCHEMA_V16 or
            candidate.get("case_id") != row["case_id"] or
            candidate.get("split") != row["split"] or
            candidate.get("authorization_id") != authorization):
        raise ValueError("candidate schema, case, split or authorization differs")
    example = learning_example_v16_from_record(candidate, split=row["split"])
    supervised_output_record_v16(example)
    if row["host_family"] != example.context.base.scope.host_instance_id:
        raise ValueError("host family differs from registered source context")
    if len(candidate["sources"]) != len(row["sources"]):
        raise ValueError("candidate source count differs from the intake")
    from base64 import b64decode
    for given, source in zip(row["sources"], candidate["sources"], strict=True):
        _source(root, given, source["ref_id"], b64decode(source["content_b64"], validate=True))
    return candidate, case_raw


def _exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def assemble(bundle_root: Path, intake_path: Path, *, intake_sha256: str,
             manifest_path: str = "owner-teacher-manifest.json",
             check_role_coverage: bool = True) -> dict:
    """Create a frozen corpus under bundle_root after exact intake validation.

    All source, prompt, teacher output, conversion code, provenance and rights
    files must already exist under the private bundle root. No external file is
    altered. Output paths must not preexist; a failed run removes only files
    that this invocation created.
    """
    root = bundle_root.resolve()
    if root.is_relative_to(ROOT.resolve()):
        raise ValueError("owner corpus must be outside the Git checkout")
    if not root.is_dir():
        raise ValueError("private bundle root does not exist")
    raw = intake_path.read_bytes()
    if _sha(raw) != intake_sha256:
        raise ValueError("intake differs from externally pinned SHA-256")
    intake = _unique_json(raw, "assembly intake")
    if (intake.get("schema") != INTAKE_SCHEMA or
            intake.get("status") != STATUS or
            not isinstance(intake.get("cases"), list) or not intake["cases"]):
        raise ValueError("intake needs owner teacher schema, status and cases")
    authorization = _identifier(intake.get("authorization_id"), "authorization_id")
    corpus_id = _identifier(intake.get("corpus_id"), "corpus_id")
    _, manifest_file = _path(root, manifest_path, "manifest")
    if manifest_file.exists() or manifest_file.is_symlink():
        raise FileExistsError("manifest path already exists")
    rows, outputs = [], {}
    for item in intake["cases"]:
        if not isinstance(item, dict) or set(item) != {
                "case_id", "split", "author_id", *FAMILY_FIELDS,
                "parent_case_ids", "candidate", "sources", "target_origin",
                "producer_version", "prompt", "response", "converter",
                "converter_provenance"}:
            raise ValueError("intake case has missing or unsupported fields")
        case_id = _identifier(item["case_id"], "case_id")
        if item["split"] not in {"train", "development"}:
            raise ValueError("owner teacher split must be train or development")
        for field in (*FAMILY_FIELDS, "author_id", "producer_version"):
            _identifier(item[field], field)
        parents = item["parent_case_ids"]
        if not isinstance(parents, list):
            raise ValueError("parent case IDs must be a list")
        for parent in parents:
            _identifier(parent, "parent_case_id")
        origin = item["target_origin"]
        if origin not in {"owner-authorized-service-teacher",
                          "licensed-deterministic-generator"}:
            raise ValueError("unsupported target origin")
        candidate, candidate_raw = _candidate(root, item, authorization)
        prompt = _ref(root, item["prompt"], "teacher prompt")
        response = _ref(root, item["response"], "raw teacher output")
        converter = _ref(root, item["converter"], "converter implementation")
        _provenance(root, item["converter_provenance"], candidate=candidate,
                    case_raw=candidate_raw, prompt=prompt,
                    response=response, converter=converter, origin=origin)
        target = {"schema": TARGET_SCHEMA_V16, "context": candidate["context"],
                  "source_ids": [ref["ref_id"] for ref in candidate["sources"]],
                  **{key: candidate["target"][key] for key in (
                      "proposals", "dispositions", "adjudications")}}
        target_raw = canonical_json_bytes(target) + b"\n"
        target_path = (f"targets/{case_id}.json" if origin ==
                       "owner-authorized-service-teacher" else response["path"])
        _, absolute = _path(root, target_path, "assembled target")
        if absolute == manifest_file or target_path in outputs:
            raise FileExistsError(f"assembled target path collides: {target_path}")
        if origin == "owner-authorized-service-teacher":
            if absolute.exists() or absolute.is_symlink():
                raise FileExistsError(f"assembled target path already exists: {target_path}")
            outputs[target_path] = target_raw
        elif response["sha256"] != _sha(target_raw) or _read_ref(root, response, "generator output") != target_raw:
            raise ValueError("deterministic generator output must be exact target bytes")
        rows.append({"case_id": case_id, "split": item["split"],
                     "author_id": item["author_id"],
                     **{name: item[name] for name in FAMILY_FIELDS},
                     "parent_case_ids": parents, "authorization_id": authorization,
                     "target_origin": origin,
                     "target_provenance": {
                         "producer_id": item["author_id"],
                         "producer_version": item["producer_version"],
                         "input": prompt, "output": response,
                         "conversion": converter},
                     "sources": item["sources"],
                     "target": {"path": target_path, "sha256": _sha(target_raw)},
                     "reviews": []})
    manifest = {"schema": TEACHER_SCHEMA, "corpus_id": corpus_id,
                "status": STATUS, "authorization_id": authorization,
                "case_count": len(rows), "cases": rows}
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    created: list[Path] = []
    try:
        for path, data in sorted(outputs.items()):
            _, absolute = _path(root, path, "assembled target")
            _exclusive(absolute, data)
            created.append(absolute)
        _exclusive(manifest_file, manifest_raw)
        created.append(manifest_file)
        admitted = admit_formation_corpus(
            manifest_file, expected_sha256=_sha(manifest_raw),
            teacher_training=True, owner_authorization_ref=authorization)
        train = tuple(admitted_rows_v16(admitted, split="train"))
        development = tuple(admitted_rows_v16(admitted, split="development"))
        if check_role_coverage:
            # The CPU data-preflight is the same guard, without importing the
            # large runtime or claiming that the labels have been reviewed.
            from scripts.mfm.train_v16_formation_specialist import (
                _require_critical_construct_coverage, _require_signed_fit_coverage,
            )
            _require_signed_fit_coverage(train, development, policy="teacher")
            _require_critical_construct_coverage(train, development, policy="teacher")
        return {"schema": "mfm-v16-owner-teacher-assembly-receipt-v1",
                "manifest_path": str(manifest_file),
                "manifest_sha256": _sha(manifest_raw),
                "intake_sha256": intake_sha256, "corpus_id": corpus_id,
                "train_cases": len(train), "development_cases": len(development),
                "role_coverage_checked": check_role_coverage,
                "qualified_for_product": False}
    except BaseException:
        for path in reversed(created):
            path.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--intake", type=Path, required=True)
    parser.add_argument("--intake-sha256", required=True)
    parser.add_argument("--manifest-path", default="owner-teacher-manifest.json")
    args = parser.parse_args()
    print(json.dumps(assemble(args.bundle_root, args.intake,
                              intake_sha256=args.intake_sha256,
                              manifest_path=args.manifest_path), sort_keys=True))


if __name__ == "__main__":
    main()
