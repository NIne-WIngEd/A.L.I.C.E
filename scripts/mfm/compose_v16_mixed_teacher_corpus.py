#!/usr/bin/env python3
"""Compose pinned synthetic and source-grounded AMI teacher cases in private custody.

Copies exact source and provenance bytes into a new private bundle. Candidate
authorization IDs and source-rights records are never rewritten. The resulting
v2 corpus is owner-authorized teacher training with diagnostic development,
not independently reviewed gold or product qualification.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cognitive_kernel.canonical import canonical_json_bytes
from cognitive_kernel.formation_dataset_admission import (
    _identifier, _path, _unique_json, admit_formation_corpus,
)
from scripts.mfm.assemble_v16_owner_teacher_corpus import (
    INTAKE_SCHEMA_V2, STATUS, assemble,
)

AMI_AUTH = "owner-directed-mfm-v16-public-teacher"
AMI_FAMILY = "ami-manual-2017"
AMI_AUTHORITY = "ami-manual-v162-ccby4-publisher-release"
AMI_ISSUER = "alice-agent-under-owner-direction"
AMI_CONVERTER = ROOT / "scripts/mfm/prepare_v16_ami_longitudinal_teacher_case.py"


def _digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _pinned(path: Path, expected: str) -> bytes:
    raw = path.read_bytes()
    if _digest(raw) != expected:
        raise ValueError(f"pinned bytes differ: {path}")
    return raw


def _ref(root: Path, relative: str, expected: str | None = None) -> dict[str, str]:
    _, path = _path(root, relative, "input reference")
    raw = path.read_bytes()
    digest = _digest(raw)
    if expected is not None and digest != expected:
        raise ValueError(f"pinned bytes differ: {relative}")
    return {"path": relative, "sha256": digest}


def _prefixed(ref: dict, prefix: str) -> dict:
    return {"path": f"{prefix}/{ref['path']}", "sha256": ref["sha256"]}


def compose(*, synthetic_root: Path, synthetic_intake_sha256: str,
            synthetic_manifest_sha256: str, ami_root: Path, ami_config: Path,
            ami_config_sha256: str, ami_issuance_sha256: str,
            output_root: Path, owner_authorization_ref: str,
            check_role_coverage: bool = True) -> dict:
    if output_root.exists() or output_root.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("output must be a new private directory outside Git")
    owner_authorization_ref = _identifier(owner_authorization_ref, "owner_authorization_ref")
    original = _unique_json(_pinned(synthetic_root / "intake.json", synthetic_intake_sha256),
                            "synthetic intake")
    admitted = admit_formation_corpus(
        synthetic_root / "owner-teacher-manifest.json",
        expected_sha256=synthetic_manifest_sha256, teacher_training=True,
        owner_authorization_ref=original.get("authorization_id"))
    if (original.get("schema") != "mfm-v16-owner-teacher-assembly-intake-v1" or
            original.get("status") != STATUS or
            len(original.get("cases", [])) != len(admitted.train) + len(admitted.development)):
        raise ValueError("original synthetic intake does not match admitted manifest")
    synthetic_auth = _identifier(original["authorization_id"], "synthetic authorization_id")
    config = _unique_json(_pinned(ami_config, ami_config_sha256), "AMI case config")
    issuance = _unique_json(_pinned(ami_root / "issued-rights/issuance-manifest.json",
                                  ami_issuance_sha256), "AMI rights issuance")
    if config.get("schema") != "mfm-v16-ami-public-teacher-expansion-v1" or \
            issuance.get("schema") != "mfm-v16-public-license-issuance-v1" or \
            not isinstance(config.get("cases"), list) or len(config["cases"]) != 2:
        raise ValueError("wrong AMI config or rights issuance")
    receipt_refs = {ref["path"]: ref["sha256"] for ref in issuance["receipts"]}
    if len(receipt_refs) != issuance.get("source_count"):
        raise ValueError("AMI issuance count or receipt paths differ")
    if not issuance["source_count"]:
        raise ValueError("AMI rights issuance empty")
    converter = _ref(ROOT, str(AMI_CONVERTER.relative_to(ROOT)))
    copied_converter = {"path": f"converters/{AMI_CONVERTER.name}",
                        "sha256": converter["sha256"]}
    rows, bindings = [], {}
    for original_case in original["cases"]:
        case = dict(original_case)
        case["authorization_id"] = synthetic_auth
        for field in ("candidate", "prompt", "response", "converter",
                      "converter_provenance"):
            case[field] = _prefixed(case[field], "synthetic")
        case["sources"] = [
            {**source, "path": f"synthetic/{source['path']}",
             "rights_path": f"synthetic/{source['rights_path']}"}
            for source in case["sources"]]
        family = _identifier(case["source_family"], "source_family")
        for source in original_case["sources"]:
            rights = _unique_json(_pinned(synthetic_root / source["rights_path"],
                                         source["rights_sha256"]), "synthetic rights")
            binding = (rights.get("authority_ref"), rights.get("issuer_id"))
            if family in bindings and bindings[family] != binding:
                raise ValueError("synthetic source family has conflicting rights authority")
            bindings[family] = binding
        rows.append(case)
    rights_seen: set[str] = set()
    for item in config["cases"]:
        case_id = _identifier(item["case_id"], "AMI case_id")
        prefix = f"ami/{case_id}"
        directory = ami_root / case_id
        candidate_ref = _ref(ami_root, f"{case_id}/teacher-candidate.json",
                             item["case_sha256"])
        candidate = _unique_json((ami_root / candidate_ref["path"]).read_bytes(), "AMI candidate")
        envelope = _unique_json(_pinned(directory / "envelope.json", item["envelope_sha256"]),
                                "AMI envelope")
        if (candidate.get("case_id") != case_id or candidate.get("split") != "train" or
                candidate.get("authorization_id") != AMI_AUTH or
                envelope.get("authorization_id") != AMI_AUTH or
                envelope.get("source_family") != AMI_FAMILY):
            raise ValueError("AMI case original authorization, family or split differs")
        prompt = _ref(ami_root, f"{case_id}/prompt.txt", item["prompt_sha256"])
        response = _ref(ami_root, f"{case_id}/teacher-response.json", item["response_sha256"])
        provenance = _ref(ami_root, f"{case_id}/teacher-provenance.json",
                          item["provenance_sha256"])
        provenance_body = _unique_json((ami_root / provenance["path"]).read_bytes(),
                                       "AMI teacher provenance")
        if (provenance_body.get("case_sha256") != candidate_ref["sha256"] or
                provenance_body.get("converter_sha256") != copied_converter["sha256"] or
                provenance_body.get("rendered_prompt_sha256") != prompt["sha256"] or
                provenance_body.get("teacher_response_sha256") != response["sha256"]):
            raise ValueError("AMI teacher provenance differs from pinned case or converter")
        sources = []
        for source in envelope["sources"]:
            source_ref = _ref(ami_root, f"{case_id}/{source['path']}", source["sha256"])
            rights_name = f"{source['ref_id']}.json"
            if rights_name not in receipt_refs:
                raise ValueError("AMI source has no pinned rights receipt")
            rights_ref = _ref(ami_root, f"issued-rights/{rights_name}",
                              receipt_refs[rights_name])
            rights = _unique_json((ami_root / rights_ref["path"]).read_bytes(), "AMI source rights")
            if (rights.get("source_id") != source["ref_id"] or
                    rights.get("source_sha256") != source_ref["sha256"] or
                    rights.get("host_family") != envelope.get("host_id") or
                    (rights.get("authority_ref"), rights.get("issuer_id")) !=
                    (AMI_AUTHORITY, AMI_ISSUER)):
                raise ValueError("AMI rights do not bind exact source and publisher authority")
            rights_seen.add(rights_name)
            sources.append({"source_id": source["ref_id"],
                            "path": f"ami/{source_ref['path']}",
                            "sha256": source_ref["sha256"],
                            "rights_path": f"ami/{rights_ref['path']}",
                            "rights_sha256": rights_ref["sha256"],
                            "parent_source_ids": []})
        if not sources or len(sources) != len(candidate.get("sources", [])):
            raise ValueError("AMI source evidence count differs from candidate")
        history = case_id.rsplit("-", 2)[0]
        rows.append({"case_id": case_id, "split": "train",
                     "author_id": "openai-codex-synthetic-teacher",
                     "host_family": envelope["host_id"], "source_family": AMI_FAMILY,
                     "generator_family": "ami-manual-openai-codex-teacher",
                     "scenario_family": history, "duplicate_group": history,
                     "parent_case_ids": [], "authorization_id": AMI_AUTH,
                     "candidate": _prefixed(candidate_ref, "ami"),
                     "sources": sources,
                     "target_origin": "owner-authorized-service-teacher",
                     "producer_version": "ami-v16-cross-stage-teacher-v1",
                     "prompt": _prefixed(prompt, "ami"),
                     "response": _prefixed(response, "ami"),
                     "converter": copied_converter,
                     "converter_provenance": _prefixed(provenance, "ami")})
    if rights_seen != set(receipt_refs):
        raise ValueError("AMI issuance contains unselected or missing source receipts")
    registrations = [
        {"authorization_id": synthetic_auth, "source_family": family,
         "rights_authority_ref": authority, "rights_issuer_id": issuer}
        for family, (authority, issuer) in sorted(bindings.items())]
    registrations.append({"authorization_id": AMI_AUTH, "source_family": AMI_FAMILY,
                          "rights_authority_ref": AMI_AUTHORITY, "rights_issuer_id": AMI_ISSUER})
    intake = {"schema": INTAKE_SCHEMA_V2, "status": STATUS,
              "corpus_id": "mfm-v16-synthetic-ami-teacher", "authorization_id": owner_authorization_ref,
              "authorizations": registrations, "cases": rows}
    for source_root in (synthetic_root, ami_root):
        if any(path.is_symlink() for path in source_root.rglob("*")):
            raise ValueError("private bundle cannot contain symlinks")
    try:
        output_root.mkdir(mode=0o700)
        shutil.copytree(synthetic_root, output_root / "synthetic")
        shutil.copytree(ami_root, output_root / "ami")
        converter_dest = output_root / copied_converter["path"]
        converter_dest.parent.mkdir(mode=0o700)
        shutil.copy2(AMI_CONVERTER, converter_dest)
        intake_raw = canonical_json_bytes(intake) + b"\n"
        intake_path = output_root / "intake.json"
        fd = os.open(intake_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(intake_raw)
        assembled = assemble(output_root, intake_path,
                             intake_sha256=_digest(intake_raw),
                             check_role_coverage=check_role_coverage)
        return {"schema": "mfm-v16-mixed-teacher-composition-receipt-v1",
                "synthetic_manifest_sha256": synthetic_manifest_sha256,
                "ami_config_sha256": ami_config_sha256,
                "ami_issuance_sha256": ami_issuance_sha256,
                "authorization_id": owner_authorization_ref,
                "authorization_count": len(registrations),
                "intake_sha256": _digest(intake_raw), **assembled}
    except BaseException:
        if output_root.exists():
            shutil.rmtree(output_root)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--synthetic-intake-sha256", required=True)
    parser.add_argument("--synthetic-manifest-sha256", required=True)
    parser.add_argument("--ami-root", type=Path, required=True)
    parser.add_argument("--ami-config", type=Path, required=True)
    parser.add_argument("--ami-config-sha256", required=True)
    parser.add_argument("--ami-issuance-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--owner-authorization-ref", required=True)
    args = parser.parse_args()
    print(json.dumps(compose(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
