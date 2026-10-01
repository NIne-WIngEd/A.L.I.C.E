#!/usr/bin/env python3
"""Issue agent-authored fictional-source rights and assemble v1.6 teacher data.

This narrow issuer covers only the pinned project-authored seed, Mira
revocation, and deterministic development bundles under the owner's explicit
chat direction. It preserves unissued drafts, identifies the agent as issuer,
and makes no claim of a real owner's utterance, signature or independent gold.
Public/third-party source families are deliberately unsupported.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import stat
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cognitive_kernel.canonical import canonical_json_bytes
from cognitive_kernel.formation_dataset_admission import _identifier, _path, _unique_json
from scripts.mfm.assemble_v16_owner_teacher_corpus import (
    INTAKE_SCHEMA, STATUS, assemble,
)


AUTHORIZATION = "owner-directed-mfm-v16-synthetic"
ISSUER = "alice-agent-under-owner-direction"
ATTESTATION_METHOD = "owner-chat-direction"
OWNER_AUTHORIZATION_DATE = "2026-10-01"
SEED_FAMILY = "assistant-authored-v16-seed-20260930"
ROLE_FAMILY = "codex-fictional-role-chronicle-20261001"
DEV_FAMILY = "deterministic-fictional-workbench-garden-20261001"
SEED_CORPUS_SHA256 = "8e967ee6494f2c5ec47ab602bad213df8c07c53b1abe9a9aff950f19c0e78e22"
ROLE_CORPUS_SHA256 = "2096cb9a148678e9a2a4a2fea9a3c09d44e9f45db448713998667f3f4219d08e"


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _raw(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _pinned_json(path: Path, pin: str, label: str) -> dict:
    raw = path.read_bytes()
    if _sha(raw) != pin:
        raise ValueError(f"{label} differs from externally pinned SHA-256")
    return _unique_json(raw, label)


def _ref(root: Path, value: dict, label: str) -> tuple[str, bytes]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} needs exact path and SHA-256")
    name, absolute = _path(root, value["path"], label)
    raw = absolute.read_bytes()
    if _sha(raw) != value["sha256"]:
        raise ValueError(f"{label} differs from exact bytes")
    return name, raw


def _copy_bundle(source: Path, target: Path) -> None:
    if not source.is_dir() or source.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("source bundle must be a private directory outside Git")
    for path in source.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("source bundle contains unsupported link or file")
    shutil.copytree(source, target)
    for path in target.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)


def _issued_rights(bundle: Path, source_ref: dict, draft_ref: dict, *,
                   host: str, split: str, issued_at_utc: str) -> tuple[dict, dict]:
    source_id = _identifier(source_ref["source_id"], "source_id")
    _, source_raw = _ref(bundle, {key: source_ref[key] for key in ("path", "sha256")},
                         "fictional source")
    _, draft_raw = _ref(bundle, draft_ref, "unissued source-rights draft")
    draft = _unique_json(draft_raw, "unissued source-rights draft")
    unissued_format = (
        (draft.get("schema") == "mfm-source-rights-v1" and
         draft.get("draft_unissued") is True) or
        (draft.get("schema") == "mfm-source-rights-unissued-draft-v1" and
         draft.get("status") == "unissued-not-admissible"))
    if (not unissued_format or
            draft.get("source_id") != source_id or
            draft.get("source_sha256") != _sha(source_raw) or
            draft.get("host_family") != host or
            any(draft.get(field) is not None for field in (
                "issuer_id", "authority_ref", "formation_training",
                "formation_evaluation", "model_distribution", "revoked"))):
        raise ValueError("draft is issued, unbound or has an asserted permission")
    record = {"schema": "mfm-source-rights-v1", "issuer_id": ISSUER,
              "authority_ref": AUTHORIZATION,
              "attestation_method": ATTESTATION_METHOD,
              "owner_authorization_date": OWNER_AUTHORIZATION_DATE,
              "issued_at_utc": issued_at_utc,
              "attestation_scope": "owner-directed-fictional-mfm-teaching",
              "source_origin": "project-authored-fictional-synthetic",
              "owner_statement_is_simulated": True,
              "owner_signature_claimed": False,
              "independent_rights_review_claimed": False,
              "source_id": source_id, "source_sha256": _sha(source_raw),
              "host_family": host,
              "formation_training": split == "train",
              "formation_evaluation": split == "development",
              "model_distribution": split == "train",
              "revoked": False,
              "supersedes_unissued_draft_sha256": _sha(draft_raw)}
    return record, {"source_id": source_id,
                    "path": source_ref["path"], "sha256": source_ref["sha256"],
                    "parent_source_ids": source_ref.get("parent_source_ids", [])}


def _issue_staged(*, seed_bundle: Path, seed_receipt_sha256: str,
                       role_bundle: Path, role_receipt_sha256: str,
                       development_bundle: Path, development_index_sha256: str,
                       output_dir: Path, issued_at_utc: str) -> dict:
    for required in (ISSUER, AUTHORIZATION, ATTESTATION_METHOD):
        _identifier(required, "synthetic rights identity")
    root = output_dir.resolve()
    if not output_dir.is_dir() or root.is_relative_to(ROOT.resolve()):
        raise ValueError("private staging directory outside Git is required")
    seed = _pinned_json(seed_bundle / "receipt.json", seed_receipt_sha256,
                        "original seed receipt")
    role = _pinned_json(role_bundle / "receipt.json", role_receipt_sha256,
                        "Mira role receipt")
    development = _pinned_json(development_bundle / "bundle-index.json",
                               development_index_sha256, "development index")
    if (seed.get("schema") != "mfm-v16-authoring-seed-train-materialization-v1" or
            seed.get("status") != "unadmitted-pending-rights" or
            seed.get("train_cases") != 15 or len(seed.get("cases", [])) != 15 or
            seed.get("generator_family") != SEED_FAMILY or
            seed.get("frozen_corpus_sha256") != SEED_CORPUS_SHA256 or
            seed.get("authorization_id") != AUTHORIZATION or
            seed.get("source_rights_created") is not False or
            role.get("schema") != "mfm-v16-role-revocation-materialization-v1" or
            role.get("status") != "unadmitted-pending-rights" or
            role.get("case_id") != "role-source-person-revocation" or
            role.get("generator_family") != ROLE_FAMILY or
            role.get("original_corpus_sha256") != ROLE_CORPUS_SHA256 or
            role.get("authorization_id") != AUTHORIZATION or
            role.get("source_rights_created") is not False or
            development.get("schema") != "mfm-v16-deterministic-development-bundle-index-v1" or
            development.get("status") != "unreviewed-unadmitted-source-rights-absent" or
            development.get("authorization_id") != AUTHORIZATION or
            development.get("generator_family") != DEV_FAMILY or
            len(development.get("cases", [])) != 9):
        raise ValueError("input families differ from the narrowly authorized synthetic scope")
    # The source-code lineage pinned by each materialization receipt is part
    # of what is being attested, even though admission validates converter
    # provenance separately.
    _ref(seed_bundle, seed["original_generator"], "original seed generator")
    _ref(role_bundle, role["original_generator"], "original role generator")
    for key, source in (("seed", seed_bundle), ("role", role_bundle),
                        ("development", development_bundle)):
        _copy_bundle(source, output_dir / key)
    cases: list[dict] = []
    issued: dict[str, tuple[bytes, str, str, dict]] = {}

    def register_case(key: str, row: dict, *, split: str, author: str,
                      producer_version: str, converter: dict,
                      prompt: dict, response: dict, provenance: dict,
                      drafts: list[dict], scenario: str, duplicate: str) -> None:
        bundle = output_dir / key
        if (row["authorization_id"] != AUTHORIZATION or
                row["generator_family"] not in {SEED_FAMILY, ROLE_FAMILY, DEV_FAMILY}):
            raise ValueError("case authorization or generator family differs")
        case_id = _identifier(row["case_id"], "case_id")
        _, candidate_raw = _ref(bundle, row["candidate"], "case candidate")
        candidate = _unique_json(candidate_raw, "case candidate")
        if (candidate.get("case_id") != case_id or candidate.get("split") != split or
                candidate.get("authorization_id") != AUTHORIZATION):
            raise ValueError("candidate split, ID or authorization differs")
        host = row["host_family"]
        source_list = row["sources"]
        if len(drafts) != len(source_list):
            raise ValueError("source draft count differs")
        incoming = []
        for source, draft in zip(source_list, drafts, strict=True):
            record, source_binding = _issued_rights(bundle, source, draft,
                                                    host=host, split=split,
                                                    issued_at_utc=issued_at_utc)
            raw = _raw(record)
            source_id = source_binding["source_id"]
            previous = issued.get(source_id)
            if previous is not None and (
                    previous[1] != source_binding["sha256"] or
                    previous[3]["host"] != host or
                    previous[3]["split"] != split or
                    previous[3]["source_path"] != key + "/" + source_binding["path"]):
                raise ValueError("one source ID has conflicting synthetic rights binding")
            if previous is None:
                issued[source_id] = (raw, source_binding["sha256"],
                                     key + "/" + source_binding["path"],
                                     {"host": host, "split": split,
                                      "bundle": key,
                                      "source_path": key + "/" + source_binding["path"]})
            else:
                # Shared source bytes can have two identical pending draft
                # files. The issued receipt refers to the first exact draft.
                raw = previous[0]
            rights_name = f"issued-rights/{source_id}.json"
            rights_ref = {"rights_path": rights_name,
                          "rights_sha256": _sha(raw)}
            incoming.append({"source_id": source_id,
                             "path": key + "/" + source_binding["path"],
                             "sha256": source_binding["sha256"],
                             "parent_source_ids": source_binding["parent_source_ids"],
                             **rights_ref})
        cases.append({"case_id": case_id, "split": split,
                      "author_id": author, "host_family": host,
                      "source_family": row["source_family"],
                      "generator_family": row["generator_family"],
                      "scenario_family": scenario, "duplicate_group": duplicate,
                      "parent_case_ids": row.get("parent_case_ids", []),
                      "candidate": {"path": key + "/" + row["candidate"]["path"],
                                    "sha256": row["candidate"]["sha256"]},
                      "sources": incoming,
                      "target_origin": "licensed-deterministic-generator",
                      "producer_version": producer_version,
                      "prompt": {"path": key + "/" + prompt["path"],
                                 "sha256": prompt["sha256"]},
                      "response": {"path": key + "/" + response["path"],
                                   "sha256": response["sha256"]},
                      "converter": {"path": key + "/" + converter["path"],
                                    "sha256": converter["sha256"]},
                      "converter_provenance": {
                          "path": key + "/" + provenance["path"],
                          "sha256": provenance["sha256"]}})

    for row in seed["cases"]:
        register_case("seed", row, split="train",
                      author="openai-codex-synthetic-teacher",
                      producer_version=SEED_FAMILY,
                      converter=seed["target_converter"],
                      prompt=row["generator_input"], response=row["generator_output"],
                      provenance=row["converter_provenance"],
                      drafts=row["rights_drafts"], scenario=row["source_family"],
                      duplicate=row["source_family"])
    register_case("role", role, split="train",
                  author="openai-codex-synthetic-teacher",
                  producer_version=ROLE_FAMILY,
                  converter=role["target_converter"], prompt=role["generator_input"],
                  response=role["generator_output"],
                  provenance=role["converter_provenance"],
                  drafts=role["rights_drafts"], scenario=role["source_family"],
                  duplicate=role["source_family"])
    for row in development["cases"]:
        register_case("development", row, split="development",
                      author=row["author_id_candidate"],
                      producer_version=DEV_FAMILY,
                      converter=row["converter"], prompt=row["prompt"],
                      response=row["response"],
                      provenance=row["converter_provenance"],
                      drafts=[source["rights_draft"] for source in row["sources"]],
                      scenario=row["scenario_family"],
                      duplicate=row["duplicate_group"])
    for source_id, (raw, _, _, _) in sorted(issued.items()):
        dest = output_dir / "issued-rights" / f"{source_id}.json"
        dest.parent.mkdir(mode=0o700, exist_ok=True)
        with dest.open("xb") as stream:
            stream.write(raw)
        dest.chmod(stat.S_IRUSR | stat.S_IWUSR)
    intake = {"schema": INTAKE_SCHEMA, "status": STATUS,
              "corpus_id": "owner-directed-fictional-v16-teacher-20261001",
              "authorization_id": AUTHORIZATION, "cases": cases}
    intake_path = output_dir / "intake.json"
    with intake_path.open("xb") as stream:
        stream.write(_raw(intake))
    intake_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    assembly = assemble(output_dir, intake_path,
                        intake_sha256=_sha(intake_path.read_bytes()))
    receipt = {"schema": "mfm-v16-chat-directed-synthetic-issuance-v1",
               "issuer_id": ISSUER, "authority_ref": AUTHORIZATION,
               "attestation_method": ATTESTATION_METHOD,
               "owner_authorization_date": OWNER_AUTHORIZATION_DATE,
               "issued_at_utc": issued_at_utc,
               "seed_receipt_sha256": seed_receipt_sha256,
               "role_receipt_sha256": role_receipt_sha256,
               "development_index_sha256": development_index_sha256,
               "intake_sha256": assembly["intake_sha256"],
               "manifest_sha256": assembly["manifest_sha256"],
               "issued_source_rights_receipts": len(issued),
               "train_cases": assembly["train_cases"],
               "development_cases": assembly["development_cases"],
               "source_rights_scope": "fictional-project-authored-only",
               "independent_gold": False, "qualified_for_product": False,
               "cpu_processor_pass_completed": False}
    receipt_path = output_dir / "issuance-receipt.json"
    with receipt_path.open("xb") as stream:
        stream.write(_raw(receipt))
    receipt_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return {"manifest_path": assembly["manifest_path"],
            "manifest_sha256": assembly["manifest_sha256"],
            "intake_sha256": assembly["intake_sha256"],
            "issuance_receipt_path": str(receipt_path),
            "issuance_receipt_sha256": _sha(receipt_path.read_bytes()),
            "train_cases": assembly["train_cases"],
            "development_cases": assembly["development_cases"],
            "issued_source_rights_receipts": len(issued),
            "qualified_for_product": False}


def issue_and_assemble(*, seed_bundle: Path, seed_receipt_sha256: str,
                       role_bundle: Path, role_receipt_sha256: str,
                       development_bundle: Path, development_index_sha256: str,
                       output_dir: Path) -> dict:
    """Publish only a complete, admitted private bundle; clean our own failed stage."""
    destination = output_dir.resolve()
    if destination.is_relative_to(ROOT.resolve()):
        raise ValueError("private output directory outside Git is required")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("new private output directory is required")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".mfm-v16-issue-", dir=output_dir.parent))
    stage.chmod(0o700)
    try:
        issued_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z")
        result = _issue_staged(
            seed_bundle=seed_bundle, seed_receipt_sha256=seed_receipt_sha256,
            role_bundle=role_bundle, role_receipt_sha256=role_receipt_sha256,
            development_bundle=development_bundle,
            development_index_sha256=development_index_sha256,
            output_dir=stage, issued_at_utc=issued_at_utc)
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError("destination appeared during private staging")
        stage.rename(output_dir)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    for key in ("manifest_path", "issuance_receipt_path"):
        result[key] = str(output_dir / Path(result[key]).relative_to(stage))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seed-bundle", "role-bundle", "development-bundle", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("seed-receipt-sha256", "role-receipt-sha256",
                 "development-index-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--issuer-id", required=True, choices=(ISSUER,))
    parser.add_argument("--authority-ref", required=True, choices=(AUTHORIZATION,))
    parser.add_argument("--attestation-method", required=True,
                        choices=(ATTESTATION_METHOD,))
    args = parser.parse_args()
    print(json.dumps(issue_and_assemble(
        seed_bundle=args.seed_bundle, seed_receipt_sha256=args.seed_receipt_sha256,
        role_bundle=args.role_bundle, role_receipt_sha256=args.role_receipt_sha256,
        development_bundle=args.development_bundle,
        development_index_sha256=args.development_index_sha256,
        output_dir=args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
