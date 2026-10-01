#!/usr/bin/env python3
"""Extend a pinned private synthetic-only teacher corpus with train histories.

The old manifest is immutable. A new private directory is published only after
exact admission/role coverage. This issuer grants source-bound permissions only
for its own fictional bytes under the owner's chat direction; it never claims
an owner signature, independent review, or real-person consent.
"""

from __future__ import annotations

import argparse
from base64 import b64decode
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
from cognitive_kernel.formation_dataset_admission import _unique_json
from scripts.mfm.assemble_v16_owner_teacher_corpus import assemble, INTAKE_SCHEMA, STATUS
from scripts.mfm import build_v16_fictional_longitudinal_curriculum as generator

AUTHORIZATION = "owner-directed-mfm-v16-synthetic"
ISSUER = "alice-agent-under-owner-direction"
AUTHOR = "openai-codex-synthetic-teacher"
BASE_MANIFEST_SHA256 = "4e59403ee2205d5219eed09a1dfd712c87b4bbae09dc0435a2275aac18ca7573"
BASE_INTAKE_SHA256 = "1ecc3fc4709db42fde700272caa811c59f2caaef60a6dddbd3207ba41f739538"


def digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def raw_json(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def write(root: Path, name: str, data: bytes) -> dict[str, str]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("xb") as out:
        out.write(data)
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return {"path": name, "sha256": digest(data)}


def _base_copy(base: Path, stage: Path) -> dict:
    if base.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("private base must be outside Git")
    if digest((base / "owner-teacher-manifest.json").read_bytes()) != BASE_MANIFEST_SHA256:
        raise ValueError("old manifest differs from pinned v3")
    if digest((base / "intake.json").read_bytes()) != BASE_INTAKE_SHA256:
        raise ValueError("old intake differs from pinned v3")
    intake = _unique_json((base / "intake.json").read_bytes(), "base intake")
    if (intake.get("schema") != INTAKE_SCHEMA or intake.get("status") != STATUS or
            intake.get("authorization_id") != AUTHORIZATION or
            len(intake.get("cases", [])) != 25):
        raise ValueError("unexpected base intake")
    for path in base.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("base contains unsupported link or file")
        rel = path.relative_to(base)
        if rel.as_posix() in {"owner-teacher-manifest.json", "intake.json", "issuance-receipt.json"}:
            continue
        target = stage / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True, mode=0o700)
        else:
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copyfile(path, target)
            target.chmod(0o600)
    return intake


def _issue(stage: Path, base: Path, issued_at: str) -> dict:
    intake = _base_copy(base, stage)
    code = Path(generator.__file__).read_bytes()
    rendered = generator.render()
    generator_ref = write(stage, "longitudinal/generator.py", code)
    converter_ref = write(stage, "longitudinal/converter.py", Path(__file__).read_bytes())
    frozen_ref = write(stage, "longitudinal/frozen-candidates.jsonl", rendered)
    sources: dict[str, dict] = {}
    new_cases = []
    for index, line in enumerate(rendered.splitlines(keepends=True)):
        row = _unique_json(line, "generated case")
        s = generator.spec(index // len(generator.SCENARIOS),
                           generator.SCENARIOS[index % len(generator.SCENARIOS)])
        case_id, host = row["case_id"], s["host"]
        if (row["split"] != "train" or row["authorization_id"] != AUTHORIZATION or
                row["lineage"]["generator_family"] != generator.GENERATOR_FAMILY or
                row["context"]["base_context"]["scope"]["host_instance_id"] != host):
            raise ValueError("generator metadata differs")
        prefix = f"longitudinal/cases/{case_id}"
        candidate = write(stage, f"{prefix}/candidate.json", line)
        prompt = write(stage, f"{prefix}/generator-input.json", raw_json({
            "schema": "mfm-v16-fictional-longitudinal-input-v1", "case_id": case_id,
            "generator_code_sha256": generator_ref["sha256"], "spec": s}))
        target = {"schema": "mfm-formation-target-v1.6", "context": row["context"],
                  "source_ids": [source["ref_id"] for source in row["sources"]],
                  **{key: row["target"][key] for key in (
                      "proposals", "dispositions", "adjudications")}}
        response = write(stage, f"{prefix}/target.json", raw_json(target))
        provenance = write(stage, f"{prefix}/provenance.json", raw_json({
            "schema": "mfm-v16-fictional-longitudinal-provenance-v1",
            "case_id": case_id, "case_sha256": candidate["sha256"],
            "rendered_prompt_sha256": prompt["sha256"],
            "generator_output_sha256": response["sha256"],
            "converter_sha256": converter_ref["sha256"],
            "generator_source_code_sha256": generator_ref["sha256"],
            "frozen_generator_corpus_sha256": frozen_ref["sha256"],
            "independent_review": False, "qualified_for_product": False}))
        refs = []
        for source in row["sources"]:
            source_id = source["ref_id"]
            payload = b64decode(source["content_b64"], validate=True)
            existing = sources.get(source_id)
            if existing:
                if existing["host"] != host or existing["sha256"] != digest(payload):
                    raise ValueError("source ID reused with different bytes or host")
                refs.append(existing["ref"])
                continue
            source_ref = write(stage, f"longitudinal/sources/{source_id}.bin", payload)
            receipt = {"schema": "mfm-source-rights-v1", "issuer_id": ISSUER,
                       "authority_ref": AUTHORIZATION, "attestation_method": "owner-chat-direction",
                       "owner_authorization_date": "2026-10-01", "issued_at_utc": issued_at,
                       "attestation_scope": "owner-directed-fictional-mfm-teaching",
                       "source_origin": "project-authored-fictional-synthetic",
                       "owner_statement_is_simulated": True, "owner_signature_claimed": False,
                       "independent_rights_review_claimed": False,
                       "source_id": source_id, "source_sha256": source_ref["sha256"],
                       "host_family": host, "formation_training": True,
                       "formation_evaluation": False, "model_distribution": True,
                       "revoked": False}
            rights = write(stage, f"longitudinal/rights/{source_id}.json", raw_json(receipt))
            ref = {"source_id": source_id, **source_ref,
                   "rights_path": rights["path"], "rights_sha256": rights["sha256"],
                   "parent_source_ids": []}
            sources[source_id] = {"sha256": source_ref["sha256"], "host": host, "ref": ref}
            refs.append(ref)
        new_cases.append({
            "case_id": case_id, "split": "train", "author_id": AUTHOR,
            "host_family": host, "source_family": s["source_family"],
            "generator_family": generator.GENERATOR_FAMILY,
            "scenario_family": f"fictional-longitudinal-{s['scenario']}",
            "duplicate_group": f"{host}-{s['pair'] or s['scenario']}",
            "parent_case_ids": [], "candidate": candidate, "sources": refs,
            "target_origin": "licensed-deterministic-generator",
            "producer_version": generator.GENERATOR_FAMILY,
            "prompt": prompt, "response": response, "converter": converter_ref,
            "converter_provenance": provenance})
    intake["corpus_id"] = "owner-directed-fictional-v16-longitudinal-teacher-20261001"
    intake["cases"].extend(new_cases)
    intake_ref = write(stage, "intake.json", raw_json(intake))
    assembly = assemble(stage, stage / "intake.json", intake_sha256=intake_ref["sha256"])
    receipt = {"schema": "mfm-v16-fictional-longitudinal-expansion-receipt-v1",
               "issuer_id": ISSUER, "authority_ref": AUTHORIZATION,
               "owner_authorization_date": "2026-10-01", "issued_at_utc": issued_at,
               "parent_manifest_sha256": BASE_MANIFEST_SHA256,
               "parent_intake_sha256": BASE_INTAKE_SHA256,
               "generator_code_sha256": generator_ref["sha256"],
               "converter_code_sha256": converter_ref["sha256"],
               "generated_cases_sha256": frozen_ref["sha256"],
               "new_train_cases": len(new_cases),
               "new_source_rights_receipts": len(sources),
               "train_cases": assembly["train_cases"],
               "development_cases": assembly["development_cases"],
               "manifest_sha256": assembly["manifest_sha256"],
               "intake_sha256": assembly["intake_sha256"],
               "synthetic_single_author": True, "independent_gold": False,
               "qualified_for_product": False, "cpu_processor_pass_completed": False}
    receipt_ref = write(stage, "longitudinal-issuance-receipt.json", raw_json(receipt))
    return {"manifest_sha256": assembly["manifest_sha256"],
            "intake_sha256": assembly["intake_sha256"],
            "receipt_sha256": receipt_ref["sha256"],
            "new_cases": len(new_cases), "new_source_rights": len(sources),
            "train_cases": assembly["train_cases"],
            "development_cases": assembly["development_cases"],
            "qualified_for_product": False}


def issue(base: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("new private destination outside Git is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".mfm-longitudinal-", dir=output.parent))
    stage.chmod(0o700)
    try:
        instant = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        receipt = _issue(stage, base, instant)
        if output.exists() or output.is_symlink():
            raise FileExistsError("destination appeared during issuance")
        stage.rename(output)
        receipt["manifest_path"] = str(output / "owner-teacher-manifest.json")
        return receipt
    except BaseException:
        shutil.rmtree(stage)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(issue(args.base_dir, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
