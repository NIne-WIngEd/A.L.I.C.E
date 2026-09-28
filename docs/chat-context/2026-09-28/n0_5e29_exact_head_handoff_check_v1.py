#!/usr/bin/env python3
"""Data-free admission check before each exact-head Magnolia Slurm submission.

This does not manufacture a receipt or authorize training. The existing N0
producers and auditors still own the full proof. No private payload is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


REVISION = "5e29f7f69ba4a5d031c7036639b67bebbcdc0bd2"
SEMANTIC_SHA256 = "6c2706984c0e05123c4d88ba456788fbd4c9e7f0fca41d43d9e575ac6e53bf43"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def data(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"receipt is not an object: {path}")
    return result


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("STOP: " + message)


def receipt(path: Path, status: str, revision: str = REVISION) -> dict:
    result = data(path)
    check(result.get("status") == status, f"receipt is not {status}: {path}")
    check(result.get("source_revision") == revision, f"source mismatch: {path}")
    check(result.get("final_results_observed") in (None, False), f"FINAL observed: {path}")
    check(result.get("private_identity_data") in (None, False), f"private input: {path}")
    return result


def matching_digest(result: dict, field: str, path: Path) -> None:
    check(result.get(field) == sha256(path), f"{field} does not match {path}")


def cpu_receipts(source: Path, root: Path, work: Path) -> None:
    matrix = receipt(root / "static_proof.json", "PASS_N0_FULL_ENVELOPE_PROOF_OBLIGATIONS_STATIC_V1")
    check(matrix.get("static_suite_executed") is True and matrix.get("static_suite_pass") is True,
          "static suite did not execute and pass")
    check(matrix.get("static_suite_exit_code") == 0 and matrix.get("pytest_skipped_cases") == 0,
          "static suite failed or skipped")
    matching_digest(matrix, "auditor_sha256", source / "scripts/eipm/n0/audit_n0_v02_full_envelope_proof_obligations_v1.py")
    matching_digest(matrix, "proof_contract_sha256", source / "configs/eipm/n0/n0_v02_full_envelope_proof_obligations_v1.json")
    matching_digest(matrix, "supersession_sha256", source / "configs/eipm/n0/n0_v02_full_envelope_supersession_map_v1.json")
    matching_digest(matrix, "retrospective_sha256", source / "configs/eipm/n0/n0_v02_full_envelope_retrospective_audit_v1.json")
    matching_digest(matrix, "ci_workflow_sha256", source / ".github/workflows/n0-full-envelope-foundation-build-v1-contract.yml")

    cpu = receipt(root / "result.json", "PASS_N0_FULL_ENVELOPE_CPU_RUNTIME_QUALIFICATION_V1")
    matching_digest(cpu, "qualifier_sha256", source / "scripts/eipm/n0/qualify_n0_v02_full_envelope_cpu_runtime_v1.py")
    matching_digest(cpu, "registered_topology_sha256", source / "configs/eipm/n0/n0_v02_full_envelope_registered_topology_v1.json")
    matching_digest(cpu, "qualification_config_sha256", source / "configs/eipm/n0/n0_v02_full_envelope_cpu_runtime_qualification_v1.json")
    matching_digest(cpu, "semantic_config_sha256", source / "configs/eipm/n0/alice_n0_semantic_v0.2.json")
    matching_digest(cpu, "semantic_checkpoint_sha256", work / "targeted-repair-v0.1/checkpoints/step-00000080/alice_n0_v02.safetensors")
    matching_digest(cpu, "tokenizer_json_sha256", work / "tokenizer-v0.2.1/tokenizer.json")
    matching_digest(cpu, "source_config_sha256", source / "configs/eipm/n0/public_corpus_v0.2.1.activated.json")
    matching_digest(cpu, "corpus_receipt_sha256", work / "tokenizer-corpus-v0.2.1-offline/corpus_receipt.json")
    for filename, status, producer in (
        ("tokenizer_stress.json", "PASS_N0_TOKENIZER_STRESS_V1", "audit_tokenizer_v02.py"),
        ("operator-evidence-alignment/token_alignment.json", "PASS_N0_OPERATOR_EVIDENCE_TOKEN_ALIGNMENT_V1", "audit_n0_v02_operator_evidence_token_alignment_v1.py"),
        ("semantic-operator-long-context/token_alignment.json", "PASS_N0_SEMANTIC_OPERATOR_LONG_TOKEN_ALIGNMENT_V1", "audit_n0_v02_semantic_operator_long_token_alignment_v1.py"),
        ("long-context/token_boundary_alignment.json", "PASS_N0_FULL_ENVELOPE_LONG_CONTEXT_TOKEN_BOUNDARY_ALIGNMENT_V1", "audit_n0_v02_full_envelope_long_context_token_boundaries_v1.py"),
    ):
        row = receipt(root / filename, status)
        matching_digest(row, "auditor_sha256", source / "scripts/eipm/n0" / producer)


def public_mixture(source: Path, root: Path, cpu: Path, teacher: Path, teacher_audit: Path, corpus: Path) -> None:
    manifest_path = root / "full_public_mixture_manifest.json"
    manifest = data(manifest_path)
    check(manifest.get("status") == "MATERIALIZED_N0_FULL_PUBLIC_MIXTURE_NO_GRADIENT", "public mixture not materialized")
    check(manifest.get("source_revision") == REVISION, "public mixture source mismatch")
    check(manifest.get("private_identity_data") is False and manifest.get("final_results_observed") is False,
          "private identity or FINAL entered mixture")
    check(manifest.get("final_rows_in_training") == 0 and manifest.get("final_rows_in_model_selection") == 0,
          "FINAL rows entered the training or selection mixture")
    audit = receipt(root / "full_public_mixture_audit.json", "PASS_N0_FULL_PUBLIC_MIXTURE_MANIFEST_AUDIT_V1")
    matching_digest(audit, "manifest_sha256", manifest_path)
    matching_digest(audit, "contract_sha256", source / "configs/eipm/n0/n0_v02_full_public_mixture_contract_v1.json")
    matching_digest(audit, "auditor_sha256", source / "scripts/eipm/n0/audit_n0_v02_full_public_mixture_manifest_v1.py")
    frozen = data(root / "final-v2/freeze_receipt.json")
    check(frozen.get("status") == "FROZEN_N0_FULL_ENVELOPE_FINAL_V2_PACKAGE_BEFORE_GRADIENT",
          "sealed FINAL package was not frozen")
    for key in ("results_observed", "training_authorized", "model_selection_authorized", "final_opening_authorized"):
        check(frozen.get(key) is False, f"sealed FINAL package claimed {key}")
    lanes = manifest.get("training_lanes") or {}
    for key, relative in (
        ("semantic_operator_intervention", "semantic/rows.jsonl"),
        ("semantic_operator_long_context", "semantic-long/rows.jsonl"),
        ("full_envelope_behavioral", "behavioral/rows.jsonl"),
        ("runtime_view_supplement", "runtime-view/rows.jsonl"),
        ("long_context_supplement", "long-context/rows.jsonl"),
        ("natural_relation", "fewrel/train_dev_rows.jsonl"),
    ):
        matching_digest(lanes.get(key) or {}, "rows_sha256", root / relative)
    matching_digest(lanes.get("natural_relation") or {}, "bank_sha256", root / "fewrel/train_dev_bank.json")
    matching_digest(lanes.get("governed_judgment_replay") or {}, "teacher_registry_sha256", teacher)
    matching_digest(lanes.get("governed_judgment_replay") or {}, "teacher_audit_sha256", teacher_audit)
    matching_digest(lanes.get("broad_semantic_replay") or {}, "corpus_receipt_sha256", corpus / "corpus_receipt.json")
    # The materializer copies this CPU evidence byte-for-byte.
    check(sha256(cpu / "static_proof.json") == sha256(root / "runtime/static_proof.json"),
          "mixture static proof differs from exact-head CPU proof")
    check(sha256(cpu / "result.json") == sha256(root / "runtime/cpu_runtime.json"),
          "mixture CPU receipt differs from exact-head CPU proof")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "mixture", "gpu"))
    args = parser.parse_args()
    env = os.environ
    source = Path(env["ALICE_N0_REPO_ROOT"]).resolve()
    work = Path(env["ALICE_N0_WORKDIR"]).resolve()
    cpu = Path(env["ALICE_N0_CPU_RUNTIME_ROOT"]).resolve()
    mixture = Path(env["ALICE_N0_FULL_MIXTURE_ROOT"]).resolve()
    evidence = Path(env["ALICE_N0_MEASURED_JOINT_ROOT"]).resolve()
    teacher = Path(env["ALICE_N0_TEACHER_REGISTRY"]).resolve()
    teacher_audit = Path(env["ALICE_N0_TEACHER_AUDIT"]).resolve()
    corpus = work / "tokenizer-corpus-v0.2.1-offline"
    tokenizer = work / "tokenizer-v0.2.1"
    semantic = work / "targeted-repair-v0.1/checkpoints/step-00000080/alice_n0_v02.safetensors"
    check(git(source, "rev-parse", "HEAD") == REVISION, "wrong checked-out N0 revision")
    check(not git(source, "status", "--porcelain"), "dirty N0 source worktree")
    check(env["ALICE_N0_EXPECTED_REVISION"] == REVISION, "expected revision differs from pinned source")
    for path in (work, tokenizer / "tokenizer.json", corpus / "corpus_receipt.json", semantic):
        check(path.exists(), f"required public input missing: {path}")
    check(sha256(semantic) == SEMANTIC_SHA256, "pretrained semantic checkpoint hash drift")
    if args.mode == "cpu":
        check(not cpu.exists(), f"preserve occupied exact-head CPU root: {cpu}")
    else:
        cpu_receipts(source, cpu, work)
        for path in (teacher, teacher_audit):
            check(path.is_file(), f"teacher input missing: {path}")
        if args.mode == "mixture":
            fewrel = Path(env["ALICE_N0_FEWREL_ROOT"]).resolve()
            check(fewrel.is_dir(), f"FewRel checkout missing: {fewrel}")
            sources = data(source / "configs/eipm/n0/n0_v02_natural_relation_sources_v1.json")
            expected = sources["sources"][0]["revision"]
            check(git(fewrel, "rev-parse", "HEAD") == expected, "FewRel revision drift")
            check(not mixture.exists(), f"preserve occupied exact-head mixture root: {mixture}")
        else:
            public_mixture(source, mixture, cpu, teacher, teacher_audit, corpus)
            check(not evidence.exists(), f"preserve occupied measured GPU root: {evidence}")
    print(f"PASS_EXACT_HEAD_{args.mode.upper()}_ADMISSION source={REVISION}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (KeyError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"STOP: exact-head handoff admission failed: {exc}") from exc
