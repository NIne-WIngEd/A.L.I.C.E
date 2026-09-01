#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import io
import getpass
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

CONTROLLER_VERSION = "1.1.0"
EXPECTED_HEAD = "0abaed85873c3f8de04765847eb7700b0e20433f"
EXPECTED_ALICE_KAGGLE_SHA = "B50A0B0060BD7378A431D97AD660D5310BC12934FEB620EA0D6641BE98763B7A"
FROZEN_HASHES = {
    "mc10b1_kaggle_worker.py": "16010EB47F103F66E1CA52FAA436ECAF3551D52130C1E9FFEE6C64BB83F1A15B",
    "build_mc10b1_portfolio_pilot_v1_1.py": "48970BE5AFF9D002C137A3FBF8F1D0AE4A5BC634C09BCD35DE341585E1733ABB",
    "mc10b1_transport_common.py": "6BEF93ECB5897AA20912388F14B5EB70C1A32064991D8C0B0651ABC55D57B913",
    "ALICE_MC10B_GENERATOR_PORTFOLIO_QUALIFICATION_v1.json": "275FE5B2BBB597CBDFE5838E1B6A8E1ACB53C3BE4FED78397AFF8E577B16FFA6",
    "ALICE_MC10B1_PILOT_AUDIT_CLOSURE_AND_FULL_GENERATION_AUTHORIZATION_v1.json": "402115CAC129473EA7D74F3E56E5702BF95B1F569E4C2CAB81BF9DD5D8BE50FD",
    "ALICE_MC10B1_CHALLENGER_AUDIT_INPUT_mc10b1-challengers-20260830T152032Z-4870b223.zip": "F6E2E54157AFBFEF0B7214F1DF0C0FDA3CC698A69CCCD4C964E923D91BFD8788",
    "ALICE_MC10B1_CHALLENGER_FIDELITY_AUDIT_20260830.md": "1870FC2245E4B12F18D04D82580DDA74DC2C6C4B0BD1390141E9FE141A91FD61",
    "ALICE_MC10B1_CHALLENGER_AUDIT_VERDICTS_20260830.jsonl": "8C91587CA45B03F90498BE94A8EA4763085B30F11132A861C072D246FE45990F",
    "ALICE_MC10B1_POST_CHALLENGER_PROVISIONAL_EINF_SET_20260830.md": "D9533F5139C5573FD0DCED7BD9FE2E0A67EEC136263C01006388F9C0FBC429CA",
}
SHARED_RUNTIME_DATASET = "mkrayanyan/alice-tournament-runtime-50539c5fe9bf"
GITHUB_REPO = "NIne-WIngEd/A.L.I.C.E"
GITHUB_BRANCH = "alice-mc10b-live"
KAGGLE_USERNAME = "mkrayanyan"
WORK_REL = Path("datasets") / "memory_stage_g2" / "alice.stage-g2.g2a.gold-semantic-decomposition.v1" / "audits"


class ControllerError(RuntimeError):
    pass


@dataclass
class CmdResult:
    code: int
    text: str


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_utf8(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise ControllerError(msg)


def run_cmd(args: list[str], *, cwd: Path | None = None, check: bool = True, timeout: int | None = None, echo: bool = False) -> CmdResult:
    if echo:
        print("+ " + " ".join(args))
    proc = subprocess.run(args, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout)
    text = (proc.stdout or "").strip()
    if check and proc.returncode != 0:
        if text:
            print(text)
        raise ControllerError(f"command failed rc={proc.returncode}: {' '.join(args)}")
    return CmdResult(proc.returncode, text)


def executable(name: str) -> str:
    p = shutil.which(name)
    require(bool(p), f"required executable not found on PATH: {name}")
    return str(p)


def load_exact_module(path: Path, expected_sha: str, module_name: str):
    require(path.is_file(), f"missing package module: {path.name}")
    actual = sha256_file(path)
    require(actual == expected_sha.upper(), f"package module SHA mismatch {path.name} expected={expected_sha} actual={actual}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def verify_package_manifest(package_root: Path) -> None:
    require(sys.version_info >= (3, 11), f"Python 3.11+ required; running {sys.version.split()[0]}")
    require(package_root.is_dir(), f"package root missing: {package_root}")
    unexpected_dirs=[p.name for p in package_root.iterdir() if p.is_dir()]
    require(not unexpected_dirs, f"package contains unexpected subdirectories: {unexpected_dirs}")
    require(not any(p.is_symlink() for p in package_root.iterdir()), "package symlinks are forbidden")
    sums_path = package_root / "SHA256SUMS.txt"
    require(sums_path.is_file(), "SHA256SUMS.txt missing")
    expected: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        require(len(parts) == 2, f"malformed SHA256SUMS line: {line!r}")
        digest, name = parts[0].upper(), parts[1]
        require(name not in expected, f"duplicate SHA256SUMS entry: {name}")
        expected[name] = digest
    actual_names = {p.name for p in package_root.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt"}
    require(set(expected) == actual_names, f"package file set mismatch expected={sorted(expected)} actual={sorted(actual_names)}")
    for name, digest in expected.items():
        actual = sha256_file(package_root / name)
        require(actual == digest, f"package SHA mismatch {name} expected={digest} actual={actual}")
    for name, digest in FROZEN_HASHES.items():
        require(name in expected, f"frozen artifact missing from manifest: {name}")
        require(expected[name] == digest, f"frozen artifact changed: {name}")
    manifest = read_json(package_root / "PACKAGE_MANIFEST.json")
    require(manifest.get("artifact_id") == "alice.MC10B.full-einf-frontier.execution-package.v1.1.0", "package manifest artifact id")
    require(manifest.get("controller_version") == CONTROLLER_VERSION, "controller version mismatch")
    print(f"package_manifest_verified=true files={len(expected)}")


def gh_api_path(repo: str, relpath: str) -> str:
    return "repos/" + repo + "/contents/" + "/".join(urllib.parse.quote(x, safe="") for x in relpath.split("/"))


def gh_put_text(gh: str, repo: str, branch: str, relpath: str, text: str, message: str) -> None:
    api = gh_api_path(repo, relpath)
    existing = run_cmd([gh, "api", f"{api}?ref={urllib.parse.quote(branch, safe='')}", "--jq", ".sha"], check=False)
    content = base64.b64encode(text.encode("utf-8")).decode("ascii")
    args = [gh, "api", "--method", "PUT", api, "-f", f"message={message}", "-f", f"content={content}", "-f", f"branch={branch}"]
    if existing.code == 0 and existing.text.strip():
        args += ["-f", f"sha={existing.text.strip()}"]
    r = run_cmd(args, check=False)
    if r.code != 0:
        if r.text:
            print(r.text)
        raise ControllerError(f"GitHub update failed: {relpath}")


def ensure_live_branch(gh: str, repo: str, branch: str) -> None:
    ref = run_cmd([gh, "api", f"repos/{repo}/git/ref/heads/{branch}", "--jq", ".object.sha"], check=False)
    if ref.code == 0:
        return
    main = run_cmd([gh, "api", f"repos/{repo}/git/ref/heads/main", "--jq", ".object.sha"]).text.strip()
    require(main == EXPECTED_HEAD, f"GitHub main baseline moved: {main}")
    run_cmd([gh, "api", "--method", "POST", f"repos/{repo}/git/refs", "-f", f"ref=refs/heads/{branch}", "-f", f"sha={main}"])


def safe_public_failure_message(msg: str) -> str:
    # Public branch must never receive packet IDs, local paths, E0, prompt text, candidate text, or tokens.
    low = msg.lower()
    if "token" in low or "packet" in low or "source" in low or ":\\" in msg or "/kaggle/" in msg:
        return "See local preserved diagnostic."
    return re.sub(r"[\r\n]+", " ", msg)[:240]


def publish_local_failure(gh: str, run_id: str | None, checkpoint: str, error: BaseException) -> None:
    if not run_id:
        return
    try:
        obj = {
            "artifact_id": "alice.MC10B.full-einf-generation.live.current.v1",
            "run_id": run_id,
            "stage": "full-einf-generation",
            "status": "LOCAL_FAILURE",
            "checkpoint": checkpoint,
            "failure_class": type(error).__name__,
            "failure_message": "See local preserved diagnostic.",
            "packet_total": 60,
            "primary_candidates_generated": 0,
            "E_INF_accepted_count": 0,
            "A_SYN_generated_count": 0,
            "model_training_enabled": False,
            "updated_at_utc": utcnow(),
            "stage_g_closed": False,
        }
        gh_put_text(gh, GITHUB_REPO, GITHUB_BRANCH, "mc10b/full/current.json", canon(obj) + "\n", f"mc10b-full local failure {run_id}")
        ctl = {
            "artifact_id": "alice.MC10B.full-einf-generation.control.v1", "run_id": run_id,
            "action": "stop", "stage": "full-einf-generation", "updated_at_utc": utcnow(),
            "note": "Run closed by local controller before/after remote execution."
        }
        gh_put_text(gh, GITHUB_REPO, GITHUB_BRANCH, "mc10b/full/control.json", canon(ctl) + "\n", f"mc10b-full close {run_id}")
    except Exception as publish_error:
        print(f"warning_public_failure_publish_failed={type(publish_error).__name__}")


def close_control(gh: str, run_id: str, note: str) -> None:
    ctl = {
        "artifact_id":"alice.MC10B.full-einf-generation.control.v1", "run_id":run_id,
        "action":"stop", "stage":"full-einf-generation", "updated_at_utc":utcnow(), "note":note,
    }
    gh_put_text(gh,GITHUB_REPO,GITHUB_BRANCH,"mc10b/full/control.json",canon(ctl)+"\n",f"mc10b-full close {run_id}")


def publish_post_gpu_local_failure(gh: str, run_id: str, checkpoint: str, candidate_count: int | None = None) -> None:
    obj = {
        "artifact_id":"alice.MC10B.full-einf-generation.live.current.v1", "run_id":run_id,
        "stage":"full-einf-generation", "status":"REMOTE_STAGE_FINISHED_LOCAL_FINALIZATION_FAILED",
        "checkpoint":checkpoint, "packet_total":60,
        "primary_candidates_generated":candidate_count if candidate_count is not None else 0,
        "E_INF_accepted_count":0, "A_SYN_generated_count":0, "model_training_enabled":False,
        "updated_at_utc":utcnow(), "stage_g_closed":False, "controller_version":CONTROLLER_VERSION,
        "failure_message":"See local preserved diagnostic."
    }
    gh_put_text(gh,GITHUB_REPO,GITHUB_BRANCH,"mc10b/full/current.json",canon(obj)+"\n",f"mc10b-full local finalization failure {run_id}")
    close_control(gh,run_id,"Remote stage ended; local finalization failed. See local diagnostic.")


def copy_exact(src: Path, dst: Path) -> None:
    require(src.is_file(), f"missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree_exact(src: Path, dst: Path) -> None:
    require(src.is_dir(), f"missing source directory: {src}")
    require(not dst.exists(), f"destination already exists: {dst}")
    shutil.copytree(src, dst, symlinks=True)


def make_source_namespace(src: Path, argparse_module=argparse):
    return argparse_module.Namespace(
        mc10a=str(src / "mc10a"), activation=str(src / "activation"), v1=str(src / "v1"), h11=str(src / "h11"),
        router=str(src / "router"), semantic=str(src / "semantic"), leak=str(src / "leak"), recon=str(src / "recon"),
        mc6=str(src / "mc6"), mc7=str(src / "mc7"), mc8=str(src / "mc8"), mc9=str(src / "mc9"),
        doctrine=str(src / "doctrine"), repo=str(src / "repo"),
        qualification_receipt=str(src / "ALICE_MC10B_GENERATOR_PORTFOLIO_QUALIFICATION_v1.json"),
    )


def build_source_tree(package_root: Path, repo_root: Path, vault_root: Path, source: Path, full_common, builder) -> dict[str, Any]:
    audit = vault_root / WORK_REL
    paths = {
        "mc10a": audit / "alice-mc10a-frontier-entry-evidence-freeze-and-eligibility-revalidation-v1",
        "activation": audit / "alice-mc10b-einf-generation-execution-activation-v1",
        "v1": audit / "alice-autonomous-personality-completion-pipeline-v1-proposal",
        "h11": audit / "alice-autonomous-personality-completion-v1-1-hardening-proposal",
        "doctrine": audit / "alice-autonomous-personality-completion-owner-ratified-v1",
        "router": audit / "eipm-global-fidelity-router-owner-ratified-v2",
        "semantic": audit / "alice-semantic-repair-v2-owner-ratified-v1",
        "leak": audit / "alice-mind-coverage-leakage-map-v1",
        "recon": audit / "alice-mc3-mc5-source-person-provenance-reconciliation-v1",
        "mc6": audit / "alice-mc6a-h-coordinated-higher-order-interaction-coverage-v1",
        "mc7": audit / "alice-mc7a-h-coordinated-adversarial-gap-discovery-v1",
        "mc8": audit / "alice-mc8a-h-coordinated-heldout-e0-reconstruction-v1",
        "mc9": audit / "alice-mc9a-h-coordinated-provenance-reality-eligibility-firewall-v1",
    }
    source.mkdir(parents=True, exist_ok=False)
    copy_tree_exact(paths["mc10a"], source / "mc10a")
    copy_tree_exact(paths["activation"], source / "activation")
    copy_exact(paths["v1"] / "e_inf_candidate_generation_contract_v1.json", source / "v1/e_inf_candidate_generation_contract_v1.json")
    copy_exact(paths["v1"] / "mc10_recursive_gap_discovery_policy_v2.json", source / "v1/mc10_recursive_gap_discovery_policy_v2.json")
    copy_exact(paths["v1"] / "a_syn_candidate_generation_contract_v1.json", source / "v1/a_syn_candidate_generation_contract_v1.json")
    for n in ["candidate_generator_judge_independence_policy_v1.json", "dual_frontier_gap_discovery_policy_v1.json",
              "evidence_independence_recursive_lineage_policy_v2.json", "heldout_pseudogap_calibration_protocol_v1.json"]:
        copy_exact(paths["h11"] / n, source / "h11" / n)
    copy_exact(paths["doctrine"] / "owner_ratified_doctrine_summary_v1.json", source / "doctrine/owner_ratified_doctrine_summary_v1.json")
    copy_exact(paths["router"] / "eipm_global_fidelity_router_v2_entries.jsonl", source / "router/eipm_global_fidelity_router_v2_entries.jsonl")
    copy_exact(paths["semantic"] / "e0_semantic_component_masks_v2.jsonl", source / "semantic/e0_semantic_component_masks_v2.jsonl")
    copy_exact(paths["semantic"] / "e0_semantic_repair_evidence_map_v2.jsonl", source / "semantic/e0_semantic_repair_evidence_map_v2.jsonl")
    copy_exact(paths["leak"] / "evidence_family_registry_v1.jsonl", source / "leak/evidence_family_registry_v1.jsonl")
    copy_exact(paths["recon"] / "mc5b_high_impact_behavioral_system_inventory_v3.jsonl", source / "recon/mc5b_high_impact_behavioral_system_inventory_v3.jsonl")
    for n in ["mc6b_core_interaction_probe_registry_v1.jsonl", "mc6e_dynamic_conditioned_four_way_interactions_v1.jsonl", "mc6g_priority_interaction_gap_frontier_v1.jsonl"]:
        copy_exact(paths["mc6"] / n, source / "mc6" / n)
    for n in ["mc7b_axis_counterfactual_challenges_v1.jsonl", "mc7g_priority_gap_candidate_frontier_v1.jsonl"]:
        copy_exact(paths["mc7"] / n, source / "mc7" / n)
    copy_exact(paths["mc8"] / "mc8g_mc9_mc10_handoff_contract_v1.json", source / "mc8/mc8g_mc9_mc10_handoff_contract_v1.json")
    for n in ["mc9e_completion_candidate_eligibility_v1.jsonl", "mc9g_mc10_frozen_eligibility_handoff_v1.json", "mc9h_closure_receipt_v1.json"]:
        copy_exact(paths["mc9"] / n, source / "mc9" / n)
    copy_exact(repo_root / "README.md", source / "repo/README.md")
    copy_exact(repo_root / "docs/MEMORY_IDENTITY_FORMATION_AND_HOST_LEARNING_ARCHITECTURE.md", source / "repo/docs/MEMORY_IDENTITY_FORMATION_AND_HOST_LEARNING_ARCHITECTURE.md")
    copy_exact(repo_root / "docs/ALICE_CLONE_AWARE_IDENTITY_STANDARD.md", source / "repo/docs/ALICE_CLONE_AWARE_IDENTITY_STANDARD.md")
    copy_exact(repo_root / "docs/PHASE2_TO_KERNEL_MEMORY_MIGRATION_PLAN.md", source / "repo/docs/PHASE2_TO_KERNEL_MEMORY_MIGRATION_PLAN.md")
    copy_exact(package_root / "ALICE_MC10B_GENERATOR_PORTFOLIO_QUALIFICATION_v1.json", source / "ALICE_MC10B_GENERATOR_PORTFOLIO_QUALIFICATION_v1.json")

    pa = source / "pilot_audit"
    pa.mkdir()
    for n in [
        "ALICE_MC10B1_PILOT_AUDIT_CLOSURE_AND_FULL_GENERATION_AUTHORIZATION_v1.json",
        "ALICE_MC10B1_CHALLENGER_AUDIT_INPUT_mc10b1-challengers-20260830T152032Z-4870b223.zip",
        "ALICE_MC10B1_CHALLENGER_FIDELITY_AUDIT_20260830.md",
        "ALICE_MC10B1_CHALLENGER_AUDIT_VERDICTS_20260830.jsonl",
        "ALICE_MC10B1_POST_CHALLENGER_PROVISIONAL_EINF_SET_20260830.md",
    ]:
        copy_exact(package_root / n, pa / n)

    work_root = audit / "alice-mc10b-full-einf-generation-v1.work"
    resume_pointer = work_root / "resume-current.json"
    resume_source: Path | None = None
    if resume_pointer.is_file():
        rp = read_json(resume_pointer)
        require(rp.get("artifact_id") == "alice.MC10B.full-einf-generation.resume-pointer.v1", "resume pointer artifact id")
        raw = str(rp.get("resume_result_path", "")).strip()
        require(raw, "resume pointer path empty")
        resume_source = Path(raw)
        require(resume_source.is_dir(), f"resume result missing: {resume_source}")
        # Fail closed against a hand-edited pointer escaping the MC10B work tree.
        wr = work_root.resolve()
        rr = resume_source.resolve()
        require(wr in rr.parents, "resume result path escapes MC10B work root")
        require(resume_source.name == 'full_result', "resume result path must end in full_result")
        copy_tree_exact(resume_source, source / "resume_result")
        print(f"resume_result_source={resume_source}")

    # Local source authority pass BEFORE any Kaggle upload.
    ns = make_source_namespace(source)
    mc10a, act, v1, h11, eligible, unitreg = builder.verify_inputs(ns)
    bound = builder.verify_bound_sources(ns, mc10a)
    qualification = builder.verify_qualification_receipt(Path(ns.qualification_receipt))
    gate = full_common.verify_pilot_audit_gate(source, builder)
    pilot, remaining = full_common.select_remaining_packets(eligible, builder, gate["pilot_selected_rows"])
    require(len(eligible) == 65 and len(pilot) == 5 and len(remaining) == 60, "local frontier count invariant")
    resume_count = 0
    if (source / "resume_result").is_dir():
        candidates: list[dict] = []
        for n in ["mc10b_full_einf_raw_candidates_v1.partial.jsonl", "mc10b_full_einf_raw_candidates_v1.jsonl"]:
            p = source / "resume_result" / n
            if p.is_file():
                if n.endswith('.partial.jsonl'):
                    candidates, recovery = full_common.read_recoverable_partial_jsonl(p)
                    if recovery.get('tail_recovered'):
                        print(f"local_resume_torn_tail_detected=true discarded_tail_bytes={recovery.get('discarded_tail_bytes',0)}")
                else:
                    candidates = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
                break
        if candidates:
            resume_count = len(full_common.validate_resume_rows(candidates, remaining, unitreg, builder, qualification["portfolio"]["primary"]))
    print("local_source_authority_verified=true")
    print(f"local_eligible_einf_packets={len(eligible)}")
    print(f"local_remaining_packets={len(remaining)}")
    print(f"local_resume_candidates={resume_count}")
    expected_source_manifest = full_common.source_manifest(bound, pilot, remaining, builder)
    pilot_baseline = full_common.build_block_telemetry(0, pilot, gate["pilot_primary_rows"], builder, scope="AUDITED_PILOT_BASELINE")
    return {
        "audit_root": audit, "work_root": work_root, "eligible": eligible, "pilot": pilot, "remaining": remaining,
        "resume_count": resume_count, "resume_source_present": resume_source is not None, "bound": bound,
        "unitreg": unitreg, "qualification": qualification, "pilot_gate": gate,
        "expected_source_manifest": expected_source_manifest, "pilot_baseline": pilot_baseline,
    }


def get_github_token() -> str:
    env = os.environ.get("ALICE_GITHUB_TOKEN", "").strip()
    if env:
        require(len(env) >= 20, "ALICE_GITHUB_TOKEN appears malformed")
        return env
    print("\nPaste fine-grained GitHub PAT. It is written only to the transient PRIVATE Kaggle input dataset.")
    token = getpass.getpass("GitHub PAT: ").strip()
    require(len(token) >= 20, "GitHub PAT appears malformed")
    return token


def dataset_expected_files(private_blob_name: str) -> set[str]:
    return {
        private_blob_name,
        "build_mc10b1_portfolio_pilot_v1_1.py",
        "mc10b1_kaggle_worker.py",
        "mc10b1_transport_common.py",
        "mc10b_full_common.py",
        "ALICE_MC10B_GENERATOR_PORTFOLIO_QUALIFICATION_v1.json",
        "mc10b-full-run-config.json",
        "alice-github-token.txt",
    }


def validate_dataset_contract(data_dir: Path, cfg: dict[str, Any], transport) -> None:
    blob_name = str(cfg.get("private_blob_name", ""))
    require(blob_name == transport.PRIVATE_BLOB_NAME, f"private blob contract drift cfg={blob_name!r} transport={transport.PRIVATE_BLOB_NAME!r}")
    expected = dataset_expected_files(blob_name)
    actual = {p.name for p in data_dir.iterdir() if p.is_file() and p.name != "dataset-metadata.json"}
    require(actual == expected, f"dataset file contract mismatch expected={sorted(expected)} actual={sorted(actual)}")
    require(sha256_file(data_dir / blob_name) == str(cfg["private_input_sha256"]).upper(), "dataset private blob hash")
    require(sha256_file(data_dir / "build_mc10b1_portfolio_pilot_v1_1.py") == str(cfg["builder_sha256"]).upper(), "dataset builder hash")
    require(sha256_file(data_dir / "mc10b1_kaggle_worker.py") == str(cfg["base_worker_sha256"]).upper(), "dataset base worker hash")
    require(sha256_file(data_dir / "mc10b1_transport_common.py") == str(cfg["transport_common_sha256"]).upper(), "dataset transport hash")
    require(sha256_file(data_dir / "mc10b_full_common.py") == str(cfg["full_common_sha256"]).upper(), "dataset full common hash")
    require(sha256_file(data_dir / "ALICE_MC10B_GENERATOR_PORTFOLIO_QUALIFICATION_v1.json") == str(cfg["qualification_sha256"]).upper(), "dataset qualification hash")
    require(int(cfg.get("expected_remaining_packets", -1)) == 60, "run config remaining packet target")
    require(int(cfg.get("expected_candidate_obligations", -1)) == 720, "run config candidate target")
    print("local_dataset_contract_verified=true")
    print(f"private_blob_name={blob_name}")


def parse_dataset_file_listing(text: str) -> set[str]:
    # Prefer Kaggle's documented CSV output (-v/--csv), but retain a conservative
    # plain-table fallback because older CLI builds have emitted human-readable tables.
    names: set[str] = set()
    try:
        rows = list(csv.reader(io.StringIO(text)))
        if rows:
            header = [str(x).strip().lower() for x in rows[0]]
            name_idx = next((i for i, x in enumerate(header) if x in {"name", "filename", "file_name"}), None)
            if name_idx is not None:
                for row in rows[1:]:
                    if len(row) > name_idx and row[name_idx].strip():
                        names.add(Path(row[name_idx].strip()).name)
    except csv.Error:
        pass
    if names:
        return names
    for token in re.findall(r"[A-Za-z0-9_.-]+", text):
        if "." in token:
            names.add(Path(token).name)
    return names


def wait_dataset_ready(kaggle: str, dataset_ref: str, expected_files: set[str], timeout_seconds: int = 900) -> None:
    deadline = time.time() + timeout_seconds
    last_status = ""
    while time.time() < deadline:
        # Keep the status probe on the same plain-text command that already succeeded
        # in the audited MC10B1 pilot. Readiness itself is established by the exact file
        # listing plus the CPU-side hash/integrity preflight, not by guessing a JSON schema.
        status = run_cmd([kaggle, "datasets", "status", dataset_ref], check=False)
        last_status = status.text
        if status.code == 0 and re.search(r"(?i)error|failed|failure|invalid", status.text):
            raise ControllerError(f"private Kaggle dataset entered failure state: {status.text}")
        listing = run_cmd([kaggle, "datasets", "files", dataset_ref, "--page-size", "100", "-v"], check=False)
        if listing.code != 0:
            listing = run_cmd([kaggle, "datasets", "files", dataset_ref, "--page-size", "100"], check=False)
        if listing.code == 0:
            present = parse_dataset_file_listing(listing.text)
            if expected_files.issubset(present):
                print(f"kaggle_private_dataset_files_visible={dataset_ref}")
                return
        time.sleep(15)
    raise ControllerError(f"private Kaggle dataset visibility timeout: {dataset_ref} last_status={last_status[:500]}")


def kernel_status(kaggle: str, ref: str) -> str:
    r = run_cmd([kaggle, "kernels", "status", ref], check=False)
    if r.code != 0:
        return "PLATFORM_RECONCILIATION"
    t = r.text.upper()
    if "COMPLETE" in t: return "COMPLETE"
    if "ERROR" in t or "FAILED" in t: return "ERROR"
    if "CANCELLED" in t or "CANCELED" in t: return "CANCELLED"
    if "RUNNING" in t: return "RUNNING"
    if "QUEUED" in t: return "QUEUED"
    return "PENDING"


def wait_kernel(kaggle: str, ref: str, timeout_seconds: int, poll_seconds: int) -> str:
    deadline = time.time() + timeout_seconds
    last = "UNKNOWN"
    reconciliation_since: float | None = None
    slug = ref.split("/", 1)[-1]
    while time.time() < deadline:
        last = kernel_status(kaggle, ref)
        print(f"kaggle_status={last}")
        if last in {"COMPLETE", "ERROR", "CANCELLED"}:
            return last
        if last == "PLATFORM_RECONCILIATION":
            if reconciliation_since is None:
                reconciliation_since = time.time()
            elif time.time() - reconciliation_since > 600:
                listed = run_cmd([kaggle, "kernels", "list", "-m", "-s", slug, "-v"], check=False)
                if listed.code != 0 or slug not in listed.text:
                    raise ControllerError(f"Kaggle kernel not discoverable for >10 minutes: {ref}")
        else:
            reconciliation_since = None
        time.sleep(poll_seconds)
    return "TIMEOUT"


def download_kernel_output(kaggle: str, ref: str, out_dir: Path) -> CmdResult:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return run_cmd([kaggle, "kernels", "output", ref, "-p", str(out_dir), "-o"], check=False)


def locate_one(root: Path, name: str) -> Path | None:
    xs = sorted(p for p in root.rglob(name) if p.is_file())
    return xs[0] if len(xs) == 1 else None


def find_dir(root: Path, name: str) -> Path | None:
    xs = sorted(p for p in root.rglob(name) if p.is_dir())
    return xs[0] if len(xs) == 1 else None


def run_cpu_preflight(kaggle: str, probe_dir: Path, probe_ref: str, probe_out: Path, run_root: Path,
                      remote_dataset_files_verified: bool) -> dict[str, Any]:
    retry_codes = {"MOUNT_PRIVATE_BLOB_MISSING", "RUNTIME_MOUNT_MISSING", "GITHUB_TRANSIENT"}
    last: dict[str, Any] | None = None
    for attempt in (1, 2):
        print(f"\n===== CPU-ONLY KAGGLE FULL PREFLIGHT {attempt}/2 =====")
        run_cmd([kaggle, "kernels", "push", "-p", str(probe_dir)], echo=True)
        status = wait_kernel(kaggle, probe_ref, timeout_seconds=600, poll_seconds=10)
        download_kernel_output(kaggle, probe_ref, probe_out)
        pf = locate_one(probe_out, "mc10b_full_mount_probe.json")
        if status == "COMPLETE" and pf is not None:
            result = read_json(pf)
            saved = run_root / f"preflight-attempt-{attempt}.json"
            shutil.copy2(pf, saved)
            last = result
            print(f"preflight_pass={bool(result.get('pass'))}")
            print(f"preflight_checkpoint={result.get('checkpoint','')}")
            print(f"preflight_failure_code={result.get('failure_code','')}")
            for field in ["source_authority_ok", "eligible_einf_packets", "pilot_audit_gate_passed", "remaining_packets",
                          "resume_candidates_validated", "runtime_extract_ok", "github_control_read_ok", "github_write_ok"]:
                print(f"preflight_{field}={result.get(field,'')}")
            if result.get("pass") is True:
                return result
            code = str(result.get("failure_code", "UNKNOWN"))
            print(f"preflight_failure_class={result.get('failure_class','')}")
            print(f"preflight_failure_message={result.get('failure_message','')}")
            print(f"preflight_diagnostic={saved}")
            retryable = bool(result.get("retryable")) and code in retry_codes and remote_dataset_files_verified
            if not retryable or attempt == 2:
                break
            print(f"preflight_retry_reason={code}; one controlled retry will be attempted")
            time.sleep(30)
            continue
        print(f"preflight_terminal_status={status}")
        # Only platform/no-output failure gets one retry. Structured invariant failures never loop blindly.
        if attempt == 2:
            break
        time.sleep(30)
    raise ControllerError("CPU-only Kaggle preflight failed. GPU kernel was NOT submitted; see preserved preflight diagnostic.")


def independent_validate_completed_result(full_result: Path, python: str, source_info: dict[str, Any], full_common, builder) -> None:
    require(full_result.is_dir(), f"completed result directory missing: {full_result}")
    actual = {p.name for p in full_result.iterdir() if p.is_file()}
    require(actual == set(full_common.FINAL_OUTPUT_FILENAMES),
            f"completed result exact file set mismatch expected={sorted(full_common.FINAL_OUTPUT_FILENAMES)} actual={sorted(actual)}")
    require(not any(p.is_dir() for p in full_result.iterdir()), "completed result contains unexpected subdirectory")
    validator = full_result / "validate_alice_mc10b_full_einf_frontier_v1.py"
    require(validator.is_file(), "downloaded full frontier validator missing")
    run_cmd([python, str(validator), str(full_result)], echo=True)

    rows = [json.loads(x) for x in (full_result / "mc10b_full_einf_raw_candidates_v1.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = full_common.validate_resume_rows(rows, source_info["remaining"], source_info["unitreg"], builder, source_info["qualification"]["portfolio"]["primary"])
    require(len(rows) == full_common.TOTAL_REMAINING_CANDIDATES, "independent candidate count")
    expected_keys = [(p["packet_id"], m, int(seed)) for p in source_info["remaining"] for m in builder.METHODS for seed in builder.SEEDS]
    actual_keys = [(r["packet_id"], r["generation_method"], int(r["seed"])) for r in rows]
    require(actual_keys == expected_keys, "candidate obligation sequence differs from canonical frontier order")

    packet_projection = [json.loads(x) for x in (full_result / "mc10b_full_remaining_packets_v1.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    require(canon(packet_projection) == canon(full_common.remaining_packet_projection(source_info["remaining"])), "remaining packet projection drift")
    require(canon(read_json(full_result / "mc10b_full_generation_source_manifest_v1.json")) == canon(source_info["expected_source_manifest"]), "source manifest drift")
    require(canon(read_json(full_result / "mc10b_full_generator_portfolio_receipt_v1.json")) == canon(source_info["qualification"]), "generator portfolio receipt drift")
    rt = read_json(full_result / "mc10b_full_generator_runtime_manifest_v1.json")
    pspec = source_info["qualification"]["portfolio"]["primary"]
    require(rt.get("model_name") == pspec["tag"] and str(rt.get("model_digest", "")).lower() == str(pspec["digest"]).lower(), "generator runtime identity drift")
    require(rt.get("backend_role") == "CANONICAL_EINF_PROPOSAL_GENERATOR_ONLY" and rt.get("generator_has_acceptance_authority") is False and rt.get("generator_is_Alice_identity_model") is False, "generator runtime authority drift")

    unknowns = [json.loads(x) for x in (full_result / "mc10b_full_unknown_competitors_v1.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    expected_unknowns = full_common.unknown_rows(source_info["remaining"])
    require(canon(unknowns) == canon(expected_unknowns), "UNKNOWN competitor set drift")
    blind = [json.loads(x) for x in (full_result / "mc10b_full_blinded_future_evaluation_handoff_v1.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    require(canon(blind) == canon(full_common.blinded_handoff_rows(rows, expected_unknowns)), "blinded handoff drift")

    baseline = read_json(full_result / "mc10b_full_telemetry_pilot_baseline_v1.json")
    require(canon(baseline) == canon(source_info["pilot_baseline"]), "pilot telemetry baseline drift")
    telemetry = [json.loads(x) for x in (full_result / "mc10b_full_block_telemetry_v1.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    require(len(telemetry) == full_common.TELEMETRY_BLOCK_COUNT, "telemetry block count")
    ack = read_json(full_result / "mc10b_full_telemetry_acknowledgements_v1.json")
    acked = {int(x) for x in ack.get("acknowledged_review_blocks", [])}
    for bi in range(1, full_common.TELEMETRY_BLOCK_COUNT + 1):
        bp = source_info["remaining"][(bi-1)*full_common.TELEMETRY_BLOCK_PACKET_COUNT:bi*full_common.TELEMETRY_BLOCK_PACKET_COUNT]
        fresh = full_common.build_block_telemetry(bi, bp, rows, builder)
        gate = full_common.evaluate_block_telemetry_gate(fresh, baseline)
        expected_status = "PASS" if not gate["review_required"] else ("ACKNOWLEDGED_REVIEW" if bi in acked else "REVIEW_REQUIRED")
        require(expected_status != "REVIEW_REQUIRED", f"unresolved telemetry review block {bi}")
        observed = telemetry[bi-1]
        observed_core = {k:v for k,v in observed.items() if k not in {"gate","gate_status","review_acknowledged"}}
        require(canon(observed_core) == canon(fresh), f"telemetry metrics drift block {bi}")
        require(canon(observed.get("gate") or {}) == canon(gate), f"telemetry gate drift block {bi}")
        require(observed.get("gate_status") == expected_status and bool(observed.get("review_acknowledged")) == (expected_status == "ACKNOWLEDGED_REVIEW"), f"telemetry resolution drift block {bi}")
    require(acked == {int(x["block_index"]) for x in telemetry if x.get("gate_status") == "ACKNOWLEDGED_REVIEW"}, "telemetry acknowledgement set drift")

    ri = read_json(full_result / "mc10b_full_resume_integrity_receipt_v1.json")
    require(ri.get("artifact_id") == "alice.MC10B.full-einf.resume-integrity.v1", "resume integrity receipt id")
    require(bool(ri.get("resume_source_present")) == bool(source_info["resume_source_present"]), "resume source presence drift")
    require(int(ri.get("resume_candidates_validated", -1)) == int(source_info["resume_count"]), "resume candidate count drift")
    cp = read_json(full_result / "mc10b_full_generation_checkpoint_v1.json")
    require(int(cp.get("completed_candidate_obligations", -1)) == 720 and int(cp.get("remaining_candidate_obligations", -1)) == 0 and cp.get("generation_soft_stop_reached") is False, "final checkpoint is not complete")
    print("independent_completed_result_revalidation=true")


def verify_and_publish_canonical(full_result: Path, audit_root: Path, python: str, run_id: str, source_info: dict[str, Any], full_common, builder) -> Path:
    independent_validate_completed_result(full_result, python, source_info, full_common, builder)
    canonical = audit_root / "alice-mc10b-full-einf-frontier-generation-v1"
    downloaded_manifest_sha = sha256_file(full_result / "SHA256SUMS.txt")
    if canonical.exists():
        require(canonical.is_dir(), f"canonical path exists but is not directory: {canonical}")
        independent_validate_completed_result(canonical, python, source_info, full_common, builder)
        require(sha256_file(canonical / "SHA256SUMS.txt") == downloaded_manifest_sha,
                "existing canonical package differs from completed run; refusing overwrite")
        print("canonical_publish_idempotent_existing_copy=true")
        return canonical
    staging = audit_root / f".alice-mc10b-full-einf-frontier-generation-v1.incoming-{run_id}"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(full_result, staging)
    independent_validate_completed_result(staging, python, source_info, full_common, builder)
    os.replace(staging, canonical)
    independent_validate_completed_result(canonical, python, source_info, full_common, builder)
    print("canonical_atomic_publish=true")
    return canonical

def delete_remote(kaggle: str, kind: str, ref: str | None, attempts: int = 3) -> bool:
    if not ref:
        return True
    require(kind in {"kernel", "dataset"}, f"unsupported remote cleanup kind: {kind}")
    noun = "kernels" if kind == "kernel" else "datasets"
    for attempt in range(1, attempts + 1):
        r = run_cmd([kaggle, noun, "delete", ref, "--yes"], check=False, timeout=120)
        if r.code == 0:
            print(f"remote_{kind}_cleanup=true ref={ref}")
            return True
        if attempt < attempts:
            time.sleep(5 * attempt)
    print(f"WARNING_remote_{kind}_cleanup_failed=true ref={ref}")
    return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="A.L.I.C.E. MC10B full E-INF frontier controller v1.1.0")
    ap.add_argument("--repo-root", default=r"C:\A.L.I.C.E-main")
    ap.add_argument("--vault-root", default=r"C:\ALICE_Vault")
    ap.add_argument("--max-generation-minutes", type=int, default=480)
    ap.add_argument("--timeout-hours", type=int, default=12)
    ap.add_argument("--poll-seconds", type=int, default=30)
    ap.add_argument("--acknowledge-telemetry-block", type=int, action="append", default=[])
    ap.add_argument("--keep-remote", action="store_true")
    args = ap.parse_args(argv)
    require(120 <= args.max_generation_minutes <= 600, "max generation minutes must be 120..600")
    require(1 <= args.timeout_hours <= 14, "timeout hours must be 1..14")
    require(10 <= args.poll_seconds <= 300, "poll seconds must be 10..300")
    acks = sorted(set(args.acknowledge_telemetry_block))
    require(all(1 <= x <= 6 for x in acks), "telemetry acknowledgement blocks must be 1..6")
    args.acknowledge_telemetry_block = acks
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    package_root = Path(__file__).resolve().parent
    repo_root = Path(args.repo_root)
    vault_root = Path(args.vault_root)
    run_id: str | None = None
    dataset_ref: str | None = None
    probe_ref: str | None = None
    kernel_ref: str | None = None
    temp_root: Path | None = None
    gpu_submitted = False
    terminal_output_retrieved = False
    remote_stage_status: str | None = None
    checkpoint = "PACKAGE_VERIFY"
    gh = ""
    kaggle = ""
    try:
        print("\n===== A.L.I.C.E. MC10B FULL E-INF FRONTIER v1.1.0 =====")
        print(f"python_version={sys.version.split()[0]}")
        print("scope=remaining_60_EINF_eligible_packets")
        print("raw_candidates_target=720")
        print("telemetry_checkpoint_packets=10")
        print(f"acknowledge_telemetry_blocks={','.join(map(str,args.acknowledge_telemetry_block))}")
        print("E_INF_acceptance=false")
        print("A_SYN=false")
        print("model_training=false")
        print(f"max_generation_minutes={args.max_generation_minutes}")
        verify_package_manifest(package_root)

        git = executable("git"); gh = executable("gh"); kaggle = executable("kaggle"); python = executable("python")
        builder = load_exact_module(package_root / "build_mc10b1_portfolio_pilot_v1_1.py", FROZEN_HASHES["build_mc10b1_portfolio_pilot_v1_1.py"], "mc10b_builder_controller")
        transport = load_exact_module(package_root / "mc10b1_transport_common.py", FROZEN_HASHES["mc10b1_transport_common.py"], "mc10b_transport_controller")
        # full_common is package-evolving, but its exact bytes are covered by SHA256SUMS + outer ZIP SHA.
        spec = importlib.util.spec_from_file_location("mc10b_full_common_controller", package_root / "mc10b_full_common.py")
        require(spec is not None and spec.loader is not None, "cannot import mc10b_full_common.py")
        full_common = importlib.util.module_from_spec(spec); sys.modules[spec.name] = full_common; spec.loader.exec_module(full_common)

        checkpoint = "REPOSITORY_BASELINE"
        require(repo_root.is_dir(), f"repo root missing: {repo_root}")
        branch = run_cmd([git, "branch", "--show-current"], cwd=repo_root).text.strip()
        head = run_cmd([git, "rev-parse", "HEAD"], cwd=repo_root).text.strip()
        require(branch == "main" and head == EXPECTED_HEAD, f"repository baseline mismatch branch={branch} head={head}")
        require(not run_cmd([git, "diff", "--name-status", "HEAD", "--"], cwd=repo_root).text.strip(), "tracked repository differs from HEAD")
        require(not run_cmd([git, "diff", "--cached", "--name-status", "--"], cwd=repo_root).text.strip(), "staged repository changes exist")
        run_cmd([git, "fetch", "origin", "--prune"], cwd=repo_root, echo=True)
        origin = run_cmd([git, "rev-parse", "origin/main"], cwd=repo_root).text.strip()
        require(origin == EXPECTED_HEAD, f"origin/main moved: {origin}")
        alice_kaggle = repo_root / "tools/alice_kaggle_v0.py"
        require(alice_kaggle.is_file() and sha256_file(alice_kaggle) == EXPECTED_ALICE_KAGGLE_SHA, "alice_kaggle_v0.py hash mismatch")
        print(f"sha256_alice_kaggle_v0.py={sha256_file(alice_kaggle)}")
        print(run_cmd([kaggle, "--version"], echo=True).text)
        run_cmd([kaggle, "kernels", "list", "-m", "--page-size", "1"], echo=True)
        print(run_cmd([gh, "auth", "status"], echo=True).text)
        gh_login = run_cmd([gh, "api", "user", "--jq", ".login"]).text.strip()
        require(gh_login == "NIne-WIngEd", f"GitHub authenticated account mismatch: {gh_login}")
        print(f"github_authenticated_login={gh_login}")
        ensure_live_branch(gh, GITHUB_REPO, GITHUB_BRANCH)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        nonce = hashlib.sha256(os.urandom(32)).hexdigest()[:8]
        run_id = f"mc10b-full-{stamp}-{nonce}"
        control = {"artifact_id":"alice.MC10B.full-einf-generation.control.v1","run_id":run_id,"action":"run","stage":"full-einf-generation","updated_at_utc":utcnow(),"note":"Set action=stop to request a graceful checkpoint before the next generation obligation."}
        gh_put_text(gh,GITHUB_REPO,GITHUB_BRANCH,"mc10b/full/control.json",canon(control)+"\n",f"mc10b-full init {run_id}")
        initial = {"artifact_id":"alice.MC10B.full-einf-generation.live.current.v1","run_id":run_id,"stage":"full-einf-generation","status":"LOCAL_PREPARATION","checkpoint":"INPUT_FREEZE","packet_total":60,"primary_candidates_generated":0,"E_INF_accepted_count":0,"A_SYN_generated_count":0,"model_training_enabled":False,"updated_at_utc":utcnow(),"stage_g_closed":False,"controller_version":CONTROLLER_VERSION}
        gh_put_text(gh,GITHUB_REPO,GITHUB_BRANCH,"mc10b/full/current.json",canon(initial)+"\n",f"mc10b-full prepare {run_id}")

        checkpoint = "LOCAL_SOURCE_AUTHORITY"
        audit_root = vault_root / WORK_REL
        work_root = audit_root / "alice-mc10b-full-einf-generation-v1.work"
        run_root = work_root / "kaggle" / run_id
        download_dir = run_root / "download"
        run_root.mkdir(parents=True, exist_ok=True)
        temp_root = Path(tempfile.mkdtemp(prefix=f"alice-mc10b-full-{run_id}-"))
        source = temp_root / "source"
        source_info = build_source_tree(package_root, repo_root, vault_root, source, full_common, builder)

        checkpoint = "PRIVATE_TRANSPORT"
        input_zip = temp_root / "mc10b-full-private-input.zip"
        pack_info = transport.pack_private_source(source, input_zip, prefix="source")
        input_hash = sha256_file(input_zip)
        require(input_hash == str(pack_info["archive_sha256"]).upper(), "transport pack hash disagreement")
        print(f"private_input_sha256={input_hash}")
        print(f"private_input_transport=deterministic_posix_zip_bytes_as_opaque_bin")

        checkpoint = "DATASET_BUILD"
        data_dir = temp_root / "dataset"; data_dir.mkdir()
        blob_name = transport.PRIVATE_BLOB_NAME
        blob_path = data_dir / blob_name
        shutil.copy2(input_zip, blob_path)
        require(sha256_file(blob_path) == input_hash, "opaque blob copy hash mismatch")
        for n in ["build_mc10b1_portfolio_pilot_v1_1.py","mc10b1_kaggle_worker.py","mc10b1_transport_common.py","mc10b_full_common.py","ALICE_MC10B_GENERATOR_PORTFOLIO_QUALIFICATION_v1.json"]:
            shutil.copy2(package_root / n, data_dir / n)
        cfg = {
            "artifact_id":"alice.MC10B.full-einf-generation.run-config.v1.1.0","controller_version":CONTROLLER_VERSION,
            "run_id":run_id,"github_branch":GITHUB_BRANCH,"private_blob_name":blob_name,"private_input_sha256":input_hash,
            "builder_sha256":sha256_file(package_root/"build_mc10b1_portfolio_pilot_v1_1.py"),
            "base_worker_sha256":sha256_file(package_root/"mc10b1_kaggle_worker.py"),
            "full_common_sha256":sha256_file(package_root/"mc10b_full_common.py"),
            "transport_common_sha256":sha256_file(package_root/"mc10b1_transport_common.py"),
            "qualification_sha256":sha256_file(package_root/"ALICE_MC10B_GENERATOR_PORTFOLIO_QUALIFICATION_v1.json"),
            "generation_timeout_seconds":1800,"model_pull_timeout_seconds":5400,"max_generation_minutes":args.max_generation_minutes,
            "telemetry_block_packet_count":10,"acknowledged_telemetry_blocks":args.acknowledge_telemetry_block,
            "expected_eligible_packets":65,"expected_pilot_packets":5,"expected_remaining_packets":60,"expected_candidate_obligations":720,
        }
        write_utf8(data_dir / "mc10b-full-run-config.json", canon(cfg)+"\n")
        token = get_github_token()
        write_utf8(data_dir / "alice-github-token.txt", token+"\n")
        token = ""  # drop controller variable as early as possible

        short = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + nonce[:6]
        slug = (f"alice-mc10b-full-{short}-data")[:50].rstrip("-")
        dataset_ref = f"{KAGGLE_USERNAME}/{slug}"
        title = (f"ALICE MC10B full {short}")[:50]
        meta = {"title":title,"id":dataset_ref,"licenses":[{"name":"other"}],"description":"Private transient MC10B full E-INF generation input."}
        write_utf8(data_dir / "dataset-metadata.json", json.dumps(meta, indent=2)+"\n")
        validate_dataset_contract(data_dir, cfg, transport)

        checkpoint = "KAGGLE_DATASET_CREATE"
        print(f"\n===== CREATE PRIVATE KAGGLE DATASET =====")
        run_cmd([kaggle,"datasets","create","-p",str(data_dir),"-r","skip","-q"], echo=True)
        expected_remote = dataset_expected_files(blob_name)
        wait_dataset_ready(kaggle, dataset_ref, expected_remote)
        # Local token copy is no longer needed after server-side dataset visibility is confirmed.
        (data_dir / "alice-github-token.txt").unlink(missing_ok=True)
        print("local_transient_github_token_copy_deleted=true")

        checkpoint = "CPU_PREFLIGHT"
        probe_dir = temp_root / "probe"; probe_dir.mkdir()
        shutil.copy2(package_root / "mc10b_full_mount_probe.py", probe_dir / "mc10b_full_mount_probe.py")
        pslug = (f"alice-mc10b-full-probe-{short}")[:50].rstrip("-")
        probe_ref = f"{KAGGLE_USERNAME}/{pslug}"
        pmeta = {"id":probe_ref,"title":(f"ALICE MC10B full probe {short}")[:50],"code_file":"mc10b_full_mount_probe.py","language":"python","kernel_type":"script","is_private":True,"enable_gpu":False,"enable_internet":True,"machine_shape":"","dataset_sources":[dataset_ref,SHARED_RUNTIME_DATASET],"competition_sources":[],"kernel_sources":[],"model_sources":[]}
        write_utf8(probe_dir / "kernel-metadata.json", json.dumps(pmeta, indent=2)+"\n")
        probe_out = temp_root / "probe-out"
        run_cpu_preflight(kaggle, probe_dir, probe_ref, probe_out, run_root, remote_dataset_files_verified=True)
        print("kaggle_cpu_full_preflight_pass=true")
        preflight_public = dict(initial)
        preflight_public.update({"status":"LOCAL_PREPARATION","checkpoint":"CPU_PREFLIGHT_PASSED","updated_at_utc":utcnow(),"controller_version":CONTROLLER_VERSION})
        gh_put_text(gh,GITHUB_REPO,GITHUB_BRANCH,"mc10b/full/current.json",canon(preflight_public)+"\n",f"mc10b-full cpu preflight passed {run_id}")

        checkpoint = "GPU_SUBMISSION"
        kernel_dir = temp_root / "kernel"; kernel_dir.mkdir()
        shutil.copy2(package_root / "mc10b_full_kaggle_worker.py", kernel_dir / "mc10b_full_kaggle_worker.py")
        kslug = (f"alice-mc10b-full-{short}")[:50].rstrip("-")
        kernel_ref = f"{KAGGLE_USERNAME}/{kslug}"
        kmeta = {"id":kernel_ref,"title":(f"ALICE MC10B full {short}")[:50],"code_file":"mc10b_full_kaggle_worker.py","language":"python","kernel_type":"script","is_private":True,"enable_gpu":True,"enable_internet":True,"machine_shape":"NvidiaTeslaT4","dataset_sources":[dataset_ref,SHARED_RUNTIME_DATASET],"competition_sources":[],"kernel_sources":[],"model_sources":[]}
        write_utf8(kernel_dir / "kernel-metadata.json", json.dumps(kmeta, indent=2)+"\n")
        print("\n===== PUSH PRIVATE KAGGLE GPU KERNEL =====")
        run_cmd([kaggle,"kernels","push","-p",str(kernel_dir),"--accelerator","NvidiaTeslaT4"], echo=True)
        gpu_submitted = True

        checkpoint = "GPU_RUN"
        status = wait_kernel(kaggle, kernel_ref, timeout_seconds=(args.timeout_hours*3600)+1200, poll_seconds=args.poll_seconds)
        print(f"kaggle_terminal_status={status}")
        # Kaggle CLI 2.2.x exposes kernel logs independently from output artifacts.
        # Preserve a private copy before any remote cleanup so an interpreter/platform
        # failure that cannot write our structured failure artifact is still diagnosable.
        log_result = run_cmd([kaggle, "kernels", "logs", kernel_ref], check=False, timeout=180)
        if log_result.text:
            write_utf8(run_root / "kernel.log", log_result.text + "\n")
            print(f"kernel_log_saved={run_root / 'kernel.log'}")
        elif log_result.code != 0:
            print(f"warning_kernel_log_capture_failed_rc={log_result.code}")
        dl = download_kernel_output(kaggle, kernel_ref, download_dir)
        if dl.text:
            print(dl.text)
        terminal_output_retrieved = dl.code == 0 and status in {"COMPLETE", "ERROR", "CANCELLED"}
        if not terminal_output_retrieved:
            raise ControllerError(f"terminal Kaggle output was not safely retrieved; status={status} output_rc={dl.code}. Remote kernel/dataset will be preserved for recovery.")
        full_result = find_dir(download_dir, "full_result")
        status_file = locate_one(download_dir, "mc10b_full_kaggle_status.json")
        failure_file = locate_one(download_dir, "mc10b_full_kaggle_failure.json")

        if full_result is not None:
            preserve = run_root / "full_result"
            if preserve.exists(): shutil.rmtree(preserve)
            shutil.copytree(full_result, preserve)
            pointer = {"artifact_id":"alice.MC10B.full-einf-generation.resume-pointer.v1","run_id":run_id,"resume_result_path":str(preserve),"updated_at_utc":utcnow()}
            work_root.mkdir(parents=True, exist_ok=True)
            write_utf8(work_root / "resume-current.json", canon(pointer)+"\n")

        if failure_file is not None:
            remote_stage_status = "FAILED"
            print("Full generation kernel failed; preserved output follows:")
            print(failure_file.read_text(encoding="utf-8", errors="replace")[:12000])
            raise ControllerError("MC10B full generation failed. Preserved candidates will be revalidated on the next run.")
        require(status_file is not None, f"Kaggle status artifact missing terminal_status={status}")
        so = read_json(status_file)
        stage_status = str(so.get("status", ""))
        remote_stage_status = stage_status
        if stage_status == "TELEMETRY_REVIEW_REQUIRED":
            block = int(so["telemetry_block_index"])
            print(f"telemetry_review_required_block={block}")
            print(f"completed_candidates={so.get('completed_candidate_obligations')}")
            print("review_reasons=" + ",".join(map(str, so.get("review_reasons", []))))
            if full_result:
                tp = full_result / "mc10b_full_block_telemetry_v1.jsonl"
                if tp.is_file():
                    for line in tp.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            obj=json.loads(line)
                            if int(obj.get("block_index",0)) == block:
                                print("telemetry_block_json="+canon(obj)); break
            print(f"next_action=REVIEW_THEN_RERUN_WITH_-AcknowledgeTelemetryBlock_{block}")
            close_control(gh,run_id,"Telemetry review checkpoint reached; remote run complete.")
            return 0
        if stage_status == "PARTIAL_CHECKPOINT":
            print(f"partial_candidates={so.get('completed_candidate_obligations')}")
            print("next_action=RERUN_SAME_PACKAGE_TO_RESUME")
            close_control(gh,run_id,"Partial checkpoint reached; remote run complete.")
            return 0
        require(stage_status == "COMPLETE", f"unexpected remote status artifact: {stage_status} terminal={status}")
        require(full_result is not None, "completed run missing full_result")

        checkpoint = "CANONICAL_FINALIZATION"
        canonical = verify_and_publish_canonical(full_result, audit_root, python, run_id, source_info, full_common, builder)
        (work_root / "resume-current.json").unlink(missing_ok=True)
        print(f"mc10b_full_canonical={canonical}")
        print("raw_EINF_candidates_generated=720")
        print("E_INF_accepted_count=0")
        print("A_SYN_generated_count=0")
        print("model_training_performed=false")
        print("next_gate=MC10B_FULL_FRONTIER_FIDELITY_EVALUATION")
        close_control(gh,run_id,"Full E-INF frontier generation complete; awaiting fidelity evaluation.")
        return 0
    except KeyboardInterrupt as e:
        print("controller_interrupted=true")
        if gh and run_id and not gpu_submitted:
            publish_local_failure(gh, run_id, checkpoint, e)
        elif gh and run_id and gpu_submitted:
            # Request a graceful remote checkpoint, but preserve the remote kernel and its
            # private input dataset so its output can be recovered even though the local
            # controller was interrupted.
            try:
                close_control(gh,run_id,"Local controller interrupted; request graceful remote checkpoint and preserve remote resources.")
            except Exception as publish_error:
                print(f"warning_interrupt_control_publish_failed={type(publish_error).__name__}")
            print(f"remote_recovery_kernel_ref={kernel_ref}")
            print(f"remote_recovery_dataset_ref={dataset_ref}")
        raise
    except Exception as e:
        print(f"controller_failure_checkpoint={checkpoint}")
        print(f"controller_failure_class={type(e).__name__}")
        print(f"controller_failure_message={e}")
        if 'run_root' in locals() and isinstance(locals().get('run_root'), Path):
            try:
                diag={"artifact_id":"alice.MC10B.full-einf.local-controller-failure.v1.1.0","run_id":run_id,"checkpoint":checkpoint,"failure_class":type(e).__name__,"failure_message":str(e),"traceback":traceback.format_exc(),"gpu_submitted":gpu_submitted,"terminal_output_retrieved":terminal_output_retrieved,"remote_stage_status":remote_stage_status,"at_utc":utcnow()}
                write_utf8(locals()['run_root']/"controller-failure.json",json.dumps(diag,sort_keys=True,indent=2)+"\n")
                print(f"controller_private_diagnostic={locals()['run_root']/ 'controller-failure.json'}")
            except Exception as diag_error:
                print(f"warning_controller_diagnostic_write_failed={type(diag_error).__name__}")
        if gh and run_id and not gpu_submitted:
            publish_local_failure(gh, run_id, checkpoint, e)
        elif gh and run_id and gpu_submitted:
            # Never overwrite an accurate worker FAILED/PARTIAL/REVIEW state. The only
            # post-GPU public override allowed here is when the worker itself completed
            # successfully and local canonical finalization then failed.
            if checkpoint == "CANONICAL_FINALIZATION" and remote_stage_status == "COMPLETE":
                try:
                    candidate_count = None
                    if 'so' in locals() and isinstance(locals().get('so'), dict):
                        candidate_count = int(locals()['so'].get('raw_EINF_candidates_generated', locals()['so'].get('completed_candidate_obligations', 0)) or 0)
                    publish_post_gpu_local_failure(gh,run_id,checkpoint,candidate_count)
                except Exception as publish_error:
                    print(f"warning_post_gpu_failure_publish_failed={type(publish_error).__name__}")
            elif not terminal_output_retrieved:
                print("remote_resources_preserved_for_recovery=true")
                print(f"remote_recovery_kernel_ref={kernel_ref}")
                print(f"remote_recovery_dataset_ref={dataset_ref}")
        return 1
    finally:
        if kaggle and not args.keep_remote:
            # The probe is disposable once the controller exits. The GPU kernel and its
            # private dataset are deleted only when no GPU was submitted, or when a
            # terminal GPU output was successfully downloaded. This prevents local
            # controller/network failures from destroying recoverable remote evidence.
            delete_remote(kaggle, "kernel", probe_ref)
            if (not gpu_submitted) or terminal_output_retrieved:
                delete_remote(kaggle, "kernel", kernel_ref)
                delete_remote(kaggle, "dataset", dataset_ref)
            else:
                print("remote_kernel_cleanup_deferred_for_recovery=true")
                print("remote_dataset_cleanup_deferred_for_recovery=true")
        elif args.keep_remote:
            print("remote_cleanup_skipped=true")
        if temp_root is not None and temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
