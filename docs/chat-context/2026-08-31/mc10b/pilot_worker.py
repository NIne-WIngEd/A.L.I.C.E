from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Runtime-loaded helpers are hash-bound by the private run config.
# They are deliberately loaded from /kaggle/input only after the config is read.
b = None
tc = None

def _load_input_module_exact(filename: str, expected_sha: str, module_name: str):
    candidates = sorted(p for p in Path("/kaggle/input").rglob(filename) if p.is_file())
    require(len(candidates) == 1, f"expected exactly one {filename}, found {len(candidates)}")
    actual = sha256_file(candidates[0])
    require(actual == expected_sha.upper(), f"{filename} SHA mismatch expected={expected_sha} actual={actual}")
    spec = importlib.util.spec_from_file_location(module_name, candidates[0])
    require(spec is not None and spec.loader is not None, f"cannot load {filename}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

GITHUB_REPO = "NIne-WIngEd/A.L.I.C.E"
DEFAULT_BRANCH = "alice-mc10b-live"
HEARTBEAT_SECONDS = 120
CONTROL_POLL_SECONDS = 30
PUBLIC_FORBIDDEN_KEYS = {
    "prompt_text", "candidate_record", "challenge_record", "payload",
    "frozen_relevant_E0", "related_E0_constraints_no_eligibility_credit",
    "E0_anchor_unit_ids", "E0_anchor_family_ids", "hypothesis_text",
    "alternative_hypotheses", "counterevidence_interpretation",
    "uncertainty_reasons", "scope_conditions",
}

def assert_public_telemetry_safe(obj: Any, path: str = "root") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k)
            require(key not in PUBLIC_FORBIDDEN_KEYS, f"PUBLIC_TELEMETRY_PRIVATE_KEY_FORBIDDEN: {path}.{key}")
            assert_public_telemetry_safe(v, path + "." + key)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            assert_public_telemetry_safe(v, f"{path}[{i}]")



def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canon(obj) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canon(x) + "\n" for x in rows), encoding="utf-8", newline="\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


class StopRequested(RuntimeError):
    pass


class GitHubLive:
    def __init__(self, token: str, repo: str, branch: str, run_id: str, stage: str):
        self.token = token.strip()
        self.repo = repo
        self.branch = branch
        self.run_id = run_id
        self.stage = stage
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.done_event = threading.Event()
        self.current: dict[str, Any] = {
            "artifact_id": "alice.MC10B1.kaggle-live.current.v1",
            "run_id": run_id,
            "stage": stage,
            "status": "STARTING",
            "checkpoint": "BOOT",
            "updated_at_utc": utcnow(),
            "heartbeat_at_utc": utcnow(),
            "model": None,
            "packet_index": 0,
            "packet_total": 5,
            "primary_candidates_generated": 0,
            "shadow_challenges_generated": 0,
            "gpu_count": None,
            "runtime_archive_verified": False,
            "runtime_binary_verified": False,
            "failure_class": None,
            "failure_message": None,
        }
        self.heartbeat_thread: threading.Thread | None = None
        self.telemetry_failure_event = threading.Event()
        self.telemetry_failure_count = 0
        self.telemetry_last_error: str | None = None
        self.telemetry_failure_threshold = 5
        self._control_poll_lock = threading.Lock()
        self._last_control_poll_monotonic = 0.0

    def _request(self, method: str, path: str, body: dict | None = None, query: dict | None = None) -> Any:
        safe_path = "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))
        url = f"https://api.github.com/repos/{self.repo}/contents/{safe_path}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        data = None if body is None else canon(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                return json.loads(raw.decode("utf-8")) if raw else None
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            if e.code == 404:
                return None
            raise RuntimeError(f"GitHub API HTTP {e.code} {method} {path}: {raw[:1000]}") from e

    def get_file(self, path: str) -> tuple[str | None, str | None]:
        obj = self._request("GET", path, query={"ref": self.branch})
        if obj is None:
            return None, None
        content = base64.b64decode(obj.get("content", "")).decode("utf-8")
        return content, obj.get("sha")

    def put_text(self, path: str, text: str, message: str) -> None:
        with self.lock:
            for attempt in range(4):
                _, current_sha = self.get_file(path)
                body = {
                    "message": message,
                    "content": base64.b64encode(text.encode("utf-8")).decode("ascii"),
                    "branch": self.branch,
                }
                if current_sha:
                    body["sha"] = current_sha
                try:
                    self._request("PUT", path, body=body)
                    return
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(1.5 * (attempt + 1))

    def put_json(self, path: str, obj: Any, message: str) -> None:
        assert_public_telemetry_safe(obj, path)
        self.put_text(path, canon(obj) + "\n", message)

    def _telemetry_success(self) -> None:
        with self.lock:
            self.telemetry_failure_count = 0
            self.telemetry_last_error = None

    def _telemetry_failure(self, exc: Exception) -> None:
        with self.lock:
            self.telemetry_failure_count += 1
            self.telemetry_last_error = str(exc)[:1000]
            count = self.telemetry_failure_count
        if count >= self.telemetry_failure_threshold:
            self.telemetry_failure_event.set()

    def publish_current(self) -> None:
        with self.lock:
            self.current["updated_at_utc"] = utcnow()
            snapshot = dict(self.current)
        self.put_json("mc10b/current.json", snapshot, f"mc10b-live: {self.run_id} {self.stage} heartbeat")
        self._telemetry_success()

    def update(self, **kwargs: Any) -> None:
        with self.lock:
            self.current.update(kwargs)
            self.current["updated_at_utc"] = utcnow()
        self.publish_current()

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        event = {
            "run_id": self.run_id,
            "stage": self.stage,
            "kind": kind,
            "at_utc": utcnow(),
            **payload,
        }
        token = hashlib.sha256((canon(event) + str(time.time_ns())).encode()).hexdigest()[:12]
        path = f"mc10b/runs/{self.run_id}/events/{int(time.time()*1000)}-{token}.json"
        self.put_json(path, event, f"mc10b-live: {self.run_id} {kind}")

    def observation(self, relpath: str, obj: Any) -> None:
        path = f"mc10b/runs/{self.run_id}/{relpath}"
        self.put_json(path, obj, f"mc10b-live: {self.run_id} output {relpath}")

    def publish_file(self, relpath: str, file_path: Path) -> None:
        raise RuntimeError("PUBLIC_TELEMETRY_RAW_ARTIFACT_PUBLICATION_FORBIDDEN")

    def poll_control(self, force: bool = False) -> None:
        now = time.monotonic()
        with self._control_poll_lock:
            if not force and now - self._last_control_poll_monotonic < CONTROL_POLL_SECONDS:
                return
            self._last_control_poll_monotonic = now
        try:
            text, _ = self.get_file("mc10b/control.json")
            self._telemetry_success()
            if not text:
                return
            ctl = json.loads(text)
            if ctl.get("run_id") != self.run_id:
                return
            if str(ctl.get("action", "run")).lower() in {"stop", "abort", "cancel"}:
                self.stop_event.set()
        except Exception as e:
            with self.lock:
                self.current["control_poll_warning"] = str(e)[:500]
            self._telemetry_failure(e)

    def start_heartbeat(self) -> None:
        def loop() -> None:
            last_push = 0.0
            last_control = 0.0
            while not self.done_event.wait(2):
                now = time.time()
                if now - last_control >= CONTROL_POLL_SECONDS:
                    self.poll_control()
                    last_control = now
                if now - last_push >= HEARTBEAT_SECONDS:
                    try:
                        with self.lock:
                            self.current["heartbeat_at_utc"] = utcnow()
                        self.publish_current()
                    except Exception as e:
                        self._telemetry_failure(e)
                    last_push = now
        self.heartbeat_thread = threading.Thread(target=loop, name="mc10b-live-heartbeat", daemon=True)
        self.heartbeat_thread.start()

    def close(self) -> None:
        self.done_event.set()
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5)

    def check_stop(self) -> None:
        self.poll_control(force=False)
        if self.telemetry_failure_event.is_set():
            detail = self.telemetry_last_error or "unknown GitHub telemetry failure"
            raise RuntimeError(f"GITHUB_TELEMETRY: repeated live GitHub failures: {detail}")
        if self.stop_event.is_set():
            raise StopRequested("GitHub control requested stop")


def get_github_token() -> str:
    # API-pushed Kaggle kernels do not reliably inherit interactive Kaggle Secrets.
    # The Windows launcher therefore places the PAT in the same transient private
    # per-run Kaggle dataset and removes its local temporary copy after upload.
    env = os.environ.get("ALICE_GITHUB_TOKEN", "").strip()
    if env:
        return env
    candidates = sorted(Path("/kaggle/input").rglob("alice-github-token.txt"))
    require(len(candidates) == 1, f"expected exactly one transient GitHub token file, found {len(candidates)}")
    token = candidates[0].read_text(encoding="utf-8").strip()
    require(bool(token), "transient GitHub token file is empty")
    require(len(token) >= 20, "transient GitHub token appears malformed")
    return token


def extract_input_zip(zip_path: Path, root: Path) -> Path:
    require(tc is not None, "transport common module not loaded")
    return tc.safe_extract_private_archive(zip_path, root)


def source_namespace(src: Path) -> argparse.Namespace:
    return argparse.Namespace(
        mc10a=str(src / "mc10a"),
        activation=str(src / "activation"),
        v1=str(src / "v1"),
        h11=str(src / "h11"),
        router=str(src / "router"),
        semantic=str(src / "semantic"),
        leak=str(src / "leak"),
        recon=str(src / "recon"),
        mc6=str(src / "mc6"),
        mc7=str(src / "mc7"),
        mc8=str(src / "mc8"),
        mc9=str(src / "mc9"),
        doctrine=str(src / "doctrine"),
        repo=str(src / "repo"),
        qualification_receipt=str(src / "ALICE_MC10B_GENERATOR_PORTFOLIO_QUALIFICATION_v1.json"),
    )


def verify_sources(ns: argparse.Namespace):
    mc10a, act, v1, h11, eligible, unitreg = b.verify_inputs(ns)
    bound = b.verify_bound_sources(ns, mc10a)
    q = b.verify_qualification_receipt(Path(ns.qualification_receipt))
    selected = b.select_pilot(eligible)
    require(len(selected) == 5, "pilot selection count")
    return mc10a, act, v1, h11, eligible, unitreg, bound, q, selected


def locate_runtime_archive() -> Path:
    require(tc is not None, "transport common module not loaded")
    return tc.locate_runtime_archive(Path("/kaggle/input"))

def install_runtime(live: GitHubLive, work: Path) -> Path:
    require(tc is not None, "transport common module not loaded")
    live.update(checkpoint="RUNTIME_ARCHIVE_VERIFY", status="PREFLIGHT")
    archive = locate_runtime_archive()
    ah = tc.sha256_file(archive)
    require(ah == tc.RUNTIME_ARCHIVE_SHA256, "runtime archive SHA mismatch")
    live.update(runtime_archive_verified=True)
    binary = tc.extract_pinned_runtime(archive, work / "runtime")
    bh = tc.sha256_file(binary)
    require(bh == tc.RUNTIME_BINARY_SHA256, "runtime binary SHA mismatch")
    live.update(runtime_binary_verified=True)
    return binary


def start_ollama(binary: Path, work: Path, live: GitHubLive) -> tuple[subprocess.Popen, dict[str, str]]:
    models = work / "models"
    models.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "OLLAMA_HOST": "127.0.0.1:11434",
        "OLLAMA_MODELS": str(models),
        "OLLAMA_KEEP_ALIVE": "0",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
    })
    log_path = work / "ollama-serve.log"
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen([str(binary), "serve"], stdout=log, stderr=subprocess.STDOUT, env=env, text=True)
    log.close()
    deadline = time.time() + 90
    while time.time() < deadline:
        live.check_stop()
        if proc.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:] if log_path.exists() else ""
            raise RuntimeError(f"ollama serve exited before readiness rc={proc.returncode}: {tail}")
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open("http://127.0.0.1:11434/api/tags", timeout=5) as r:
                json.loads(r.read().decode("utf-8"))
                live.update(checkpoint="OLLAMA_READY")
                return proc, env
        except Exception:
            time.sleep(2)
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:] if log_path.exists() else ""
    raise RuntimeError(f"ollama serve readiness timeout: {tail}")


def run_process_with_stop(cmd: list[str], env: dict[str, str], live: GitHubLive, label: str, timeout_s: int = 3600) -> str:
    live.event("PROCESS_START", {"label": label, "argv0": cmd[0], "arg1": cmd[1] if len(cmd) > 1 else None})
    fd, log_name = tempfile.mkstemp(prefix="mc10b-proc-", suffix=".log", dir="/tmp")
    os.close(fd)
    log_path = Path(log_name)
    start = time.time()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
        while True:
            rc = p.poll()
            live.check_stop()
            if live.stop_event.is_set() and rc is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                raise StopRequested(f"stop requested during {label}")
            if time.time() - start > timeout_s and rc is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                raise RuntimeError(f"{label} timeout")
            if rc is not None:
                text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
                if rc != 0:
                    raise RuntimeError(f"{label} failed rc={rc}: " + text[-6000:])
                live.event("PROCESS_COMPLETE", {"label": label, "elapsed_seconds": round(time.time() - start, 2)})
                return text[-12000:]
            time.sleep(1)
    finally:
        log_path.unlink(missing_ok=True)

MODEL_APPROX_BYTES = {
    "gpt-oss:20b": 13793441244,
    "gemma4:31b-it-q4_K_M": 19868981791,
    "glm-4.7-flash:q4_K_M": 19019270897,
}

def pull_and_verify_model(binary: Path, env: dict[str, str], spec: dict, live: GitHubLive, pull_timeout_s: int) -> dict:
    tag = spec["tag"]
    expected_bytes = MODEL_APPROX_BYTES.get(tag)
    models_dir = Path(env["OLLAMA_MODELS"])
    free_bytes = shutil.disk_usage(models_dir).free
    live.event("MODEL_DISK_PREFLIGHT", {"model": tag, "free_bytes": free_bytes, "approx_model_bytes": expected_bytes})
    if expected_bytes is not None:
        require(free_bytes >= expected_bytes + 1024**3,
                f"insufficient ephemeral disk before model pull: model={tag} free={free_bytes} required_min={expected_bytes + 1024**3}")
    live.update(checkpoint="MODEL_PULL", model=tag, status="MODEL_SETUP")
    run_process_with_stop([str(binary), "pull", tag], env, live, f"ollama pull {tag}", timeout_s=pull_timeout_s)
    live.check_stop()
    rt = b.runtime_info("http://127.0.0.1:11434", spec)
    safe_tag = tag.replace(':','_').replace('/','_')
    # Publish runtime identity/version before the smoke so a response-contract failure remains diagnosable.
    live.observation(f"runtime/{safe_tag}.json", rt)
    smoke = b.smoke_model(rt, min(pull_timeout_s, 900))
    live.observation(f"runtime/{safe_tag}.structured-smoke.json", smoke)
    live.update(checkpoint="MODEL_VERIFIED", model=tag)
    return rt


def remove_model(binary: Path, env: dict[str, str], tag: str, live: GitHubLive) -> None:
    try:
        b.unload_model({"base_url": "http://127.0.0.1:11434", "model_name": tag})
    except Exception:
        pass
    try:
        subprocess.run([str(binary), "rm", tag], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    except Exception as e:
        live.event("MODEL_REMOVE_WARNING", {"model": tag, "warning": str(e)[:500]})


def primary_source_manifest(bound: dict, selected: list[dict]) -> dict:
    return {
        "artifact_id": "alice.MC10B1.einf-portfolio-pilot.source-manifest.v1.1",
        "MC10A_manifest_sha256": b.MC10A_MANIFEST_SHA,
        "activation_manifest_sha256": b.ACT_MANIFEST_SHA,
        "EINF_contract_sha256": b.EINF_CONTRACT_SHA,
        "generator_judge_policy_sha256": b.GENJUDGE_SHA,
        "generator_portfolio_qualification_sha256": b.QUAL_RECEIPT_SHA,
        "bound_source_hashes": bound,
        "selected_packet_ids": [p["packet_id"] for p in selected],
        "selected_packet_content_sha256": {p["packet_id"]: p["packet_content_sha256"] for p in selected},
        "candidate_visible_MC8_hidden_evaluator_material_loaded": 0,
        "canonical_pool_unchanged": True,
        "challengers_outside_canonical_pool": True,
        "reserve_model_calls": 0,
        "execution_backend": "KAGGLE_T4X2_PINNED_OLLAMA_RUNTIME",
    }


def run_primary(src: Path, out: Path, live: GitHubLive, binary: Path, env: dict[str, str], generation_timeout_s: int, model_pull_timeout_s: int) -> None:
    ns = source_namespace(src)
    mc10a, act, v1, h11, eligible, unitreg, bound, q, selected = verify_sources(ns)
    spec = q["portfolio"]["primary"]
    # The GitHub live branch lives in a public repository. Publish operational metadata only;
    # private candidate-visible evidence, prompts, and generated hypotheses stay in Kaggle output.
    selection_commitment = hashlib.sha256(canon([
        {"packet_id": p["packet_id"], "packet_content_sha256": p["packet_content_sha256"]}
        for p in selected
    ]).encode("utf-8")).hexdigest().upper()
    live.observation("selected_packets_summary.json", {
        "run_id": live.run_id, "stage": "primary", "selected_packet_count": len(selected),
        "selection_commitment_sha256": selection_commitment,
        "private_packet_metadata_published": False, "private_evidence_published": False,
    })
    rt = pull_and_verify_model(binary, env, spec, live, model_pull_timeout_s)
    out.mkdir(parents=True, exist_ok=False)
    source_manifest = primary_source_manifest(bound, selected)
    write_json(out / "mc10b1_generation_source_manifest_v1.json", source_manifest)
    write_json(out / "mc10b1_generator_portfolio_receipt_v1.json", q)
    write_json(out / "mc10b1_generator_runtime_manifest_v1.json", {
        "artifact_id": "alice.MC10B1.primary-generator-runtime.v1.1",
        **rt,
        "backend_role": "CANONICAL_EINF_PROPOSAL_GENERATOR_ONLY",
        "private_evidence_transport": "PRIVATE_KAGGLE_DATASET_CANDIDATE_VISIBLE_ONLY",
        "generator_has_acceptance_authority": False,
        "generator_is_Alice_identity_model": False,
        "A_SYN_generation_enabled": False,
        "model_training_enabled": False,
    })
    selected_rows = [{
        "packet_id": p["packet_id"], "cluster_id": p["cluster_id"], "graph_id": p["graph_id"],
        "system": p["system"], "candidate_kind": p["candidate_kind"], "priority": p["MC4_priority_tier"],
        "family_count": p["candidate_relevant_independent_E0_family_count"],
        "post_freeze_A_SYN_eligible": p["post_freeze_A_SYN_eligible"],
        "packet_content_sha256": p["packet_content_sha256"],
    } for p in selected]
    write_jsonl(out / "mc10b1_selected_pilot_packets_v1.jsonl", selected_rows)

    rows: list[dict] = []
    packets = {p["packet_id"]: p for p in selected}
    live.update(status="RUNNING", checkpoint="PRIMARY_GENERATION", model=spec["tag"])
    for pi, p in enumerate(selected, 1):
        live.check_stop()
        live.update(packet_index=pi, checkpoint="PRIMARY_PACKET")
        packet_rows: list[dict] = []
        for method in b.METHODS:
            for seed in b.SEEDS:
                live.check_stop()
                rec = b.generate_primary(rt, p, unitreg, method, seed, generation_timeout_s)
                b.validate_primary_record(rec, packets, unitreg, rt)
                rows.append(rec)
                packet_rows.append(rec)
                write_jsonl(out / "mc10b1_einf_raw_candidates_v1.jsonl", rows)
                obs_name = f"primary/packet-{pi:02d}/{method}-{seed}.json"
                live.observation(obs_name, {
                    "run_id": live.run_id, "stage": "primary", "packet_index": pi,
                    "method": method, "seed": seed,
                    "generation_attempts": rec["generation_attempts"],
                    "generation_num_predict": rec["generation_num_predict"],
                    "ollama_done_reason": rec.get("ollama_done_reason"),
                    "prompt_eval_count": rec.get("prompt_eval_count"), "eval_count": rec.get("eval_count"),
                    "unknown_preferred": bool(rec["payload"].get("unknown_preferred")),
                    "prompt_sha256": rec["prompt_sha256"],
                    "response_canonical_sha256": rec["response_canonical_sha256"],
                    "private_prompt_published": False, "private_candidate_payload_published": False,
                })
                live.update(primary_candidates_generated=len(rows))
        distinct = len({b.normalized_text(r["payload"]["hypothesis_text"]) for r in packet_rows})
        packet_summary = {
            "packet_index": pi,
            "packet_id": p["packet_id"],
            "candidate_count": len(packet_rows),
            "distinct_normalized_hypotheses": distinct,
            "unknown_preferred_count": sum(1 for r in packet_rows if r["payload"].get("unknown_preferred")),
            "retry_count": sum(int(r.get("generation_attempts", 1)) - 1 for r in packet_rows),
        }
        live.observation(f"primary/packet-{pi:02d}/summary.json", packet_summary)
        if distinct < 4:
            raise RuntimeError(f"DIVERSITY_FLOOR: packet {pi} has only {distinct} distinct normalized hypotheses")

    require(len(rows) == 60, "primary total must be 60")
    receipt = {
        "artifact_id": "alice.MC10B1.kaggle-primary-stage-receipt.v1",
        "run_id": live.run_id,
        "stage": "primary",
        "complete": True,
        "pilot_packets": 5,
        "raw_EINF_candidates_generated": 60,
        "shadow_challenges_generated": 0,
        "reserve_model_calls": 0,
        "E_INF_accepted_count": 0,
        "A_SYN_generated_count": 0,
        "model_training_performed": False,
        "MC10_saturation_rounds_credited": 0,
        "MC10B_full_generation_start_allowed": False,
        "MC10C_start_allowed": False,
        "stage_g_closed": False,
        "completed_at_utc": utcnow(),
    }
    write_json(out / "mc10b1_primary_stage_receipt_v1.json", receipt)
    b.manifest(out)
    artifact_manifest = [{
        "name": f.name, "bytes": f.stat().st_size, "sha256": sha256_file(f)
    } for f in sorted(out.iterdir()) if f.is_file()]
    live.observation("primary_artifact_manifest.json", {
        "run_id": live.run_id, "stage": "primary", "artifact_count": len(artifact_manifest),
        "artifacts": artifact_manifest, "private_artifact_contents_published": False,
    })
    live.update(status="PRIMARY_COMPLETE_AWAITING_AUDIT", checkpoint="PRIMARY_COMPLETE", packet_index=5, primary_candidates_generated=60)
    live.event("PRIMARY_COMPLETE", {"raw_EINF_candidates_generated": 60})
    remove_model(binary, env, spec["tag"], live)


def synthetic_runtime(spec: dict) -> dict:
    return {
        "base_url": "http://127.0.0.1:11434",
        "model_name": spec["tag"],
        "model_digest": spec["digest"],
        "qualified_profile": spec["qualified_profile"],
    }


def finalize_full_package(src: Path, primary_dir: Path, out: Path, challenge_rows: list[dict], challenge_runtimes: list[dict], q: dict, selected: list[dict], unitreg: dict) -> None:
    out.mkdir(parents=True, exist_ok=False)
    primary_rows = read_jsonl(primary_dir / "mc10b1_einf_raw_candidates_v1.jsonl")
    source_manifest = read_json(primary_dir / "mc10b1_generation_source_manifest_v1.json")
    primary_runtime_manifest = read_json(primary_dir / "mc10b1_generator_runtime_manifest_v1.json")
    primary_spec = q["portfolio"]["primary"]
    rt_primary = synthetic_runtime(primary_spec)
    packets = {p["packet_id"]: p for p in selected}
    require(len(primary_rows) == 60, "primary input count")
    for rec in primary_rows:
        b.validate_primary_record(rec, packets, unitreg, rt_primary)
    spec_by_tag = {s["tag"]: s for s in q["portfolio"]["challengers"]}
    rt_by_tag = {rt["model_name"]: rt for rt in challenge_runtimes}
    require(len(challenge_rows) == 10, "challenge count")
    for rec in challenge_rows:
        b.validate_challenge_record(rec, packets, unitreg, rt_by_tag[rec["generator_model"]], spec_by_tag[rec["generator_model"]])

    shutil.copy2(primary_dir / "mc10b1_generation_source_manifest_v1.json", out / "mc10b1_generation_source_manifest_v1.json")
    shutil.copy2(primary_dir / "mc10b1_generator_portfolio_receipt_v1.json", out / "mc10b1_generator_portfolio_receipt_v1.json")
    shutil.copy2(primary_dir / "mc10b1_generator_runtime_manifest_v1.json", out / "mc10b1_generator_runtime_manifest_v1.json")
    shutil.copy2(primary_dir / "mc10b1_selected_pilot_packets_v1.jsonl", out / "mc10b1_selected_pilot_packets_v1.jsonl")
    shutil.copy2(primary_dir / "mc10b1_einf_raw_candidates_v1.jsonl", out / "mc10b1_einf_raw_candidates_v1.jsonl")
    write_jsonl(out / "mc10b1_shadow_challenges_v1.jsonl", challenge_rows)
    write_jsonl(out / "mc10b1_challenger_runtime_manifests_v1.jsonl", [{
        "artifact_id": "alice.MC10B1.shadow-challenger-runtime.v1.1",
        **rt,
        "backend_role": spec["role"],
        "counts_as_EINF_candidate": False,
        "eligible_for_promotion": False,
        "generator_has_acceptance_authority": False,
        "generator_is_Alice_identity_model": False,
    } for rt, spec in zip(challenge_runtimes, q["portfolio"]["challengers"])])

    unknown = [{
        "candidate_id": "UNKNOWN-" + b.sha_bytes((p["packet_id"] + "|UNKNOWN").encode())[:24],
        "candidate_state": "RAW_NULL_COMPETITOR", "candidate_type": "UNKNOWN",
        "packet_id": p["packet_id"], "cluster_id": p["cluster_id"], "graph_id": p["graph_id"],
        "provenance_class": "UNKNOWN", "historical_Elaina_truth": False,
        "unknown_remains_competitor": True, "candidate_visible_MC8_hidden_evaluator_material_loaded": 0,
    } for p in selected]
    write_jsonl(out / "mc10b1_unknown_competitors_v1.jsonl", unknown)
    blind = [{
        "blinded_candidate_id": "BLIND-" + b.sha_bytes(("MC10B1|" + r["candidate_id"]).encode())[:24],
        "packet_id": r["packet_id"], "cluster_id": r["cluster_id"], "graph_id": r["graph_id"],
        "candidate_state": "RAW_UNEVALUATED", "provenance_class": "E-INF", "historical_Elaina_truth": False,
        "E0_anchor_family_ids": r["E0_anchor_family_ids"], "payload": r["payload"],
        "generator_metadata_visible_to_judge": False,
    } for r in primary_rows] + [{
        "blinded_candidate_id": "BLIND-" + b.sha_bytes(("MC10B1|" + u["candidate_id"]).encode())[:24],
        "packet_id": u["packet_id"], "cluster_id": u["cluster_id"], "graph_id": u["graph_id"],
        "candidate_state": "RAW_NULL_COMPETITOR", "provenance_class": "UNKNOWN", "historical_Elaina_truth": False,
        "payload": {"unknown": True}, "generator_metadata_visible_to_judge": False,
    } for u in unknown]
    write_jsonl(out / "mc10b1_blinded_future_evaluation_handoff_v1.jsonl", blind)
    cblind = [{
        "blinded_challenge_id": "BLIND-CHALLENGE-" + b.sha_bytes(("MC10B1|" + r["challenge_id"]).encode())[:24],
        "packet_id": r["packet_id"], "cluster_id": r["cluster_id"], "graph_id": r["graph_id"],
        "candidate_state": "SHADOW_UNEVALUATED_NOT_EINF", "provenance_class": "SHADOW-CHALLENGE",
        "payload": r["payload"], "generator_metadata_visible_to_judge": False,
        "counts_as_EINF_candidate": False, "eligible_for_promotion": False,
    } for r in challenge_rows]
    write_jsonl(out / "mc10b1_blinded_challenger_handoff_v1.jsonl", cblind)
    distinct = {p["packet_id"]: len({b.normalized_text(r["payload"]["hypothesis_text"]) for r in primary_rows if r["packet_id"] == p["packet_id"]}) for p in selected}
    require(all(v >= 4 for v in distinct.values()), "exact diversity floor")
    retry_primary = sum(r["generation_attempts"] - 1 for r in primary_rows)
    retry_ch = sum(r["generation_attempts"] - 1 for r in challenge_rows)
    primary_budgets = sorted({r["generation_num_predict"] for r in primary_rows})
    challenge_budgets = sorted({r["generation_num_predict"] for r in challenge_rows})
    unknown_pref = sum(1 for r in primary_rows if r["payload"]["unknown_preferred"])
    write_json(out / "mc10b1_generation_checkpoint_receipt_v1.json", {
        "artifact_id": "alice.MC10B1.portfolio-generation-checkpoint-receipt.v1.1",
        "resume_supported": True, "resume_count": 0,
        "completed_primary_generation_obligations": 60,
        "completed_shadow_challenge_obligations": 10,
        "partial_files_finalized": True, "checkpoint_state_removed_after_completion": True,
        "execution_backend": "KAGGLE_T4X2_PINNED_OLLAMA_RUNTIME",
    })
    summary = {
        "artifact_id": "alice.MC10B1.einf-portfolio-pilot-summary.v1.1", "pilot_packets": 5,
        "eligible_EINF_packets_total": 65, "raw_EINF_candidates_generated": 60,
        "unknown_competitors_created": 5, "shadow_challenge_proposals_generated": 10,
        "reserve_model_calls": 0, "generation_methods": 4, "seeds_per_method": 3,
        "candidate_ensemble_size_per_packet": 12, "primary_model": primary_spec["tag"],
        "primary_model_digest": primary_spec["digest"],
        "challenger_models": [s["tag"] for s in q["portfolio"]["challengers"]],
        "reserve_model": q["portfolio"]["reserve"]["tag"],
        "normalized_distinct_non_null_candidates_by_packet": distinct,
        "exact_normalized_diversity_failures": 0, "semantic_dedup_not_yet_claimed": True,
        "primary_generation_retry_count": retry_primary, "challenger_generation_retry_count": retry_ch,
        "primary_generation_predict_budgets_used": primary_budgets,
        "challenger_generation_predict_budgets_used": challenge_budgets,
        "raw_candidates_preferring_UNKNOWN": unknown_pref,
        "candidate_visible_MC8_hidden_evaluator_material_loaded": 0,
        "challenger_outputs_count_as_EINF_candidates": False,
        "challenger_outputs_eligible_for_promotion": False,
        "generator_is_Alice_identity_model": False, "generator_has_acceptance_authority": False,
        "A_SYN_generation_enabled": False, "autonomous_A_SYN_promotion_enabled": False,
        "model_training_enabled": False, "E_INF_generated_count": 60, "E_INF_accepted_count": 0,
        "A_SYN_generated_count": 0, "MC10_saturation_rounds_credited": 0,
        "MC10B_started": True, "MC10B_complete": False, "MC10C_start_allowed": False,
        "stage_g_closed": False, "stage_h_activated": False, "phase2_replaced": False,
        "future_completion_objective": b.OBJ, "generic_safe_neutral_default_allowed": False,
        "unknown_remains_competitor": True, "pilot_only_no_full_generation_authority_inferred": True,
        "execution_backend": "KAGGLE_T4X2_PINNED_OLLAMA_RUNTIME",
    }
    write_json(out / "mc10b1_summary_v1.json", summary)
    write_json(out / "mc10b1_pilot_closure_receipt_v1.json", {
        "artifact_id": "alice.MC10B1.einf-portfolio-pilot-closure.v1.1", "pilot_generation_complete": True,
        "pilot_packets": 5, "raw_EINF_candidates_generated": 60, "shadow_challenge_proposals_generated": 10,
        "unknown_competitors_created": 5, "all_primary_candidates_raw_unevaluated": True,
        "all_challenges_shadow_unevaluated_not_EINF": True, "generator_judge_separation_preserved": True,
        "candidate_blinded_handoff_created": True, "challenger_blinded_handoff_created": True,
        "MC10B_pilot_audit_start_allowed": True, "MC10B_full_generation_start_allowed": False,
        "MC10B_complete": False, "MC10C_start_allowed": False, "E_INF_accepted_count": 0,
        "A_SYN_generated_count": 0, "model_training_performed": False, "MC10_saturation_rounds_credited": 0,
        "stage_g_closed": False, "phase2_replaced": False,
    })
    (out / "ALICE_MC10B1_EINF_PORTFOLIO_PILOT_V1_1.md").write_text(
        "# A.L.I.C.E. MC10B1 — E-INF Portfolio Pilot v1.1\n\n"
        "Cloud execution: private Kaggle T4x2 with the pinned tournament Ollama runtime. "
        "GPT-OSS 20B generated the canonical 4×3 candidate pool on five packets. "
        "Gemma 4 31B and GLM-4.7-Flash generated shadow challenges outside the E-INF pool. "
        "Qwen3.8 remained reserve with zero calls. No output has acceptance, A-SYN, training, saturation, Stage-G-closure, or Phase-2 authority.\n",
        encoding="utf-8", newline="\n")
    (out / "validate_alice_mc10b1_einf_portfolio_pilot_v1_1.py").write_text(b.validator_source(), encoding="utf-8", newline="\n")
    b.manifest(out)


def run_challengers(src: Path, primary_dir: Path, out: Path, live: GitHubLive, binary: Path, env: dict[str, str], generation_timeout_s: int, model_pull_timeout_s: int) -> None:
    ns = source_namespace(src)
    mc10a, act, v1, h11, eligible, unitreg, bound, q, selected = verify_sources(ns)
    require(primary_dir.is_dir(), "challenger stage missing primary_result directory")
    require((primary_dir / "SHA256SUMS.txt").is_file(), "primary stage manifest missing")
    # Verify primary stage manifest byte-for-byte.
    manifest_lines = (primary_dir / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    for line in manifest_lines:
        if not line.strip():
            continue
        h, name = line.split("  ", 1)
        require(sha256_file(primary_dir / name) == h.upper(), f"primary artifact hash mismatch: {name}")
    source_manifest = read_json(primary_dir / "mc10b1_generation_source_manifest_v1.json")
    expected = primary_source_manifest(bound, selected)
    require(source_manifest == expected, "primary source manifest drift")
    primary_rows = read_jsonl(primary_dir / "mc10b1_einf_raw_candidates_v1.jsonl")
    require(len(primary_rows) == 60, "primary rows count")
    packets = {p["packet_id"]: p for p in selected}
    rt_primary = synthetic_runtime(q["portfolio"]["primary"])
    for rec in primary_rows:
        b.validate_primary_record(rec, packets, unitreg, rt_primary)

    live.update(status="RUNNING", checkpoint="CHALLENGER_SETUP", primary_candidates_generated=60)
    challenge_rows: list[dict] = []
    challenge_runtimes: list[dict] = []
    for spec in q["portfolio"]["challengers"]:
        live.check_stop()
        rt = pull_and_verify_model(binary, env, spec, live, model_pull_timeout_s)
        challenge_runtimes.append(rt)
        live.update(checkpoint="CHALLENGER_GENERATION", model=spec["tag"])
        for pi, p in enumerate(selected, 1):
            live.check_stop()
            rec = b.generate_challenge(rt, spec, p, unitreg, generation_timeout_s)
            b.validate_challenge_record(rec, packets, unitreg, rt, spec)
            challenge_rows.append(rec)
            obs_name = f"challengers/{spec['family']}/packet-{pi:02d}.json"
            live.observation(obs_name, {
                "run_id": live.run_id, "stage": "challengers", "packet_index": pi,
                "challenger_role": spec["role"],
                "generation_attempts": rec["generation_attempts"],
                "generation_num_predict": rec["generation_num_predict"],
                "ollama_done_reason": rec.get("ollama_done_reason"),
                "prompt_eval_count": rec.get("prompt_eval_count"), "eval_count": rec.get("eval_count"),
                "unknown_preferred": bool(rec["payload"].get("unknown_preferred")),
                "prompt_sha256": rec["prompt_sha256"],
                "response_canonical_sha256": rec["response_canonical_sha256"],
                "private_prompt_published": False, "private_challenge_payload_published": False,
            })
            live.update(packet_index=pi, shadow_challenges_generated=len(challenge_rows))
        remove_model(binary, env, spec["tag"], live)
    require(len(challenge_rows) == 10, "challenge total")
    final_dir = out / "final_package"
    finalize_full_package(src, primary_dir, final_dir, challenge_rows, challenge_runtimes, q, selected, unitreg)
    # Run generated validator in cloud before publishing/downloading.
    proc = subprocess.run([sys.executable, str(final_dir / "validate_alice_mc10b1_einf_portfolio_pilot_v1_1.py"), str(final_dir)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    require(proc.returncode == 0, "generated final validator failed: " + proc.stdout[-3000:])
    final_artifact_manifest = [{
        "name": f.name, "bytes": f.stat().st_size, "sha256": sha256_file(f)
    } for f in sorted(final_dir.iterdir()) if f.is_file()]
    live.observation("final_artifact_manifest.json", {
        "run_id": live.run_id, "stage": "challengers", "artifact_count": len(final_artifact_manifest),
        "artifacts": final_artifact_manifest, "private_artifact_contents_published": False,
    })
    write_json(out / "mc10b1_challenger_stage_receipt_v1.json", {
        "artifact_id": "alice.MC10B1.kaggle-challenger-stage-receipt.v1",
        "run_id": live.run_id, "stage": "challengers", "complete": True,
        "shadow_challenges_generated": 10, "final_package_validated": True,
        "MC10B_pilot_audit_start_allowed": True, "MC10B_full_generation_start_allowed": False,
        "MC10C_start_allowed": False, "stage_g_closed": False, "completed_at_utc": utcnow(),
    })
    live.observation("challenger_stage_receipt_summary.json", {
        "run_id": live.run_id, "stage": "challengers", "complete": True,
        "shadow_challenges_generated": 10, "final_package_validated": True,
        "MC10B_full_generation_start_allowed": False, "stage_g_closed": False,
        "private_artifact_contents_published": False,
    })
    live.update(status="PILOT_GENERATION_COMPLETE_AWAITING_AUDIT", checkpoint="FINAL_PACKAGE_COMPLETE", packet_index=5, shadow_challenges_generated=10)
    live.event("CHALLENGER_COMPLETE", {"shadow_challenges_generated": 10, "final_package_validated": True})


def locate_private_source(work: Path, live: GitHubLive, expected_sha256: str) -> Path:
    require(tc is not None, "transport common module not loaded")
    root = Path("/kaggle/input")
    blobs = sorted(p for p in root.rglob(tc.PRIVATE_BLOB_NAME) if p.is_file())
    require(len(blobs) == 1, f"expected exactly one opaque private input blob, found {len(blobs)}")
    actual = tc.sha256_file(blobs[0])
    require(actual == str(expected_sha256).upper(),
            f"opaque private input SHA mismatch expected={expected_sha256} actual={actual}")
    info = tc.inspect_private_archive(blobs[0])
    live.event("PRIVATE_INPUT_TRANSPORT", {
        "mode": "opaque_deterministic_posix_zip_bytes_bin", "path": str(blobs[0]),
        "sha256": actual, "source_prefix": info["source_prefix"], "member_count": info["member_count"],
    })
    src = tc.safe_extract_private_archive(blobs[0], work / "input")
    tc.validate_source_tree(src)
    return src


def main() -> int:
    global b, tc
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["primary", "challengers"])
    ap.add_argument("--run-id")
    ap.add_argument("--github-branch", default=DEFAULT_BRANCH)
    args = ap.parse_args()

    configs = sorted(p for p in Path("/kaggle/input").rglob("mc10b1-run-config.json") if p.is_file())
    require(len(configs) == 1, f"expected one mc10b1-run-config.json, found {len(configs)}")
    cfg = read_json(configs[0])
    cfg_stage = str(cfg["stage"])
    cfg_run_id = str(cfg["run_id"])
    if args.stage is not None:
        require(args.stage == cfg_stage, f"CLI/config stage mismatch {args.stage} != {cfg_stage}")
    if args.run_id is not None:
        require(args.run_id == cfg_run_id, f"CLI/config run_id mismatch {args.run_id} != {cfg_run_id}")
    args.stage = cfg_stage
    args.run_id = cfg_run_id
    args.github_branch = str(cfg.get("github_branch", args.github_branch))
    private_input_sha256 = str(cfg["private_input_sha256"]).upper()
    generation_timeout_seconds = int(cfg.get("generation_timeout_seconds", 1800))
    model_pull_timeout_seconds = int(cfg.get("model_pull_timeout_seconds", 5400))
    require(60 <= generation_timeout_seconds <= 3600, "generation timeout out of bounds")
    require(600 <= model_pull_timeout_seconds <= 7200, "model pull timeout out of bounds")

    tc = _load_input_module_exact("mc10b1_transport_common.py", str(cfg["transport_common_sha256"]), "mc10b1_transport_common_runtime")
    b = _load_input_module_exact("build_mc10b1_portfolio_pilot_v1_1.py", str(cfg["builder_sha256"]), "mc10b1_builder_runtime")
    qualification_path = tc.locate_exact_file(Path("/kaggle/input"), "ALICE_MC10B_GENERATOR_PORTFOLIO_QUALIFICATION_v1.json")
    require(tc.sha256_file(qualification_path) == str(cfg["qualification_sha256"]).upper(), "qualification receipt dataset SHA mismatch")

    output_root = Path("/kaggle/working/output")
    output_root.mkdir(parents=True, exist_ok=True)
    fatal_path = output_root / "mc10b1_kaggle_failure.json"
    live: GitHubLive | None = None
    serve_proc: subprocess.Popen | None = None
    work = Path(tempfile.mkdtemp(prefix="alice-mc10b1-", dir="/tmp"))
    try:
        token = get_github_token()
        live = GitHubLive(token, GITHUB_REPO, args.github_branch, args.run_id, args.stage)
        # Fail before model work if branch/token/control are wrong. The CPU preflight already proved
        # this path once; T4 rechecks it as defense in depth.
        live.publish_current()
        live.poll_control(force=True)
        live.check_stop()
        live.start_heartbeat()
        live.event("KAGGLE_WORKER_STARTED", {
            "python": sys.version.split()[0],
            "generation_timeout_seconds": generation_timeout_seconds,
            "model_pull_timeout_seconds": model_pull_timeout_seconds,
        })

        gpu_count = 0
        gpu_inventory = []
        try:
            import torch
            gpu_count = torch.cuda.device_count()
            gpu_inventory = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        except Exception:
            proc = subprocess.run(["bash", "-lc", "nvidia-smi --query-gpu=name --format=csv,noheader"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            gpu_inventory = [x.strip() for x in proc.stdout.splitlines() if x.strip()]
            gpu_count = len(gpu_inventory)
        live.update(gpu_count=gpu_count, gpu_inventory=gpu_inventory, checkpoint="GPU_PREFLIGHT", status="PREFLIGHT")
        require(gpu_count >= 2, f"GPU_COUNT: expected at least 2 GPUs, got {gpu_count} inventory={gpu_inventory}")

        src_root = locate_private_source(work, live, private_input_sha256)
        ns = source_namespace(src_root)
        verify_sources(ns)
        live.update(checkpoint="SOURCE_AUTHORITY_VERIFIED")

        binary = install_runtime(live, work)
        serve_proc, env = start_ollama(binary, work, live)

        stage_out = output_root / ("primary_result" if args.stage == "primary" else "challenger_result")
        if args.stage == "primary":
            run_primary(src_root, stage_out, live, binary, env, generation_timeout_seconds, model_pull_timeout_seconds)
        else:
            primary_dir = src_root / "primary_result"
            run_challengers(src_root, primary_dir, stage_out, live, binary, env, generation_timeout_seconds, model_pull_timeout_seconds)

        write_json(output_root / "mc10b1_kaggle_status.json", {
            "run_id": args.run_id, "stage": args.stage, "status": "COMPLETE",
            "completed_at_utc": utcnow(), "stage_g_closed": False,
        })
        return 0
    except StopRequested as e:
        payload = {"run_id": args.run_id, "stage": args.stage, "status": "STOPPED_BY_CONTROL", "message": str(e), "at_utc": utcnow()}
        write_json(fatal_path, payload)
        if live:
            try:
                live.update(status="STOPPED_BY_CONTROL", checkpoint="STOPPED", failure_class="STOP_REQUESTED", failure_message=str(e))
                live.event("STOPPED_BY_CONTROL", payload)
            except Exception:
                pass
        return 75
    except Exception as e:
        msg = str(e)
        if msg.startswith("DIVERSITY_FLOOR"):
            failure_class = "DIVERSITY_FLOOR"
        elif "digest" in msg.lower():
            failure_class = "MODEL_DIGEST_MISMATCH"
        elif "runtime" in msg.lower() and "sha" in msg.lower():
            failure_class = "RUNTIME_HASH_MISMATCH"
        elif "structured output smoke" in msg.lower():
            failure_class = "STRUCTURED_OUTPUT_SMOKE"
        elif "GPU_COUNT" in msg:
            failure_class = "GPU_COUNT"
        elif "GitHub" in msg or "github" in msg:
            failure_class = "GITHUB_TELEMETRY"
        elif "private" in msg.lower() and ("archive" in msg.lower() or "source" in msg.lower() or "blob" in msg.lower()):
            failure_class = "PRIVATE_INPUT_INTEGRITY"
        else:
            failure_class = type(e).__name__
        payload = {
            "run_id": args.run_id, "stage": args.stage, "status": "FAILED",
            "failure_class": failure_class, "message": msg[:4000],
            "traceback": traceback.format_exc()[-12000:], "at_utc": utcnow(),
            "stage_g_closed": False, "MC10B_complete": False,
        }
        write_json(fatal_path, payload)
        if live:
            try:
                live.update(status="FAILED", checkpoint="FAILED", failure_class=failure_class, failure_message=msg[:1500])
                live.event("FAILURE", payload)
                live.observation("failure.json", payload)
            except Exception:
                pass
        return 1
    finally:
        if live:
            live.close()
        if serve_proc is not None and serve_proc.poll() is None:
            try:
                serve_proc.terminate()
                serve_proc.wait(timeout=10)
            except Exception:
                try:
                    serve_proc.kill()
                except Exception:
                    pass
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
