#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

INPUT_ROOT = Path("/kaggle/input")
OUT = Path("/kaggle/working/mc10b_full_mount_probe.json")
GITHUB_REPO = "NIne-WIngEd/A.L.I.C.E"


class ProbeError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def mark(payload: dict[str, Any], checkpoint: str, **fields: Any) -> None:
    payload["checkpoint"] = checkpoint
    payload.update(fields)


def find_exact_basename(name: str) -> list[Path]:
    return sorted(p for p in INPUT_ROOT.rglob(name) if p.is_file())


def one(name: str, *, missing_code: str = "MOUNT_FILE_COUNT") -> Path:
    xs = find_exact_basename(name)
    if len(xs) != 1:
        code = missing_code if len(xs) == 0 else "MOUNT_FILE_DUPLICATE"
        raise ProbeError(code, f"expected exactly one mounted {name}, found {len(xs)}", retryable=(len(xs) == 0))
    return xs[0]


def load_exact(name: str, expected_sha: str, module_name: str):
    path = one(name)
    actual = sha(path)
    if actual != expected_sha.upper():
        raise ProbeError("MODULE_HASH_MISMATCH", f"{name} SHA mismatch expected={expected_sha} actual={actual}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ProbeError("MODULE_IMPORT_SPEC", f"cannot load {name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def github_request(token: str, method: str, path: str, body: dict | None = None, query: dict | None = None) -> Any:
    safe_path = "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{safe_path}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    data = None if body is None else canon(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8")) if raw else None
    except urllib.error.HTTPError as e:
        retryable = e.code in {502, 503, 504}
        code = "GITHUB_TRANSIENT" if retryable else "GITHUB_HTTP_ERROR"
        raise ProbeError(code, f"GitHub HTTP {e.code} for {path}", retryable=retryable) from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise ProbeError("GITHUB_TRANSIENT", f"GitHub network error: {type(e).__name__}", retryable=True) from e


def source_namespace(src: Path) -> argparse.Namespace:
    return argparse.Namespace(
        mc10a=str(src / "mc10a"), activation=str(src / "activation"), v1=str(src / "v1"), h11=str(src / "h11"),
        router=str(src / "router"), semantic=str(src / "semantic"), leak=str(src / "leak"), recon=str(src / "recon"),
        mc6=str(src / "mc6"), mc7=str(src / "mc7"), mc8=str(src / "mc8"), mc9=str(src / "mc9"),
        doctrine=str(src / "doctrine"), repo=str(src / "repo"),
        qualification_receipt=str(src / "ALICE_MC10B_GENERATOR_PORTFOLIO_QUALIFICATION_v1.json"),
    )


def mounted_inventory() -> list[str]:
    items=[]
    for p in sorted(INPUT_ROOT.rglob("*")):
        if p.is_file():
            try: items.append(str(p.relative_to(INPUT_ROOT)).replace("\\", "/"))
            except Exception: pass
    return items[:500]


def main() -> int:
    payload: dict[str, Any] = {
        "artifact_id":"alice.MC10B.full-einf.cpu-preflight.v3",
        "pass":False,
        "retryable":False,
        "checkpoint":"BOOT",
        "at_utc":time.time(),
        "mounted_input_inventory":mounted_inventory(),
    }
    work = Path(tempfile.mkdtemp(prefix="alice-mc10b-full-probe-", dir="/tmp"))
    try:
        cfg_path = one("mc10b-full-run-config.json")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg.get("artifact_id") != "alice.MC10B.full-einf-generation.run-config.v1.1.0":
            raise ProbeError("RUN_CONFIG_SCHEMA", "unexpected run config artifact id")
        mark(payload,"CONFIG_LOADED",run_id=cfg["run_id"],stage="full-einf-generation",github_branch=cfg["github_branch"],controller_version=cfg.get("controller_version"))

        tc = load_exact("mc10b1_transport_common.py", str(cfg["transport_common_sha256"]), "mc10b_transport_probe")
        builder = load_exact("build_mc10b1_portfolio_pilot_v1_1.py", str(cfg["builder_sha256"]), "mc10b_builder_probe")
        fc = load_exact("mc10b_full_common.py", str(cfg["full_common_sha256"]), "mc10b_full_common_probe")
        base_worker_path = one("mc10b1_kaggle_worker.py")
        if sha(base_worker_path) != str(cfg["base_worker_sha256"]).upper():
            raise ProbeError("MODULE_HASH_MISMATCH", "mc10b1_kaggle_worker.py SHA mismatch")
        qualification_path = one("ALICE_MC10B_GENERATOR_PORTFOLIO_QUALIFICATION_v1.json")
        if sha(qualification_path) != str(cfg["qualification_sha256"]).upper():
            raise ProbeError("MODULE_HASH_MISMATCH", "qualification receipt SHA mismatch")
        if str(cfg.get("private_blob_name")) != str(tc.PRIVATE_BLOB_NAME):
            raise ProbeError("PRIVATE_BLOB_CONTRACT_DRIFT", "cfg private_blob_name != transport constant")
        mark(payload,"MODULES_VERIFIED",private_blob_name=tc.PRIVATE_BLOB_NAME,base_worker_sha256=sha(base_worker_path),qualification_sha256=sha(qualification_path))

        blobs = find_exact_basename(tc.PRIVATE_BLOB_NAME)
        if len(blobs) == 0:
            raise ProbeError("MOUNT_PRIVATE_BLOB_MISSING", f"mounted private blob missing: {tc.PRIVATE_BLOB_NAME}", retryable=True)
        if len(blobs) != 1:
            raise ProbeError("MOUNT_FILE_DUPLICATE", f"expected one private blob, found {len(blobs)}")
        blob = blobs[0]
        blob_sha = tc.sha256_file(blob)
        if blob_sha != str(cfg["private_input_sha256"]).upper():
            raise ProbeError("PRIVATE_BLOB_HASH_MISMATCH", f"private blob SHA mismatch expected={cfg['private_input_sha256']} actual={blob_sha}")
        src = tc.safe_extract_private_archive(blob, work / "private-input")
        tc.validate_source_tree(src)
        mark(payload,"PRIVATE_INPUT_VERIFIED",private_blob_sha256=blob_sha,private_extract_ok=True)

        ns = source_namespace(src)
        try:
            mc10a, act, v1, h11, eligible, unitreg = builder.verify_inputs(ns)
            bound = builder.verify_bound_sources(ns, mc10a)
            qualification = builder.verify_qualification_receipt(Path(ns.qualification_receipt))
        except Exception as e:
            raise ProbeError("SOURCE_AUTHORITY_FAILURE", f"source authority verification failed: {type(e).__name__}: {e}") from e
        if len(eligible) != int(cfg.get("expected_eligible_packets", -1)):
            raise ProbeError("FRONTIER_COUNT_DRIFT", f"eligible packet count={len(eligible)}")
        mark(payload,"SOURCE_AUTHORITY_VERIFIED",source_authority_ok=True,eligible_einf_packets=len(eligible),bound_source_count=len(bound),primary_model=qualification["portfolio"]["primary"]["tag"])

        try:
            gate = fc.verify_pilot_audit_gate(src, builder)
            pilot, remaining = fc.select_remaining_packets(eligible, builder, gate["pilot_selected_rows"])
        except Exception as e:
            raise ProbeError("PILOT_AUDIT_GATE_FAILURE", f"pilot audit gate failed: {type(e).__name__}: {e}") from e
        if len(pilot) != int(cfg.get("expected_pilot_packets", -1)) or len(remaining) != int(cfg.get("expected_remaining_packets", -1)):
            raise ProbeError("FRONTIER_PARTITION_DRIFT", f"pilot={len(pilot)} remaining={len(remaining)}")
        mark(payload,"PILOT_AND_FRONTIER_VERIFIED",pilot_audit_gate_passed=True,pilot_packets_verified=len(pilot),remaining_packets=len(remaining))

        resume_count=0
        rr=src/"resume_result"
        if rr.is_dir():
            rows=[]
            for n in ["mc10b_full_einf_raw_candidates_v1.partial.jsonl","mc10b_full_einf_raw_candidates_v1.jsonl"]:
                p=rr/n
                if p.is_file():
                    if n.endswith('.partial.jsonl'):
                        rows,recovery=fc.read_recoverable_partial_jsonl(p)
                        payload['resume_tail_recovered']=bool(recovery.get('tail_recovered'))
                        payload['resume_discarded_tail_bytes']=int(recovery.get('discarded_tail_bytes',0))
                    else:
                        rows=[json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
                    break
            if rows:
                try:
                    resume_count=len(fc.validate_resume_rows(rows,remaining,unitreg,builder,qualification["portfolio"]["primary"]))
                except Exception as e:
                    raise ProbeError("RESUME_VALIDATION_FAILURE", f"resume validation failed: {type(e).__name__}: {e}") from e
        mark(payload,"RESUME_VERIFIED",resume_candidates_validated=resume_count)

        runtime_candidates=sorted(p for p in INPUT_ROOT.rglob(tc.RUNTIME_ARCHIVE_NAME) if p.is_file())
        if not runtime_candidates:
            raise ProbeError("RUNTIME_MOUNT_MISSING", f"runtime archive not mounted: {tc.RUNTIME_ARCHIVE_NAME}", retryable=True)
        exact=[]
        observed=[]
        for p in runtime_candidates:
            size=p.stat().st_size
            digest=tc.sha256_file(p) if size==tc.RUNTIME_ARCHIVE_SIZE else None
            observed.append({"path":str(p),"size":size,"sha256":digest})
            if size==tc.RUNTIME_ARCHIVE_SIZE and digest==tc.RUNTIME_ARCHIVE_SHA256:
                exact.append(p)
        if len(exact)!=1:
            raise ProbeError("RUNTIME_HASH_OR_DUPLICATE", f"expected exactly one pinned runtime archive; observed={observed[:8]}")
        try:
            runtime_bin=tc.extract_pinned_runtime(exact[0],work/"runtime")
        except Exception as e:
            raise ProbeError("RUNTIME_EXTRACT_FAILURE", f"runtime extraction failed: {type(e).__name__}: {e}") from e
        binary_sha=tc.sha256_file(runtime_bin)
        if binary_sha!=tc.RUNTIME_BINARY_SHA256:
            raise ProbeError("RUNTIME_BINARY_HASH_MISMATCH", f"runtime binary SHA mismatch actual={binary_sha}")
        mark(payload,"RUNTIME_VERIFIED",runtime_archive_sha256=tc.RUNTIME_ARCHIVE_SHA256,runtime_binary_sha256=binary_sha,runtime_extract_ok=True)

        token=one("alice-github-token.txt").read_text(encoding="utf-8").strip()
        if len(token)<20:
            raise ProbeError("GITHUB_TOKEN_MALFORMED","transient GitHub token appears malformed")
        ctl_obj=github_request(token,"GET","mc10b/full/control.json",query={"ref":cfg["github_branch"]})
        if not isinstance(ctl_obj,dict):
            raise ProbeError("GITHUB_CONTROL_UNREADABLE","control file response invalid")
        ctl=json.loads(base64.b64decode(ctl_obj["content"]).decode("utf-8"))
        if ctl.get("run_id")!=cfg["run_id"] or str(ctl.get("action","run")).lower()!="run":
            raise ProbeError("GITHUB_CONTROL_MISMATCH","control run/action mismatch")
        mark(payload,"GITHUB_CONTROL_VERIFIED",github_control_read_ok=True)
        event={"artifact_id":"alice.MC10B.full-einf.cpu-preflight-event.v3","run_id":cfg["run_id"],"kind":"CPU_PREFLIGHT_GITHUB_WRITE_OK","at_utc":time.time()}
        nonce=hashlib.sha256((cfg["run_id"]+"|"+str(time.time_ns())).encode()).hexdigest()[:12]
        rel=f"mc10b/full/runs/{cfg['run_id']}/events/cpu-preflight-{nonce}.json"
        github_request(token,"PUT",rel,body={"message":"mc10b-full cpu preflight","content":base64.b64encode((canon(event)+"\n").encode()).decode(),"branch":cfg["github_branch"]})
        mark(payload,"COMPLETE",github_write_ok=True,github_preflight_event_path=rel)
        payload["pass"]=True
        payload["retryable"]=False
    except ProbeError as e:
        payload.update({"pass":False,"failure_code":e.code,"failure_class":type(e).__name__,"failure_message":str(e)[:4000],"retryable":bool(e.retryable),"traceback":traceback.format_exc()[-12000:]})
    except Exception as e:
        payload.update({"pass":False,"failure_code":"UNCLASSIFIED_INTERNAL_ERROR","failure_class":type(e).__name__,"failure_message":str(e)[:4000],"retryable":False,"traceback":traceback.format_exc()[-12000:]})
    finally:
        shutil.rmtree(work,ignore_errors=True)
        OUT.write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        print(canon(payload))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
