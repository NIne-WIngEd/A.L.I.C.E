from __future__ import annotations
import json, uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_MODEL_CACHE: dict[tuple[str,str], Any] = {}

@dataclass(frozen=True)
class ResponseRerankerPolicy:
    policy_id: str
    enabled: bool
    model_id: str
    revision: str
    candidate_pool_per_source: int
    passages_per_source: int
    batch_size: int
    maximum_characters_per_source: int
    minimum_passage_characters: int
    private_text_uploaded: bool
    source_path: Path

def default_policy_path() -> Path:
    return Path(__file__).resolve().parents[2] / "policies" / "response_reranker_policy.json"

def load_response_reranker_policy(path: Path|None=None) -> ResponseRerankerPolicy:
    source=(path or default_policy_path()).expanduser().resolve(strict=True)
    data=json.loads(source.read_text(encoding="utf-8"))
    if int(data.get("response_reranker_policy_schema_version",-1)) != 1:
        raise ValueError("Unsupported response-reranker policy schema")
    p=ResponseRerankerPolicy(
        policy_id=str(data["policy_id"]),
        enabled=bool(data["enabled"]),
        model_id=str(data["model_id"]),
        revision=str(data.get("revision","main")),
        candidate_pool_per_source=int(data["candidate_pool_per_source"]),
        passages_per_source=int(data["passages_per_source"]),
        batch_size=int(data["batch_size"]),
        maximum_characters_per_source=int(data["maximum_characters_per_source"]),
        minimum_passage_characters=int(data["minimum_passage_characters"]),
        private_text_uploaded=bool(data["private_text_uploaded"]),
        source_path=source,
    )
    if p.candidate_pool_per_source < p.passages_per_source:
        raise ValueError("candidate_pool_per_source must be >= passages_per_source")
    if p.private_text_uploaded:
        raise ValueError("Private text may not be uploaded by reranker")
    return p

def model_directory(*, vault_root: Path, policy: ResponseRerankerPolicy) -> Path:
    safe=policy.model_id.replace("/","__")
    rev=policy.revision.replace("/","_").replace(":","_")
    return vault_root / "models" / "rerankers" / f"{safe}__{rev}"

def prepare_response_reranker(*, vault_root: Path, policy_path: Path|None=None) -> dict[str,Any]:
    from huggingface_hub import HfApi, snapshot_download
    vault_root=vault_root.expanduser().resolve(strict=True)
    policy=load_response_reranker_policy(policy_path)
    root=model_directory(vault_root=vault_root, policy=policy)
    root.parent.mkdir(parents=True, exist_ok=True)
    info=HfApi().model_info(policy.model_id, revision=policy.revision)
    resolved=str(info.sha)
    snapshot_download(repo_id=policy.model_id, revision=resolved, local_dir=str(root))
    out={
        "response_reranker_model_manifest_schema_version":1,
        "model_id":policy.model_id,
        "requested_revision":policy.revision,
        "resolved_revision":resolved,
        "prepared_at":datetime.now(timezone.utc).isoformat(),
        "private_data_used_during_download":False,
        "model_path":str(root),
    }
    export=vault_root/"manifests"/"exports"/f"response-reranker-model-summary-{uuid.uuid4().hex}.json"
    export.parent.mkdir(parents=True, exist_ok=True)
    export.write_text(json.dumps(out, indent=2), encoding="utf-8")
    out["summary_path"]=str(export)
    return out

def load_local_response_reranker(*, vault_root: Path, policy_path: Path|None=None, device: str="auto"):
    policy=load_response_reranker_policy(policy_path)
    root=model_directory(vault_root=vault_root.expanduser().resolve(strict=True), policy=policy)
    if not root.is_dir():
        raise FileNotFoundError("Local response reranker is missing. Run scripts/prepare_response_reranker.py first.")
    key=(str(root),device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key], policy
    from sentence_transformers import CrossEncoder
    kwargs={}
    if device!="auto":
        kwargs["device"]=device
    model=CrossEncoder(str(root), **kwargs)
    _MODEL_CACHE[key]=model
    return model, policy

def rerank_candidates(*, query: str, candidates: list[dict[str,Any]], reranker, batch_size: int) -> list[dict[str,Any]]:
    if not candidates:
        return []
    pairs=[[query,str(c["text"])] for c in candidates]
    scores=reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    out=[]
    for c,s in zip(candidates,scores):
        item=dict(c); item["reranker_score"]=float(s); out.append(item)
    out.sort(key=lambda x:(-x["reranker_score"],-float(x.get("selection_score",0.0)),str(x.get("semantic_segment_id",""))))
    return out
