from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class QueryClaimRelevanceGatePolicy:
    policy_id: str
    enabled: bool
    model_id: str
    frozen_threshold: float
    objective: str
    fallback_answer: str
    private_output_only: bool
    memory_write_allowed: bool
    external_action_allowed: bool
    tool_calling_allowed: bool
    web_access_allowed: bool
    source_path: Path


def default_policy_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "policies"
        / "query_claim_relevance_gate_policy.json"
    )


def load_query_claim_relevance_gate_policy(
    path: Path | None = None,
) -> QueryClaimRelevanceGatePolicy:
    source = (path or default_policy_path()).expanduser().resolve(strict=True)
    data = json.loads(source.read_text(encoding="utf-8"))

    if int(data.get("query_claim_relevance_gate_policy_schema_version", -1)) != 1:
        raise ValueError("Unsupported query-claim relevance gate policy schema")

    policy = QueryClaimRelevanceGatePolicy(
        policy_id=str(data["policy_id"]),
        enabled=bool(data["enabled"]),
        model_id=str(data["model_id"]),
        frozen_threshold=float(data["frozen_threshold"]),
        objective=str(data["objective"]),
        fallback_answer=str(data["fallback_answer"]),
        private_output_only=bool(data["private_output_only"]),
        memory_write_allowed=bool(data["memory_write_allowed"]),
        external_action_allowed=bool(data["external_action_allowed"]),
        tool_calling_allowed=bool(data["tool_calling_allowed"]),
        web_access_allowed=bool(data["web_access_allowed"]),
        source_path=source,
    )

    if policy.objective != "non_irrelevance_filter":
        raise ValueError("Relevance gate must use the validated non-irrelevance objective")
    if not policy.private_output_only:
        raise ValueError("Relevance gate output must remain private")
    if any(
        (
            policy.memory_write_allowed,
            policy.external_action_allowed,
            policy.tool_calling_allowed,
            policy.web_access_allowed,
        )
    ):
        raise ValueError("Relevance gate must remain read-only and offline")

    return policy


@lru_cache(maxsize=4)
def _cached_local_model(vault_root_text: str, device: str):
    from .response_reranker import load_local_response_reranker

    return load_local_response_reranker(
        vault_root=Path(vault_root_text),
        device=device,
    )


def load_local_query_claim_relevance_model(
    *,
    vault_root: Path,
    device: str = "auto",
):
    vault_root = vault_root.expanduser().resolve(strict=True)
    return _cached_local_model(str(vault_root), str(device or "auto"))


def filter_model_output_by_query_claim_relevance(
    *,
    model_output: dict[str, Any],
    context_package: dict[str, Any],
    model,
    policy: QueryClaimRelevanceGatePolicy,
    answer_renderer: Callable[[list[dict[str, Any]]], str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = copy.deepcopy(model_output)
    claims = [
        claim
        for claim in output.get("claims", [])
        if isinstance(claim, dict) and str(claim.get("text", "")).strip()
    ]

    summary: dict[str, Any] = {
        "enabled": bool(policy.enabled),
        "policy_id": policy.policy_id,
        "model_id": policy.model_id,
        "objective": policy.objective,
        "frozen_threshold": policy.frozen_threshold,
        "input_claim_count": len(claims),
        "kept_claim_count": 0,
        "dropped_claim_count": 0,
        "decisions": [],
        "private_output_only": True,
        "memory_write_allowed": False,
        "external_action_allowed": False,
        "tool_calling_allowed": False,
        "web_access_allowed": False,
    }

    if not policy.enabled or not claims:
        summary["kept_claim_count"] = len(claims)
        return output, summary

    question = str(context_package.get("query", "")).strip()
    pairs = [[question, str(claim.get("text", "")).strip()] for claim in claims]
    scores = model.predict(pairs, show_progress_bar=False)

    kept: list[dict[str, Any]] = []
    for claim, score in zip(claims, scores):
        numeric_score = float(score)
        keep = numeric_score >= policy.frozen_threshold
        summary["decisions"].append(
            {
                "claim_text": str(claim.get("text", "")),
                "score": round(numeric_score, 6),
                "kept": keep,
            }
        )
        if keep:
            kept.append(claim)

    summary["kept_claim_count"] = len(kept)
    summary["dropped_claim_count"] = len(claims) - len(kept)
    output["claims"] = kept

    actual_contradictions = [
        group
        for group in context_package.get("contradiction_groups", [])
        if str(group.get("label", "")).strip()
    ]

    if kept:
        if not (
            output.get("answer_type") == "contradictory_evidence"
            and actual_contradictions
        ):
            output["answer_type"] = "grounded"
        output["answer"] = answer_renderer(kept)
    else:
        if not (
            output.get("answer_type") == "contradictory_evidence"
            and actual_contradictions
        ):
            output["answer_type"] = "insufficient_evidence"
            output["answer"] = policy.fallback_answer

    return output, summary
