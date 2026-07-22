"""Semantic and hybrid retrieval over Phase 2 memory.

Every candidate is re-hydrated from the authoritative Memory Core and passed
through the same deterministic authorization, classification, lifecycle,
temporal, correction, and conflict rules established by P2.5a.
"""

from __future__ import annotations

from pathlib import Path

from alice_vault.semantic_retrieval import load_semantic_policy

from .lexical_index import (
    memory_lexical_index_path,
    search_memory_lexical_candidates,
    verify_memory_lexical_index,
)
from .retrieval import (
    _classification_allowed,
    _conflict_ids,
    _corrected_targets,
    _result_from_record,
    _require_retrieval_authorization,
    _temporally_eligible,
    _validate_request,
)
from .retrieval_models import (
    MemoryRetrievalAuthorization,
    MemorySearchRequest,
    MemorySearchResponse,
)
from .semantic_index import (
    semantic_memory_candidates,
)
from .service import load_memory


def _filtered_ranked_records(
    connection,
    *,
    ranked_candidates: list[
        tuple[str, float, str]
    ],
    request: MemorySearchRequest,
    authorization: MemoryRetrievalAuthorization,
) -> tuple[
    list[tuple[object, float, str]],
    set[str],
]:
    corrected = _corrected_targets(
        connection
    )
    ranked = []
    seen: set[str] = set()

    for memory_id, score, matched_by in ranked_candidates:
        if memory_id in seen:
            continue
        record = load_memory(
            connection,
            memory_id=memory_id,
        )
        if not _classification_allowed(
            record,
            authorization,
        ):
            continue
        if not _temporally_eligible(
            record,
            request=request,
            corrected_targets=corrected,
        ):
            continue

        ranked.append(
            (
                record,
                score,
                matched_by,
            )
        )
        seen.add(memory_id)
        if len(ranked) >= request.limit:
            break

    if request.expand_conflicts:
        expanded = list(ranked)
        for record, score, matched_by in tuple(
            ranked
        ):
            for other_id in _conflict_ids(
                connection,
                memory_id=record.memory_id,
            ):
                if other_id in seen:
                    continue
                other = load_memory(
                    connection,
                    memory_id=other_id,
                )
                if not _classification_allowed(
                    other,
                    authorization,
                ):
                    continue
                if not _temporally_eligible(
                    other,
                    request=request,
                    corrected_targets=corrected,
                ):
                    continue
                expanded.append(
                    (
                        other,
                        score,
                        matched_by,
                    )
                )
                seen.add(other_id)
        ranked = expanded

    return ranked, seen


def search_memories_semantic(
    connection,
    vault_root: str | Path,
    *,
    request: MemorySearchRequest,
    authorization: MemoryRetrievalAuthorization,
    model,
    policy_path: Path | None = None,
    repository_root: str | Path | None = None,
) -> MemorySearchResponse:
    _require_retrieval_authorization(
        authorization
    )
    _validate_request(request)

    candidate_limit = min(
        max(request.limit * 8, request.limit),
        800,
    )
    manifest, semantic = (
        semantic_memory_candidates(
            connection,
            vault_root,
            query=request.query,
            model=model,
            policy_path=policy_path,
            repository_root=repository_root,
            limit=candidate_limit,
        )
    )

    ranked, seen = _filtered_ranked_records(
        connection,
        ranked_candidates=[
            (
                memory_id,
                score,
                "semantic",
            )
            for memory_id, score in semantic
        ],
        request=request,
        authorization=authorization,
    )

    results = tuple(
        _result_from_record(
            record,
            score=score,
            conflict_memory_ids=tuple(
                other_id
                for other_id in _conflict_ids(
                    connection,
                    memory_id=record.memory_id,
                )
                if other_id in seen
            ),
        ).__class__(
            **{
                **_result_from_record(
                    record,
                    score=score,
                    conflict_memory_ids=tuple(
                        other_id
                        for other_id in _conflict_ids(
                            connection,
                            memory_id=record.memory_id,
                        )
                        if other_id in seen
                    ),
                ).__dict__,
                "matched_by": matched_by,
            }
        )
        for record, score, matched_by in ranked
    )

    return MemorySearchResponse(
        query=request.query,
        results=results,
        index_id=manifest.index_id,
        authoritative_digest=manifest.authoritative_digest,
    )


def hybrid_search_memories(
    connection,
    vault_root: str | Path,
    *,
    request: MemorySearchRequest,
    authorization: MemoryRetrievalAuthorization,
    model,
    policy_path: Path | None = None,
    repository_root: str | Path | None = None,
) -> MemorySearchResponse:
    """Fuse lexical and semantic memory candidates with weighted RRF."""
    _require_retrieval_authorization(
        authorization
    )
    _validate_request(request)

    policy = load_semantic_policy(
        policy_path
    )
    lexical_manifest = verify_memory_lexical_index(
        connection,
        memory_lexical_index_path(
            vault_root,
            repository_root=repository_root,
        ),
    )

    lexical = search_memory_lexical_candidates(
        memory_lexical_index_path(
            vault_root,
            repository_root=repository_root,
        ),
        query=request.query,
        limit=policy.search.lexical_candidate_k,
    )
    semantic_manifest, semantic = (
        semantic_memory_candidates(
            connection,
            vault_root,
            query=request.query,
            model=model,
            policy_path=policy_path,
            repository_root=repository_root,
            limit=policy.search.semantic_candidate_k,
        )
    )

    fused: dict[str, dict[str, float | int | None]] = {}
    for method, candidates, weight in (
        (
            "lexical",
            lexical,
            policy.search.lexical_weight,
        ),
        (
            "semantic",
            semantic,
            policy.search.semantic_weight,
        ),
    ):
        for rank, (
            memory_id,
            _score,
        ) in enumerate(
            candidates,
            start=1,
        ):
            entry = fused.setdefault(
                memory_id,
                {
                    "rrf_score": 0.0,
                    "lexical_rank": None,
                    "semantic_rank": None,
                },
            )
            entry["rrf_score"] = float(
                entry["rrf_score"]
            ) + (
                float(weight)
                / (
                    policy.search.rrf_k
                    + rank
                )
            )
            entry[f"{method}_rank"] = rank

    ordered = sorted(
        fused.items(),
        key=lambda item: (
            -float(
                item[1]["rrf_score"]
            ),
            item[0],
        ),
    )

    candidates = []
    for memory_id, entry in ordered:
        lexical_rank = entry[
            "lexical_rank"
        ]
        semantic_rank = entry[
            "semantic_rank"
        ]
        if (
            lexical_rank is not None
            and semantic_rank is not None
        ):
            matched_by = "hybrid"
        elif semantic_rank is not None:
            matched_by = "semantic"
        else:
            matched_by = "lexical"

        candidates.append(
            (
                memory_id,
                float(
                    entry["rrf_score"]
                ),
                matched_by,
            )
        )

    ranked, seen = _filtered_ranked_records(
        connection,
        ranked_candidates=candidates,
        request=request,
        authorization=authorization,
    )

    results = []
    for record, score, matched_by in ranked:
        base = _result_from_record(
            record,
            score=score,
            conflict_memory_ids=tuple(
                other_id
                for other_id in _conflict_ids(
                    connection,
                    memory_id=record.memory_id,
                )
                if other_id in seen
            ),
        )
        results.append(
            base.__class__(
                **{
                    **base.__dict__,
                    "matched_by": matched_by,
                }
            )
        )

    combined_id = (
        f"lexical:{lexical_manifest.index_id}|"
        f"semantic:{semantic_manifest.index_id}"
    )
    return MemorySearchResponse(
        query=request.query,
        results=tuple(results),
        index_id=combined_id,
        authoritative_digest=semantic_manifest.authoritative_digest,
    )
