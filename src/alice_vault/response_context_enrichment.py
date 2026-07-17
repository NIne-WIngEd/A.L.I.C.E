from __future__ import annotations

import copy
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .grounded_context import _fingerprint
from .response_reranker import load_local_response_reranker, load_response_reranker_policy, rerank_candidates
from .semantic_retrieval import (
    _dot_scores,
    _encode_query,
    _load_local_model,
    _read_float32_matrix,
    _semantic_paths,
    load_semantic_policy,
    locate_chunk_set,
)


def _query_terms(query: str) -> set[str]:
    return {
        token
        for token in re.findall(
            r"[A-Za-z0-9][A-Za-z0-9_-]+",
            query.casefold(),
        )
        if len(token) >= 3
    }


def _lexical_overlap(
    query_terms: set[str],
    text: str,
) -> float:
    if not query_terms:
        return 0.0
    text_terms = set(
        re.findall(
            r"[A-Za-z0-9][A-Za-z0-9_-]+",
            text.casefold(),
        )
    )
    return len(query_terms.intersection(text_terms)) / len(
        query_terms
    )


def _interval_overlap_ratio(
    a_start: int,
    a_end: int,
    b_start: int,
    b_end: int,
) -> float:
    overlap = max(
        0,
        min(a_end, b_end) - max(a_start, b_start),
    )
    if overlap <= 0:
        return 0.0
    shorter = max(
        1,
        min(a_end - a_start, b_end - b_start),
    )
    return overlap / shorter


def _select_nonredundant(
    candidates: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        redundant = False
        for previous in selected:
            if (
                candidate["chunk_id"]
                == previous["chunk_id"]
                and _interval_overlap_ratio(
                    candidate["segment_start_char"],
                    candidate["segment_end_char"],
                    previous["segment_start_char"],
                    previous["segment_end_char"],
                )
                >= 0.72
            ):
                redundant = True
                break
        if redundant:
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _truncate_passages(
    passages: list[dict[str, Any]],
    *,
    maximum_characters: int,
) -> tuple[str, list[dict[str, Any]]]:
    if not passages:
        return "", []

    remaining = maximum_characters
    rendered: list[str] = []
    metadata: list[dict[str, Any]] = []

    for index, passage in enumerate(passages, start=1):
        prefix = f"Passage {index}: "
        if remaining <= len(prefix) + 20:
            break

        text = " ".join(
            str(passage["text"]).split()
        )
        allowed = remaining - len(prefix)
        if len(text) > allowed:
            text = text[: max(0, allowed - 2)].rstrip() + " …"

        rendered_piece = prefix + text
        rendered.append(rendered_piece)
        remaining -= len(rendered_piece) + 2

        metadata.append(
            {
                "semantic_segment_id": passage[
                    "semantic_segment_id"
                ],
                "chunk_id": passage["chunk_id"],
                "chunk_index": passage["chunk_index"],
                "segment_index": passage[
                    "segment_index"
                ],
                "semantic_cosine_similarity": round(
                    float(
                        passage[
                            "semantic_cosine_similarity"
                        ]
                    ),
                    8,
                ),
                "lexical_overlap": round(
                    float(passage["lexical_overlap"]),
                    6,
                ),
                "selection_score": round(
                    float(passage["selection_score"]),
                    8,
                ),
                "reranker_score": (
                    round(float(passage["reranker_score"]), 8)
                    if "reranker_score" in passage
                    else None
                ),
            }
        )

    return "\n\n".join(rendered), metadata


def enrich_response_context(
    *,
    vault_root: Path,
    context_package: dict[str, Any],
    passages_per_source: int = 3,
    maximum_characters_per_source: int = 2600,
    lexical_overlap_weight: float = 0.18,
    minimum_passage_characters: int = 80,
    semantic_policy_path: Path | None = None,
    reranker_policy_path: Path | None = None,
    device: str = "auto",
    model_loader=None,
    reranker_model_loader=None,
) -> dict[str, Any]:
    """Expand each retrieved source with its best internal semantic passages.

    P1.10 retrieves and ranks sources. This function keeps those source choices
    unchanged, but searches the already-built local semantic segment index for
    the most query-relevant passages inside each selected source.

    No new source is introduced, no benchmark label is consulted, and no
    private text leaves the local vault.
    """
    vault_root = vault_root.expanduser().resolve(strict=True)
    package = copy.deepcopy(context_package)
    query = str(package.get("query", "")).strip()
    if not query:
        raise ValueError("Context package has no query")

    evidence = list(package.get("evidence", []))
    source_hashes = {
        str(item["source_content_sha256"])
        for item in evidence
    }
    if not source_hashes:
        return package

    policy = load_semantic_policy(
        semantic_policy_path
    )
    root, manifest, segment_map = _semantic_paths(
        vault_root=vault_root,
        pilot_name=str(
            package.get("pilot_name", "pilot-v1")
        ),
        policy=policy,
    )
    model, _ = _load_local_model(
        vault_root=vault_root,
        policy=policy,
        device=device,
        model_loader=model_loader,
    )
    query_vector = _encode_query(
        model,
        query,
        policy,
    )

    reranker_policy = load_response_reranker_policy(reranker_policy_path)
    reranker = None
    if reranker_policy.enabled:
        if reranker_model_loader is not None:
            reranker = reranker_model_loader(
                vault_root=vault_root,
                policy_path=reranker_policy_path,
                device=device,
            )
            if isinstance(reranker, tuple):
                reranker = reranker[0]
        else:
            reranker, _ = load_local_response_reranker(
                vault_root=vault_root,
                policy_path=reranker_policy_path,
                device=device,
            )

    count = int(manifest["embedding_count"])
    dimension = int(manifest["embedding_dimension"])
    values = _read_float32_matrix(
        root / "embeddings.f32",
        rows=count,
        dimension=dimension,
    )
    scores = _dot_scores(
        values,
        rows=count,
        dimension=dimension,
        query=query_vector,
    )

    chunk_root = locate_chunk_set(
        vault_root=vault_root,
        pilot_name=str(
            package.get("pilot_name", "pilot-v1")
        ),
    )
    query_terms = _query_terms(query)
    by_source: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for semantic_score, record in zip(
        scores,
        segment_map,
    ):
        source = str(
            record["source_content_sha256"]
        )
        if source not in source_hashes:
            continue

        chunk_id = str(record["chunk_id"])
        parent_text = (
            chunk_root
            / "text"
            / f"{chunk_id}.txt"
        ).read_text(encoding="utf-8")
        start = int(
            record["segment_start_char"]
        )
        end = int(record["segment_end_char"])
        text = parent_text[start:end]
        if len(text.strip()) < minimum_passage_characters:
            continue

        overlap = _lexical_overlap(
            query_terms,
            text,
        )
        combined = float(
            semantic_score
        ) + lexical_overlap_weight * overlap

        by_source[source].append(
            {
                "semantic_segment_id": str(
                    record[
                        "semantic_segment_id"
                    ]
                ),
                "chunk_id": chunk_id,
                "chunk_index": int(
                    record["chunk_index"]
                ),
                "segment_index": int(
                    record["segment_index"]
                ),
                "segment_start_char": start,
                "segment_end_char": end,
                "semantic_cosine_similarity": float(
                    semantic_score
                ),
                "lexical_overlap": overlap,
                "selection_score": combined,
                "text": text,
            }
        )

    enriched_source_count = 0
    total_passages = 0
    for item in evidence:
        source = str(
            item["source_content_sha256"]
        )
        candidates = sorted(
            by_source.get(source, []),
            key=lambda candidate: (
                -candidate["selection_score"],
                candidate["semantic_segment_id"],
            ),
        )
        effective_passage_limit = passages_per_source
        effective_character_limit = maximum_characters_per_source

        if reranker_policy.enabled and reranker is not None:
            candidates = candidates[:reranker_policy.candidate_pool_per_source]
            candidates = rerank_candidates(
                query=query,
                candidates=candidates,
                reranker=reranker,
                batch_size=reranker_policy.batch_size,
            )
            effective_passage_limit = reranker_policy.passages_per_source
            effective_character_limit = reranker_policy.maximum_characters_per_source

        selected = _select_nonredundant(
            candidates,
            limit=effective_passage_limit,
        )
        rendered, metadata = _truncate_passages(
            selected,
            maximum_characters=effective_character_limit,
        )
        if rendered:
            item["context_text"] = rendered
            item[
                "response_evidence_passages"
            ] = metadata
            item[
                "response_evidence_expanded"
            ] = True
            enriched_source_count += 1
            total_passages += len(metadata)
        else:
            item[
                "response_evidence_passages"
            ] = []
            item[
                "response_evidence_expanded"
            ] = False

    package["evidence"] = evidence
    package[
        "response_context_enrichment"
    ] = {
        "strategy": (
            "source_preserving_semantic_passage_expansion_v1"
        ),
        "source_count": len(evidence),
        "enriched_source_count": (
            enriched_source_count
        ),
        "selected_passage_count": (
            total_passages
        ),
        "passages_per_source": passages_per_source,
        "maximum_characters_per_source": (
            maximum_characters_per_source
        ),
        "lexical_overlap_weight": (
            lexical_overlap_weight
        ),
        "benchmark_labels_used": False,
        "new_sources_introduced": False,
        "private_text_uploaded": False,
        "reranker_enabled": bool(reranker_policy.enabled),
        "reranker_policy_id": reranker_policy.policy_id,
        "reranker_model_id": (
            reranker_policy.model_id if reranker_policy.enabled else ""
        ),
        "reranker_candidate_pool_per_source": (
            reranker_policy.candidate_pool_per_source
            if reranker_policy.enabled else 0
        ),
    }
    package["package_fingerprint"] = _fingerprint(
        package
    )
    return package
