from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor

from alice_personality.n0.qsre_schema_matcher import QSRESchemaMatcher
from alice_personality.n0.qsre_semantic_authority import (
    combine_authority_components,
)
from prepare_n0_v02_qsre_production_cache_v1 import encode_texts


SEMANTIC_EMBED_PROMPT = (
    "Represent the semantic meaning of the supplied text for comparison with "
    "another possible interpretation."
)
JOINT_MATCH_PROMPT = (
    "Decide how well the candidate semantic interpretation matches this request. "
    "Judge the meaning, argument roles, direction, scope, and stated constraints. "
    "Request: "
)


def _chunks(count: int, batch_size: int):
    for start in range(0, count, batch_size):
        yield start, min(start + batch_size, count)


@torch.inference_mode()
def semantic_embeddings(
    *,
    texts: list[str],
    model,
    tokenizer,
    device: torch.device,
    max_length: int = 160,
    batch_size: int = 32,
) -> Tensor:
    if not texts:
        raise ValueError("semantic embedding text list is empty")
    output: list[Tensor] = []
    for start, stop in _chunks(len(texts), batch_size):
        batch_text = texts[start:stop]
        tokenized = tokenizer(
            [SEMANTIC_EMBED_PROMPT] * len(batch_text),
            batch_text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ids = tokenized["input_ids"].to(device)
        mask = tokenized["attention_mask"].to(device)
        pooled = model.encode(ids, mask)
        projected = model.project_semantic_pooled(pooled)
        output.append(projected.detach().float().cpu())
    return torch.cat(output, dim=0)


@torch.inference_mode()
def joint_preference_scores(
    *,
    queries: list[str],
    candidates: list[str],
    model,
    tokenizer,
    device: torch.device,
    max_length: int = 192,
    batch_size: int = 32,
) -> Tensor:
    if not queries or not candidates:
        raise ValueError("joint preference scoring requires queries and candidates")
    prompts: list[str] = []
    candidate_text: list[str] = []
    for query in queries:
        prompt = JOINT_MATCH_PROMPT + str(query)
        for candidate in candidates:
            prompts.append(prompt)
            candidate_text.append(str(candidate))

    scores: list[Tensor] = []
    for start, stop in _chunks(len(prompts), batch_size):
        tokenized = tokenizer(
            prompts[start:stop],
            candidate_text[start:stop],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ids = tokenized["input_ids"].to(device)
        mask = tokenized["attention_mask"].to(device)
        value = model.score_candidates(ids, mask)
        scores.append(value.detach().float().cpu())
    return torch.cat(scores, dim=0).reshape(len(queries), len(candidates))


@torch.inference_mode()
def semantic_projection_scores(
    *,
    queries: list[str],
    candidates: list[str],
    model,
    tokenizer,
    device: torch.device,
    max_length: int = 160,
    batch_size: int = 32,
) -> Tensor:
    query = semantic_embeddings(
        texts=queries,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    candidate = semantic_embeddings(
        texts=candidates,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    return torch.einsum("qd,cd->qc", query, candidate)


@torch.inference_mode()
def token_evidence_scores(
    *,
    queries: list[str],
    candidates: list[str],
    model,
    tokenizer,
    device: torch.device,
    semantic_dim: int,
    num_hidden_states: int,
    max_length: int = 128,
    batch_size: int = 32,
) -> Tensor:
    query_states, query_mask = encode_texts(
        texts=queries,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        all_hidden_states=True,
    )
    candidate_states, candidate_mask = encode_texts(
        texts=candidates,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        all_hidden_states=True,
    )
    matcher = QSRESchemaMatcher(
        semantic_dim=semantic_dim,
        model_dim=semantic_dim,
        num_hidden_states=num_hidden_states,
    )
    result = matcher(
        query_hidden_states=query_states.float(),
        query_token_mask=query_mask.bool(),
        schema_hidden_states=candidate_states.float(),
        schema_token_mask=candidate_mask.bool(),
    )
    return result["token_score"].detach().float().cpu()


@torch.inference_mode()
def frozen_authority_scores(
    *,
    queries: list[str],
    candidates: list[str],
    model,
    tokenizer,
    device: torch.device,
    semantic_dim: int,
    num_hidden_states: int,
    max_length: int = 160,
    batch_size: int = 32,
    include_token_evidence: bool = True,
) -> dict[str, Tensor]:
    joint = joint_preference_scores(
        queries=queries,
        candidates=candidates,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    semantic = semantic_projection_scores(
        queries=queries,
        candidates=candidates,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    if include_token_evidence:
        token = token_evidence_scores(
            queries=queries,
            candidates=candidates,
            model=model,
            tokenizer=tokenizer,
            device=device,
            semantic_dim=semantic_dim,
            num_hidden_states=num_hidden_states,
            max_length=min(max_length, 128),
            batch_size=batch_size,
        )
        combined = combine_authority_components(
            joint_preference=joint,
            semantic_projection=semantic,
            token_evidence=token,
        )
    else:
        token = torch.zeros_like(joint)
        combined = None
    return {
        "joint_preference": joint,
        "semantic_projection": semantic,
        "token_evidence": token,
        "combined": combined,
    }
