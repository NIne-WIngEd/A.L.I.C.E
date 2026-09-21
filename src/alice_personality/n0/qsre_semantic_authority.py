from __future__ import annotations

import torch
from torch import Tensor


AUTHORITY_COMPONENTS = (
    "joint_preference",
    "semantic_projection",
    "principle_alignment",
    "token_evidence",
)


def candidate_zscore(values: Tensor) -> Tensor:
    """Normalize only across the candidates that are active in this call.

    Caches store raw component scores. Production TRAIN can therefore slice to
    the six core candidates before normalization, while DEV/final can normalize
    over their larger runtime schemas without allowing held-out candidates to
    influence a training gradient.
    """
    if values.ndim < 2:
        raise ValueError("authority score tensor requires a candidate axis")
    if values.size(-1) <= 1:
        return torch.zeros_like(values)
    centered = values - values.mean(dim=-1, keepdim=True)
    scale = centered.square().mean(dim=-1, keepdim=True).sqrt()
    return centered / scale.clamp_min(1.0e-6)


def combine_authority_components(
    *,
    joint_preference: Tensor,
    semantic_projection: Tensor,
    principle_alignment: Tensor,
    token_evidence: Tensor,
) -> Tensor:
    shape = joint_preference.shape
    if (
        semantic_projection.shape != shape
        or principle_alignment.shape != shape
        or token_evidence.shape != shape
    ):
        raise ValueError("frozen semantic authority component shape drift")
    return (
        candidate_zscore(joint_preference)
        + candidate_zscore(semantic_projection)
        + candidate_zscore(principle_alignment)
        + candidate_zscore(token_evidence)
    ) / 4.0


def slice_relation_authority(
    cache_split: dict,
    *,
    indices: Tensor,
    view: int,
    relation_count: int,
    device: torch.device,
) -> dict[str, Tensor]:
    if relation_count <= 0:
        raise ValueError("relation_count must be positive")
    relation = cache_split["relation"]
    result = {}
    for key in ("joint_preference", "semantic_projection", "principle_alignment"):
        value = relation[key]
        if value.ndim != 3:
            raise ValueError(f"relation authority {key} must be [N,V,R]")
        if view < 0 or view >= value.size(1):
            raise ValueError("authority query-view index drift")
        if relation_count > value.size(2):
            raise ValueError("authority cache relation cardinality too small")
        result[key] = value[indices, view, :relation_count].to(
            device=device,
            dtype=torch.float32,
        )
    return result


def slice_factor_authority(
    cache_split: dict,
    *,
    indices: Tensor,
    view: int,
    device: torch.device,
) -> dict[str, dict[str, Tensor]]:
    factors = cache_split["factors"]
    result: dict[str, dict[str, Tensor]] = {}
    for category, row in factors.items():
        result[category] = {}
        for key in ("joint_preference", "semantic_projection", "principle_alignment"):
            value = row[key]
            if value.ndim < 3:
                raise ValueError(
                    f"factor authority {category}/{key} requires query and candidate axes"
                )
            if view < 0 or view >= value.size(1):
                raise ValueError("factor authority query-view index drift")
            result[category][key] = value[indices, view].to(
                device=device,
                dtype=torch.float32,
            )
    return result


def validate_authority_cache(
    payload: dict,
    *,
    expected_schema: str,
) -> None:
    if payload.get("schema") != expected_schema:
        raise ValueError("frozen semantic authority cache schema drift")
    if payload.get("gradient") is not False:
        raise ValueError("frozen semantic authority cache must be gradient-free")
    if payload.get("optimizer") is not False:
        raise ValueError("frozen semantic authority cache must be optimizer-free")
    if payload.get("private_identity_data") is not False:
        raise ValueError("private identity data entered frozen semantic authority cache")
    if payload.get("relation_keys_used_as_semantic_tokens") is not False:
        raise ValueError("relation keys entered frozen semantic authority text")
