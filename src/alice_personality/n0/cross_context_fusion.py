from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class CrossContextFusionConfig:
    """Identity-neutral fusion over semantic, structured, and evidence views.

    Parameter counts are measurements, not ceilings. The module preserves each
    contextualized view separately so later multi-view pooling can retain
    distinct evidence rather than relying on one collapsed representation.
    """

    semantic_size: int = 640
    fusion_size: int = 640
    num_layers: int = 3
    num_heads: int = 10
    feedforward_size: int = 2560
    dropout: float = 0.0
    num_views: int = 3
    reliability_prior_scale: float = 0.5

    def validate(self) -> None:
        if self.semantic_size < 1 or self.fusion_size < 1:
            raise ValueError("semantic_size and fusion_size must be positive")
        if self.num_layers < 1 or self.num_heads < 1:
            raise ValueError("num_layers and num_heads must be positive")
        if self.fusion_size % self.num_heads != 0:
            raise ValueError("fusion_size must be divisible by num_heads")
        if self.feedforward_size < self.fusion_size:
            raise ValueError("feedforward_size must be >= fusion_size")
        if self.num_views != 3:
            raise ValueError("N0 v0.2 fusion currently defines exactly three public input views")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.reliability_prior_scale < 0.0:
            raise ValueError("reliability_prior_scale must be non-negative")


class CrossContextFusion(nn.Module):
    """Fuse public semantic, typed-state, and evidence contexts.

    Each view contributes a learned summary token plus its contextual tokens.
    The encoder has no new positional embeddings: semantic tokens already carry
    upstream positional information, while structured/evidence fields retain
    their order-robust semantics. View identity and reliability are explicit.
    """

    SEMANTIC_VIEW = 0
    STRUCTURED_VIEW = 1
    EVIDENCE_VIEW = 2

    def __init__(self, config: CrossContextFusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or CrossContextFusionConfig()
        self.config.validate()

        d = self.config.fusion_size
        self.input_projection = nn.Linear(self.config.semantic_size, d, bias=False)
        self.query_projection = nn.Linear(self.config.semantic_size, d, bias=False)
        self.view_embedding = nn.Embedding(self.config.num_views, d)
        self.reliability_projection = nn.Sequential(
            nn.Linear(1, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.summary_tokens = nn.Parameter(torch.empty(self.config.num_views, d))
        nn.init.normal_(self.summary_tokens, mean=0.0, std=d ** -0.5)
        self.input_norm = nn.LayerNorm(d)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.feedforward_size,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.config.num_layers)
        self.output_norm = nn.LayerNorm(d)

        self.fusion_query = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.fusion_query, mean=0.0, std=d ** -0.5)
        self.view_score = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.SiLU(),
            nn.Linear(d, 1),
        )
        self.token_output_projection = nn.Linear(d, self.config.semantic_size, bias=False)
        self.summary_output_projection = nn.Linear(d, self.config.semantic_size, bias=False)
        self.fused_output_projection = nn.Linear(d, self.config.semantic_size, bias=False)

    def _validate_view(
        self,
        name: str,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        batch: int,
    ) -> None:
        if tokens.ndim != 3:
            raise ValueError(f"{name}_tokens must have shape [batch, tokens, semantic_size]")
        if tokens.shape[0] != batch or tokens.shape[2] != self.config.semantic_size:
            raise ValueError(f"{name}_tokens shape mismatch")
        if valid_mask.shape != tokens.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError(f"{name}_valid_mask must be bool with shape [batch, tokens]")
        if not torch.isfinite(tokens).all():
            raise ValueError(f"{name}_tokens contains non-finite values")

    def _validate_inputs(
        self,
        *,
        semantic_tokens: torch.Tensor,
        semantic_valid_mask: torch.Tensor,
        structured_tokens: torch.Tensor,
        structured_valid_mask: torch.Tensor,
        evidence_tokens: torch.Tensor,
        evidence_valid_mask: torch.Tensor,
        query_semantic: torch.Tensor,
        view_reliability: torch.Tensor,
    ) -> int:
        if query_semantic.ndim != 2 or query_semantic.shape[1] != self.config.semantic_size:
            raise ValueError("query_semantic must have shape [batch, semantic_size]")
        batch = query_semantic.shape[0]
        if not torch.isfinite(query_semantic).all():
            raise ValueError("query_semantic contains non-finite values")
        self._validate_view("semantic", semantic_tokens, semantic_valid_mask, batch)
        self._validate_view("structured", structured_tokens, structured_valid_mask, batch)
        self._validate_view("evidence", evidence_tokens, evidence_valid_mask, batch)
        if view_reliability.shape != (batch, self.config.num_views):
            raise ValueError("view_reliability must have shape [batch, 3]")
        if not torch.isfinite(view_reliability).all():
            raise ValueError("view_reliability contains non-finite values")
        if (view_reliability < 0).any() or (view_reliability > 1).any():
            raise ValueError("view_reliability must be in [0, 1]")
        available = torch.stack(
            [
                semantic_valid_mask.any(dim=1),
                structured_valid_mask.any(dim=1),
                evidence_valid_mask.any(dim=1),
            ],
            dim=1,
        )
        if not torch.all(available.any(dim=1)):
            raise ValueError("every example must provide at least one fusion view")
        return batch

    def _prepare_view(
        self,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        view_id: int,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = tokens.shape[0]
        projected = self.input_projection(tokens)
        view = self.view_embedding.weight[view_id].view(1, 1, -1)
        rel = self.reliability_projection(reliability.view(batch, 1, 1))
        projected = projected + view + rel

        summary = self.summary_tokens[view_id].view(1, 1, -1).expand(batch, -1, -1)
        summary = summary + view + rel
        available = valid_mask.any(dim=1, keepdim=True)
        group = torch.cat([summary, projected], dim=1)
        mask = torch.cat([available, valid_mask], dim=1)
        return group, mask

    def forward(
        self,
        *,
        semantic_tokens: torch.Tensor,
        semantic_valid_mask: torch.Tensor,
        structured_tokens: torch.Tensor,
        structured_valid_mask: torch.Tensor,
        evidence_tokens: torch.Tensor,
        evidence_valid_mask: torch.Tensor,
        query_semantic: torch.Tensor,
        view_reliability: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = self._validate_inputs(
            semantic_tokens=semantic_tokens,
            semantic_valid_mask=semantic_valid_mask,
            structured_tokens=structured_tokens,
            structured_valid_mask=structured_valid_mask,
            evidence_tokens=evidence_tokens,
            evidence_valid_mask=evidence_valid_mask,
            query_semantic=query_semantic,
            view_reliability=view_reliability,
        )

        groups: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        lengths: list[int] = []
        for view_id, (tokens, mask) in enumerate(
            (
                (semantic_tokens, semantic_valid_mask),
                (structured_tokens, structured_valid_mask),
                (evidence_tokens, evidence_valid_mask),
            )
        ):
            group, group_mask = self._prepare_view(
                tokens,
                mask,
                view_id=view_id,
                reliability=view_reliability[:, view_id],
            )
            groups.append(group)
            masks.append(group_mask)
            lengths.append(group.shape[1])

        x = self.input_norm(torch.cat(groups, dim=1))
        combined_mask = torch.cat(masks, dim=1)
        x = self.encoder(x, src_key_padding_mask=~combined_mask)
        x = self.output_norm(x)

        offsets = [0, lengths[0], lengths[0] + lengths[1]]
        summaries = torch.stack([x[:, offset] for offset in offsets], dim=1)
        available = torch.stack(
            [semantic_valid_mask.any(dim=1), structured_valid_mask.any(dim=1), evidence_valid_mask.any(dim=1)],
            dim=1,
        )
        query = torch.nn.functional.normalize(
            self.fusion_query.unsqueeze(0).expand(batch, -1) + self.query_projection(query_semantic),
            dim=-1,
        )
        query_expand = query.unsqueeze(1).expand(-1, self.config.num_views, -1)
        score_features = torch.cat(
            [summaries, query_expand, summaries * query_expand], dim=-1
        )
        scores = self.view_score(score_features).squeeze(-1)
        if self.config.reliability_prior_scale > 0.0:
            scores = scores + self.config.reliability_prior_scale * view_reliability.clamp_min(1e-6).log()
        scores = scores.masked_fill(~available, torch.finfo(scores.dtype).min)
        view_weights = torch.softmax(scores, dim=-1)
        fused = torch.einsum("bv,bvd->bd", view_weights, summaries)

        semantic_start = 1
        structured_start = offsets[1] + 1
        evidence_start = offsets[2] + 1
        semantic_context = x[:, semantic_start : semantic_start + semantic_tokens.shape[1]]
        structured_context = x[:, structured_start : structured_start + structured_tokens.shape[1]]
        evidence_context = x[:, evidence_start : evidence_start + evidence_tokens.shape[1]]

        normalized = torch.nn.functional.normalize(summaries, dim=-1)
        cross_view_cosine = torch.einsum("bvd,bwd->bvw", normalized, normalized)

        return {
            "semantic_tokens": self.token_output_projection(semantic_context),
            "structured_tokens": self.token_output_projection(structured_context),
            "evidence_tokens": self.token_output_projection(evidence_context),
            "view_summaries": self.summary_output_projection(summaries),
            "view_weights": view_weights,
            "fused_state": self.fused_output_projection(fused),
            "cross_view_cosine": cross_view_cosine,
            "view_available": available,
        }

    def parameter_report(self) -> dict[str, Any]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "semantic_size": self.config.semantic_size,
            "fusion_size": self.config.fusion_size,
            "layers": self.config.num_layers,
            "heads": self.config.num_heads,
            "feedforward_size": self.config.feedforward_size,
            "input_views": 3,
            "view_identity_explicit": True,
            "missing_view_supported": True,
            "preserves_contextualized_views": True,
            "position_embeddings_added": 0,
            "private_identity_parameters": 0,
            "hard_parameter_ceiling": None,
        }
