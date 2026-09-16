from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class CrossContextFusionConfig:
    """Full-scale identity-neutral N0 cross-context fusion.

    This is the intended N0 fusion architecture, not a reduced rehearsal model.
    It preserves semantic, structured, and evidence streams separately while
    allowing repeated gated bidirectional cross-attention between them.

    Parameter counts are measurements, never capability ceilings.
    """

    semantic_size: int = 640
    fusion_size: int = 640
    num_layers: int = 4
    num_heads: int = 10
    feedforward_size: int = 2560
    dropout: float = 0.0
    num_views: int = 3
    reliability_prior_scale: float = 0.5
    cross_gate_bias: float = -1.0

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
            raise ValueError("N0 fusion defines exactly three public input views")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.reliability_prior_scale < 0.0:
            raise ValueError("reliability_prior_scale must be non-negative")


class _GEGLU(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.in_proj = nn.Linear(width, 2 * hidden)
        self.out_proj = nn.Linear(hidden, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(self.dropout(value * F.gelu(gate)))


class _TriStreamFusionBlock(nn.Module):
    """One full interaction stage over three preserved context streams."""

    def __init__(self, config: CrossContextFusionConfig) -> None:
        super().__init__()
        d = config.fusion_size
        self.num_views = config.num_views
        self.self_norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(self.num_views))
        self.self_attention = nn.ModuleList(
            nn.MultiheadAttention(
                d,
                config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            for _ in range(self.num_views)
        )
        self.cross_query_norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(self.num_views))
        self.cross_source_norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(self.num_views))
        self.cross_attention = nn.ModuleList(
            nn.MultiheadAttention(
                d,
                config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            for _ in range(self.num_views)
        )
        self.gate_reliability = nn.ModuleList(nn.Linear(2, d) for _ in range(self.num_views))
        self.cross_gate = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * d, d),
                nn.SiLU(),
                nn.Linear(d, d),
            )
            for _ in range(self.num_views)
        )
        for gate in self.cross_gate:
            final = gate[-1]
            assert isinstance(final, nn.Linear)
            nn.init.constant_(final.bias, config.cross_gate_bias)

        self.ffn_norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(self.num_views))
        self.ffn = nn.ModuleList(
            _GEGLU(d, config.feedforward_size, config.dropout)
            for _ in range(self.num_views)
        )

    @staticmethod
    def _safe_attention(
        attention: nn.MultiheadAttention,
        query: torch.Tensor,
        key_value: torch.Tensor,
        *,
        query_valid: torch.Tensor,
        key_valid: torch.Tensor,
    ) -> torch.Tensor:
        if key_value.size(1) < 1:
            return torch.zeros_like(query)
        safe_key_valid = key_valid.clone()
        missing_source = ~safe_key_valid.any(dim=1)
        safe_key_value = key_value
        if missing_source.any():
            safe_key_value = key_value.clone()
            safe_key_value[missing_source, 0] = 0.0
            safe_key_valid[missing_source, 0] = True
        output, _ = attention(
            query,
            safe_key_value,
            safe_key_value,
            key_padding_mask=~safe_key_valid,
            need_weights=False,
        )
        active = query_valid & (~missing_source).unsqueeze(1)
        return torch.where(active.unsqueeze(-1), output, torch.zeros_like(output))

    def forward(
        self,
        streams: list[torch.Tensor],
        masks: list[torch.Tensor],
        *,
        query: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        refined: list[torch.Tensor] = []
        for view_id in range(self.num_views):
            normed = self.self_norm[view_id](streams[view_id])
            attended = self._safe_attention(
                self.self_attention[view_id],
                normed,
                normed,
                query_valid=masks[view_id],
                key_valid=masks[view_id],
            )
            value = streams[view_id] + attended
            value = torch.where(masks[view_id].unsqueeze(-1), value, torch.zeros_like(value))
            refined.append(value)

        exchanged: list[torch.Tensor] = []
        gate_means: list[torch.Tensor] = []
        available = torch.stack([mask.any(dim=1) for mask in masks], dim=1)
        for target_id in range(self.num_views):
            source_ids = [idx for idx in range(self.num_views) if idx != target_id]
            target = refined[target_id]
            target_norm = self.cross_query_norm[target_id](target)
            source = torch.cat(
                [self.cross_source_norm[target_id](refined[idx]) for idx in source_ids],
                dim=1,
            )
            source_mask = torch.cat([masks[idx] for idx in source_ids], dim=1)
            cross = self._safe_attention(
                self.cross_attention[target_id],
                target_norm,
                source,
                query_valid=masks[target_id],
                key_valid=source_mask,
            )

            source_available = available[:, source_ids].to(reliability.dtype)
            source_reliability = (
                reliability[:, source_ids] * source_available
            ).sum(dim=1) / source_available.sum(dim=1).clamp_min(1.0)
            rel_features = torch.stack(
                [reliability[:, target_id], source_reliability], dim=-1
            )
            gate_context = query + self.gate_reliability[target_id](rel_features)
            gate_context = gate_context.unsqueeze(1).expand(-1, target.size(1), -1)
            gate = torch.sigmoid(
                self.cross_gate[target_id](torch.cat([target_norm, gate_context], dim=-1))
            )
            has_cross_source = source_mask.any(dim=1, keepdim=True).unsqueeze(-1)
            active = masks[target_id].unsqueeze(-1) & has_cross_source
            gate = torch.where(active, gate, torch.zeros_like(gate))
            value = target + gate * cross
            value = value + self.ffn[target_id](self.ffn_norm[target_id](value))
            value = torch.where(masks[target_id].unsqueeze(-1), value, torch.zeros_like(value))
            exchanged.append(value)
            denom = masks[target_id].sum(dim=1).clamp_min(1).to(gate.dtype)
            gate_mean = (gate.mean(dim=-1) * masks[target_id].to(gate.dtype)).sum(dim=1) / denom
            gate_means.append(gate_mean)

        return exchanged, torch.stack(gate_means, dim=1)


class CrossContextFusion(nn.Module):
    """Frontier tri-stream fusion for semantic, typed-state, and evidence context.

    The three streams remain distinct through every stage. Each stream first
    self-refines, then queries the other two streams through gated cross-attention.
    Reliability and the current semantic query influence the exchange gates.
    The output preserves each contextualized stream for the later adaptive
    multi-view latent pool while also exposing a fused public N0 state.
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
        self.input_norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(self.config.num_views))
        self.blocks = nn.ModuleList(
            _TriStreamFusionBlock(self.config) for _ in range(self.config.num_layers)
        )
        self.output_norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(self.config.num_views))

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
        self.token_residual_scale = nn.Parameter(torch.full((self.config.num_views,), 0.10))

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
        if tokens.shape[1] < 1:
            raise ValueError(f"{name}_tokens must reserve at least one token slot")
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
    ) -> tuple[int, torch.Tensor]:
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
        return batch, available

    def _prepare_stream(
        self,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        view_id: int,
        reliability: torch.Tensor,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = tokens.shape[0]
        view = self.view_embedding.weight[view_id].view(1, 1, -1)
        rel = self.reliability_projection(reliability.view(batch, 1, 1))
        projected = self.input_projection(tokens) + view + rel
        summary = self.summary_tokens[view_id].view(1, 1, -1).expand(batch, -1, -1)
        summary = summary + view + rel + query.unsqueeze(1)
        available = valid_mask.any(dim=1, keepdim=True)
        stream = torch.cat([summary, projected], dim=1)
        mask = torch.cat([available, valid_mask], dim=1)
        stream = self.input_norm[view_id](stream)
        stream = torch.where(mask.unsqueeze(-1), stream, torch.zeros_like(stream))
        return stream, mask

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
        batch, available = self._validate_inputs(
            semantic_tokens=semantic_tokens,
            semantic_valid_mask=semantic_valid_mask,
            structured_tokens=structured_tokens,
            structured_valid_mask=structured_valid_mask,
            evidence_tokens=evidence_tokens,
            evidence_valid_mask=evidence_valid_mask,
            query_semantic=query_semantic,
            view_reliability=view_reliability,
        )
        source_tokens = [semantic_tokens, structured_tokens, evidence_tokens]
        source_masks = [semantic_valid_mask, structured_valid_mask, evidence_valid_mask]
        query = self.query_projection(query_semantic)

        streams: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for view_id, (tokens, mask) in enumerate(zip(source_tokens, source_masks)):
            stream, stream_mask = self._prepare_stream(
                tokens,
                mask,
                view_id=view_id,
                reliability=view_reliability[:, view_id],
                query=query,
            )
            streams.append(stream)
            masks.append(stream_mask)

        block_gate_means: list[torch.Tensor] = []
        for block in self.blocks:
            streams, gate_means = block(
                streams,
                masks,
                query=query,
                reliability=view_reliability,
            )
            block_gate_means.append(gate_means)

        streams = [self.output_norm[idx](stream) for idx, stream in enumerate(streams)]
        summaries = torch.stack([stream[:, 0] for stream in streams], dim=1)
        route_query = F.normalize(
            self.fusion_query.unsqueeze(0).expand(batch, -1) + query,
            dim=-1,
        )
        query_expand = route_query.unsqueeze(1).expand(-1, self.config.num_views, -1)
        score_features = torch.cat(
            [summaries, query_expand, summaries * query_expand], dim=-1
        )
        scores = self.view_score(score_features).squeeze(-1)
        if self.config.reliability_prior_scale > 0.0:
            scores = scores + self.config.reliability_prior_scale * view_reliability.clamp_min(1e-6).log()
        scores = scores.masked_fill(~available, torch.finfo(scores.dtype).min)
        view_weights = torch.softmax(scores, dim=-1)
        fused = torch.einsum("bv,bvd->bd", view_weights, summaries)

        contextualized_tokens: list[torch.Tensor] = []
        for view_id, (stream, source, valid) in enumerate(zip(streams, source_tokens, source_masks)):
            delta = self.token_output_projection(stream[:, 1:])
            scale = torch.tanh(self.token_residual_scale[view_id])
            output = source + scale * delta
            output = torch.where(valid.unsqueeze(-1), output, source)
            contextualized_tokens.append(output)

        summary_output = self.summary_output_projection(summaries)
        normalized = F.normalize(summary_output, dim=-1)
        cross_view_cosine = torch.einsum("bvd,bwd->bvw", normalized, normalized)

        return {
            "semantic_tokens": contextualized_tokens[self.SEMANTIC_VIEW],
            "structured_tokens": contextualized_tokens[self.STRUCTURED_VIEW],
            "evidence_tokens": contextualized_tokens[self.EVIDENCE_VIEW],
            "view_summaries": summary_output,
            "view_weights": view_weights,
            "fused_state": self.fused_output_projection(fused),
            "cross_view_cosine": cross_view_cosine,
            "view_available": available,
            "cross_gate_means": torch.stack(block_gate_means, dim=1),
        }

    def parameter_report(self) -> dict[str, Any]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "semantic_size": self.config.semantic_size,
            "fusion_size": self.config.fusion_size,
            "fusion_stages": self.config.num_layers,
            "heads": self.config.num_heads,
            "feedforward_size": self.config.feedforward_size,
            "input_views": 3,
            "fusion_family": "tri_stream_self_refinement_plus_gated_bidirectional_cross_attention",
            "full_scale_n0_candidate": True,
            "reduced_pilot_model": False,
            "view_identity_explicit": True,
            "view_reliability_explicit": True,
            "missing_view_supported": True,
            "gated_cross_view_exchange": True,
            "bidirectional_cross_attention": True,
            "preserves_contextualized_views": True,
            "position_embeddings_added": 0,
            "private_identity_parameters": 0,
            "hard_parameter_ceiling": None,
        }
