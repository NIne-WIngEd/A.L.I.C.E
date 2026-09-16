from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class AdaptiveMultiViewLatentPoolConfig:
    """Identity-neutral adaptive multi-view latent pooling for N0.

    ``num_slots`` and ``num_views`` are checkpoint tensor shapes. They are not
    personality ontologies and they are not permanent product ceilings. A
    successor checkpoint may expand either when measured capability or later
    identity fidelity requires more representational capacity.
    """

    semantic_size: int = 640
    latent_size: int = 640
    num_slots: int = 12
    num_layers: int = 3
    num_heads: int = 10
    feedforward_size: int = 2560
    dropout: float = 0.0
    num_views: int = 3
    reliability_feature_scale: float = 1.0
    route_feature_scale: float = 1.0

    def validate(self) -> None:
        if self.semantic_size < 1 or self.latent_size < 1:
            raise ValueError("semantic_size and latent_size must be positive")
        if self.num_slots < 2:
            raise ValueError("adaptive latent pooling requires at least two slots")
        if self.num_layers < 1 or self.num_heads < 1:
            raise ValueError("num_layers and num_heads must be positive")
        if self.latent_size % self.num_heads != 0:
            raise ValueError("latent_size must be divisible by num_heads")
        if self.feedforward_size < self.latent_size:
            raise ValueError("feedforward_size must be >= latent_size")
        if self.num_views < 2:
            raise ValueError("adaptive latent pooling requires at least two views")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.reliability_feature_scale < 0.0 or self.route_feature_scale < 0.0:
            raise ValueError("feature scales must be non-negative")


class _GEGLU(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.in_proj = nn.Linear(width, 2 * hidden)
        self.out_proj = nn.Linear(hidden, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(self.dropout(value * F.gelu(gate)))


class _LatentPoolBlock(nn.Module):
    def __init__(self, config: AdaptiveMultiViewLatentPoolConfig) -> None:
        super().__init__()
        d = config.latent_size
        self.cross_norm = nn.LayerNorm(d)
        self.bank_norm = nn.LayerNorm(d)
        self.cross_attention = nn.MultiheadAttention(
            d,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(d)
        self.self_attention = nn.MultiheadAttention(
            d,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn = _GEGLU(d, config.feedforward_size, config.dropout)

    def forward(
        self,
        latents: torch.Tensor,
        bank: torch.Tensor,
        bank_valid: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query = self.cross_norm(latents)
        key_value = self.bank_norm(bank)
        cross, attention = self.cross_attention(
            query,
            key_value,
            key_value,
            key_padding_mask=~bank_valid,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        latents = latents + cross

        normed = self.self_norm(latents)
        self_update, _ = self.self_attention(
            normed,
            normed,
            normed,
            need_weights=False,
        )
        latents = latents + self_update
        latents = latents + self.ffn(self.ffn_norm(latents))
        return latents, attention


class AdaptiveMultiViewLatentPool(nn.Module):
    """Adaptive latent-slot readout over ratified heterogeneous fusion views.

    The pool consumes both exact source tokens and learned contextualized tokens.
    It does not collapse the identity/judgment state into a single vector. The
    latent slots are the primary output. ``pooled_state`` is only a convenience
    readout for downstream heads that explicitly need one aggregate state.

    Fusion routing weights are metadata features. They are not interpreted as
    calibrated probabilities or as ground-truth personality labels.
    """

    SOURCE_CHANNEL = 0
    CONTEXTUALIZED_CHANNEL = 1

    def __init__(self, config: AdaptiveMultiViewLatentPoolConfig | None = None) -> None:
        super().__init__()
        self.config = config or AdaptiveMultiViewLatentPoolConfig()
        self.config.validate()
        d = self.config.latent_size

        self.input_projection = nn.Linear(self.config.semantic_size, d, bias=False)
        self.query_projection = nn.Linear(self.config.semantic_size, d, bias=False)
        self.view_embedding = nn.Embedding(self.config.num_views, d)
        self.channel_embedding = nn.Embedding(2, d)
        self.reliability_projection = nn.Sequential(
            nn.Linear(1, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.route_projection = nn.Sequential(
            nn.Linear(1, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )

        self.latent_slots = nn.Parameter(torch.empty(self.config.num_slots, d))
        nn.init.normal_(self.latent_slots, mean=0.0, std=d ** -0.5)
        self.slot_query_gate = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.SiLU(),
            nn.Linear(d, d),
            nn.Sigmoid(),
        )
        self.blocks = nn.ModuleList(_LatentPoolBlock(self.config) for _ in range(self.config.num_layers))
        self.output_norm = nn.LayerNorm(d)
        self.slot_output_projection = nn.Linear(d, self.config.semantic_size, bias=False)
        self.slot_score = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.SiLU(),
            nn.Linear(d, 1),
        )
        self.pooled_output_projection = nn.Linear(d, self.config.semantic_size, bias=False)

    def _validate_inputs(
        self,
        *,
        contextualized_view_tokens: list[torch.Tensor],
        source_view_tokens: list[torch.Tensor],
        view_valid_masks: list[torch.Tensor],
        query_semantic: torch.Tensor,
        view_reliability: torch.Tensor,
        fusion_view_weights: torch.Tensor,
    ) -> tuple[int, torch.Tensor]:
        if len(contextualized_view_tokens) != self.config.num_views:
            raise ValueError("contextualized view count does not match checkpoint topology")
        if len(source_view_tokens) != self.config.num_views:
            raise ValueError("source view count does not match checkpoint topology")
        if len(view_valid_masks) != self.config.num_views:
            raise ValueError("view mask count does not match checkpoint topology")
        if query_semantic.ndim != 2 or query_semantic.shape[1] != self.config.semantic_size:
            raise ValueError("query_semantic must have shape [batch, semantic_size]")
        batch = query_semantic.shape[0]
        if not torch.isfinite(query_semantic).all():
            raise ValueError("query_semantic contains non-finite values")
        if view_reliability.shape != (batch, self.config.num_views):
            raise ValueError("view_reliability shape mismatch")
        if fusion_view_weights.shape != (batch, self.config.num_views):
            raise ValueError("fusion_view_weights shape mismatch")
        if not torch.isfinite(view_reliability).all() or not torch.isfinite(fusion_view_weights).all():
            raise ValueError("view metadata contains non-finite values")
        if (view_reliability < 0).any() or (view_reliability > 1).any():
            raise ValueError("view_reliability must be in [0, 1]")
        if (fusion_view_weights < 0).any() or (fusion_view_weights > 1).any():
            raise ValueError("fusion_view_weights must be in [0, 1]")
        if not torch.allclose(
            fusion_view_weights.sum(dim=-1),
            torch.ones(batch, device=fusion_view_weights.device, dtype=fusion_view_weights.dtype),
            atol=1e-4,
            rtol=1e-4,
        ):
            raise ValueError("fusion_view_weights must sum to one")

        availability: list[torch.Tensor] = []
        for view_id, (contextualized, source, mask) in enumerate(
            zip(contextualized_view_tokens, source_view_tokens, view_valid_masks)
        ):
            if contextualized.ndim != 3 or source.ndim != 3:
                raise ValueError(f"view {view_id} token tensors must be rank 3")
            if contextualized.shape != source.shape:
                raise ValueError(f"view {view_id} source/contextualized token shape mismatch")
            if contextualized.shape[0] != batch or contextualized.shape[2] != self.config.semantic_size:
                raise ValueError(f"view {view_id} token shape mismatch")
            if mask.shape != contextualized.shape[:2] or mask.dtype != torch.bool:
                raise ValueError(f"view {view_id} mask must be bool with shape [batch, tokens]")
            if contextualized.shape[1] < 1:
                raise ValueError("each view must reserve at least one token position")
            if not torch.isfinite(contextualized).all() or not torch.isfinite(source).all():
                raise ValueError(f"view {view_id} contains non-finite tokens")
            availability.append(mask.any(dim=1))

        available = torch.stack(availability, dim=1)
        if not torch.all(available.any(dim=1)):
            raise ValueError("every example must provide at least one latent-pool view")
        unavailable_mass = fusion_view_weights.masked_select(~available)
        if unavailable_mass.numel() and unavailable_mass.abs().max() > 1e-6:
            raise ValueError("fusion_view_weights assign mass to unavailable views")
        return batch, available

    def _build_bank(
        self,
        *,
        contextualized_view_tokens: list[torch.Tensor],
        source_view_tokens: list[torch.Tensor],
        view_valid_masks: list[torch.Tensor],
        query: torch.Tensor,
        view_reliability: torch.Tensor,
        fusion_view_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bank_parts: list[torch.Tensor] = []
        mask_parts: list[torch.Tensor] = []
        view_ids: list[torch.Tensor] = []
        channel_ids: list[torch.Tensor] = []
        batch = query.shape[0]

        for view_id in range(self.config.num_views):
            mask = view_valid_masks[view_id]
            token_count = mask.shape[1]
            view = self.view_embedding.weight[view_id].view(1, 1, -1)
            reliability = self.reliability_projection(view_reliability[:, view_id].view(batch, 1, 1))
            route = self.route_projection(fusion_view_weights[:, view_id].view(batch, 1, 1))
            metadata = (
                view
                + self.config.reliability_feature_scale * reliability
                + self.config.route_feature_scale * route
                + query.unsqueeze(1)
            )

            for channel_id, tokens in (
                (self.SOURCE_CHANNEL, source_view_tokens[view_id]),
                (self.CONTEXTUALIZED_CHANNEL, contextualized_view_tokens[view_id]),
            ):
                channel = self.channel_embedding.weight[channel_id].view(1, 1, -1)
                projected = self.input_projection(tokens) + metadata + channel
                projected = torch.where(mask.unsqueeze(-1), projected, torch.zeros_like(projected))
                bank_parts.append(projected)
                mask_parts.append(mask)
                view_ids.append(
                    torch.full(
                        (batch, token_count),
                        view_id,
                        dtype=torch.long,
                        device=tokens.device,
                    )
                )
                channel_ids.append(
                    torch.full(
                        (batch, token_count),
                        channel_id,
                        dtype=torch.long,
                        device=tokens.device,
                    )
                )

        return (
            torch.cat(bank_parts, dim=1),
            torch.cat(mask_parts, dim=1),
            torch.cat(view_ids, dim=1),
            torch.cat(channel_ids, dim=1),
        )

    @staticmethod
    def _attention_mass_by_id(
        attention: torch.Tensor,
        ids: torch.Tensor,
        valid: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        # MultiheadAttention weights: [batch, heads, slots, bank_tokens].
        weights = attention.mean(dim=1)
        weights = torch.where(valid.unsqueeze(1), weights, torch.zeros_like(weights))
        output = []
        for index in range(count):
            selector = (ids == index) & valid
            output.append((weights * selector.unsqueeze(1).to(weights.dtype)).sum(dim=-1))
        return torch.stack(output, dim=-1)

    def forward(
        self,
        *,
        contextualized_view_tokens: list[torch.Tensor],
        source_view_tokens: list[torch.Tensor],
        view_valid_masks: list[torch.Tensor],
        query_semantic: torch.Tensor,
        view_reliability: torch.Tensor,
        fusion_view_weights: torch.Tensor,
    ) -> dict[str, Any]:
        batch, available = self._validate_inputs(
            contextualized_view_tokens=contextualized_view_tokens,
            source_view_tokens=source_view_tokens,
            view_valid_masks=view_valid_masks,
            query_semantic=query_semantic,
            view_reliability=view_reliability,
            fusion_view_weights=fusion_view_weights,
        )
        query = self.query_projection(query_semantic)
        base_slots = self.latent_slots.unsqueeze(0).expand(batch, -1, -1)
        query_expand = query.unsqueeze(1).expand(-1, self.config.num_slots, -1)
        query_gate = self.slot_query_gate(torch.cat([base_slots, query_expand], dim=-1))
        latents = base_slots + query_gate * query_expand

        bank, bank_valid, bank_view_ids, bank_channel_ids = self._build_bank(
            contextualized_view_tokens=contextualized_view_tokens,
            source_view_tokens=source_view_tokens,
            view_valid_masks=view_valid_masks,
            query=query,
            view_reliability=view_reliability,
            fusion_view_weights=fusion_view_weights,
        )

        final_attention: torch.Tensor | None = None
        for index, block in enumerate(self.blocks):
            latents, attention = block(
                latents,
                bank,
                bank_valid,
                return_attention=index == len(self.blocks) - 1,
            )
            if attention is not None:
                final_attention = attention
        if final_attention is None:
            raise RuntimeError("latent pool failed to expose final cross-attention")

        latents = self.output_norm(latents)
        slot_states = self.slot_output_projection(latents)
        route_query = F.normalize(query, dim=-1)
        normalized_latents = F.normalize(latents, dim=-1)
        query_for_slots = route_query.unsqueeze(1).expand(-1, self.config.num_slots, -1)
        score_features = torch.cat(
            [latents, query_for_slots, normalized_latents * query_for_slots],
            dim=-1,
        )
        slot_scores = self.slot_score(score_features).squeeze(-1)
        slot_weights = torch.softmax(slot_scores, dim=-1)
        pooled = torch.einsum("bs,bsd->bd", slot_weights, latents)

        normalized_slot_states = F.normalize(slot_states, dim=-1)
        slot_cosine = torch.einsum("bsd,btd->bst", normalized_slot_states, normalized_slot_states)
        view_attention_mass = self._attention_mass_by_id(
            final_attention,
            bank_view_ids,
            bank_valid,
            self.config.num_views,
        )
        channel_attention_mass = self._attention_mass_by_id(
            final_attention,
            bank_channel_ids,
            bank_valid,
            2,
        )

        return {
            "latent_slots": slot_states,
            "slot_weights": slot_weights,
            "pooled_state": self.pooled_output_projection(pooled),
            "slot_cosine": slot_cosine,
            "view_attention_mass": view_attention_mass,
            "channel_attention_mass": channel_attention_mass,
            "view_available": available,
            "fusion_view_weights_as_features": fusion_view_weights,
            "source_channel_preserved": True,
        }

    def parameter_report(self) -> dict[str, Any]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "semantic_size": self.config.semantic_size,
            "latent_size": self.config.latent_size,
            "instantiated_slots": self.config.num_slots,
            "slot_count_ceiling": None,
            "slot_count_role": "migratable_checkpoint_shape_not_personality_taxonomy_or_capacity_ceiling",
            "layers": self.config.num_layers,
            "heads": self.config.num_heads,
            "feedforward_size": self.config.feedforward_size,
            "instantiated_view_count": self.config.num_views,
            "view_count_ceiling": None,
            "exact_source_channel": True,
            "contextualized_channel": True,
            "routing_weights_are_features_not_probability_targets": True,
            "primary_output_is_multi_slot": True,
            "single_vector_is_convenience_readout_only": True,
            "private_identity_parameters": 0,
            "hard_parameter_ceiling": None,
        }
