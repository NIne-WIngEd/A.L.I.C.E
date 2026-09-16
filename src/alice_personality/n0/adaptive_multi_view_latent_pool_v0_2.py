from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class AdaptiveMultiViewLatentPoolV02Config:
    """Failure-driven successor to the N0 adaptive latent pool.

    Slot count and view count are checkpoint tensor shapes only. They are not
    personality ontologies or permanent capability ceilings.

    v0.2 changes the slot/input interaction from independent cross-attention to
    competitive cross-attention: each input token first allocates mass across
    slots, then each slot normalizes its assigned evidence for aggregation.
    This makes duplicate slots compete for evidence instead of allowing every
    slot to consume the same bank independently.
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
    competition_temperature: float = 1.0
    initial_self_exchange_gate: float = -2.0

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
        if self.competition_temperature <= 0.0:
            raise ValueError("competition_temperature must be positive")


class _GEGLU(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.in_proj = nn.Linear(width, 2 * hidden)
        self.out_proj = nn.Linear(hidden, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(self.dropout(value * F.gelu(gate)))


class _CompetitiveCrossAttention(nn.Module):
    """Multi-head Slot-Attention-style competitive evidence allocation.

    Standard cross-attention normalizes over input tokens independently for
    each query slot. That lets every slot attend to the same evidence. Here the
    raw attention logits are first normalized across slots for each input token,
    creating competition, and then re-normalized over tokens for each slot for
    stable aggregation.
    """

    def __init__(
        self,
        width: int,
        heads: int,
        *,
        dropout: float,
        temperature: float,
    ) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError("competitive attention width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.temperature = temperature
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def _split(self, value: torch.Tensor) -> torch.Tensor:
        batch, length, _ = value.shape
        return value.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        slots: torch.Tensor,
        bank: torch.Tensor,
        bank_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self._split(self.q_proj(slots))
        key = self._split(self.k_proj(bank))
        value = self._split(self.v_proj(bank))
        logits = torch.einsum("bhsd,bhnd->bhsn", query, key)
        logits = logits / math.sqrt(self.head_dim)
        logits = logits / self.temperature
        valid = bank_valid[:, None, None, :]
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)

        # Competition: for each input token/head, distribute ownership across slots.
        assignment = torch.softmax(logits, dim=2)
        assignment = torch.where(valid, assignment, torch.zeros_like(assignment))

        # Stable per-slot evidence aggregation after competitive ownership.
        attention = assignment / assignment.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        attention = self.dropout(attention)
        update = torch.einsum("bhsn,bhnd->bhsd", attention, value)
        update = update.transpose(1, 2).contiguous().view(slots.shape)
        return self.out_proj(update), attention, assignment


class _CompetitiveLatentPoolBlock(nn.Module):
    def __init__(self, config: AdaptiveMultiViewLatentPoolV02Config) -> None:
        super().__init__()
        d = config.latent_size
        self.cross_norm = nn.LayerNorm(d)
        self.bank_norm = nn.LayerNorm(d)
        self.cross_attention = _CompetitiveCrossAttention(
            d,
            config.num_heads,
            dropout=config.dropout,
            temperature=config.competition_temperature,
        )
        self.self_norm = nn.LayerNorm(d)
        self.self_attention = nn.MultiheadAttention(
            d,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        # Slot-to-slot exchange remains available, but starts gated down so the
        # new competitive evidence assignment can establish distinct slot state
        # before unrestricted slot mixing is useful. This gate is learned.
        self.self_exchange_gate = nn.Parameter(
            torch.tensor(float(config.initial_self_exchange_gate))
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
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        cross, attention, assignment = self.cross_attention(
            self.cross_norm(latents),
            self.bank_norm(bank),
            bank_valid,
        )
        latents = latents + cross

        normed = self.self_norm(latents)
        self_update, _ = self.self_attention(
            normed,
            normed,
            normed,
            need_weights=False,
        )
        latents = latents + torch.sigmoid(self.self_exchange_gate) * self_update
        latents = latents + self.ffn(self.ffn_norm(latents))
        if return_attention:
            return latents, attention, assignment
        return latents, None, None


class AdaptiveMultiViewLatentPoolV02(nn.Module):
    """Competitive multi-slot readout over the ratified fusion representation."""

    SOURCE_CHANNEL = 0
    CONTEXTUALIZED_CHANNEL = 1

    def __init__(self, config: AdaptiveMultiViewLatentPoolV02Config | None = None) -> None:
        super().__init__()
        self.config = config or AdaptiveMultiViewLatentPoolV02Config()
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
        self.blocks = nn.ModuleList(
            _CompetitiveLatentPoolBlock(self.config)
            for _ in range(self.config.num_layers)
        )
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
        # attention: [batch, heads, slots, bank_tokens], normalized over tokens.
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
        final_assignment: torch.Tensor | None = None
        for index, block in enumerate(self.blocks):
            latents, attention, assignment = block(
                latents,
                bank,
                bank_valid,
                return_attention=index == len(self.blocks) - 1,
            )
            if attention is not None:
                final_attention = attention
                final_assignment = assignment
        if final_attention is None or final_assignment is None:
            raise RuntimeError("latent pool failed to expose competitive attention")

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
            "slot_token_attention": final_attention.mean(dim=1),
            "token_slot_assignment": final_assignment.mean(dim=1),
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
            "competitive_cross_attention": True,
            "competitive_normalization_axis": "slots_then_tokens",
            "learned_gated_slot_self_exchange": True,
            "exact_source_channel": True,
            "contextualized_channel": True,
            "routing_weights_are_features_not_probability_targets": True,
            "primary_output_is_multi_slot": True,
            "single_vector_is_convenience_readout_only": True,
            "private_identity_parameters": 0,
            "hard_parameter_ceiling": None,
        }
