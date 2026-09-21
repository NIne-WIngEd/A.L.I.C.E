from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class PublicJudgmentProbeConfig:
    semantic_dim: int = 640
    latent_dim: int = 640
    model_dim: int = 640
    num_hidden_states: int = 17

    def validate(self) -> None:
        for name, value in (
            ("semantic_dim", self.semantic_dim),
            ("latent_dim", self.latent_dim),
            ("model_dim", self.model_dim),
            ("num_hidden_states", self.num_hidden_states),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")


class PublicJudgmentProbeV1(nn.Module):
    """Behavioral readout for public N0 judgment supervision.

    Candidate meaning comes from runtime candidate text hidden states. The head
    has no candidate-ID or candidate-count parameter axis. It gives the final
    fusion/latent fabric an independently labeled behavioral objective instead
    of training only to reconstruct an upstream embedding geometry.
    """

    def __init__(self, config: PublicJudgmentProbeConfig | None = None) -> None:
        super().__init__()
        self.config = config or PublicJudgmentProbeConfig()
        self.config.validate()
        d = self.config.model_dim

        self.latent_projection = nn.Linear(self.config.latent_dim, d)
        self.candidate_projection = nn.Linear(
            self.config.semantic_dim,
            d,
            bias=False,
        )
        self.layer_gate = nn.Sequential(
            nn.Linear(3, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        self.score = nn.Sequential(
            nn.Linear(4 * d, d),
            nn.SiLU(),
            nn.Linear(d, 1),
        )

    def forward(
        self,
        *,
        pooled_state: Tensor,
        candidate_hidden_states: Tensor,
        candidate_token_mask: Tensor,
    ) -> dict[str, Tensor]:
        if pooled_state.ndim != 2 or pooled_state.size(-1) != self.config.latent_dim:
            raise ValueError("pooled_state must be [B,D_latent]")
        if candidate_hidden_states.ndim != 5:
            raise ValueError(
                "candidate_hidden_states must be [B,C,L,T,D_semantic]"
            )
        batch, candidates, layers, tokens, width = candidate_hidden_states.shape
        if pooled_state.size(0) != batch:
            raise ValueError("candidate/latent batch drift")
        if layers != self.config.num_hidden_states:
            raise ValueError("candidate hidden-state depth drift")
        if width != self.config.semantic_dim:
            raise ValueError("candidate semantic width drift")
        if candidate_token_mask.shape != (batch, candidates, tokens):
            raise ValueError("candidate_token_mask shape drift")
        if candidate_token_mask.dtype != torch.bool:
            raise ValueError("candidate_token_mask must be bool")
        if bool((candidate_token_mask.sum(dim=-1) == 0).any()):
            raise ValueError("every candidate requires at least one content token")

        latent = self.latent_projection(pooled_state.float())
        candidate = self.candidate_projection(candidate_hidden_states.float())
        normalized_latent = F.normalize(latent, dim=-1)
        normalized_candidate = F.normalize(candidate, dim=-1)

        token_score = torch.einsum(
            "bd,bcltd->bclt",
            normalized_latent,
            normalized_candidate,
        )
        token_mask = candidate_token_mask[:, :, None, :].expand(
            batch,
            candidates,
            layers,
            tokens,
        )
        token_score = token_score.masked_fill(~token_mask, -1.0e4)
        token_weight = torch.softmax(4.0 * token_score, dim=-1)
        token_weight = token_weight * token_mask.to(token_weight.dtype)
        token_weight = token_weight / token_weight.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0e-12)
        layer_candidate = torch.einsum(
            "bclt,bcltd->bcld",
            token_weight,
            candidate,
        )
        layer_similarity = F.cosine_similarity(
            layer_candidate,
            latent[:, None, None, :],
            dim=-1,
        )
        layer_position = torch.linspace(
            -1.0,
            1.0,
            layers,
            device=layer_similarity.device,
            dtype=layer_similarity.dtype,
        ).view(1, 1, layers).expand_as(layer_similarity)
        layer_features = torch.stack(
            [
                layer_similarity,
                layer_similarity.square(),
                layer_position,
            ],
            dim=-1,
        )
        layer_logit = self.layer_gate(layer_features).squeeze(-1)
        layer_weight = torch.softmax(layer_logit, dim=-1)
        summary = torch.einsum(
            "bcl,bcld->bcd",
            layer_weight,
            layer_candidate,
        )

        latent_expanded = latent[:, None, :].expand(batch, candidates, -1)
        feature = torch.cat(
            [
                latent_expanded,
                summary,
                latent_expanded * summary,
                (latent_expanded - summary).abs(),
            ],
            dim=-1,
        )
        logit = self.score(feature).squeeze(-1)
        return {
            "candidate_logits": logit,
            "candidate_summary": summary,
            "candidate_layer_weight": layer_weight,
            "candidate_token_weight": token_weight,
        }

    def parameter_report(self) -> dict[str, Any]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "candidate_identity_parameters": 0,
            "candidate_count_dependent_parameters": 0,
            "candidate_count_ceiling": None,
            "multi_layer_candidate_read": True,
            "behavioral_supervision_required": True,
        }
