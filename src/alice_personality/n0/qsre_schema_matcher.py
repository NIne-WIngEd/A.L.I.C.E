from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _masked_softmax_flat(logits: Tensor, mask: Tensor) -> Tensor:
    if logits.shape != mask.shape:
        raise ValueError("masked-softmax shape drift")
    flat_logits = logits.reshape(*logits.shape[:-2], -1)
    flat_mask = mask.reshape(*mask.shape[:-2], -1)
    masked = flat_logits.masked_fill(~flat_mask, -1.0e4)
    probability = torch.softmax(masked, dim=-1)
    probability = probability * flat_mask.to(probability.dtype)
    probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    return probability.reshape_as(logits)


class QSRESchemaMatcher(nn.Module):
    """Relation-identity-independent query/schema semantic matcher.

    The matcher owns no parameter indexed by relation key, factor label, schema
    cardinality, or reasoning step. It learns only the reusable operation
    "does this query evidence express this supplied schema description?".

    Scoring is candidate-conditioned. Each candidate gets its own query-token
    evidence distribution rather than sharing one global query attention map.
    This is important for ordered programs because the selected candidate's
    evidence map can be softly consumed before the next recurrent step.
    """

    def __init__(
        self,
        *,
        semantic_dim: int,
        model_dim: int,
        num_hidden_states: int,
    ) -> None:
        super().__init__()
        self.semantic_dim = int(semantic_dim)
        self.model_dim = int(model_dim)
        self.num_hidden_states = int(num_hidden_states)
        self.semantic_norm = nn.LayerNorm(self.semantic_dim)
        self.semantic_projection = nn.Linear(
            self.semantic_dim,
            self.model_dim,
            bias=False,
        )
        self.layer_embedding = nn.Embedding(
            self.num_hidden_states,
            self.model_dim,
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(8.0)))
        self.query_evidence_scale = nn.Parameter(torch.tensor(4.0))
        self.schema_evidence_scale = nn.Parameter(torch.tensor(4.0))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.semantic_dim == self.model_dim:
            nn.init.eye_(self.semantic_projection.weight)
        else:
            nn.init.orthogonal_(self.semantic_projection.weight)
        nn.init.zeros_(self.layer_embedding.weight)

    def project(self, states: Tensor) -> Tensor:
        if states.ndim != 4:
            raise ValueError("semantic states must be [N,L,T,D]")
        _, layers, _, width = states.shape
        if layers != self.num_hidden_states:
            raise ValueError("semantic hidden-state depth drift")
        if width != self.semantic_dim:
            raise ValueError("semantic width drift")
        value = self.semantic_projection(self.semantic_norm(states))
        layer_ids = torch.arange(layers, device=states.device)
        return value + self.layer_embedding(layer_ids).view(
            1,
            layers,
            1,
            self.model_dim,
        )

    def match_projected(
        self,
        *,
        query_projected: Tensor,
        query_token_mask: Tensor,
        schema_projected: Tensor,
        schema_token_mask: Tensor,
        query_remaining: Tensor | None = None,
        query_prior: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if query_projected.ndim != 4 or schema_projected.ndim != 4:
            raise ValueError("projected query/schema states must be rank four")
        batch, layers, query_tokens, width = query_projected.shape
        candidates, schema_layers, schema_tokens, schema_width = schema_projected.shape
        if layers != schema_layers or width != schema_width:
            raise ValueError("query/schema projected geometry drift")
        if query_token_mask.shape != (batch, query_tokens):
            raise ValueError("query token-mask shape drift")
        if schema_token_mask.shape != (candidates, schema_tokens):
            raise ValueError("schema token-mask shape drift")
        if query_token_mask.dtype != torch.bool or schema_token_mask.dtype != torch.bool:
            raise ValueError("query/schema token masks must be boolean")
        if bool((query_token_mask.sum(dim=-1) == 0).any()):
            raise ValueError("every query requires at least one content token")
        if bool((schema_token_mask.sum(dim=-1) == 0).any()):
            raise ValueError("every schema candidate requires at least one content token")

        if query_remaining is None:
            query_remaining = torch.ones(
                batch,
                query_tokens,
                dtype=query_projected.dtype,
                device=query_projected.device,
            )
        if query_remaining.shape != (batch, query_tokens):
            raise ValueError("query remaining-evidence shape drift")
        query_remaining = query_remaining.clamp(min=0.05, max=1.0)

        if query_prior is None:
            query_prior = torch.ones(
                batch,
                layers,
                query_tokens,
                dtype=query_projected.dtype,
                device=query_projected.device,
            )
        if query_prior.shape != (batch, layers, query_tokens):
            raise ValueError("query prior shape drift")
        query_prior = query_prior.clamp_min(1.0e-6)

        q = F.normalize(query_projected, dim=-1)
        s = F.normalize(schema_projected, dim=-1)
        similarity = torch.einsum("bltd,clsd->bclts", q, s)

        query_valid = query_token_mask[:, None, None, :, None].expand(
            batch,
            candidates,
            layers,
            query_tokens,
            schema_tokens,
        )
        schema_valid = schema_token_mask[None, :, None, None, :].expand(
            batch,
            candidates,
            layers,
            query_tokens,
            schema_tokens,
        )
        pair_valid = query_valid & schema_valid
        similarity = similarity.masked_fill(~pair_valid, -1.0e4)

        # Candidate-specific query evidence. A relation description determines
        # which query tokens matter; one global attention map is not reused for
        # every candidate.
        query_to_schema = similarity.max(dim=-1).values
        query_position_valid = query_token_mask[:, None, None, :].expand(
            batch,
            candidates,
            layers,
            query_tokens,
        )
        remaining = query_remaining[:, None, None, :]
        prior = query_prior[:, None, :, :]
        query_weight_logits = (
            self.query_evidence_scale.clamp(min=0.5, max=12.0)
            * query_to_schema
            + torch.log(remaining)
            + torch.log(prior)
        )
        query_weight = _masked_softmax_flat(
            query_weight_logits,
            query_position_valid,
        )
        query_score = (query_weight * query_to_schema).sum(dim=(2, 3))

        # Candidate-specific schema evidence. This avoids uniformly averaging
        # generic function words in long relation descriptions.
        schema_to_query = similarity.max(dim=-2).values
        schema_position_valid = schema_token_mask[None, :, None, :].expand(
            batch,
            candidates,
            layers,
            schema_tokens,
        )
        schema_weight_logits = (
            self.schema_evidence_scale.clamp(min=0.5, max=12.0)
            * schema_to_query
        )
        schema_weight = _masked_softmax_flat(
            schema_weight_logits,
            schema_position_valid,
        )
        schema_score = (schema_weight * schema_to_query).sum(dim=(2, 3))

        semantic_score = 0.6 * query_score + 0.4 * schema_score
        scale = self.logit_scale.exp().clamp(min=1.0, max=100.0)

        schema_mask = schema_token_mask[:, None, :, None].to(
            schema_projected.dtype
        )
        schema_per_layer = (
            schema_projected * schema_mask
        ).sum(dim=2) / schema_mask.sum(dim=2).clamp_min(1.0)
        schema_summary = schema_per_layer.mean(dim=1)

        return {
            "logits": scale * semantic_score,
            "semantic_score": semantic_score,
            "query_evidence": query_weight,
            "schema_evidence": schema_weight,
            "schema_summary": schema_summary,
        }

    def forward(
        self,
        *,
        query_hidden_states: Tensor,
        query_token_mask: Tensor,
        schema_hidden_states: Tensor,
        schema_token_mask: Tensor,
        query_remaining: Tensor | None = None,
        query_prior: Tensor | None = None,
    ) -> dict[str, Tensor]:
        return self.match_projected(
            query_projected=self.project(query_hidden_states),
            query_token_mask=query_token_mask,
            schema_projected=self.project(schema_hidden_states),
            schema_token_mask=schema_token_mask,
            query_remaining=query_remaining,
            query_prior=query_prior,
        )

    def parameter_report(self) -> dict[str, int | bool | None]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "relation_identity_parameters": 0,
            "candidate_count_dependent_parameters": 0,
            "reasoning_step_dependent_parameters": 0,
            "shared_query_schema_projection": True,
            "candidate_conditioned_query_evidence": True,
            "symmetric_late_interaction": True,
            "runtime_schema_cardinality_ceiling": None,
        }
