from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _masked_softmax(logits: Tensor, mask: Tensor, *, dim: int) -> Tensor:
    if logits.shape != mask.shape:
        raise ValueError("masked-softmax shape drift")
    masked = logits.masked_fill(~mask, -1.0e4)
    probability = torch.softmax(masked, dim=dim)
    probability = probability * mask.to(probability.dtype)
    return probability / probability.sum(dim=dim, keepdim=True).clamp_min(1.0e-12)


class QSRESchemaMatcher(nn.Module):
    """Parameter-free token evidence matcher for frozen semantic authority.

    Relation/factor identity is not learned here. The frozen N0 semantic model
    supplies the semantic authority scores. This module only extracts local
    token evidence from the already materialized hidden states so the recurrent
    operator can consume ordered evidence without learning another relation
    classifier on top of a small schema set.

    The layer mixture is deterministic, candidate conditioning is runtime-only,
    and there are no trainable parameters or relation/factor indexed tensors.
    """

    def __init__(
        self,
        *,
        semantic_dim: int,
        model_dim: int,
        num_hidden_states: int,
    ) -> None:
        super().__init__()
        if int(semantic_dim) != int(model_dim):
            raise ValueError(
                "parameter-free schema evidence matcher requires semantic_dim == model_dim"
            )
        self.semantic_dim = int(semantic_dim)
        self.model_dim = int(model_dim)
        self.num_hidden_states = int(num_hidden_states)

    def _layer_weights(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        # Deterministic mild preference for later contextual layers. No learned
        # layer identity can become a hidden relation ontology.
        position = torch.linspace(
            -1.0,
            1.0,
            self.num_hidden_states,
            dtype=dtype,
            device=device,
        )
        return torch.softmax(position, dim=0)

    def project(self, states: Tensor) -> Tensor:
        """Compatibility surface: validate and preserve frozen hidden states."""
        if states.ndim != 4:
            raise ValueError("semantic states must be [N,L,T,D]")
        if states.size(1) != self.num_hidden_states:
            raise ValueError("semantic hidden-state depth drift")
        if states.size(-1) != self.semantic_dim:
            raise ValueError("semantic width drift")
        return states

    def _mix_layers(self, states: Tensor) -> tuple[Tensor, Tensor]:
        weights = self._layer_weights(
            dtype=states.dtype,
            device=states.device,
        )
        mixed = torch.einsum("l,nltd->ntd", weights, states)
        return mixed, weights

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
        if layers != self.num_hidden_states or width != self.semantic_dim:
            raise ValueError("semantic matcher geometry drift")
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

        query_mixed, layer_weights = self._mix_layers(query_projected)
        schema_mixed, _ = self._mix_layers(schema_projected)
        q = F.normalize(query_mixed, dim=-1)
        s = F.normalize(schema_mixed, dim=-1)
        similarity = torch.einsum("btd,csd->bcts", q, s)

        query_valid = query_token_mask[:, None, :, None].expand(
            batch,
            candidates,
            query_tokens,
            schema_tokens,
        )
        schema_valid = schema_token_mask[None, :, None, :].expand(
            batch,
            candidates,
            query_tokens,
            schema_tokens,
        )
        pair_valid = query_valid & schema_valid
        similarity = similarity.masked_fill(~pair_valid, -1.0e4)

        query_compatibility = similarity.max(dim=-1).values
        schema_compatibility = similarity.max(dim=-2).values

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
            prior_token = torch.ones(
                batch,
                query_tokens,
                dtype=query_projected.dtype,
                device=query_projected.device,
            )
        else:
            if query_prior.shape != (batch, layers, query_tokens):
                raise ValueError("query prior shape drift")
            prior_token = torch.einsum(
                "l,blt->bt",
                layer_weights,
                query_prior.to(query_projected.dtype),
            ).clamp_min(1.0e-6)

        query_position_valid = query_token_mask[:, None, :].expand(
            batch,
            candidates,
            query_tokens,
        )
        base_query_logits = (
            4.0 * query_compatibility
            + torch.log(prior_token[:, None, :])
        )
        base_query_weight = _masked_softmax(
            base_query_logits,
            query_position_valid,
            dim=-1,
        )
        remaining_support = (
            base_query_weight * query_remaining[:, None, :]
        ).sum(dim=-1).clamp(min=0.05, max=1.0)

        query_weight = _masked_softmax(
            base_query_logits + torch.log(query_remaining[:, None, :]),
            query_position_valid,
            dim=-1,
        )
        query_score = (query_weight * query_compatibility).sum(dim=-1)

        schema_position_valid = schema_token_mask[None, :, :].expand(
            batch,
            candidates,
            schema_tokens,
        )
        schema_weight = _masked_softmax(
            4.0 * schema_compatibility,
            schema_position_valid,
            dim=-1,
        )
        schema_score = (
            schema_weight * schema_compatibility
        ).sum(dim=-1)

        token_score = 0.5 * query_score + 0.5 * schema_score

        schema_mask_float = schema_token_mask.to(
            schema_mixed.dtype
        ).unsqueeze(-1)
        schema_summary = (
            schema_mixed * schema_mask_float
        ).sum(dim=1) / schema_mask_float.sum(dim=1).clamp_min(1.0)

        # Preserve [B,C,L,T] for ordered-coverage code. The evidence mass is
        # token-local while layer mass is deterministic and shared.
        query_evidence = (
            query_weight[:, :, None, :]
            * layer_weights[None, None, :, None]
        )
        schema_evidence = (
            schema_weight[:, :, None, :]
            * layer_weights[None, None, :, None]
        )

        return {
            "logits": token_score,
            "semantic_score": token_score,
            "token_score": token_score,
            "query_evidence": query_evidence,
            "schema_evidence": schema_evidence,
            "schema_summary": schema_summary,
            "remaining_support": remaining_support,
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
            "total_parameters": 0,
            "trainable_parameters": 0,
            "relation_identity_parameters": 0,
            "factor_identity_parameters": 0,
            "candidate_count_dependent_parameters": 0,
            "reasoning_step_dependent_parameters": 0,
            "learned_semantic_metric": False,
            "frozen_semantic_authority_required": True,
            "deterministic_layer_mixture": True,
            "candidate_conditioned_query_evidence": True,
            "ordered_evidence_remaining_support": True,
            "runtime_schema_cardinality_ceiling": None,
        }
