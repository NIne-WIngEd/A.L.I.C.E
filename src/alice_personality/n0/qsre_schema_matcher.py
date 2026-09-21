from __future__ import annotations

import math

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
    """Identity-neutral candidate-conditioned semantic schema matcher.

    Runtime schema candidates are data, not learned class IDs. The matcher keeps
    a frozen semantic anchor and learns only a shared residual metric plus a
    lightweight bidirectional query/schema interaction. No parameter is indexed
    by relation key, factor label, candidate cardinality, or reasoning step.

    The interaction is deliberately pair-conditioned: every candidate refines
    which query tokens are evidence for that candidate, and every query refines
    which schema tokens matter. The same weights are reused for relations,
    operator factors, unseen runtime schemas, and every recurrent step.
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
        hidden = max(32, min(160, self.model_dim))

        self.semantic_norm = nn.LayerNorm(self.semantic_dim)

        # Preserve a stable semantic path. The learned projection is residual,
        # so P2S cannot rotate the whole frozen N0 semantic space into another
        # closed relation ontology.
        self.anchor_projection = nn.Linear(
            self.semantic_dim,
            self.model_dim,
            bias=False,
        )
        for parameter in self.anchor_projection.parameters():
            parameter.requires_grad = False
        self.semantic_residual_projection = nn.Linear(
            self.semantic_dim,
            self.model_dim,
            bias=False,
        )

        # One relation-independent layer mixture replaces a separately learned
        # representation for every relation or hop.
        self.layer_logits = nn.Parameter(torch.zeros(self.num_hidden_states))

        # Efficient bidirectional cross refinement. Instead of materializing a
        # huge [query;schema] concatenation, four shared projections compose
        # ordered token-pair features.
        self.token_left = nn.Linear(self.model_dim, hidden, bias=False)
        self.token_right = nn.Linear(self.model_dim, hidden, bias=False)
        self.token_product = nn.Linear(self.model_dim, hidden, bias=False)
        self.token_difference = nn.Linear(self.model_dim, hidden, bias=False)
        self.token_score = nn.Linear(hidden, 1)

        self.pair_left = nn.Linear(self.model_dim, hidden, bias=False)
        self.pair_right = nn.Linear(self.model_dim, hidden, bias=False)
        self.pair_product = nn.Linear(self.model_dim, hidden, bias=False)
        self.pair_difference = nn.Linear(self.model_dim, hidden, bias=False)
        self.pair_score = nn.Linear(hidden, 1)

        self.logit_scale = nn.Parameter(torch.tensor(math.log(8.0)))
        self.query_evidence_scale = nn.Parameter(torch.tensor(4.0))
        self.schema_evidence_scale = nn.Parameter(torch.tensor(4.0))
        self.cross_alignment_scale = nn.Parameter(torch.tensor(4.0))
        self.token_residual_scale = nn.Parameter(torch.tensor(0.35))
        self.pair_residual_scale = nn.Parameter(torch.tensor(0.35))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.semantic_dim == self.model_dim:
                self.anchor_projection.weight.copy_(
                    torch.eye(self.semantic_dim)
                )
            else:
                nn.init.orthogonal_(self.anchor_projection.weight)
            nn.init.zeros_(self.semantic_residual_projection.weight)
            self.layer_logits.zero_()
            # Prefer later contextual layers without hard-coding one layer.
            if self.num_hidden_states > 1:
                self.layer_logits.copy_(
                    torch.linspace(
                        -0.5,
                        0.5,
                        self.num_hidden_states,
                    )
                )
        for module in (
            self.token_left,
            self.token_right,
            self.token_product,
            self.token_difference,
            self.pair_left,
            self.pair_right,
            self.pair_product,
            self.pair_difference,
        ):
            nn.init.xavier_uniform_(module.weight)
        nn.init.xavier_uniform_(self.token_score.weight, gain=0.25)
        nn.init.zeros_(self.token_score.bias)
        nn.init.xavier_uniform_(self.pair_score.weight, gain=0.25)
        nn.init.zeros_(self.pair_score.bias)

    def project(self, states: Tensor) -> Tensor:
        if states.ndim != 4:
            raise ValueError("semantic states must be [N,L,T,D]")
        _, layers, _, width = states.shape
        if layers != self.num_hidden_states:
            raise ValueError("semantic hidden-state depth drift")
        if width != self.semantic_dim:
            raise ValueError("semantic width drift")
        normalized = self.semantic_norm(states)
        anchor = self.anchor_projection(normalized)
        residual = self.semantic_residual_projection(normalized)
        return anchor + 0.25 * residual

    def _layer_weights(self, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        return torch.softmax(
            self.layer_logits.to(device=device, dtype=dtype),
            dim=0,
        )

    def _mix_layers(self, projected: Tensor) -> Tensor:
        weights = self._layer_weights(
            dtype=projected.dtype,
            device=projected.device,
        )
        return torch.einsum("l,nltd->ntd", weights, projected)

    def _token_pair_score(self, left: Tensor, right: Tensor) -> Tensor:
        value = (
            self.token_left(left)
            + self.token_right(right)
            + self.token_product(left * right)
            + self.token_difference((left - right).abs())
        )
        return self.token_score(F.gelu(value)).squeeze(-1)

    def _pair_summary_score(self, left: Tensor, right: Tensor) -> Tensor:
        value = (
            self.pair_left(left)
            + self.pair_right(right)
            + self.pair_product(left * right)
            + self.pair_difference((left - right).abs())
        )
        return self.pair_score(F.gelu(value)).squeeze(-1)

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

        layer_weights = self._layer_weights(
            dtype=query_projected.dtype,
            device=query_projected.device,
        )
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

        query_mixed = self._mix_layers(query_projected)
        schema_mixed = self._mix_layers(schema_projected)
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

        alignment_scale = self.cross_alignment_scale.exp().clamp(
            min=1.0,
            max=30.0,
        )

        # Query -> schema soft alignment.
        q_to_s_logits = alignment_scale * similarity
        q_to_s_weight = _masked_softmax(
            q_to_s_logits,
            schema_valid,
            dim=-1,
        )
        attended_schema = torch.einsum(
            "bcts,csd->bctd",
            q_to_s_weight,
            schema_mixed,
        )
        query_expanded = query_mixed[:, None, :, :].expand(
            batch,
            candidates,
            query_tokens,
            width,
        )
        query_anchor = similarity.max(dim=-1).values
        query_residual = self._token_pair_score(
            query_expanded,
            attended_schema,
        )
        token_residual_scale = torch.tanh(self.token_residual_scale)
        query_compatibility = (
            query_anchor
            + token_residual_scale * torch.tanh(query_residual)
        )

        query_position_valid = query_token_mask[:, None, :].expand(
            batch,
            candidates,
            query_tokens,
        )
        query_weight_logits = (
            self.query_evidence_scale.clamp(min=0.5, max=12.0)
            * query_compatibility
            + torch.log(query_remaining[:, None, :])
            + torch.log(prior_token[:, None, :])
        )
        query_weight = _masked_softmax(
            query_weight_logits,
            query_position_valid,
            dim=-1,
        )
        query_score = (query_weight * query_compatibility).sum(dim=-1)

        # Schema -> query soft alignment using the same pairwise semantic field.
        s_to_q_logits = alignment_scale * similarity.transpose(-1, -2)
        s_to_q_valid = query_valid.transpose(-1, -2)
        s_to_q_weight = _masked_softmax(
            s_to_q_logits,
            s_to_q_valid,
            dim=-1,
        )
        attended_query = torch.einsum(
            "bcst,btd->bcsd",
            s_to_q_weight,
            query_mixed,
        )
        schema_expanded = schema_mixed[None, :, :, :].expand(
            batch,
            candidates,
            schema_tokens,
            width,
        )
        schema_anchor = similarity.max(dim=-2).values
        schema_residual = self._token_pair_score(
            schema_expanded,
            attended_query,
        )
        schema_compatibility = (
            schema_anchor
            + token_residual_scale * torch.tanh(schema_residual)
        )

        schema_position_valid = schema_token_mask[None, :, :].expand(
            batch,
            candidates,
            schema_tokens,
        )
        schema_weight_logits = (
            self.schema_evidence_scale.clamp(min=0.5, max=12.0)
            * schema_compatibility
        )
        schema_weight = _masked_softmax(
            schema_weight_logits,
            schema_position_valid,
            dim=-1,
        )
        schema_score = (
            schema_weight * schema_compatibility
        ).sum(dim=-1)

        query_summary = torch.einsum(
            "bct,btd->bcd",
            query_weight,
            query_mixed,
        )
        schema_summary_for_pair = torch.einsum(
            "bcs,csd->bcd",
            schema_weight,
            schema_mixed,
        )
        pair_interaction = self._pair_summary_score(
            query_summary,
            schema_summary_for_pair,
        )

        anchor_score = 0.5 * query_score + 0.5 * schema_score
        semantic_score = (
            anchor_score
            + torch.tanh(self.pair_residual_scale)
            * torch.tanh(pair_interaction)
        )
        scale = self.logit_scale.exp().clamp(min=1.0, max=100.0)

        # The recurrent operator requires a candidate-only summary. Do not let
        # one query rewrite runtime schema identity.
        schema_mask_float = schema_token_mask.to(
            schema_mixed.dtype
        ).unsqueeze(-1)
        schema_summary = (
            schema_mixed * schema_mask_float
        ).sum(dim=1) / schema_mask_float.sum(dim=1).clamp_min(1.0)

        # Preserve the historical [B,C,L,T] evidence interface. Layer mass is
        # globally shared and therefore cannot encode a relation identity.
        query_evidence = (
            query_weight[:, :, None, :]
            * layer_weights[None, None, :, None]
        )
        schema_evidence = (
            schema_weight[:, :, None, :]
            * layer_weights[None, None, :, None]
        )

        return {
            "logits": scale * semantic_score,
            "semantic_score": semantic_score,
            "query_evidence": query_evidence,
            "schema_evidence": schema_evidence,
            "schema_summary": schema_summary,
            "pair_interaction": pair_interaction,
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
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "relation_identity_parameters": 0,
            "factor_identity_parameters": 0,
            "candidate_count_dependent_parameters": 0,
            "reasoning_step_dependent_parameters": 0,
            "fixed_semantic_anchor": True,
            "learned_projection_is_residual": True,
            "shared_layer_mixture": True,
            "bidirectional_pair_refinement": True,
            "candidate_conditioned_query_evidence": True,
            "symmetric_late_interaction": True,
            "runtime_schema_cardinality_ceiling": None,
        }
