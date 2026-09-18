from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .evidence_graph import EvidenceRelationType


RELATION_ID_TO_NAME = {
    int(EvidenceRelationType.SUPPORTS): "supports",
    int(EvidenceRelationType.CORRECTS): "corrects",
    int(EvidenceRelationType.SUPERSEDES): "supersedes",
    int(EvidenceRelationType.DERIVED_FROM): "derived_from",
    int(EvidenceRelationType.CAUSES): "causes",
    int(EvidenceRelationType.TEMPORAL_SUCCESSOR): "temporal_successor",
}


class RelationConditionedMultiLayerQueryInterface(nn.Module):
    """Token-level language↔relation interface over frozen semantic layers.

    This module sits *before* evidence-graph field selection. It does not replace
    the frozen semantic backbone and it does not hardcode one universal layer.

    For each directed evidence edge:
      1. relation type selects a small audit-derived candidate layer set;
      2. relation-conditioned attention extracts relevant query-token evidence
         separately from every candidate layer;
      3. relation-conditioned layer mixing combines those summaries;
      4. the resulting edge-specific semantic state is projected to graph space.

    The audit map is treated as an initialization prior, not permanent truth.
    Candidate-layer masks are an operating policy for the first causal study,
    not a product capability ceiling.
    """

    def __init__(
        self,
        *,
        semantic_size: int,
        graph_size: int,
        num_relation_types: int,
        num_hidden_states: int,
        layer_map: dict[str, Any],
        interface_size: int | None = None,
        dropout: float = 0.0,
        audit_prior_scale: float = 2.0,
    ) -> None:
        super().__init__()
        if semantic_size < 1 or graph_size < 1 or num_relation_types < 1:
            raise ValueError("semantic/graph/relation sizes must be positive")
        if num_hidden_states < 2:
            raise ValueError("multi-layer interface requires hidden-state depth")
        if not isinstance(layer_map, dict) or not layer_map:
            raise ValueError("relation-conditioned layer map is required")

        width = int(interface_size or graph_size)
        self.semantic_size = int(semantic_size)
        self.graph_size = int(graph_size)
        self.num_relation_types = int(num_relation_types)
        self.num_hidden_states = int(num_hidden_states)
        self.interface_size = width
        self.audit_prior_scale = float(audit_prior_scale)

        self.token_norm = nn.LayerNorm(semantic_size)
        self.token_key = nn.Linear(semantic_size, width, bias=False)
        self.token_value = nn.Linear(semantic_size, width, bias=False)

        self.relation_embedding = nn.Embedding(
            num_relation_types,
            width,
            padding_idx=int(EvidenceRelationType.PAD),
        )
        self.relation_query = nn.Linear(width, width, bias=False)

        self.layer_embedding = nn.Embedding(num_hidden_states, width)
        self.layer_score = nn.Sequential(
            nn.Linear(4 * width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, 1),
        )
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, graph_size)

        candidate_mask = torch.zeros(
            num_relation_types,
            num_hidden_states,
            dtype=torch.bool,
        )
        audit_prior = torch.full(
            (num_relation_types, num_hidden_states),
            -1.0e4,
            dtype=torch.float32,
        )

        compiled: dict[int, dict[str, Any]] = {}
        for relation_id, relation_name in RELATION_ID_TO_NAME.items():
            if relation_id >= num_relation_types:
                continue
            info = layer_map.get(relation_name)
            if info is None:
                continue
            candidates = sorted({int(x) for x in info.get("candidate_layers", [])})
            ranked = {
                int(item["layer_index"]): float(item["token_accuracy"])
                for item in info.get("ranked_layers", [])
            }
            if not candidates:
                continue
            for layer in candidates:
                if not 0 <= layer < num_hidden_states:
                    raise ValueError(
                        f"candidate layer {layer} for {relation_name} exceeds "
                        f"hidden-state depth {num_hidden_states}"
                    )
                candidate_mask[relation_id, layer] = True

            values = torch.tensor(
                [ranked.get(layer, 0.0) for layer in candidates],
                dtype=torch.float32,
            )
            centered = values - values.mean()
            for layer, value in zip(candidates, centered.tolist()):
                audit_prior[relation_id, layer] = (
                    self.audit_prior_scale * float(value)
                )
            compiled[relation_id] = {
                "relation_name": relation_name,
                "candidate_layers": candidates,
                "audit_accuracy": {
                    str(layer): ranked.get(layer)
                    for layer in candidates
                },
            }

        # CONFLICTS_WITH is symmetric for this endpoint-role capability and is
        # intentionally left without a directional language interface.
        candidate_mask[int(EvidenceRelationType.CONFLICTS_WITH)] = False
        audit_prior[int(EvidenceRelationType.CONFLICTS_WITH)] = -1.0e4

        self.register_buffer(
            "relation_layer_candidate_mask",
            candidate_mask,
            persistent=True,
        )
        self.register_buffer(
            "relation_layer_audit_prior",
            audit_prior,
            persistent=True,
        )
        self.compiled_relation_policy = compiled

    @classmethod
    def from_layer_map_file(
        cls,
        *,
        layer_map_path: str | Path,
        semantic_size: int,
        graph_size: int,
        num_relation_types: int,
        num_hidden_states: int,
        interface_size: int | None = None,
        dropout: float = 0.0,
        audit_prior_scale: float = 2.0,
    ) -> "RelationConditionedMultiLayerQueryInterface":
        payload = json.loads(Path(layer_map_path).read_text(encoding="utf-8"))
        if payload.get("schema") != "alice.eipm.n0.relation-conditioned-layer-map.v0.1":
            raise ValueError("relation-conditioned layer-map schema mismatch")
        if payload.get("training_authorized") is not False:
            raise ValueError("layer-map governance drift")
        return cls(
            semantic_size=semantic_size,
            graph_size=graph_size,
            num_relation_types=num_relation_types,
            num_hidden_states=num_hidden_states,
            layer_map=payload["relation_map"],
            interface_size=interface_size,
            dropout=dropout,
            audit_prior_scale=audit_prior_scale,
        )

    def _candidate_layers_for_edges(
        self,
        edge_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        if edge_type_ids.ndim != 2:
            raise ValueError("edge_type_ids must be [batch, edges]")
        edge_type_ids = edge_type_ids.clamp(
            0, self.num_relation_types - 1
        )
        return self.relation_layer_candidate_mask[edge_type_ids]

    def forward(
        self,
        *,
        hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor],
        attention_mask: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if len(hidden_states) != self.num_hidden_states:
            raise ValueError(
                f"expected {self.num_hidden_states} hidden states, "
                f"received {len(hidden_states)}"
            )
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must be [batch, tokens]")
        if edge_type_ids.shape != edge_valid_mask.shape:
            raise ValueError("edge type/mask shape mismatch")

        batch, tokens = attention_mask.shape
        edge_batch, edges = edge_type_ids.shape
        if batch != edge_batch:
            raise ValueError("query batch and edge batch mismatch")

        stack = torch.stack(list(hidden_states), dim=1)
        if stack.ndim != 4:
            raise ValueError("hidden states must stack to [batch,layers,tokens,width]")
        if stack.shape[2] != tokens or stack.shape[3] != self.semantic_size:
            raise ValueError("hidden-state shape mismatch")

        candidate_mask = self._candidate_layers_for_edges(edge_type_ids)
        relation_ids = edge_type_ids.clamp(0, self.num_relation_types - 1)
        relation = self.relation_embedding(relation_ids)
        relation_query = self.relation_query(relation)

        normalized = self.token_norm(stack)
        token_keys = self.token_key(normalized)
        token_values = self.token_value(normalized)

        # [B,E,L,T]: each edge relation asks each candidate layer which query
        # tokens carry evidence relevant to that relation.
        token_scores = torch.einsum(
            "bew,bltw->belt",
            relation_query,
            token_keys,
        ) / (self.interface_size ** 0.5)

        token_valid = attention_mask.bool()[:, None, None, :]
        token_scores = token_scores.masked_fill(~token_valid, -1.0e4)
        token_weights = torch.softmax(token_scores, dim=-1)

        # [B,E,L,W]
        layer_summaries = torch.einsum(
            "belt,bltw->belw",
            token_weights,
            token_values,
        )

        layer_ids = torch.arange(
            self.num_hidden_states,
            device=stack.device,
        )
        layer_emb = self.layer_embedding(layer_ids)[None, None, :, :]
        layer_emb = layer_emb.expand(batch, edges, -1, -1)
        relation_expanded = relation[:, :, None, :].expand(
            -1, -1, self.num_hidden_states, -1
        )

        layer_features = torch.cat(
            [
                layer_summaries,
                relation_expanded,
                layer_summaries * relation_expanded,
                layer_emb,
            ],
            dim=-1,
        )
        learned_layer_score = self.layer_score(layer_features).squeeze(-1)
        prior = self.relation_layer_audit_prior[relation_ids]
        layer_logits = learned_layer_score + prior

        active_edge = edge_valid_mask.bool()
        active_candidate = candidate_mask & active_edge[:, :, None]
        layer_logits = layer_logits.masked_fill(~active_candidate, -1.0e4)

        has_candidates = active_candidate.any(dim=-1)
        safe_logits = torch.where(
            has_candidates[:, :, None],
            layer_logits,
            torch.zeros_like(layer_logits),
        )
        layer_weights = torch.softmax(safe_logits, dim=-1)
        layer_weights = layer_weights * active_candidate.to(layer_weights.dtype)
        layer_weights = layer_weights / layer_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-8)

        edge_state = torch.einsum(
            "bel,belw->bew",
            layer_weights,
            layer_summaries,
        )
        edge_state = self.output_projection(self.output_norm(edge_state))
        edge_state = edge_state * has_candidates.unsqueeze(-1).to(edge_state.dtype)

        return {
            "edge_relation_query_state": edge_state,
            "relation_layer_weights": layer_weights,
            "relation_token_attention": token_weights,
            "relation_has_interface": has_candidates,
            "relation_layer_candidate_mask": candidate_mask,
        }

    def parameter_report(self) -> dict[str, Any]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "semantic_size": self.semantic_size,
            "graph_size": self.graph_size,
            "interface_size": self.interface_size,
            "num_hidden_states": self.num_hidden_states,
            "relation_conditioned_layer_selection": True,
            "token_level_relation_attention": True,
            "audit_scores_are_initialization_priors": True,
            "candidate_layer_mask_is_permanent_architecture_limit": False,
            "single_universal_layer": False,
            "parent_semantic_backbone_mutated": False,
            "hard_parameter_ceiling": None,
        }
