from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .evidence_graph import EvidenceRelationType
from .relation_conditioned_multilayer_query import RELATION_ID_TO_NAME


class QueryEdgeCrossAttentionBridge(nn.Module):
    """Edge-specific language↔graph bridge over frozen semantic hidden states.

    The bridge is deliberately not an endpoint classifier.  For each directed
    edge, its token-attention query is constructed from the relation plus the
    *actual source and target graph states*.  Two edges with the same relation
    type can therefore attend to different query tokens and produce different
    specialist states.

    The bridge emits independent source and target residual proposals and a
    separate edge-specific gate.  The gate's final projection is exactly
    zero-initialized, so the containing graph is an exact parent at
    initialization while the residual heads remain non-zero and can provide a
    learning signal to the gate on the first optimization step.

    The compiled relation→layer map is an audit-derived operating prior for the
    first study, not a permanent capability ceiling.
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
            raise ValueError("query-edge bridge requires hidden-state depth")
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
        self.source_projection = nn.Linear(graph_size, width, bias=False)
        self.target_projection = nn.Linear(graph_size, width, bias=False)

        # The token query is edge-specific: relation + source + target + their
        # directional difference jointly determine which query tokens matter.
        self.edge_query = nn.Sequential(
            nn.Linear(4 * width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
        )

        self.layer_embedding = nn.Embedding(num_hidden_states, width)
        self.layer_score = nn.Sequential(
            nn.Linear(4 * width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, 1),
        )

        # Fused specialist state retains both language and actual edge context.
        self.fusion = nn.Sequential(
            nn.Linear(4 * width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
            nn.LayerNorm(width),
        )
        self.source_residual_read = nn.Linear(width, 1)
        self.target_residual_read = nn.Linear(width, 1)

        # Zero-init only the final gate projection.  The independent residual
        # heads remain live, avoiding a doubly-zero path that would suppress
        # the gate's first-step gradient.
        self.gate = nn.Sequential(
            nn.Linear(4 * width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        gate_final = self.gate[-1]
        assert isinstance(gate_final, nn.Linear)
        nn.init.zeros_(gate_final.weight)
        nn.init.zeros_(gate_final.bias)

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

        # Symmetric conflict edges do not receive a directional specialist.
        candidate_mask[int(EvidenceRelationType.CONFLICTS_WITH)] = False
        audit_prior[int(EvidenceRelationType.CONFLICTS_WITH)] = -1.0e4
        candidate_mask[int(EvidenceRelationType.PAD)] = False
        audit_prior[int(EvidenceRelationType.PAD)] = -1.0e4

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
    ) -> "QueryEdgeCrossAttentionBridge":
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

    def forward(
        self,
        *,
        hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor],
        attention_mask: torch.Tensor,
        graph_states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_valid_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if len(hidden_states) != self.num_hidden_states:
            raise ValueError(
                f"expected {self.num_hidden_states} hidden states, "
                f"received {len(hidden_states)}"
            )
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must be [batch,tokens]")
        if graph_states.ndim != 3 or graph_states.size(-1) != self.graph_size:
            raise ValueError("graph_states must be [batch,fields,graph_size]")
        if edge_index.ndim != 3 or edge_index.size(-1) != 2:
            raise ValueError("edge_index must be [batch,edges,2]")
        if edge_type_ids.shape != edge_valid_mask.shape:
            raise ValueError("edge type/mask shape mismatch")

        batch, fields, _ = graph_states.shape
        if attention_mask.size(0) != batch or edge_index.size(0) != batch:
            raise ValueError("query/graph/edge batch mismatch")
        if valid_mask.shape != (batch, fields):
            raise ValueError("valid_mask shape mismatch")

        stack = torch.stack(list(hidden_states), dim=1)
        if stack.ndim != 4:
            raise ValueError("hidden states must stack to [batch,layers,tokens,width]")
        if stack.size(0) != batch or stack.size(2) != attention_mask.size(1):
            raise ValueError("hidden/query shape mismatch")
        if stack.size(3) != self.semantic_size:
            raise ValueError("semantic hidden width mismatch")

        edges = edge_index.size(1)
        source = edge_index[..., 0].clamp(0, fields - 1)
        target = edge_index[..., 1].clamp(0, fields - 1)
        batch_index = torch.arange(batch, device=graph_states.device).unsqueeze(1)
        batch_index = batch_index.expand_as(source)

        source_valid = valid_mask[batch_index, source]
        target_valid = valid_mask[batch_index, target]
        relation_ids = edge_type_ids.clamp(0, self.num_relation_types - 1)
        relation = self.relation_embedding(relation_ids)
        source_state = graph_states[batch_index, source]
        target_state = graph_states[batch_index, target]
        source_latent = self.source_projection(source_state)
        target_latent = self.target_projection(target_state)

        edge_query_input = torch.cat(
            [
                relation,
                source_latent,
                target_latent,
                source_latent - target_latent,
            ],
            dim=-1,
        )
        edge_query = self.edge_query(edge_query_input)

        normalized = self.token_norm(stack)
        token_keys = self.token_key(normalized)
        token_values = self.token_value(normalized)

        # Crucially, E is present in the query: same-relation edges with
        # different endpoint states can attend to different language tokens.
        token_scores = torch.einsum(
            "bew,bltw->belt",
            edge_query,
            token_keys,
        ) / (self.interface_size ** 0.5)
        token_valid = attention_mask.bool()[:, None, None, :]
        token_scores = token_scores.masked_fill(~token_valid, -1.0e4)
        token_weights = torch.softmax(token_scores, dim=-1)
        layer_summaries = torch.einsum(
            "belt,bltw->belw",
            token_weights,
            token_values,
        )

        candidate_mask = self.relation_layer_candidate_mask[relation_ids]
        layer_ids = torch.arange(self.num_hidden_states, device=stack.device)
        layer_emb = self.layer_embedding(layer_ids)[None, None, :, :]
        layer_emb = layer_emb.expand(batch, edges, -1, -1)
        edge_query_expanded = edge_query[:, :, None, :].expand(
            -1, -1, self.num_hidden_states, -1
        )
        layer_features = torch.cat(
            [
                layer_summaries,
                edge_query_expanded,
                layer_summaries * edge_query_expanded,
                layer_emb,
            ],
            dim=-1,
        )
        learned_layer_score = self.layer_score(layer_features).squeeze(-1)
        layer_logits = learned_layer_score + self.relation_layer_audit_prior[relation_ids]

        conflict = edge_type_ids.eq(int(EvidenceRelationType.CONFLICTS_WITH))
        pad = edge_type_ids.eq(int(EvidenceRelationType.PAD))
        active_edge = (
            edge_valid_mask.bool()
            & source_valid
            & target_valid
            & ~conflict
            & ~pad
        )
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

        query_edge_language = torch.einsum(
            "bel,belw->bew",
            layer_weights,
            layer_summaries,
        )
        fused = self.fusion(
            torch.cat(
                [
                    query_edge_language,
                    edge_query,
                    source_latent,
                    target_latent,
                ],
                dim=-1,
            )
        )

        source_proposal = self.source_residual_read(fused).squeeze(-1)
        target_proposal = self.target_residual_read(fused).squeeze(-1)
        gate_logit = self.gate(
            torch.cat(
                [
                    query_edge_language,
                    edge_query,
                    source_latent,
                    target_latent,
                ],
                dim=-1,
            )
        ).squeeze(-1)
        gate = torch.tanh(gate_logit)
        gate = gate * has_candidates.to(gate.dtype)

        source_delta = gate * source_proposal
        target_delta = gate * target_proposal

        return {
            "edge_binding_query": edge_query,
            "edge_query_language_state": query_edge_language,
            "edge_fused_state": fused,
            "source_residual_proposal": source_proposal,
            "target_residual_proposal": target_proposal,
            "edge_specialist_gate": gate,
            "source_residual": source_delta,
            "target_residual": target_delta,
            "relation_token_attention": token_weights,
            "relation_layer_weights": layer_weights,
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
            "edge_specific_query_conditioning": True,
            "source_graph_state_conditioning": True,
            "target_graph_state_conditioning": True,
            "independent_source_target_residuals": True,
            "forced_antisymmetry": False,
            "zero_initialized_specialist_gate": True,
            "generic_endpoint_classifier": False,
            "raw_mean_pool_router": False,
            "hardcoded_single_layer": False,
            "hard_parameter_ceiling": None,
        }
