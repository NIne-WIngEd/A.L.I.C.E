from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from alice_personality.n0.dynamic_competitive_latent_pool_v3 import (
    DynamicCompetitiveLatentPoolV3,
    DynamicLatentPoolConfig,
)
from alice_personality.n0.dynamic_cross_context_fusion_v3 import (
    DynamicCrossContextFusionConfig,
    DynamicCrossContextFusionV3,
)
from alice_personality.n0.dynamic_evidence_view_v2 import (
    DynamicEvidenceViewConfig,
    DynamicEvidenceViewV2,
)
from alice_personality.n0.dynamic_schema_evidence_graph_v1 import (
    DynamicSchemaEvidenceGraphConfig,
    DynamicSchemaEvidenceGraphV1,
)
from alice_personality.n0.dynamic_structured_state_v2 import (
    DynamicStructuredStateConfig,
    DynamicStructuredStateV2,
)
from alice_personality.n0.qsre_full_envelope_binder_v1 import (
    FullEnvelopeBinderConfig,
    FullEnvelopeQSREBinderV1,
)
from alice_personality.n0.qsre_full_envelope_executor_v1 import (
    FullEnvelopeExecutorConfig,
    FullEnvelopeQSREExecutorV1,
)
from alice_personality.n0.full_envelope_structural_types import CONTROL_RELATIONAL
from alice_personality.n0.public_judgment_probe_v1 import (
    PublicJudgmentProbeConfig,
    PublicJudgmentProbeV1,
)
from alice_personality.n0.semantic_operator_foundation import (
    DynamicRelationSchema,
    DynamicSemanticSchema,
    SchemaConditionedSemanticOperator,
    SemanticOperatorFoundationConfig,
)
from alice_personality.n0.semantic_operator_qsre_adapter import (
    SemanticOperatorQSREAdapter,
)


@dataclass(frozen=True)
class N0FullEnvelopeStackConfig:
    semantic_dim: int = 640
    model_dim: int = 640
    num_hidden_states: int = 17
    num_attention_heads: int = 10
    structured_layers: int = 2
    field_metadata_dim: int = 3
    edge_metadata_dim: int = 4
    dropout: float = 0.0

    def validate(self) -> None:
        for name, value in (
            ("semantic_dim", self.semantic_dim),
            ("model_dim", self.model_dim),
            ("num_hidden_states", self.num_hidden_states),
            ("num_attention_heads", self.num_attention_heads),
            ("structured_layers", self.structured_layers),
            ("field_metadata_dim", self.field_metadata_dim),
            ("edge_metadata_dim", self.edge_metadata_dim),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.model_dim != self.semantic_dim:
            raise ValueError(
                "v1 full-envelope stack requires semantic_dim == model_dim; "
                "this is an implementation interface choice, not a product ceiling"
            )
        if self.model_dim % self.num_attention_heads:
            raise ValueError("model_dim must be divisible by num_attention_heads")


class N0FullEnvelopeStackV1(nn.Module):
    """Static full N0 successor wiring over semantic-backbone hidden states.

    The semantic backbone itself remains external so this module can be tested
    without downloading weights. During authorized training, backbone hidden
    states must be passed without detach so gradients can co-adapt the semantic
    representation with every reopened N0 interface.
    """

    INTERNAL_VIEW_COUNT = 6

    def __init__(self, config: N0FullEnvelopeStackConfig | None = None) -> None:
        super().__init__()
        self.config = config or N0FullEnvelopeStackConfig()
        self.config.validate()
        d = self.config.model_dim

        self.raw_semantic_layer_gate = nn.Sequential(
            nn.Linear(d + 1, d),
            nn.SiLU(),
            nn.Linear(d, 1),
        )
        self.semantic_operator = SchemaConditionedSemanticOperator(
            SemanticOperatorFoundationConfig(
                semantic_dim=d,
                model_dim=d,
                num_hidden_states=self.config.num_hidden_states,
                dropout=self.config.dropout,
            )
        )
        self.operator_adapter = SemanticOperatorQSREAdapter()
        self.structured = DynamicStructuredStateV2(
            DynamicStructuredStateConfig(
                semantic_dim=d,
                model_dim=d,
                num_hidden_states=self.config.num_hidden_states,
                num_attention_heads=self.config.num_attention_heads,
                num_layers=self.config.structured_layers,
                continuous_metadata_dim=self.config.field_metadata_dim,
                dropout=self.config.dropout,
            )
        )
        self.binder = FullEnvelopeQSREBinderV1(
            FullEnvelopeBinderConfig(
                semantic_dim=d,
                model_dim=d,
                num_hidden_states=self.config.num_hidden_states,
                edge_metadata_dim=self.config.edge_metadata_dim,
            )
        )
        self.executor = FullEnvelopeQSREExecutorV1(
            FullEnvelopeExecutorConfig(
                field_dim=d,
                model_dim=d,
                field_metadata_dim=self.config.field_metadata_dim,
                edge_metadata_dim=self.config.edge_metadata_dim,
                dropout=self.config.dropout,
            )
        )
        self.evidence_view = DynamicEvidenceViewV2(
            DynamicEvidenceViewConfig(
                semantic_dim=d,
                model_dim=d,
                num_hidden_states=self.config.num_hidden_states,
                dropout=self.config.dropout,
            )
        )
        self.evidence_graph = DynamicSchemaEvidenceGraphV1(
            DynamicSchemaEvidenceGraphConfig(
                field_dim=d,
                relation_dim=d,
                operator_dim=d,
                model_dim=d,
                edge_metadata_dim=self.config.edge_metadata_dim,
                dropout=self.config.dropout,
            )
        )
        self.fusion = DynamicCrossContextFusionV3(
            DynamicCrossContextFusionConfig(
                semantic_dim=d,
                model_dim=d,
                num_attention_heads=self.config.num_attention_heads,
                recurrent_refinement_steps=2,
                dropout=self.config.dropout,
            )
        )
        self.latent = DynamicCompetitiveLatentPoolV3(
            DynamicLatentPoolConfig(
                semantic_dim=d,
                model_dim=d,
                dropout=self.config.dropout,
            )
        )
        self.public_judgment_probe = PublicJudgmentProbeV1(
            PublicJudgmentProbeConfig(
                semantic_dim=d,
                latent_dim=d,
                model_dim=d,
                num_hidden_states=self.config.num_hidden_states,
            )
        )

    def _raw_field_semantic(
        self,
        field_hidden_states: Tensor,
        field_token_mask: Tensor,
        field_valid_mask: Tensor,
    ) -> Tensor:
        weight = field_token_mask[:, :, None, :, None].to(
            field_hidden_states.dtype
        )
        per_layer_field = (
            (field_hidden_states * weight).sum(dim=3)
            / weight.sum(dim=3).clamp_min(1.0)
        ).float()
        layers = per_layer_field.size(2)
        layer_position = torch.linspace(
            -1.0,
            1.0,
            layers,
            device=per_layer_field.device,
            dtype=per_layer_field.dtype,
        ).view(1,1,layers,1).expand(
            per_layer_field.size(0),
            per_layer_field.size(1),
            layers,
            1,
        )
        layer_logit = self.raw_semantic_layer_gate(
            torch.cat([per_layer_field, layer_position], dim=-1)
        ).squeeze(-1)
        layer_weight = torch.softmax(layer_logit, dim=-1)
        field_summary = torch.einsum(
            "bfl,bfld->bfd",
            layer_weight,
            per_layer_field,
        )
        field_weight = field_valid_mask.to(field_summary.dtype)
        return (
            (field_summary * field_weight.unsqueeze(-1)).sum(dim=1)
            / field_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        )

    def forward(
        self,
        *,
        query_hidden_states: Tensor,
        query_token_mask: Tensor,
        relation_schema: DynamicRelationSchema,
        factor_schemas: Mapping[str, DynamicSemanticSchema],
        factor_opcodes: Mapping[str, list[str]],
        relation_candidate_mask: Tensor | None = None,
        factor_candidate_masks: Mapping[str, Tensor] | None = None,
        field_hidden_states: Tensor,
        field_token_mask: Tensor,
        field_valid_mask: Tensor,
        field_confidence: Tensor,
        field_missing: Tensor,
        field_reliability: Tensor,
        descriptor_banks: Mapping[str, DynamicSemanticSchema],
        descriptor_indices: Mapping[str, Tensor],
        field_type_index: Tensor,
        field_metadata: Tensor,
        edge_index: Tensor,
        edge_relation_index: Tensor,
        edge_valid_mask: Tensor,
        edge_metadata: Tensor,
        edge_reliability: Tensor,
        edge_recency: Tensor,
        edge_temporal_match: Tensor,
        edge_provenance_match: Tensor,
        internal_view_descriptor_states: Tensor,
        internal_view_reliability: Tensor,
        max_reasoning_steps: int,
        graph_message_steps: int,
        fusion_refinement_steps: int,
        latent_slot_count: int,
        latent_refinement_steps: int,
        additional_source_views: Tensor | None = None,
        additional_view_descriptor_states: Tensor | None = None,
        additional_view_available: Tensor | None = None,
        additional_view_reliability: Tensor | None = None,
        candidate_hidden_states: Tensor | None = None,
        candidate_token_mask: Tensor | None = None,
        candidate_valid_mask: Tensor | None = None,
    ) -> dict[str, Any]:
        semantic = self.semantic_operator(
            query_hidden_states=query_hidden_states,
            query_token_mask=query_token_mask,
            relation_schema=relation_schema,
            factor_schemas=factor_schemas,
            max_steps=max_reasoning_steps,
            relation_candidate_mask=relation_candidate_mask,
            factor_candidate_masks=factor_candidate_masks,
        )
        adapted = self.operator_adapter(
            semantic_operator=semantic["operator"],
            relation_schema_states=semantic["relation_schema_states"],
            factor_opcodes=factor_opcodes,
        )
        operator = adapted["operator"]
        relation_state = adapted["relation_schema_state"]

        structured = self.structured(
            field_hidden_states=field_hidden_states,
            field_token_mask=field_token_mask,
            field_valid_mask=field_valid_mask,
            field_confidence=field_confidence,
            field_missing=field_missing,
            field_metadata=field_metadata,
            descriptor_banks=descriptor_banks,
            descriptor_indices=descriptor_indices,
        )

        batch = query_hidden_states.size(0)
        relation_symmetric = relation_schema.symmetric[None, :].expand(
            batch, -1
        )
        raw_relation_mass = torch.einsum(
            "bsr,bs->br",
            operator.relation_distribution,
            operator.relation_step_mass,
        )
        relation_program_mass = operator.relation_step_mass.sum(
            dim=1,
            keepdim=True,
        )
        relation_mass = raw_relation_mass / relation_program_mass.clamp_min(
            1.0e-6
        )
        known_mass = (
            1.0 - operator.unknown_probability.sum(dim=1).clamp(max=1.0)
        ).clamp(0.0, 1.0)
        semantic_activity = (
            relation_program_mass.squeeze(1).clamp(0.0, 1.0)
            * operator.applicability.clamp(0.0, 1.0)
            * operator.control_distribution[:, CONTROL_RELATIONAL]
            * known_mass
        ).clamp(0.0, 1.0)

        graph = self.evidence_graph(
            field_state=structured["field_states"],
            field_valid_mask=field_valid_mask,
            edge_index=edge_index,
            edge_relation_index=edge_relation_index,
            edge_metadata=edge_metadata,
            edge_valid_mask=edge_valid_mask,
            relation_schema_state=relation_state,
            relation_mass=relation_mass,
            relation_symmetric=relation_symmetric,
            semantic_activity=semantic_activity,
            operator_state=operator.continuous_state,
            message_steps=graph_message_steps,
        )

        domain = relation_schema.domain_type_mask[None, :, :].expand(
            batch, -1, -1
        )
        range_mask = relation_schema.range_type_mask[None, :, :].expand(
            batch, -1, -1
        )
        binder = self.binder(
            query_hidden_states=query_hidden_states,
            query_token_mask=query_token_mask,
            field_hidden_states=field_hidden_states,
            field_token_mask=field_token_mask,
            field_state=graph["field_states"],
            field_valid_mask=field_valid_mask,
            field_type_index=field_type_index,
            edge_index=edge_index,
            edge_relation_index=edge_relation_index,
            edge_valid_mask=edge_valid_mask,
            edge_reliability=edge_reliability,
            edge_recency=edge_recency,
            relation_domain_type_mask=domain,
            relation_range_type_mask=range_mask,
            relation_symmetric=relation_symmetric,
            relation_schema_state=relation_state,
            operator=operator,
        )

        executor = self.executor(
            field_state=graph["field_states"],
            field_metadata=field_metadata,
            field_valid_mask=field_valid_mask,
            edge_index=edge_index,
            edge_relation_index=edge_relation_index,
            edge_valid_mask=edge_valid_mask,
            edge_support_weight=binder["edge_support_weight"],
            support_available=binder["support_available"],
            edge_reliability=edge_reliability,
            edge_recency=edge_recency,
            edge_temporal_match=edge_temporal_match,
            edge_provenance_match=edge_provenance_match,
            relation_schema_state=relation_state,
            relation_symmetric=relation_symmetric,
            operator=operator,
            focus_field_weight=binder["focus_field_weight"],
        )

        evidence_view = self.evidence_view(
            query_hidden_states=query_hidden_states,
            query_token_mask=query_token_mask,
            field_hidden_states=field_hidden_states,
            field_token_mask=field_token_mask,
            structured_field_state=graph["field_states"],
            field_valid_mask=field_valid_mask,
            field_confidence=field_confidence,
            field_missing=field_missing,
            field_reliability=field_reliability,
            relation_schema_state=relation_state,
            relation_mass=relation_mass,
            semantic_activity=semantic_activity,
            operator_state=operator.continuous_state,
        )

        raw_semantic = self._raw_field_semantic(
            field_hidden_states,
            field_token_mask,
            field_valid_mask,
        )
        internal_views = torch.stack(
            [
                raw_semantic,
                structured["pooled_state"],
                evidence_view["evidence_summary"],
                graph["source_summary"],
                graph["target_summary"],
                executor["relational_summary"],
            ],
            dim=1,
        )
        if internal_view_descriptor_states.shape != internal_views.shape:
            raise ValueError(
                "internal_view_descriptor_states must be [B,6,D]"
            )
        if internal_view_reliability.shape != (
            batch,
            self.INTERNAL_VIEW_COUNT,
        ):
            raise ValueError("internal_view_reliability must be [B,6]")
        internal_available = torch.ones(
            batch,
            self.INTERNAL_VIEW_COUNT,
            device=internal_views.device,
            dtype=torch.bool,
        )

        source_views = internal_views
        descriptors = internal_view_descriptor_states
        available = internal_available
        reliability = internal_view_reliability

        extras = (
            additional_source_views,
            additional_view_descriptor_states,
            additional_view_available,
            additional_view_reliability,
        )
        if any(x is not None for x in extras):
            if not all(x is not None for x in extras):
                raise ValueError(
                    "all additional-view tensors must be supplied together"
                )
            if additional_source_views.ndim != 3:
                raise ValueError("additional_source_views must be [B,V,D]")
            extra_views = additional_source_views.size(1)
            if additional_source_views.shape != (
                batch,
                extra_views,
                self.config.model_dim,
            ):
                raise ValueError("additional source-view geometry drift")
            if additional_view_descriptor_states.shape != additional_source_views.shape:
                raise ValueError("additional view descriptor geometry drift")
            if additional_view_available.shape != (batch, extra_views):
                raise ValueError("additional view availability geometry drift")
            if additional_view_reliability.shape != (batch, extra_views):
                raise ValueError("additional view reliability geometry drift")
            source_views = torch.cat(
                [source_views, additional_source_views],
                dim=1,
            )
            descriptors = torch.cat(
                [descriptors, additional_view_descriptor_states],
                dim=1,
            )
            available = torch.cat(
                [available, additional_view_available],
                dim=1,
            )
            reliability = torch.cat(
                [reliability, additional_view_reliability],
                dim=1,
            )

        fusion = self.fusion(
            source_view_summaries=source_views,
            view_descriptor_states=descriptors,
            view_available=available,
            query_state=operator.continuous_state,
            view_reliability=reliability,
            refinement_steps=fusion_refinement_steps,
        )
        latent = self.latent(
            source_view_summaries=fusion["source_view_summaries"],
            contextualized_view_summaries=fusion[
                "contextualized_view_summaries"
            ],
            view_descriptor_states=descriptors,
            view_available=available,
            query_state=operator.continuous_state,
            view_reliability=reliability,
            slot_count=latent_slot_count,
            refinement_steps=latent_refinement_steps,
        )

        public_judgment = None
        if (
            candidate_hidden_states is not None
            or candidate_token_mask is not None
            or candidate_valid_mask is not None
        ):
            if candidate_hidden_states is None or candidate_token_mask is None:
                raise ValueError(
                    "candidate_hidden_states and candidate_token_mask must be supplied together"
                )
            public_judgment = self.public_judgment_probe(
                pooled_state=latent["pooled_state"],
                candidate_hidden_states=candidate_hidden_states,
                candidate_token_mask=candidate_token_mask,
                candidate_valid_mask=candidate_valid_mask,
            )

        return {
            "semantic_operator": semantic,
            "operator": operator,
            "relation_schema_state": relation_state,
            "semantic_activity": semantic_activity,
            "structured": structured,
            "binder": binder,
            "executor": executor,
            "evidence_view": evidence_view,
            "evidence_graph": graph,
            "fusion": fusion,
            "latent": latent,
            "public_judgment": public_judgment,
            "source_views": source_views,
            "view_descriptors": descriptors,
            "view_available": available,
        }

    def parameter_report(self) -> dict[str, Any]:
        reports = {
            "semantic_operator": self.semantic_operator.parameter_report(),
            "operator_adapter": self.operator_adapter.parameter_report(),
            "structured": self.structured.parameter_report(),
            "binder": self.binder.parameter_report(),
            "executor": self.executor.parameter_report(),
            "evidence_view": self.evidence_view.parameter_report(),
            "evidence_graph": self.evidence_graph.parameter_report(),
            "fusion": self.fusion.parameter_report(),
            "latent": self.latent.parameter_report(),
            "public_judgment_probe": self.public_judgment_probe.parameter_report(),
        }
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "module_reports": reports,
            "semantic_backbone_included": False,
            "semantic_backbone_gradient_must_remain_connected": True,
            "continuous_graph_before_exact_binder_sparsity": True,
            "raw_semantic_view_static_layer_mean": False,
            "raw_semantic_view_content_conditioned_layer_read": True,
            "pre_binder_graph_soft_activity_gated": True,
            "runtime_relation_ceiling": None,
            "per_example_candidate_subset_supported": True,
            "runtime_factor_ceiling": None,
            "runtime_field_ceiling": None,
            "runtime_edge_ceiling": None,
            "runtime_view_ceiling": None,
            "runtime_slot_ceiling": None,
            "runtime_reasoning_step_ceiling": None,
            "exact_structural_sparsity_boundary": "FullEnvelopeQSREBinderV1",
        }
