from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from alice_personality.n0.full_envelope_semantic_input_v1 import (
    FullEnvelopeSemanticInputConfig,
    FullEnvelopeSemanticInputV1,
)
from alice_personality.n0.n0_full_envelope_stack_v1 import (
    N0FullEnvelopeStackConfig,
    N0FullEnvelopeStackV1,
)
from alice_personality.n0.semantic_operator_foundation import (
    DynamicRelationSchema,
    DynamicSemanticSchema,
)


@dataclass(frozen=True)
class N0FullEnvelopeTrainableSystemConfig:
    semantic_dim: int = 640
    model_dim: int = 640
    num_hidden_states: int = 17
    num_attention_heads: int = 10
    structured_layers: int = 2
    field_metadata_dim: int = 3
    edge_metadata_dim: int = 4
    native_window_tokens: int = 4096
    overlap_tokens: int = 256
    segment_bridge_layers: int = 2
    segment_query_chunk: int = 32
    segment_key_chunk: int = 64
    special_token_ids: tuple[int, ...] = (0, 1, 2, 3, 4)
    pad_token_id: int = 0
    dropout: float = 0.0

    def validate(self) -> None:
        if self.semantic_dim != self.model_dim:
            raise ValueError("semantic_dim and model_dim must match in v1")
        for name, value in (
            ("semantic_dim", self.semantic_dim),
            ("num_hidden_states", self.num_hidden_states),
            ("num_attention_heads", self.num_attention_heads),
            ("structured_layers", self.structured_layers),
            ("field_metadata_dim", self.field_metadata_dim),
            ("edge_metadata_dim", self.edge_metadata_dim),
            ("native_window_tokens", self.native_window_tokens),
            ("segment_bridge_layers", self.segment_bridge_layers),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")


class N0FullEnvelopeTrainableSystemV1(nn.Module):
    """One registered module tree for joint public-N0 training.

    This closes the previous integration gap where the semantic backbone,
    long-context bridge, and full-envelope successor were individually valid
    but there was no single trainable topology proving that all gradients,
    text surfaces, and runtime schemas travel through one shared system.

    The supplied semantic model must expose a shared backbone property and its
    normal forward(task=...) replay interface. Historical semantic heads
    therefore remain available for regression replay while the same backbone
    co-adapts with the successor.
    """

    def __init__(
        self,
        *,
        semantic_model: nn.Module,
        config: N0FullEnvelopeTrainableSystemConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or N0FullEnvelopeTrainableSystemConfig()
        self.config.validate()
        self.semantic_model = semantic_model
        if not hasattr(semantic_model, "backbone"):
            raise ValueError("semantic_model must expose one shared backbone")

        self.semantic_input = FullEnvelopeSemanticInputV1(
            FullEnvelopeSemanticInputConfig(
                semantic_dim=self.config.semantic_dim,
                num_hidden_states=self.config.num_hidden_states,
                num_attention_heads=self.config.num_attention_heads,
                native_window_tokens=self.config.native_window_tokens,
                overlap_tokens=self.config.overlap_tokens,
                segment_bridge_layers=self.config.segment_bridge_layers,
                segment_query_chunk=self.config.segment_query_chunk,
                segment_key_chunk=self.config.segment_key_chunk,
                special_token_ids=self.config.special_token_ids,
                pad_token_id=self.config.pad_token_id,
                dropout=self.config.dropout,
            )
        )
        self.stack = N0FullEnvelopeStackV1(
            N0FullEnvelopeStackConfig(
                semantic_dim=self.config.semantic_dim,
                model_dim=self.config.model_dim,
                num_hidden_states=self.config.num_hidden_states,
                num_attention_heads=self.config.num_attention_heads,
                structured_layers=self.config.structured_layers,
                field_metadata_dim=self.config.field_metadata_dim,
                edge_metadata_dim=self.config.edge_metadata_dim,
                dropout=self.config.dropout,
            )
        )

    @property
    def backbone(self) -> nn.Module:
        return getattr(self.semantic_model, "backbone")

    def _encode_factor_schemas(
        self,
        batch: Mapping[str, Any],
    ) -> tuple[dict[str, DynamicSemanticSchema], dict[str, dict[str, int | bool]]]:
        ids = batch["factor_input_ids"]
        masks = batch["factor_attention_mask"]
        if set(ids) != set(masks):
            raise ValueError("factor input/mask bank names must match")
        schemas: dict[str, DynamicSemanticSchema] = {}
        metadata: dict[str, dict[str, int | bool]] = {}
        for raw_name in ids:
            name = str(raw_name)
            encoded = self.semantic_input.encode_items(
                backbone=self.backbone,
                input_ids=ids[raw_name],
                attention_mask=masks[raw_name],
            )
            schema = DynamicSemanticSchema(
                token_states=encoded["hidden_states"],
                token_mask=encoded["content_mask"],
            )
            schema.validate(
                num_hidden_states=self.config.num_hidden_states,
                semantic_dim=self.config.semantic_dim,
            )
            schemas[name] = schema
            metadata[name] = {
                "used_virtualization": bool(encoded["used_virtualization"]),
                "segment_count_max": int(encoded["segment_count_max"]),
            }
        return schemas, metadata

    def _encode_descriptor_banks(
        self,
        batch: Mapping[str, Any],
    ) -> tuple[dict[str, DynamicSemanticSchema], dict[str, dict[str, int | bool]]]:
        ids = batch["descriptor_input_ids"]
        masks = batch["descriptor_attention_mask"]
        if set(ids) != set(masks):
            raise ValueError("descriptor input/mask bank names must match")
        schemas: dict[str, DynamicSemanticSchema] = {}
        metadata: dict[str, dict[str, int | bool]] = {}
        for raw_name in ids:
            name = str(raw_name)
            encoded = self.semantic_input.encode_items(
                backbone=self.backbone,
                input_ids=ids[raw_name],
                attention_mask=masks[raw_name],
            )
            schema = DynamicSemanticSchema(
                token_states=encoded["hidden_states"],
                token_mask=encoded["content_mask"],
            )
            schema.validate(
                num_hidden_states=self.config.num_hidden_states,
                semantic_dim=self.config.semantic_dim,
            )
            schemas[name] = schema
            metadata[name] = {
                "used_virtualization": bool(encoded["used_virtualization"]),
                "segment_count_max": int(encoded["segment_count_max"]),
            }
        return schemas, metadata

    def _encode_view_descriptors(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, dict[str, int | bool]]:
        encoded = self.semantic_input.encode_items(
            backbone=self.backbone,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        summary = self.semantic_input.summarize_items(
            hidden_states=encoded["hidden_states"],
            token_mask=encoded["content_mask"],
        )
        return summary, {
            "used_virtualization": bool(encoded["used_virtualization"]),
            "segment_count_max": int(encoded["segment_count_max"]),
        }

    def _forward_full_envelope(
        self,
        batch: Mapping[str, Any],
    ) -> dict[str, Any]:
        query = self.semantic_input.encode_items(
            backbone=self.backbone,
            input_ids=batch["query_input_ids"],
            attention_mask=batch["query_attention_mask"],
        )
        relation_encoded = self.semantic_input.encode_items(
            backbone=self.backbone,
            input_ids=batch["relation_input_ids"],
            attention_mask=batch["relation_attention_mask"],
        )
        relation_schema = DynamicRelationSchema(
            token_states=relation_encoded["hidden_states"],
            token_mask=relation_encoded["content_mask"],
            domain_type_mask=batch["relation_domain_type_mask"],
            range_type_mask=batch["relation_range_type_mask"],
            symmetric=batch["relation_symmetric"],
        )
        relation_schema.validate(
            num_hidden_states=self.config.num_hidden_states,
            semantic_dim=self.config.semantic_dim,
        )
        factor_schemas, factor_metadata = self._encode_factor_schemas(batch)

        field_encoded = self.semantic_input.encode_padded_items(
            backbone=self.backbone,
            input_ids=batch["field_input_ids"],
            attention_mask=batch["field_attention_mask"],
            item_valid_mask=batch["field_valid_mask"],
        )
        descriptor_banks, descriptor_metadata = self._encode_descriptor_banks(batch)

        internal_descriptor, internal_descriptor_metadata = self._encode_view_descriptors(
            input_ids=batch["internal_view_descriptor_input_ids"],
            attention_mask=batch["internal_view_descriptor_attention_mask"],
        )
        batch_size = batch["query_input_ids"].size(0)
        if internal_descriptor.size(0) != self.stack.INTERNAL_VIEW_COUNT:
            raise ValueError("internal view descriptor count drift")
        internal_descriptor = internal_descriptor[None, :, :].expand(
            batch_size,
            -1,
            -1,
        ).contiguous()

        candidate_hidden = None
        candidate_token_mask = None
        candidate_valid_mask = batch.get("candidate_valid_mask")
        if candidate_valid_mask is not None:
            candidate = self.semantic_input.encode_padded_items(
                backbone=self.backbone,
                input_ids=batch["candidate_input_ids"],
                attention_mask=batch["candidate_attention_mask"],
                item_valid_mask=candidate_valid_mask,
            )
            candidate_hidden = candidate["hidden_states"]
            candidate_token_mask = candidate["token_mask"]

        additional_source_views = batch.get("additional_source_views")
        additional_view_available = batch.get("additional_view_available")
        additional_view_reliability = batch.get("additional_view_reliability")
        additional_view_descriptor_states = None
        additional_descriptor_ids = batch.get(
            "additional_view_descriptor_input_ids"
        )
        if additional_source_views is not None:
            if (
                additional_view_available is None
                or additional_view_reliability is None
                or additional_descriptor_ids is None
                or batch.get("additional_view_descriptor_attention_mask") is None
            ):
                raise ValueError(
                    "additional views require availability, reliability, and descriptor tokens"
                )
            additional_descriptor_encoded = self.semantic_input.encode_padded_items(
                backbone=self.backbone,
                input_ids=additional_descriptor_ids,
                attention_mask=batch[
                    "additional_view_descriptor_attention_mask"
                ],
                item_valid_mask=additional_view_available,
            )
            b, views, layers, tokens, width = (
                additional_descriptor_encoded["hidden_states"].shape
            )
            flat_hidden = additional_descriptor_encoded[
                "hidden_states"
            ].reshape(b * views, layers, tokens, width)
            flat_mask = additional_descriptor_encoded["token_mask"].reshape(
                b * views,
                tokens,
            )
            flat_valid = additional_view_available.reshape(-1)
            flat_summary = torch.zeros(
                b * views,
                width,
                device=flat_hidden.device,
                dtype=flat_hidden.dtype,
            )
            if bool(flat_valid.any()):
                flat_summary[flat_valid] = self.semantic_input.summarize_items(
                    hidden_states=flat_hidden[flat_valid],
                    token_mask=flat_mask[flat_valid],
                )
            additional_view_descriptor_states = flat_summary.reshape(
                b,
                views,
                width,
            )

        outputs = self.stack(
            query_hidden_states=query["hidden_states"],
            query_token_mask=query["content_mask"],
            relation_schema=relation_schema,
            factor_schemas=factor_schemas,
            factor_opcodes=batch["factor_opcodes"],
            relation_candidate_mask=batch.get("relation_candidate_mask"),
            factor_candidate_masks=batch.get("factor_candidate_masks"),
            field_hidden_states=field_encoded["hidden_states"],
            field_token_mask=field_encoded["token_mask"],
            field_valid_mask=batch["field_valid_mask"],
            field_confidence=batch["field_confidence"],
            field_missing=batch["field_missing"],
            field_reliability=batch["field_reliability"],
            descriptor_banks=descriptor_banks,
            descriptor_indices=batch["descriptor_indices"],
            field_type_index=batch["field_type_index"],
            field_metadata=batch["field_metadata"],
            edge_index=batch["edge_index"],
            edge_relation_index=batch["edge_relation_index"],
            edge_valid_mask=batch["edge_valid_mask"],
            edge_metadata=batch["edge_metadata"],
            edge_reliability=batch["edge_reliability"],
            edge_recency=batch["edge_recency"],
            edge_temporal_match=batch["edge_temporal_match"],
            edge_provenance_match=batch["edge_provenance_match"],
            internal_view_descriptor_states=internal_descriptor,
            internal_view_reliability=batch["internal_view_reliability"],
            max_reasoning_steps=int(batch["max_reasoning_steps"]),
            graph_message_steps=int(batch["graph_message_steps"]),
            fusion_refinement_steps=int(batch["fusion_refinement_steps"]),
            latent_slot_count=int(batch["latent_slot_count"]),
            latent_refinement_steps=int(batch["latent_refinement_steps"]),
            additional_source_views=additional_source_views,
            additional_view_descriptor_states=additional_view_descriptor_states,
            additional_view_available=additional_view_available,
            additional_view_reliability=additional_view_reliability,
            candidate_hidden_states=candidate_hidden,
            candidate_token_mask=candidate_token_mask,
            candidate_valid_mask=candidate_valid_mask,
        )
        outputs["semantic_input_metadata"] = {
            "query": {
                "used_virtualization": bool(query["used_virtualization"]),
                "segment_count_max": int(query["segment_count_max"]),
            },
            "relation_schema": {
                "used_virtualization": bool(relation_encoded["used_virtualization"]),
                "segment_count_max": int(relation_encoded["segment_count_max"]),
            },
            "factor_schema": factor_metadata,
            "field_text": {
                "used_virtualization": bool(field_encoded["used_virtualization"]),
                "segment_count_max": int(field_encoded["segment_count_max"]),
            },
            "descriptor_text": descriptor_metadata,
            "internal_view_descriptor": internal_descriptor_metadata,
            "candidate_text": (
                {
                    "used_virtualization": bool(candidate["used_virtualization"]),
                    "segment_count_max": int(candidate["segment_count_max"]),
                }
                if candidate_valid_mask is not None
                else None
            ),
            "additional_view_descriptor": (
                {
                    "used_virtualization": bool(
                        additional_descriptor_encoded["used_virtualization"]
                    ),
                    "segment_count_max": int(
                        additional_descriptor_encoded["segment_count_max"]
                    ),
                }
                if additional_source_views is not None
                else None
            ),
        }
        return outputs

    def forward(
        self,
        *,
        task: str,
        batch: Mapping[str, Any],
    ) -> Any:
        if task == "full_envelope":
            return self._forward_full_envelope(batch)
        if task in {"mlm", "teacher"}:
            return self.semantic_model(task=task, **dict(batch))
        raise ValueError(f"unsupported full-envelope system task: {task!r}")

    def parameter_report(self) -> dict[str, Any]:
        semantic = (
            self.semantic_model.parameter_report()
            if hasattr(self.semantic_model, "parameter_report")
            else {
                "total_parameters": sum(
                    p.numel() for p in self.semantic_model.parameters()
                )
            }
        )
        semantic_input = self.semantic_input.parameter_report()
        stack = self.stack.parameter_report()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        expected = (
            int(semantic["total_parameters"])
            + int(semantic_input["total_parameters"])
            + int(stack["total_parameters"])
        )
        if total != expected:
            raise RuntimeError(
                f"full system parameter accounting drift total={total} expected={expected}"
            )
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "semantic_model_parameters": int(semantic["total_parameters"]),
            "semantic_input_parameters": int(semantic_input["total_parameters"]),
            "successor_stack_parameters": int(stack["total_parameters"]),
            "single_shared_backbone": True,
            "semantic_replay_and_full_envelope_share_backbone": True,
            "full_envelope_gradient_path_registered": True,
            "all_text_surfaces_share_semantic_input": True,
            "native_window_is_product_ceiling": False,
            "runtime_relation_ceiling": None,
            "runtime_factor_ceiling": None,
            "runtime_field_ceiling": None,
            "runtime_edge_ceiling": None,
            "runtime_view_ceiling": None,
            "runtime_slot_ceiling": None,
            "runtime_reasoning_step_ceiling": None,
            "product_context_token_ceiling": None,
            "optimizer_owned": False,
            "scheduler_owned": False,
        }
