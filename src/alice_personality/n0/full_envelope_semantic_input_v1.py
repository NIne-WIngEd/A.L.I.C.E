from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from alice_personality.n0.semantic_backbone_interface_v1 import (
    FullEnvelopeSemanticBackboneInterfaceV1,
    SemanticBackboneInterfaceConfig,
)
from alice_personality.n0.semantic_context_virtualizer_v1 import (
    SemanticContextVirtualizerConfig,
    SemanticContextVirtualizerV1,
)
from alice_personality.n0.semantic_operator_foundation import (
    DynamicRelationSchema,
    DynamicSemanticSchema,
)
from alice_personality.n0.semantic_segment_context_bridge_v1 import (
    SemanticSegmentContextBridgeConfig,
    SemanticSegmentContextBridgeV1,
)


@dataclass(frozen=True)
class FullEnvelopeSemanticInputConfig:
    semantic_dim: int = 640
    num_hidden_states: int = 17
    num_attention_heads: int = 10
    native_window_tokens: int = 4096
    overlap_tokens: int = 256
    segment_bridge_layers: int = 2
    segment_query_chunk: int = 32
    segment_key_chunk: int = 64
    special_token_ids: tuple[int, ...] = (0, 1, 2, 3, 4)
    pad_token_id: int = 0
    dropout: float = 0.0

    def validate(self) -> None:
        for name, value in (
            ("semantic_dim", self.semantic_dim),
            ("num_hidden_states", self.num_hidden_states),
            ("num_attention_heads", self.num_attention_heads),
            ("native_window_tokens", self.native_window_tokens),
            ("segment_bridge_layers", self.segment_bridge_layers),
            ("segment_query_chunk", self.segment_query_chunk),
            ("segment_key_chunk", self.segment_key_chunk),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.semantic_dim % self.num_attention_heads:
            raise ValueError("semantic_dim must be divisible by num_attention_heads")
        if not 0 <= self.overlap_tokens < self.native_window_tokens:
            raise ValueError("overlap_tokens must be in [0,native_window_tokens)")
        if len(set(int(x) for x in self.special_token_ids)) != len(
            self.special_token_ids
        ):
            raise ValueError("special_token_ids must be unique")
        if int(self.pad_token_id) not in set(int(x) for x in self.special_token_ids):
            raise ValueError("pad_token_id must be one of special_token_ids")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")


class FullEnvelopeSemanticInputV1(nn.Module):
    """Shared semantic input path for every text-bearing N0 surface.

    The historical system virtualized long query text but still depended on the
    native backbone window for relation descriptions, factor descriptions,
    fields, candidates, and descriptor banks. This module makes the same
    token-lossless + cross-window semantic path available to all of them.

    It owns only the trainable cross-window bridge. The semantic backbone is
    supplied by the caller so there remains one shared encoder and one gradient
    path during joint training.
    """

    def __init__(
        self,
        config: FullEnvelopeSemanticInputConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or FullEnvelopeSemanticInputConfig()
        self.config.validate()
        self.interface = FullEnvelopeSemanticBackboneInterfaceV1(
            SemanticBackboneInterfaceConfig(
                num_hidden_states=self.config.num_hidden_states,
                semantic_dim=self.config.semantic_dim,
                special_token_ids=self.config.special_token_ids,
            )
        )
        self.virtualizer = SemanticContextVirtualizerV1(
            SemanticContextVirtualizerConfig(
                native_window_tokens=self.config.native_window_tokens,
                overlap_tokens=self.config.overlap_tokens,
                pad_token_id=self.config.pad_token_id,
            )
        )
        self.segment_bridge = SemanticSegmentContextBridgeV1(
            SemanticSegmentContextBridgeConfig(
                semantic_dim=self.config.semantic_dim,
                num_hidden_states=self.config.num_hidden_states,
                num_attention_heads=self.config.num_attention_heads,
                num_layers=self.config.segment_bridge_layers,
                metadata_dim=3,
                query_chunk_segments=self.config.segment_query_chunk,
                key_chunk_segments=self.config.segment_key_chunk,
                dropout=self.config.dropout,
            )
        )

    def _content_mask(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        content = attention_mask.bool().clone()
        for token_id in self.config.special_token_ids:
            content &= input_ids.ne(int(token_id))
        if bool((content.sum(dim=-1) == 0).any()):
            raise ValueError("every valid semantic item requires content tokens")
        return content

    def _encode_segment_batch(
        self,
        *,
        backbone: Any,
        input_ids: Tensor,
        attention_mask: Tensor,
        segment_valid_mask: Tensor,
    ) -> Tensor:
        if input_ids.ndim != 3:
            raise ValueError("segmented input_ids must be [B,G,W]")
        batch, segments, window = input_ids.shape
        if attention_mask.shape != input_ids.shape:
            raise ValueError("segmented attention geometry drift")
        if segment_valid_mask.shape != (batch, segments):
            raise ValueError("segment_valid_mask must be [B,G]")

        flat_ids = input_ids.reshape(batch * segments, window)
        flat_attention = attention_mask.reshape(batch * segments, window).bool()
        flat_valid = segment_valid_mask.reshape(batch * segments)

        # Invalid segment slots are padding introduced by batching. Give the
        # backbone one mechanically safe attended token, then zero all exposed
        # states before the bridge sees them.
        safe_attention = flat_attention.clone()
        if bool((~flat_valid).any()):
            safe_attention[~flat_valid, 0] = True

        outputs = backbone(
            input_ids=flat_ids,
            attention_mask=safe_attention,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = getattr(outputs, "hidden_states", None)
        if hidden is None or len(hidden) != self.config.num_hidden_states:
            raise RuntimeError("semantic backbone hidden-state depth drift")
        stack = torch.stack(tuple(hidden), dim=1)
        expected = (
            batch * segments,
            self.config.num_hidden_states,
            window,
            self.config.semantic_dim,
        )
        if stack.shape != expected:
            raise ValueError(
                f"segmented semantic hidden-state geometry drift: {tuple(stack.shape)} != {expected}"
            )
        stack = stack.reshape(
            batch,
            segments,
            self.config.num_hidden_states,
            window,
            self.config.semantic_dim,
        )
        return stack * segment_valid_mask[:, :, None, None, None].to(
            stack.dtype
        )

    def encode_items(
        self,
        *,
        backbone: Any,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> dict[str, Tensor | bool | int]:
        """Encode N independent semantic text items with no product token ceiling."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [N,T]")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask geometry drift")
        attention_mask = attention_mask.bool()
        if bool((attention_mask.sum(dim=-1) == 0).any()):
            raise ValueError("every semantic item requires at least one attended token")

        lengths = attention_mask.long().sum(dim=-1)
        max_length = int(lengths.max().item())
        if max_length <= self.config.native_window_tokens:
            encoded = self.interface.encode_batch(
                backbone=backbone,
                input_ids=input_ids[:, :max_length],
                attention_mask=attention_mask[:, :max_length],
            )
            return {
                **encoded,
                "used_virtualization": False,
                "segment_count_max": 1,
                "original_lengths": lengths,
            }

        segmented = self.virtualizer.segment(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        segment_hidden = self._encode_segment_batch(
            backbone=backbone,
            input_ids=segmented["segment_input_ids"],
            attention_mask=segmented["segment_attention_mask"],
            segment_valid_mask=segmented["segment_valid_mask"],
        )
        bridged = self.segment_bridge(
            segment_hidden_states=segment_hidden,
            segment_attention_mask=segmented["segment_attention_mask"],
            segment_valid_mask=segmented["segment_valid_mask"],
            segment_metadata=segmented["field_metadata"],
        )
        stitched, stitched_mask = self.virtualizer.stitch_owned_content(
            segment_hidden_states=bridged[
                "contextualized_segment_hidden_states"
            ],
            segmented=segmented,
        )
        width = stitched.size(2)
        original_ids = input_ids[:, :width]
        original_attention = attention_mask[:, :width]
        if not torch.equal(stitched_mask, original_attention):
            raise RuntimeError(
                "virtualized semantic item token ownership drifted from original attention mask"
            )
        content = self._content_mask(
            input_ids=original_ids,
            attention_mask=original_attention,
        )
        return {
            "hidden_states": stitched,
            "content_mask": content,
            "attention_mask": original_attention,
            "used_virtualization": True,
            "segment_count_max": int(
                segmented["segment_valid_mask"].sum(dim=-1).max().item()
            ),
            "original_lengths": lengths,
        }

    def encode_semantic_bank(
        self,
        *,
        backbone: Any,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> DynamicSemanticSchema:
        encoded = self.encode_items(
            backbone=backbone,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        schema = DynamicSemanticSchema(
            token_states=encoded["hidden_states"],
            token_mask=encoded["content_mask"],
        )
        schema.validate(
            num_hidden_states=self.config.num_hidden_states,
            semantic_dim=self.config.semantic_dim,
        )
        return schema

    def encode_relation_bank(
        self,
        *,
        backbone: Any,
        input_ids: Tensor,
        attention_mask: Tensor,
        domain_type_mask: Tensor,
        range_type_mask: Tensor,
        symmetric: Tensor,
    ) -> DynamicRelationSchema:
        encoded = self.encode_items(
            backbone=backbone,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        schema = DynamicRelationSchema(
            token_states=encoded["hidden_states"],
            token_mask=encoded["content_mask"],
            domain_type_mask=domain_type_mask,
            range_type_mask=range_type_mask,
            symmetric=symmetric,
        )
        schema.validate(
            num_hidden_states=self.config.num_hidden_states,
            semantic_dim=self.config.semantic_dim,
        )
        return schema

    def encode_padded_items(
        self,
        *,
        backbone: Any,
        input_ids: Tensor,
        attention_mask: Tensor,
        item_valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        """Encode [B,N,T] items while keeping padded invalid items exactly inert."""
        if input_ids.ndim != 3:
            raise ValueError("padded semantic items must be [B,N,T]")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("padded semantic attention geometry drift")
        batch, items, _ = input_ids.shape
        if item_valid_mask.shape != (batch, items):
            raise ValueError("item_valid_mask must be [B,N]")
        if item_valid_mask.dtype != torch.bool:
            raise ValueError("item_valid_mask must be bool")
        if bool((item_valid_mask.sum(dim=-1) == 0).any()):
            raise ValueError("every example requires at least one valid semantic item")

        flat_valid = item_valid_mask.reshape(-1)
        flat_ids = input_ids.reshape(batch * items, -1)
        flat_attention = attention_mask.reshape(batch * items, -1).bool()
        selected_ids = flat_ids[flat_valid]
        selected_attention = flat_attention[flat_valid]
        if bool((selected_attention.sum(dim=-1) == 0).any()):
            raise ValueError("valid semantic item has no attended tokens")

        encoded = self.encode_items(
            backbone=backbone,
            input_ids=selected_ids,
            attention_mask=selected_attention,
        )
        hidden_valid = encoded["hidden_states"]
        token_valid = encoded["content_mask"]
        tokens = hidden_valid.size(2)

        hidden = torch.zeros(
            batch * items,
            self.config.num_hidden_states,
            tokens,
            self.config.semantic_dim,
            dtype=hidden_valid.dtype,
            device=hidden_valid.device,
        )
        token_mask = torch.zeros(
            batch * items,
            tokens,
            dtype=torch.bool,
            device=hidden_valid.device,
        )
        hidden[flat_valid] = hidden_valid
        token_mask[flat_valid] = token_valid
        return {
            "hidden_states": hidden.reshape(
                batch,
                items,
                self.config.num_hidden_states,
                tokens,
                self.config.semantic_dim,
            ),
            "token_mask": token_mask.reshape(batch, items, tokens),
            "item_valid_mask": item_valid_mask,
        }

    def parameter_report(self) -> dict[str, Any]:
        bridge = self.segment_bridge.parameter_report()
        return {
            "total_parameters": int(bridge["total_parameters"]),
            "trainable_parameters": int(bridge["trainable_parameters"]),
            "shared_backbone_owned_parameters": 0,
            "single_shared_backbone_reference": True,
            "long_query_supported": True,
            "long_relation_description_supported": True,
            "long_factor_description_supported": True,
            "long_field_supported": True,
            "long_candidate_supported": True,
            "long_descriptor_supported": True,
            "native_window_is_operating_point": True,
            "all_text_surfaces_share_virtualization_policy": True,
            "cross_window_semantic_bridge_required_when_virtualized": True,
            "item_count_dependent_parameters": 0,
            "token_count_dependent_parameters": 0,
            "item_count_ceiling": None,
            "product_context_token_ceiling": None,
            "dense_unbounded_native_attention_equivalence_claimed": False,
            "segment_bridge": bridge,
        }
