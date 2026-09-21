from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class SemanticContextVirtualizerConfig:
    native_window_tokens: int = 4096
    overlap_tokens: int = 256
    pad_token_id: int = 0

    def validate(self) -> None:
        if self.native_window_tokens <= 0:
            raise ValueError("native_window_tokens must be positive")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative")
        if self.overlap_tokens >= self.native_window_tokens:
            raise ValueError("overlap_tokens must be smaller than native window")

    @property
    def stride(self) -> int:
        return self.native_window_tokens - self.overlap_tokens


class SemanticContextVirtualizerV1:
    """Parameter-free lossless product-context virtualization.

    A finite native encoder window is treated as an operating window, not a
    product context ceiling. Arbitrarily long token sequences are partitioned
    into as many overlapping encoder windows as required at runtime.

    attention_mask exposes overlap to the semantic encoder for local boundary
    continuity. content_mask assigns every original token to exactly one
    downstream field, so overlap is never double-counted by structured/fusion
    layers. Cross-window interaction then occurs through the variable-field N0
    structured/evidence/fusion stack.
    """

    def __init__(
        self,
        config: SemanticContextVirtualizerConfig | None = None,
    ) -> None:
        self.config = config or SemanticContextVirtualizerConfig()
        self.config.validate()

    def _starts(self, length: int) -> list[int]:
        if length <= 0:
            return [0]
        window = self.config.native_window_tokens
        stride = self.config.stride
        if length <= window:
            return [0]
        starts = list(range(0, max(length - window, 0) + 1, stride))
        final_start = max(0, length - window)
        if starts[-1] != final_start:
            starts.append(final_start)
        return starts

    def segment(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> dict[str, Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask shape drift")
        if attention_mask.dtype != torch.bool:
            raise ValueError("attention_mask must be bool")
        batch, total_tokens = input_ids.shape
        if total_tokens <= 0:
            raise ValueError("token axis must be non-empty")

        lengths = attention_mask.long().sum(dim=-1)
        if bool((lengths == 0).any()):
            raise ValueError("every example requires at least one valid token")

        starts_by_row = [
            self._starts(int(length.item()))
            for length in lengths
        ]
        segments = max(len(x) for x in starts_by_row)
        window = self.config.native_window_tokens

        segment_ids = torch.full(
            (batch, segments, window),
            fill_value=int(self.config.pad_token_id),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        segment_attention = torch.zeros(
            batch,
            segments,
            window,
            dtype=torch.bool,
            device=input_ids.device,
        )
        content_mask = torch.zeros_like(segment_attention)
        segment_valid = torch.zeros(
            batch,
            segments,
            dtype=torch.bool,
            device=input_ids.device,
        )
        absolute_start = torch.zeros(
            batch,
            segments,
            dtype=torch.long,
            device=input_ids.device,
        )
        absolute_end = torch.zeros_like(absolute_start)
        metadata = torch.zeros(
            batch,
            segments,
            3,
            dtype=torch.float32,
            device=input_ids.device,
        )

        for b, starts in enumerate(starts_by_row):
            length = int(lengths[b].item())
            for index, start in enumerate(starts):
                end = min(start + window, length)
                count = end - start
                segment_ids[b, index, :count] = input_ids[b, start:end]
                segment_attention[b, index, :count] = True
                segment_valid[b, index] = True
                absolute_start[b, index] = start
                absolute_end[b, index] = end

                if index + 1 < len(starts):
                    own_end = starts[index + 1]
                else:
                    own_end = end
                own_start = start
                own_count = max(0, own_end - own_start)
                content_mask[b, index, :own_count] = True

                denom = max(float(length - 1), 1.0)
                metadata[b, index, 0] = float(start) / denom
                metadata[b, index, 1] = float(max(end - 1, start)) / denom
                metadata[b, index, 2] = float(count) / max(float(length), 1.0)

        ownership = torch.zeros(
            batch,
            total_tokens,
            dtype=torch.long,
            device=input_ids.device,
        )
        for b, starts in enumerate(starts_by_row):
            for index, start in enumerate(starts):
                owned = int(content_mask[b, index].sum().item())
                if owned:
                    ownership[b, start : start + owned] += 1
        for b in range(batch):
            length = int(lengths[b].item())
            if not bool((ownership[b, :length] == 1).all()):
                raise RuntimeError(
                    "context virtualization did not assign every valid token exactly once"
                )
            if bool((ownership[b, length:] != 0).any()):
                raise RuntimeError("padding token was assigned content ownership")

        return {
            "segment_input_ids": segment_ids,
            "segment_attention_mask": segment_attention,
            "segment_content_mask": content_mask,
            "segment_valid_mask": segment_valid,
            "segment_absolute_start": absolute_start,
            "segment_absolute_end": absolute_end,
            "field_metadata": metadata,
            "original_lengths": lengths,
        }

    def validate_encoded_segments(
        self,
        *,
        segment_hidden_states: Tensor,
        segmented: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        if segment_hidden_states.ndim != 5:
            raise ValueError(
                "segment_hidden_states must be [B,G,L,W,D]"
            )
        batch, segments, _, window, _ = segment_hidden_states.shape
        expected = segmented["segment_attention_mask"].shape
        if expected != (batch, segments, window):
            raise ValueError("encoded segment geometry drift")
        return {
            "field_hidden_states": segment_hidden_states,
            "field_token_mask": segmented["segment_content_mask"],
            "field_valid_mask": segmented["segment_valid_mask"],
            "field_metadata": segmented["field_metadata"],
            "field_confidence": segmented["segment_valid_mask"].float(),
            "field_missing": torch.zeros(
                batch,
                segments,
                dtype=torch.float32,
                device=segment_hidden_states.device,
            ),
        }

    def parameter_report(self) -> dict[str, Any]:
        return {
            "total_parameters": 0,
            "trainable_parameters": 0,
            "native_window_is_operating_point": True,
            "overlap_is_operating_point": True,
            "segment_count_dependent_parameters": 0,
            "segment_count_ceiling": None,
            "product_context_token_ceiling": None,
            "lossless_content_ownership": True,
        }
