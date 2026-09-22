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
    """Parameter-free token-lossless native-window virtualization.

    A finite native encoder window is treated as an operating window, not a
    storage or routing ceiling. Arbitrarily long token sequences are
    partitioned into as many overlapping encoder windows as required.

    This class guarantees unique downstream token ownership and overlap-aware
    local encoding. It does *not* by itself make independently encoded windows
    semantically equivalent to one arbitrarily long native transformer pass.
    Long query/schema semantics therefore require the trainable
    SemanticSegmentContextBridgeV1 (or an explicit structured-field route)
    before stitched token states are treated as globally contextualized.
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
        expected_prefix = (
            torch.arange(total_tokens, device=input_ids.device)[None, :]
            < lengths[:, None]
        )
        if not torch.equal(attention_mask, expected_prefix):
            raise ValueError(
                "context virtualizer requires right-padded contiguous token masks"
            )

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
        content_absolute_start = torch.zeros_like(absolute_start)
        content_absolute_end = torch.zeros_like(absolute_start)
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

                # Split every overlap at its midpoint. Earlier windows own
                # the left half; later windows own the right half. This keeps
                # unique ownership while avoiding the old defect where an
                # overlap token was always retained from the later window and
                # therefore could lose useful left context.
                if index == 0:
                    own_start = start
                else:
                    previous_start = starts[index - 1]
                    previous_end = min(previous_start + window, length)
                    own_start = (start + previous_end) // 2

                if index + 1 < len(starts):
                    next_start = starts[index + 1]
                    own_end = (next_start + end) // 2
                else:
                    own_end = end

                if not (start <= own_start <= own_end <= end):
                    raise RuntimeError("invalid overlap ownership partition")
                local_own_start = own_start - start
                local_own_end = own_end - start
                content_mask[
                    b,
                    index,
                    local_own_start:local_own_end,
                ] = True
                content_absolute_start[b, index] = own_start
                content_absolute_end[b, index] = own_end

                denom = max(float(length - 1), 1.0)
                metadata[b, index, 0] = float(own_start) / denom
                metadata[b, index, 1] = float(max(own_end - 1, own_start)) / denom
                metadata[b, index, 2] = float(own_end - own_start) / max(float(length), 1.0)

        ownership = torch.zeros(
            batch,
            total_tokens,
            dtype=torch.long,
            device=input_ids.device,
        )
        for b, starts in enumerate(starts_by_row):
            for index, start in enumerate(starts):
                local = torch.nonzero(
                    content_mask[b, index],
                    as_tuple=False,
                ).flatten()
                if local.numel():
                    absolute = local + int(start)
                    ownership[b, absolute] += 1
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
            "content_absolute_start": content_absolute_start,
            "content_absolute_end": content_absolute_end,
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

    def stitch_owned_content(
        self,
        *,
        segment_hidden_states: Tensor,
        segmented: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Reconstruct unique downstream token states from encoded segments.

        This is intended for long queries/schema text that exceed one native
        encoder window. Overlap participates in local encoding but only the
        unique owned token state is retained downstream.
        """
        if segment_hidden_states.ndim != 5:
            raise ValueError(
                "segment_hidden_states must be [B,G,L,W,D]"
            )
        batch, segments, layers, window, width = segment_hidden_states.shape
        if segmented["segment_content_mask"].shape != (
            batch,
            segments,
            window,
        ):
            raise ValueError("segment content-mask geometry drift")
        lengths = segmented["original_lengths"]
        if lengths.shape != (batch,):
            raise ValueError("original_lengths shape drift")
        max_tokens = int(lengths.max().item())
        stitched = torch.zeros(
            batch,
            layers,
            max_tokens,
            width,
            dtype=segment_hidden_states.dtype,
            device=segment_hidden_states.device,
        )
        stitched_mask = torch.zeros(
            batch,
            max_tokens,
            dtype=torch.bool,
            device=segment_hidden_states.device,
        )

        for b in range(batch):
            cursor = 0
            for g in range(segments):
                if not bool(segmented["segment_valid_mask"][b, g]):
                    continue
                owned = segmented["segment_content_mask"][b, g]
                count = int(owned.sum().item())
                if count == 0:
                    continue
                values = segment_hidden_states[b, g, :, owned, :]
                stitched[b, :, cursor : cursor + count, :] = values
                stitched_mask[b, cursor : cursor + count] = True
                cursor += count
            if cursor != int(lengths[b].item()):
                raise RuntimeError(
                    "stitched query content length does not match original token length"
                )

        return stitched, stitched_mask

    def parameter_report(self) -> dict[str, Any]:
        return {
            "total_parameters": 0,
            "trainable_parameters": 0,
            "native_window_is_operating_point": True,
            "overlap_is_operating_point": True,
            "segment_count_dependent_parameters": 0,
            "segment_count_ceiling": None,
            "product_context_token_ceiling": None,
            "token_ownership_lossless": True,
            "lossless_content_ownership": True,
            "long_query_stitching": True,
            "standalone_cross_window_semantics_complete": False,
            "segment_context_bridge_required_for_long_query_semantics": True,
            "dense_unbounded_native_attention_equivalence_claimed": False,
            "overlap_midpoint_ownership": True,
            "boundary_context_bias_to_later_window": False,
        }
