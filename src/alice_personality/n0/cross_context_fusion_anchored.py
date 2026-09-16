from __future__ import annotations

from typing import Any

import torch

from alice_personality.n0.cross_context_fusion import (
    CrossContextFusion,
    CrossContextFusionConfig,
)


class SourceAnchoredCrossContextFusion(CrossContextFusion):
    """Full-scale N0 fusion with an explicit non-destructive source channel.

    The original full-scale fusion correctly keeps heterogeneous token streams
    separate and learns contextualized view summaries.  The first untouched
    challenge showed that using those contextualized summaries *also* as the
    only preservation interface conflates two different requirements:

    1. fusion must be free to contextualize a view; and
    2. downstream identity reasoning must never lose the ratified parent view.

    This migration keeps the learned contextualized summaries unchanged and
    adds the exact parent summaries as first-class outputs.  No information is
    discarded and no capacity ceiling is introduced.  Later adaptive pooling
    may consume either or both channels.
    """

    def __init__(self, config: CrossContextFusionConfig | None = None) -> None:
        super().__init__(config)

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(tokens.dtype).unsqueeze(-1)
        return (tokens * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    def _resolve_source_summaries(
        self,
        *,
        view_tokens: list[torch.Tensor],
        view_valid_masks: list[torch.Tensor],
        source_view_summaries: torch.Tensor | None,
        available: torch.Tensor,
    ) -> torch.Tensor:
        batch = view_tokens[0].shape[0]
        if source_view_summaries is None:
            source = torch.stack(
                [
                    self._masked_mean(tokens, mask)
                    for tokens, mask in zip(view_tokens, view_valid_masks)
                ],
                dim=1,
            )
        else:
            expected = (batch, self.config.num_views, self.config.semantic_size)
            if source_view_summaries.shape != expected:
                raise ValueError(
                    "source_view_summaries must have shape "
                    f"[batch, {self.config.num_views}, {self.config.semantic_size}]"
                )
            if not torch.isfinite(source_view_summaries).all():
                raise ValueError("source_view_summaries contains non-finite values")
            source = source_view_summaries

        # Missing views must not smuggle stale parent state into later stages.
        return torch.where(
            available.unsqueeze(-1),
            source,
            torch.zeros_like(source),
        )

    def forward_views(
        self,
        *,
        view_tokens: list[torch.Tensor],
        view_valid_masks: list[torch.Tensor],
        query_semantic: torch.Tensor,
        view_reliability: torch.Tensor,
        source_view_summaries: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        result = super().forward_views(
            view_tokens=view_tokens,
            view_valid_masks=view_valid_masks,
            query_semantic=query_semantic,
            view_reliability=view_reliability,
        )
        contextualized = result["view_summaries"]
        source = self._resolve_source_summaries(
            view_tokens=view_tokens,
            view_valid_masks=view_valid_masks,
            source_view_summaries=source_view_summaries,
            available=result["view_available"],
        )

        # Backward-compatible alias: view_summaries remains the learned fusion
        # representation.  The source anchor is now explicit instead of being
        # approximated through a cosine penalty.
        result["contextualized_view_summaries"] = contextualized
        result["source_view_summaries"] = source
        return result

    def forward(
        self,
        *,
        semantic_tokens: torch.Tensor,
        semantic_valid_mask: torch.Tensor,
        structured_tokens: torch.Tensor,
        structured_valid_mask: torch.Tensor,
        evidence_tokens: torch.Tensor,
        evidence_valid_mask: torch.Tensor,
        query_semantic: torch.Tensor,
        view_reliability: torch.Tensor,
        source_view_summaries: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if self.config.num_views != 3:
            raise ValueError(
                "named semantic/structured/evidence forward is for the three-view N0 checkpoint; "
                "use forward_views for an expanded checkpoint"
            )
        result = self.forward_views(
            view_tokens=[semantic_tokens, structured_tokens, evidence_tokens],
            view_valid_masks=[
                semantic_valid_mask,
                structured_valid_mask,
                evidence_valid_mask,
            ],
            query_semantic=query_semantic,
            view_reliability=view_reliability,
            source_view_summaries=source_view_summaries,
        )
        contextualized_tokens = result["contextualized_view_tokens"]
        result["semantic_tokens"] = contextualized_tokens[self.SEMANTIC_VIEW]
        result["structured_tokens"] = contextualized_tokens[self.STRUCTURED_VIEW]
        result["evidence_tokens"] = contextualized_tokens[self.EVIDENCE_VIEW]
        return result

    def parameter_report(self) -> dict[str, Any]:
        report = super().parameter_report()
        report.update(
            {
                "fusion_family": (
                    "multi_stream_self_refinement_plus_gated_bidirectional_cross_attention"
                    "_with_explicit_source_anchors"
                ),
                "explicit_source_view_anchor_channel": True,
                "source_anchor_parameter_cost": 0,
                "contextualized_and_source_views_both_exposed": True,
                "source_anchor_capacity_ceiling": None,
                "full_scale_n0_candidate": True,
                "reduced_pilot_model": False,
            }
        )
        return report
