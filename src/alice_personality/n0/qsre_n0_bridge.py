from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class QSREN0BridgeOutput:
    evidence_tokens: Tensor
    evidence_mask: Tensor
    relational_semantic_token: Tensor
    relational_summary_token: Tensor
    relational_token_valid: Tensor
    relational_execution_confidence: Tensor
    relational_field_distribution: Tensor


class QSREN0EvidenceBridge(nn.Module):
    """Zero-parameter bridge from QSRE structural readout into N0 evidence view.

    The bridge does not replace or rewrite the ratified evidence stream. It
    appends exactly one derived semantic token whose content is the calibrated
    QSRE field distribution applied to the original semantic field states.

    This deliberately avoids a learned adapter bottleneck between a proven
    relational decision and the existing N0 fusion/latent fabric. When QSRE is
    non-applicable or fail-closed, the appended slot is masked and the original
    evidence tokens are byte-for-byte unchanged.
    """

    def __init__(
        self,
        semantic_dim: int = 640,
        activation_threshold: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if semantic_dim <= 0:
            raise ValueError("semantic_dim must be positive")
        if not 0.0 <= activation_threshold < 1.0:
            raise ValueError("activation_threshold must be in [0,1)")
        self.semantic_dim = int(semantic_dim)
        self.activation_threshold = float(activation_threshold)

    def forward(
        self,
        *,
        base_evidence_tokens: Tensor,
        base_evidence_mask: Tensor,
        field_semantic_state: Tensor,
        field_valid_mask: Tensor,
        relational_probability: Tensor,
        relational_summary: Tensor | None = None,
    ) -> QSREN0BridgeOutput:
        if base_evidence_tokens.ndim != 3:
            raise ValueError("base_evidence_tokens must be [B,E,D]")
        batch, _, width = base_evidence_tokens.shape
        if width != self.semantic_dim:
            raise ValueError("base evidence semantic width drift")
        if base_evidence_mask.shape != base_evidence_tokens.shape[:2]:
            raise ValueError("base_evidence_mask shape drift")
        if base_evidence_mask.dtype != torch.bool:
            raise ValueError("base_evidence_mask must be bool")
        if field_semantic_state.ndim != 3:
            raise ValueError("field_semantic_state must be [B,F,D]")
        if field_semantic_state.size(0) != batch or field_semantic_state.size(2) != width:
            raise ValueError("field semantic shape drift")
        if field_valid_mask.shape != field_semantic_state.shape[:2]:
            raise ValueError("field_valid_mask shape drift")
        if field_valid_mask.dtype != torch.bool:
            raise ValueError("field_valid_mask must be bool")
        if relational_probability.shape != field_semantic_state.shape[:2]:
            raise ValueError("relational_probability shape drift")
        if not bool(torch.isfinite(base_evidence_tokens).all()):
            raise ValueError("base evidence tokens contain nonfinite values")
        if not bool(torch.isfinite(field_semantic_state).all()):
            raise ValueError("field semantic state contains nonfinite values")
        if not bool(torch.isfinite(relational_probability).all()):
            raise ValueError("relational probability contains nonfinite values")
        if bool((relational_probability < -1.0e-8).any()):
            raise ValueError("relational probability must be nonnegative")
        if relational_summary is not None:
            if relational_summary.shape != (batch, self.semantic_dim):
                raise ValueError(
                    "relational_summary must match N0 semantic width; "
                    "Production QSRE is configured at the full semantic width"
                )
            if not bool(torch.isfinite(relational_summary).all()):
                raise ValueError("relational_summary contains nonfinite values")

        valid_probability = (
            relational_probability
            * field_valid_mask.to(relational_probability.dtype)
        )
        execution_confidence = valid_probability.sum(dim=-1).clamp(
            min=0.0,
            max=1.0,
        )
        normalized = torch.where(
            execution_confidence[:, None] > 0,
            valid_probability
            / execution_confidence[:, None].clamp_min(1.0e-12),
            torch.zeros_like(valid_probability),
        )
        semantic_token = torch.einsum(
            "bf,bfd->bd",
            normalized,
            field_semantic_state,
        )
        # The QSRE executor has already calibrated relation applicability,
        # control, UNKNOWN and program-presence mass. Do not introduce another
        # arbitrary 0.5 router at the N0 boundary. Preserve that confidence in
        # the token itself and mask only exact/nearly-exact zero execution.
        token = semantic_token * execution_confidence.unsqueeze(-1)
        token_valid = execution_confidence > self.activation_threshold

        # Preserve two complementary relational signals when the executor
        # exposes its native summary: (1) a source-grounded semantic answer
        # token and (2) the executor's relation/path/operator state. Both carry
        # the same calibrated execution confidence and neither replaces the
        # exact parent evidence tokens.
        if relational_summary is None:
            summary_token = torch.zeros_like(token)
            extra_tokens = token.unsqueeze(1)
            extra_mask = token_valid.unsqueeze(1)
        else:
            summary_token = (
                relational_summary * execution_confidence.unsqueeze(-1)
            )
            extra_tokens = torch.stack([token, summary_token], dim=1)
            extra_mask = torch.stack([token_valid, token_valid], dim=1)

        augmented_tokens = torch.cat(
            [base_evidence_tokens, extra_tokens],
            dim=1,
        )
        augmented_mask = torch.cat(
            [base_evidence_mask, extra_mask],
            dim=1,
        )

        return QSREN0BridgeOutput(
            evidence_tokens=augmented_tokens,
            evidence_mask=augmented_mask,
            relational_semantic_token=token,
            relational_summary_token=summary_token,
            relational_token_valid=token_valid,
            relational_execution_confidence=execution_confidence,
            relational_field_distribution=normalized,
        )

    def parameter_report(self) -> dict[str, object]:
        return {
            "total_parameters": 0,
            "trainable_parameters": 0,
            "base_evidence_tokens_preserved_exactly": True,
            "nonrelational_pass_through_exact": True,
            "relational_token_is_grounded_in_original_field_semantics": True,
            "native_relational_summary_channel_supported": True,
            "native_relational_summary_requires_full_semantic_width": True,
            "learned_adapter_bottleneck": False,
            "field_count_ceiling": None,
            "support_count_ceiling": None,
            "candidate_count_ceiling": None,
            "activation_threshold": self.activation_threshold,
            "activation_threshold_role": "numeric_zero_tolerance_only_not_a_second_router_or_model_capacity_ceiling",
            "relational_token_scales_with_calibrated_execution_confidence": True,
            "hard_relational_confidence_gate": False,
        }
