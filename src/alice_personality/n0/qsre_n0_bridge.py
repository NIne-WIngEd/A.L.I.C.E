from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class QSREN0BridgeOutput:
    evidence_tokens: Tensor
    evidence_mask: Tensor
    relational_semantic_token: Tensor
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

    def __init__(self, semantic_dim: int = 640) -> None:
        super().__init__()
        if semantic_dim <= 0:
            raise ValueError("semantic_dim must be positive")
        self.semantic_dim = int(semantic_dim)

    def forward(
        self,
        *,
        base_evidence_tokens: Tensor,
        base_evidence_mask: Tensor,
        field_semantic_state: Tensor,
        field_valid_mask: Tensor,
        relational_probability: Tensor,
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
        token = torch.einsum(
            "bf,bfd->bd",
            normalized,
            field_semantic_state,
        )
        token_valid = execution_confidence > 0

        augmented_tokens = torch.cat(
            [base_evidence_tokens, token.unsqueeze(1)],
            dim=1,
        )
        augmented_mask = torch.cat(
            [base_evidence_mask, token_valid.unsqueeze(1)],
            dim=1,
        )

        return QSREN0BridgeOutput(
            evidence_tokens=augmented_tokens,
            evidence_mask=augmented_mask,
            relational_semantic_token=token,
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
            "learned_adapter_bottleneck": False,
            "field_count_ceiling": None,
            "support_count_ceiling": None,
            "candidate_count_ceiling": None,
        }
