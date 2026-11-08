"""Unqualified first-party semantic correction above a measured fixed path.

All upstream banks and the original scorer implementation stay immutable.
Preserving initial scores is not a guarantee of retention after training, nor
suppression of inherited persona. No identity judgment authority is supplied.
"""
from __future__ import annotations
import torch
from .semantic_readout import SemanticReadout, fixed_semantic_scores


class AnchoredSemanticReadout(SemanticReadout):
    """Fixed semantic scores plus a shared, initially zero learned correction."""

    def __init__(self, *, state_count: int, hidden_size: int, width: int = 64):
        super().__init__(state_count=state_count, hidden_size=hidden_size, width=width)
        with torch.no_grad():
            self.scorer[-1].weight.zero_()
            self.scorer[-1].bias.zero_()

    def forward(self, source, candidates):
        fixed = fixed_semantic_scores(source, candidates)
        return fixed + super().forward(source, candidates)
