"""Calibrated fresh embeddings, unchanged full specialist architecture.

This versioned construction candidate applies the actual-runtime causal
intervention to fresh token/position embeddings only. No source weights or
language head are imported, and no output is masked or repaired.
"""
import torch
from .formation_v1_specialist import FormationSpecialist

INITIALIZATION = 'fresh-token-position-scale-0.02-v1'


class FormationDecisionSpecialist(FormationSpecialist):
    def __init__(self, config):
        super().__init__(config)
        with torch.no_grad():
            self.token_embedding.weight.mul_(0.02)
            self.position_embedding.weight.mul_(0.02)
