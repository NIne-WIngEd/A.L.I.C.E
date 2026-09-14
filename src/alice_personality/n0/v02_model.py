from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .config import N0BuildConfig
from .model import build_masked_lm


class AliceN0V02Model(nn.Module):
    """Native N0 v0.2 semantic representation model.

    One random-initialized ModernBERT-family encoder supports four public N0
    learning signals:
      1. span MLM for broad language/context modeling,
      2. candidate preference scoring,
      3. candidate/rationale compatibility,
      4. semantic/rationale contrastive geometry.

    The rationale and ranking heads are generic semantic-learning heads. They do
    not authorize or contain private Elaina identity gradients.
    """

    def __init__(self, config: N0BuildConfig, projection_size: int = 256) -> None:
        super().__init__()
        self.config = config
        self.mlm = build_masked_lm(config)
        self.preference_scorer = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1),
        )
        self.semantic_projection = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, projection_size),
        )
        self.rationale_projection = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, projection_size),
        )
        self.principle_scale = nn.Parameter(torch.tensor(1.0))
        self.principle_bias = nn.Parameter(torch.tensor(0.0))

    @property
    def backbone(self) -> nn.Module:
        model = getattr(self.mlm, "model", None)
        if model is None:
            raise RuntimeError("ModernBertForMaskedLM no longer exposes .model backbone")
        return model

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self._mean_pool(outputs.last_hidden_state, attention_mask)

    def forward_mlm(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Any:
        return self.mlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def score_candidates(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.encode(input_ids, attention_mask)
        return self.preference_scorer(pooled).squeeze(-1)

    def project_semantic(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return F.normalize(self.semantic_projection(self.encode(input_ids, attention_mask)), dim=-1)

    def project_rationale(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return F.normalize(self.rationale_projection(self.encode(input_ids, attention_mask)), dim=-1)

    def principle_alignment_logits(
        self,
        semantic_input_ids: torch.Tensor,
        semantic_attention_mask: torch.Tensor,
        rationale_input_ids: torch.Tensor,
        rationale_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        semantic = self.project_semantic(semantic_input_ids, semantic_attention_mask)
        rationale = self.project_rationale(rationale_input_ids, rationale_attention_mask)
        if semantic.shape != rationale.shape:
            raise ValueError("semantic and rationale batches must align one-to-one")
        similarity = (semantic * rationale).sum(dim=-1)
        return similarity * self.principle_scale.exp() + self.principle_bias

    def parameter_report(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        heads = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith("mlm.")
        )
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "auxiliary_head_parameters": heads,
        }
