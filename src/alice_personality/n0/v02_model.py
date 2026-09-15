from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .config import N0BuildConfig
from .model import build_masked_lm


class AliceN0V02Model(nn.Module):
    """Native N0 v0.2 semantic representation model.

    One random-initialized ModernBERT-family encoder supports span MLM,
    candidate preference, rationale compatibility, and semantic/rationale
    contrastive learning. These heads remain public semantic machinery and do
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

    def score_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.preference_scorer(pooled).squeeze(-1)

    def project_semantic_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.semantic_projection(pooled), dim=-1)

    def project_rationale_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.rationale_projection(pooled), dim=-1)

    def principle_alignment_from_projected(
        self,
        semantic: torch.Tensor,
        rationale: torch.Tensor,
    ) -> torch.Tensor:
        if semantic.shape != rationale.shape:
            raise ValueError("semantic and rationale projected batches must align one-to-one")
        similarity = (semantic * rationale).sum(dim=-1)
        return similarity * self.principle_scale.exp() + self.principle_bias

    def score_candidates(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.score_pooled(self.encode(input_ids, attention_mask))

    def project_semantic(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.project_semantic_pooled(self.encode(input_ids, attention_mask))

    def project_rationale(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.project_rationale_pooled(self.encode(input_ids, attention_mask))

    def principle_alignment_logits(
        self,
        semantic_input_ids: torch.Tensor,
        semantic_attention_mask: torch.Tensor,
        rationale_input_ids: torch.Tensor,
        rationale_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        semantic = self.project_semantic(semantic_input_ids, semantic_attention_mask)
        rationale = self.project_rationale(rationale_input_ids, rationale_attention_mask)
        return self.principle_alignment_from_projected(semantic, rationale)

    def forward(
        self,
        *,
        task: str,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        candidate_input_ids: torch.Tensor | None = None,
        candidate_attention_mask: torch.Tensor | None = None,
        rationale_input_ids: torch.Tensor | None = None,
        rationale_attention_mask: torch.Tensor | None = None,
        candidate_rationale_index: torch.Tensor | None = None,
    ) -> Any:
        """DDP-safe forward entry point for both public N0 v0.2 paths."""
        if task == "mlm":
            if input_ids is None or attention_mask is None or labels is None:
                raise ValueError("task=mlm requires input_ids, attention_mask, and labels")
            return self.forward_mlm(input_ids, attention_mask, labels)

        if task == "teacher":
            if (
                candidate_input_ids is None
                or candidate_attention_mask is None
                or rationale_input_ids is None
                or rationale_attention_mask is None
                or candidate_rationale_index is None
            ):
                raise ValueError("task=teacher requires candidate/rationale tensors and index map")
            if candidate_input_ids.size(1) != rationale_input_ids.size(1):
                raise ValueError("teacher candidate/rationale sequence widths must match")

            candidate_count = candidate_input_ids.size(0)
            joined_ids = torch.cat([candidate_input_ids, rationale_input_ids], dim=0)
            joined_mask = torch.cat([candidate_attention_mask, rationale_attention_mask], dim=0)
            pooled = self.encode(joined_ids, joined_mask)
            candidate_pooled = pooled[:candidate_count]
            rationale_pooled = pooled[candidate_count:]
            semantic = self.project_semantic_pooled(candidate_pooled)
            rationale = self.project_rationale_pooled(rationale_pooled)
            rationale_for_candidate = rationale[candidate_rationale_index]
            return {
                "scores": self.score_pooled(candidate_pooled),
                "semantic": semantic,
                "rationale": rationale,
                "alignment_logits": self.principle_alignment_from_projected(
                    semantic, rationale_for_candidate
                ),
            }

        raise ValueError(f"unsupported N0 v0.2 task: {task!r}")

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
