from __future__ import annotations

import torch
from torch import nn


class AliceN0Ranker(nn.Module):
    """Universal candidate ranker on top of the N0 semantic backbone.

    N0 uses this head to teach the operation that the later personality model
    depends on: interpret a context, compare plausible candidates, and select or
    tie the candidate(s) best supported by the input. The head is generic; N1/N2
    can later replace public semantic candidates with governed identity candidates.
    """

    def __init__(self, backbone: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.scorer(pooled).squeeze(-1)


def build_ranker_from_mlm_checkpoint(checkpoint: str, hidden_size: int) -> AliceN0Ranker:
    try:
        from transformers import ModernBertModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install requirements-n0.txt before curriculum training") from exc

    backbone = ModernBertModel.from_pretrained(checkpoint)
    return AliceN0Ranker(backbone=backbone, hidden_size=hidden_size)


def listwise_preference_loss(
    scores: torch.Tensor,
    group_sizes: list[int],
    preferred_masks: list[torch.Tensor],
) -> torch.Tensor:
    """Cross-entropy against a uniform distribution over all valid preferred ties."""
    losses: list[torch.Tensor] = []
    offset = 0
    for size, preferred in zip(group_sizes, preferred_masks):
        group_scores = scores[offset : offset + size]
        offset += size
        preferred = preferred.to(device=group_scores.device, dtype=torch.bool)
        if preferred.numel() != size or not preferred.any():
            raise ValueError("each candidate group must contain at least one preferred candidate")
        log_probs = torch.log_softmax(group_scores, dim=0)
        losses.append(-log_probs[preferred].mean())
    return torch.stack(losses).mean()
