from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class CurriculumDataset(Dataset):
    def __init__(self, paths: str | Path | Sequence[str | Path], split: str) -> None:
        if isinstance(paths, (str, Path)):
            path_list = [Path(paths)]
        else:
            path_list = [Path(path) for path in paths]
        if not path_list:
            raise ValueError("at least one curriculum path is required")

        self.rows: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for path in path_list:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row_id = str(row.get("id", "")).strip()
                    if not row_id:
                        raise ValueError(f"curriculum row in {path} is missing id")
                    if row_id in seen_ids:
                        raise ValueError(f"duplicate curriculum id across shards: {row_id}")
                    seen_ids.add(row_id)
                    if row.get("split", "train") == split:
                        self.rows.append(row)
        if not self.rows:
            raise ValueError(f"no curriculum rows found for split={split}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class CurriculumCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        prompts: list[str] = []
        candidates: list[str] = []
        group_sizes: list[int] = []
        preferred_masks: list[torch.Tensor] = []
        ids: list[str] = []
        competencies: list[str] = []

        for row in rows:
            row_candidates = list(row["candidates"])
            preferred = set(int(x) for x in row["preferred_indices"])
            if not row_candidates or not preferred:
                raise ValueError(f"invalid curriculum row {row.get('id')}")
            if max(preferred) >= len(row_candidates) or min(preferred) < 0:
                raise ValueError(f"preferred index out of range in {row.get('id')}")

            ids.append(str(row["id"]))
            competencies.append(str(row["competency"]))
            group_sizes.append(len(row_candidates))
            preferred_masks.append(
                torch.tensor(
                    [index in preferred for index in range(len(row_candidates))],
                    dtype=torch.bool,
                )
            )
            prompts.extend([str(row["prompt"])] * len(row_candidates))
            candidates.extend(str(candidate) for candidate in row_candidates)

        encoded = self.tokenizer(
            prompts,
            candidates,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "group_sizes": group_sizes,
            "preferred_masks": preferred_masks,
            "ids": ids,
            "competencies": competencies,
        }


def load_tokenizer(tokenizer_dir: str | Path):
    try:
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install requirements-n0.txt before curriculum work") from exc

    tokenizer_path = Path(tokenizer_dir) / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"tokenizer.json not found under {tokenizer_dir}")
    return PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


def score_group(
    group_scores: torch.Tensor,
    preferred: torch.Tensor,
) -> dict[str, Any]:
    preferred = preferred.to(device=group_scores.device, dtype=torch.bool)
    if preferred.numel() != group_scores.numel() or not preferred.any():
        raise ValueError("invalid preferred mask")

    top_index = int(torch.argmax(group_scores).item())
    top_supported = bool(preferred[top_index].item())
    nonpreferred = ~preferred

    if nonpreferred.any():
        min_preferred = group_scores[preferred].min()
        max_nonpreferred = group_scores[nonpreferred].max()
        margin = float((min_preferred - max_nonpreferred).detach().cpu())
        supported_set_separated = bool((min_preferred > max_nonpreferred).item())
    else:
        margin = 0.0
        supported_set_separated = True

    return {
        "predicted_top_index": top_index,
        "top_supported": top_supported,
        "supported_set_separated": supported_set_separated,
        "separation_margin": margin,
    }
