from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import IterableDataset


def stable_document_split(text_sha256: str) -> str:
    bucket = int(text_sha256[:8], 16) % 10_000
    if bucket < 10:
        return "test"
    if bucket < 20:
        return "dev"
    return "train"


class PackedJSONLIterableDataset(IterableDataset):
    """Read normalized JSONL and pack documents into fixed-length sequences.

    ``shuffle_seed`` is optional so legacy N0 behavior remains unchanged. When
    supplied, all rows in the requested split are ordered by a deterministic
    hash of ``seed:text_sha256`` before packing. This is important for bounded
    v0.2 tranches: the materialized shard paths are grouped by source, so simply
    stopping after a few million tokens would otherwise expose only the first
    alphabetical source families instead of a representative slice of the
    already-balanced 21-source corpus.
    """

    def __init__(
        self,
        paths: Sequence[str | Path],
        tokenizer: Any,
        sequence_length: int,
        split: str = "train",
        shuffle_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.paths = [Path(path) for path in paths]
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.split = split
        self.shuffle_seed = shuffle_seed

    def _source_rows(self) -> Iterator[dict[str, Any]]:
        for path in self.paths:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row_split = row.get("split") or stable_document_split(row["text_sha256"])
                    if row_split == self.split:
                        yield row

    def _rows(self) -> Iterator[dict[str, Any]]:
        if self.shuffle_seed is None:
            yield from self._source_rows()
            return

        keyed: list[tuple[str, dict[str, Any]]] = []
        prefix = f"{self.shuffle_seed}:"
        for row in self._source_rows():
            text_sha = str(row["text_sha256"])
            order_key = hashlib.sha256((prefix + text_sha).encode("utf-8")).hexdigest()
            keyed.append((order_key, row))
        keyed.sort(key=lambda item: item[0])
        for _, row in keyed:
            yield row

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        buffer: list[int] = [self.tokenizer.cls_token_id]
        sep_id = self.tokenizer.sep_token_id

        for row in self._rows():
            ids = self.tokenizer.encode(row["text"], add_special_tokens=False)
            if not ids:
                continue
            ids.append(sep_id)

            offset = 0
            while offset < len(ids):
                room = self.sequence_length - len(buffer)
                if room == 0:
                    yield self._emit(buffer)
                    buffer = [self.tokenizer.cls_token_id]
                    room = self.sequence_length - 1
                take = min(room, len(ids) - offset)
                buffer.extend(ids[offset : offset + take])
                offset += take

        if len(buffer) > 1:
            yield self._emit(buffer)

    def _emit(self, ids: list[int]) -> dict[str, torch.Tensor]:
        attention = [1] * len(ids)
        if len(ids) < self.sequence_length:
            pad = self.sequence_length - len(ids)
            ids = ids + [self.tokenizer.pad_token_id] * pad
            attention += [0] * pad
        return {
            "input_ids": torch.tensor(ids[: self.sequence_length], dtype=torch.long),
            "attention_mask": torch.tensor(attention[: self.sequence_length], dtype=torch.long),
        }


class SpanMLMCollator:
    """Dynamic contiguous-span masking with BERT-style 80/10/10 replacement."""

    def __init__(
        self,
        tokenizer: Any,
        mlm_probability: float = 0.15,
        mean_span: float = 3.0,
        max_span: int = 10,
        seed: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mean_span = mean_span
        self.max_span = max_span
        self.rng = random.Random(seed)
        self.special_ids = {
            tokenizer.pad_token_id,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.mask_token_id,
        }

    def _sample_span_length(self) -> int:
        p = min(1.0, max(1e-6, 1.0 / self.mean_span))
        length = 1
        while length < self.max_span and self.rng.random() > p:
            length += 1
        return length

    def _mask_positions(self, ids: torch.Tensor, attention: torch.Tensor) -> list[int]:
        candidates = [
            index
            for index, (token_id, visible) in enumerate(zip(ids.tolist(), attention.tolist()))
            if visible and token_id not in self.special_ids
        ]
        if not candidates:
            return []

        target = max(1, round(len(candidates) * self.mlm_probability))
        candidate_set = set(candidates)
        masked: set[int] = set()
        attempts = 0
        max_attempts = max(32, target * 8)

        while len(masked) < target and attempts < max_attempts:
            start = self.rng.choice(candidates)
            span = self._sample_span_length()
            for index in range(start, min(start + span, len(ids))):
                if index in candidate_set:
                    masked.add(index)
                    if len(masked) >= target:
                        break
            attempts += 1

        if len(masked) < target:
            remaining = [index for index in candidates if index not in masked]
            self.rng.shuffle(remaining)
            masked.update(remaining[: target - len(masked)])
        return sorted(masked)

    def __call__(self, examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = torch.stack([example["input_ids"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        labels = torch.full_like(input_ids, -100)

        for row_index in range(input_ids.size(0)):
            positions = self._mask_positions(input_ids[row_index], attention_mask[row_index])
            for position in positions:
                original = input_ids[row_index, position].item()
                labels[row_index, position] = original
                roll = self.rng.random()
                if roll < 0.8:
                    input_ids[row_index, position] = self.tokenizer.mask_token_id
                elif roll < 0.9:
                    input_ids[row_index, position] = self.rng.randrange(len(self.tokenizer))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
