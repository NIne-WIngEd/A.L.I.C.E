from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class N0BuildConfig:
    model_id: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    train_sequence_lengths: tuple[int, ...]
    local_attention: int
    global_attention_every: int
    attention_dropout: float
    embedding_dropout: float
    mlp_dropout: float
    tokenizer_type: str
    tokenizer_vocab_size: int
    mlm_probability: float
    mean_mask_span: float
    max_mask_span: int
    pad_token_id: int
    unk_token_id: int
    cls_token_id: int
    sep_token_id: int
    mask_token_id: int

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "N0BuildConfig":
        architecture = raw["architecture"]
        tokenizer = raw["tokenizer"]
        training = raw["training"]
        attention = architecture["attention_pattern"]
        special = tokenizer.get(
            "special_token_ids",
            {"pad": 0, "unk": 1, "cls": 2, "sep": 3, "mask": 4},
        )
        sequence_lengths = training.get("sequence_lengths", [512, 1024, 2048, 4096])
        return cls(
            model_id=raw["model_id"],
            vocab_size=int(tokenizer["vocab_size"]),
            hidden_size=int(architecture["hidden_size"]),
            num_hidden_layers=int(architecture["num_hidden_layers"]),
            num_attention_heads=int(architecture["num_attention_heads"]),
            intermediate_size=int(architecture["intermediate_size"]),
            max_position_embeddings=int(
                architecture.get(
                    "max_position_embeddings",
                    architecture.get("long_context_continuation_target_if_needed", 8192),
                )
            ),
            train_sequence_lengths=tuple(int(x) for x in sequence_lengths),
            local_attention=int(attention["local_window_tokens"]),
            global_attention_every=int(attention["global_every_n_layers"]),
            attention_dropout=float(architecture.get("attention_dropout", 0.0)),
            embedding_dropout=float(architecture.get("embedding_dropout", 0.0)),
            mlp_dropout=float(architecture.get("mlp_dropout", 0.0)),
            tokenizer_type=str(tokenizer["type"]),
            tokenizer_vocab_size=int(tokenizer["vocab_size"]),
            mlm_probability=float(training.get("mlm_probability", 0.15)),
            mean_mask_span=float(training.get("mean_mask_span", 3.0)),
            max_mask_span=int(training.get("max_mask_span", 10)),
            pad_token_id=int(special["pad"]),
            unk_token_id=int(special["unk"]),
            cls_token_id=int(special["cls"]),
            sep_token_id=int(special["sep"]),
            mask_token_id=int(special["mask"]),
        )

    def validate(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.vocab_size != self.tokenizer_vocab_size:
            raise ValueError("model and tokenizer vocab sizes must match")
        if self.global_attention_every < 1:
            raise ValueError("global_attention_every must be >= 1")
        if self.local_attention < 1:
            raise ValueError("local_attention must be >= 1")
        if max(self.train_sequence_lengths) > self.max_position_embeddings:
            raise ValueError("training sequence length exceeds max_position_embeddings")
        special_ids = {
            self.pad_token_id,
            self.unk_token_id,
            self.cls_token_id,
            self.sep_token_id,
            self.mask_token_id,
        }
        if len(special_ids) != 5:
            raise ValueError("special token ids must be distinct")
        if min(special_ids) < 0 or max(special_ids) >= self.vocab_size:
            raise ValueError("special token id outside vocabulary")
        if not 0.0 < self.mlm_probability < 1.0:
            raise ValueError("mlm_probability must be between zero and one")
        if self.mean_mask_span <= 0 or self.max_mask_span < 1:
            raise ValueError("invalid span masking parameters")


def load_n0_config(path: str | Path) -> N0BuildConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    config = N0BuildConfig.from_dict(raw)
    config.validate()
    return config
