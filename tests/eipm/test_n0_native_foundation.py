from __future__ import annotations

import json
from pathlib import Path

import pytest

from alice_personality.n0.config import N0BuildConfig


CONFIG = Path("configs/eipm/n0/alice_n0_semantic_v0.1.json")


def test_n0_config_contract() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    config = N0BuildConfig.from_dict(raw)
    config.validate()

    assert config.model_id == "alice-n0-semantic-v0.1"
    assert config.vocab_size == 48_000
    assert config.hidden_size == 896
    assert config.num_hidden_layers == 24
    assert config.num_attention_heads == 14
    assert config.max_position_embeddings == 8192
    assert max(config.train_sequence_lengths) == 4096


def test_n0_forbids_private_identity_text_in_tokenizer() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert raw["tokenizer"]["private_identity_text_in_tokenizer_training"] is False
    assert raw["architecture"]["third_party_weight_initialization"] is False
    assert raw["data"]["private_identity_data_stage"] == "N1_or_later"


def test_modernbert_random_init_parameter_band() -> None:
    pytest.importorskip("transformers")
    from alice_personality.n0.config import load_n0_config
    from alice_personality.n0.model import build_masked_lm, count_parameters

    model = build_masked_lm(load_n0_config(CONFIG))
    total, trainable = count_parameters(model)
    assert 300_000_000 <= total <= 400_000_000
    assert trainable == total
