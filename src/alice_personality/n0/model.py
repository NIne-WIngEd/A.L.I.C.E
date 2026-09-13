from __future__ import annotations

from typing import Any

from .config import N0BuildConfig


def _layer_types(config: N0BuildConfig) -> list[str]:
    return [
        "full_attention" if index % config.global_attention_every == 0 else "sliding_attention"
        for index in range(config.num_hidden_layers)
    ]


def build_masked_lm(config: N0BuildConfig) -> Any:
    """Create Alice's N0 model from random initialization only.

    Hugging Face supplies tested ModernBERT mechanics. No pretrained
    checkpoint or inherited third-party weights are loaded here.
    """
    try:
        from transformers import ModernBertConfig, ModernBertForMaskedLM
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "N0 requires transformers with ModernBERT support. Install requirements-n0.txt."
        ) from exc

    hf_config = ModernBertConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        max_position_embeddings=config.max_position_embeddings,
        pad_token_id=config.pad_token_id,
        bos_token_id=config.cls_token_id,
        eos_token_id=config.sep_token_id,
        cls_token_id=config.cls_token_id,
        sep_token_id=config.sep_token_id,
        attention_bias=False,
        attention_dropout=config.attention_dropout,
        layer_types=_layer_types(config),
        local_attention=config.local_attention,
        embedding_dropout=config.embedding_dropout,
        mlp_bias=False,
        mlp_dropout=config.mlp_dropout,
        decoder_bias=True,
        tie_word_embeddings=True,
        sparse_prediction=False,
    )
    return ModernBertForMaskedLM(hf_config)


def count_parameters(model: Any) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
