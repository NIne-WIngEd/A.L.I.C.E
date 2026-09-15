from __future__ import annotations

import torch

from alice_personality.n0.structured_state import StructuredStateConfig, StructuredStateEncoder


def make_batch(*, batch: int = 2, fields: int = 5, semantic_size: int = 32):
    torch.manual_seed(7)
    return {
        "semantic_values": torch.randn(batch, fields, semantic_size),
        "field_type_ids": torch.randint(1, 8, (batch, fields)),
        "provenance_ids": torch.randint(1, 5, (batch, fields)),
        "relation_role_ids": torch.randint(1, 8, (batch, fields)),
        "temporal_scope_ids": torch.randint(1, 5, (batch, fields)),
        "confidence": torch.rand(batch, fields, 1),
        "missing_mask": torch.zeros(batch, fields, dtype=torch.bool),
        "valid_mask": torch.ones(batch, fields, dtype=torch.bool),
    }


def tiny_model() -> StructuredStateEncoder:
    config = StructuredStateConfig(
        semantic_size=32,
        state_size=16,
        num_layers=1,
        num_heads=4,
        feedforward_size=32,
        num_field_types=16,
        num_provenance_classes=8,
        num_relation_roles=16,
        num_temporal_scopes=8,
        dropout=0.0,
        max_fields=16,
    )
    model = StructuredStateEncoder(config)
    model.eval()
    return model


def permute_fields(batch: dict[str, torch.Tensor], order: torch.Tensor):
    permuted = {}
    for key, value in batch.items():
        if value.ndim >= 2:
            permuted[key] = value[:, order, ...]
        else:
            permuted[key] = value
    return permuted


def test_pooled_state_is_field_order_invariant() -> None:
    model = tiny_model()
    batch = make_batch()
    order = torch.tensor([3, 0, 4, 1, 2])

    with torch.inference_mode():
        original = model(**batch)
        permuted = model(**permute_fields(batch, order))

    torch.testing.assert_close(original["pooled_state"], permuted["pooled_state"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        original["field_states"][:, order, :],
        permuted["field_states"],
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        original["field_weights"][:, order],
        permuted["field_weights"],
        atol=1e-5,
        rtol=1e-5,
    )


def test_padding_fields_do_not_change_valid_state() -> None:
    model = tiny_model()
    batch = make_batch(batch=1)
    batch["valid_mask"][:, -1] = False

    changed = {key: value.clone() for key, value in batch.items()}
    changed["semantic_values"][:, -1, :] = 999.0
    changed["field_type_ids"][:, -1] = 15
    changed["provenance_ids"][:, -1] = 7
    changed["relation_role_ids"][:, -1] = 15
    changed["temporal_scope_ids"][:, -1] = 7
    changed["confidence"][:, -1, :] = 1.0
    changed["missing_mask"][:, -1] = True

    with torch.inference_mode():
        original = model(**batch)
        mutated_padding = model(**changed)

    torch.testing.assert_close(original["pooled_state"], mutated_padding["pooled_state"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        original["field_states"][:, :-1, :],
        mutated_padding["field_states"][:, :-1, :],
        atol=1e-5,
        rtol=1e-5,
    )
    assert float(original["field_weights"][0, -1]) == 0.0
    assert float(mutated_padding["field_weights"][0, -1]) == 0.0


def test_all_padding_is_rejected() -> None:
    model = tiny_model()
    batch = make_batch(batch=1)
    batch["valid_mask"].zero_()

    try:
        model(**batch)
    except ValueError as exc:
        assert "at least one valid structured field" in str(exc)
    else:
        raise AssertionError("all-padding structured state must be rejected")


def test_parameter_report_is_descriptive_not_a_capacity_gate() -> None:
    model = StructuredStateEncoder()
    report = model.parameter_report()
    assert report["total_parameters"] > 0
    assert report["trainable_parameters"] == report["total_parameters"]
    assert report["private_identity_parameters"] == 0
    assert report["position_embeddings"] == 0
