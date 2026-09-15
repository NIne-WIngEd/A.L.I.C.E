from __future__ import annotations

import torch

from alice_personality.n0.evidence_view_adapter import (
    EvidenceViewAdapter,
    EvidenceViewAdapterConfig,
)


def tiny_adapter() -> EvidenceViewAdapter:
    torch.manual_seed(17)
    model = EvidenceViewAdapter(
        EvidenceViewAdapterConfig(
            semantic_size=32,
            adapter_size=16,
            num_layers=1,
            num_heads=4,
            feedforward_size=64,
            dropout=0.0,
            max_fields=16,
        )
    )
    model.eval()
    return model


def make_inputs(fields: int = 5):
    torch.manual_seed(19)
    return {
        "parent_field_states": torch.randn(1, fields, 32),
        "valid_mask": torch.ones(1, fields, dtype=torch.bool),
        "query_semantic": torch.randn(1, 32),
        "parent_field_weights": torch.softmax(torch.randn(1, fields), dim=-1),
    }


def test_consistent_field_permutation_preserves_adapter_read() -> None:
    model = tiny_adapter()
    batch = make_inputs()
    order = torch.tensor([3, 0, 4, 1, 2])
    permuted = {
        "parent_field_states": batch["parent_field_states"][:, order, :],
        "valid_mask": batch["valid_mask"][:, order],
        "query_semantic": batch["query_semantic"],
        "parent_field_weights": batch["parent_field_weights"][:, order],
    }

    with torch.inference_mode():
        original = model(**batch)
        changed = model(**permuted)

    torch.testing.assert_close(
        original["field_states"][:, order, :],
        changed["field_states"],
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        original["field_weights"][:, order],
        changed["field_weights"],
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        original["pooled_state"],
        changed["pooled_state"],
        atol=1e-5,
        rtol=1e-5,
    )


def test_query_changes_specialized_view_without_mutating_parent_tensor() -> None:
    model = tiny_adapter()
    batch = make_inputs()
    parent_before = batch["parent_field_states"].clone()

    with torch.inference_mode():
        first = model(**batch)
        changed_batch = dict(batch)
        changed_batch["query_semantic"] = torch.arange(32, dtype=torch.float32).unsqueeze(0)
        second = model(**changed_batch)

    torch.testing.assert_close(batch["parent_field_states"], parent_before)
    assert not torch.allclose(first["field_weights"], second["field_weights"])
    assert not torch.allclose(first["field_states"], second["field_states"])


def test_padding_content_does_not_change_valid_outputs() -> None:
    model = tiny_adapter()
    batch = make_inputs(fields=4)
    batch["valid_mask"][:, -1] = False
    changed = {key: value.clone() for key, value in batch.items()}
    changed["parent_field_states"][:, -1, :] = 999.0
    changed["parent_field_weights"][:, -1] = 100.0

    with torch.inference_mode():
        first = model(**batch)
        second = model(**changed)

    torch.testing.assert_close(
        first["field_states"][:, :-1, :],
        second["field_states"][:, :-1, :],
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(first["pooled_state"], second["pooled_state"], atol=1e-5, rtol=1e-5)
    assert float(first["field_weights"][0, -1]) == 0.0
    assert float(second["field_weights"][0, -1]) == 0.0


def test_parameter_report_has_no_architecture_ceiling() -> None:
    report = EvidenceViewAdapter().parameter_report()
    assert report["total_parameters"] > 0
    assert report["hard_parameter_ceiling"] is None
    assert report["private_identity_parameters"] == 0
    assert report["position_embeddings"] == 0
