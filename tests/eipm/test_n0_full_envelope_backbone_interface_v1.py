from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from alice_personality.n0.semantic_backbone_interface_v1 import (
    FullEnvelopeSemanticBackboneInterfaceV1,
    SemanticBackboneInterfaceConfig,
)


class FakeBackbone(nn.Module):
    def __init__(self, vocab: int = 32, dim: int = 12, layers: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(layers - 1)]
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
        return_dict: bool,
    ):
        assert output_hidden_states is True
        assert return_dict is True
        x = self.embedding(input_ids)
        hidden = [x]
        for layer in self.layers:
            x = torch.tanh(layer(x))
            hidden.append(x)
        return SimpleNamespace(hidden_states=tuple(hidden))


def interface() -> FullEnvelopeSemanticBackboneInterfaceV1:
    return FullEnvelopeSemanticBackboneInterfaceV1(
        SemanticBackboneInterfaceConfig(
            num_hidden_states=4,
            semantic_dim=12,
            special_token_ids=(0,1,2,3,4),
        )
    )


def test_backbone_interface_preserves_all_layers_and_gradient() -> None:
    torch.manual_seed(301)
    backbone = FakeBackbone()
    adapter = interface()
    ids = torch.tensor(
        [
            [2,5,6,7,3,0],
            [2,8,9,10,3,0],
        ]
    )
    attention = ids.ne(0)
    out = adapter.encode_batch(
        backbone=backbone,
        input_ids=ids,
        attention_mask=attention,
    )
    assert out["hidden_states"].shape == (2,4,6,12)
    assert out["content_mask"].tolist() == [
        [False,True,True,True,False,False],
        [False,True,True,True,False,False],
    ]
    out["hidden_states"].square().mean().backward()
    assert backbone.embedding.weight.grad is not None
    assert float(backbone.embedding.weight.grad.abs().sum()) > 0.0
    report = adapter.parameter_report()
    assert report["owned_parameters"] == 0
    assert report["all_hidden_states_preserved"] is True
    assert report["final_layer_only"] is False
    assert report["pooled_only"] is False
    assert report["semantic_backbone_gradient_detached"] is False


def test_backbone_interface_builds_runtime_relation_schema_without_id_semantics() -> None:
    torch.manual_seed(302)
    backbone = FakeBackbone()
    adapter = interface()
    ids = torch.tensor(
        [
            [2,5,6,3,0],
            [2,7,8,3,0],
            [2,9,10,3,0],
        ]
    )
    attention = ids.ne(0)
    relation = adapter.encode_relation_bank(
        backbone=backbone,
        input_ids=ids,
        attention_mask=attention,
        domain_type_mask=torch.ones(3,5,dtype=torch.bool),
        range_type_mask=torch.ones(3,5,dtype=torch.bool),
        symmetric=torch.tensor([False,True,False]),
    )
    assert relation.token_states.shape == (3,4,5,12)
    assert relation.token_mask.shape == (3,5)
    assert relation.domain_type_mask.shape == (3,5)
    assert relation.symmetric.tolist() == [False,True,False]


def test_backbone_interface_padded_fields_are_zero_and_inert() -> None:
    torch.manual_seed(303)
    backbone = FakeBackbone()
    adapter = interface()
    ids = torch.tensor(
        [
            [
                [2,5,6,3,0],
                [2,7,8,3,0],
                [0,0,0,0,0],
            ],
            [
                [2,9,10,3,0],
                [0,0,0,0,0],
                [0,0,0,0,0],
            ],
        ]
    )
    attention = ids.ne(0)
    valid = torch.tensor(
        [
            [True,True,False],
            [True,False,False],
        ]
    )
    out = adapter.encode_fields(
        backbone=backbone,
        input_ids=ids,
        attention_mask=attention,
        field_valid_mask=valid,
    )
    assert out["field_hidden_states"].shape == (2,3,4,5,12)
    assert torch.equal(
        out["field_hidden_states"][~valid],
        torch.zeros_like(out["field_hidden_states"][~valid]),
    )
    assert torch.equal(
        out["field_token_mask"][~valid],
        torch.zeros_like(out["field_token_mask"][~valid]),
    )


def test_backbone_interface_fails_closed_when_special_tokens_are_all_content() -> None:
    backbone = FakeBackbone()
    adapter = interface()
    ids = torch.tensor([[2,3,0,0]])
    attention = ids.ne(0)
    try:
        adapter.encode_batch(
            backbone=backbone,
            input_ids=ids,
            attention_mask=attention,
        )
    except ValueError as exc:
        assert "content tokens" in str(exc)
    else:
        raise AssertionError("special-token-only semantic item did not fail closed")
