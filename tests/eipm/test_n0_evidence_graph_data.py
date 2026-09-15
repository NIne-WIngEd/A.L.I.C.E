from __future__ import annotations

import torch

from alice_memory.temporal import MemoryRelation
from alice_personality.n0.evidence_graph_data import compile_memory_relation_graph
from alice_personality.n0.structured_state import EvidenceRelationType


def relation(
    relation_id: str,
    source: str,
    target: str,
    relation_type: str,
) -> MemoryRelation:
    return MemoryRelation(
        relation_id=relation_id,
        from_memory_id=source,
        to_memory_id=target,
        relation_type=relation_type,
        created_at="2026-09-15T00:00:00Z",
    )


def test_compiler_uses_authoritative_memory_relation_direction() -> None:
    graph = compile_memory_relation_graph(
        memory_ids=["new", "old", "other"],
        relations=[
            relation("r2", "new", "old", "supersedes"),
            relation("r1", "other", "old", "conflicts_with"),
        ],
        max_edges=4,
    )

    assert graph.relation_ids == ("r1", "r2")
    torch.testing.assert_close(
        graph.edge_index[0, :2],
        torch.tensor([[2, 1], [0, 1]]),
    )
    torch.testing.assert_close(
        graph.edge_type_ids[0, :2],
        torch.tensor(
            [
                int(EvidenceRelationType.CONFLICTS_WITH),
                int(EvidenceRelationType.SUPERSEDES),
            ]
        ),
    )
    assert graph.edge_valid_mask.tolist() == [[True, True, False, False]]


def test_external_relations_are_not_included_in_graph_slice() -> None:
    graph = compile_memory_relation_graph(
        memory_ids=["a", "b"],
        relations=[
            relation("inside", "a", "b", "corrects"),
            relation("outside", "a", "c", "supports"),
        ],
        max_edges=3,
    )

    assert graph.relation_ids == ("inside",)
    assert graph.edge_valid_mask.tolist() == [[True, False, False]]
    assert int(graph.edge_type_ids[0, 0]) == int(EvidenceRelationType.CORRECTS)


def test_relation_confidence_is_explicit_and_bounded() -> None:
    graph = compile_memory_relation_graph(
        memory_ids=["a", "b"],
        relations=[relation("r1", "a", "b", "supports")],
        relation_confidence={"r1": 0.25},
        max_edges=1,
    )
    assert float(graph.edge_confidence[0, 0, 0]) == 0.25

    try:
        compile_memory_relation_graph(
            memory_ids=["a", "b"],
            relations=[relation("r1", "a", "b", "supports")],
            relation_confidence={"r1": 1.5},
            max_edges=1,
        )
    except ValueError as exc:
        assert "relation confidence" in str(exc)
    else:
        raise AssertionError("out-of-range relation confidence must be rejected")


def test_compiler_refuses_silent_edge_truncation() -> None:
    relations = [
        relation("r1", "a", "b", "supports"),
        relation("r2", "b", "c", "supports"),
    ]
    try:
        compile_memory_relation_graph(
            memory_ids=["a", "b", "c"],
            relations=relations,
            max_edges=1,
        )
    except ValueError as exc:
        assert "may not be silently truncated" in str(exc)
    else:
        raise AssertionError("graph overflow must fail closed")


def test_duplicate_memory_ids_are_rejected() -> None:
    try:
        compile_memory_relation_graph(
            memory_ids=["a", "a"],
            relations=[],
            max_edges=1,
        )
    except ValueError as exc:
        assert "must be unique" in str(exc)
    else:
        raise AssertionError("duplicate memory ids must be rejected")
