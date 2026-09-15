from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from alice_memory.temporal import MemoryRelation

from .structured_state import relation_type_id


@dataclass(frozen=True)
class CompiledEvidenceGraph:
    """One padded public relation graph ready for ``StructuredStateEncoder``.

    Tensor shapes include a leading batch dimension of one so the result can
    be passed directly to the encoder or concatenated by a future collator.
    Node semantics and private plaintext are intentionally outside this object.
    """

    edge_index: torch.Tensor
    edge_type_ids: torch.Tensor
    edge_confidence: torch.Tensor
    edge_valid_mask: torch.Tensor
    relation_ids: tuple[str, ...]

    def as_model_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "edge_index": self.edge_index,
            "edge_type_ids": self.edge_type_ids,
            "edge_confidence": self.edge_confidence,
            "edge_valid_mask": self.edge_valid_mask,
        }


def compile_memory_relation_graph(
    *,
    memory_ids: Sequence[str],
    relations: Sequence[MemoryRelation],
    max_edges: int,
    relation_confidence: Mapping[str, float] | None = None,
    device: torch.device | str | None = None,
) -> CompiledEvidenceGraph:
    """Compile a deterministic graph slice from authoritative memory relations.

    Only relations whose two endpoints are present in ``memory_ids`` are part
    of the graph slice. Relations are sorted by ``relation_id`` so database or
    retrieval ordering cannot leak into model behavior. The compiler refuses
    overflow instead of silently truncating evidence.
    """
    if max_edges < 1:
        raise ValueError("max_edges must be positive")
    if not memory_ids:
        raise ValueError("memory_ids must contain at least one node")
    if any(not memory_id.strip() for memory_id in memory_ids):
        raise ValueError("memory_ids may not contain empty identifiers")
    if len(set(memory_ids)) != len(memory_ids):
        raise ValueError("memory_ids must be unique")

    node_index = {memory_id: index for index, memory_id in enumerate(memory_ids)}
    confidence_by_relation = relation_confidence or {}

    selected: list[tuple[MemoryRelation, float]] = []
    for relation in relations:
        if relation.from_memory_id not in node_index or relation.to_memory_id not in node_index:
            continue
        confidence = float(confidence_by_relation.get(relation.relation_id, 1.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"relation confidence must be in [0, 1] for {relation.relation_id!r}"
            )
        selected.append((relation, confidence))

    selected.sort(key=lambda item: item[0].relation_id)
    if len(selected) > max_edges:
        raise ValueError(
            f"selected relation count {len(selected)} exceeds max_edges={max_edges}; "
            "evidence may not be silently truncated"
        )

    edge_index = torch.zeros((1, max_edges, 2), dtype=torch.long, device=device)
    edge_type_ids = torch.zeros((1, max_edges), dtype=torch.long, device=device)
    edge_confidence = torch.zeros((1, max_edges, 1), dtype=torch.float32, device=device)
    edge_valid_mask = torch.zeros((1, max_edges), dtype=torch.bool, device=device)

    relation_ids: list[str] = []
    for edge_number, (relation, confidence) in enumerate(selected):
        edge_index[0, edge_number, 0] = node_index[relation.from_memory_id]
        edge_index[0, edge_number, 1] = node_index[relation.to_memory_id]
        edge_type_ids[0, edge_number] = relation_type_id(relation.relation_type)
        edge_confidence[0, edge_number, 0] = confidence
        edge_valid_mask[0, edge_number] = True
        relation_ids.append(relation.relation_id)

    return CompiledEvidenceGraph(
        edge_index=edge_index,
        edge_type_ids=edge_type_ids,
        edge_confidence=edge_confidence,
        edge_valid_mask=edge_valid_mask,
        relation_ids=tuple(relation_ids),
    )
