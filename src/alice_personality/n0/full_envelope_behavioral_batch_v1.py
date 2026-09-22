from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

INTERNAL_VIEW_NAMES = (
    "raw_semantic",
    "structured",
    "evidence_specialist",
    "graph_source",
    "graph_target",
    "relational_executor",
)
INTERNAL_VIEW_DESCRIPTIONS = (
    "raw semantic field evidence",
    "structured contextual field representation",
    "query-conditioned evidence-specialist view",
    "graph source-endpoint summary",
    "graph target-endpoint summary",
    "relational executor summary",
)
EVENT_INDEX = {"CONTINUE": 0, "STOP": 1, "UNKNOWN": 2}


@dataclass(frozen=True)
class BehavioralBatchCompileConfig:
    graph_message_steps: int = 3
    fusion_refinement_steps: int = 2
    latent_slot_count: int = 8
    latent_refinement_steps: int = 2

    def validate(self) -> None:
        for name, value in (
            ("graph_message_steps", self.graph_message_steps),
            ("fusion_refinement_steps", self.fusion_refinement_steps),
            ("latent_slot_count", self.latent_slot_count),
            ("latent_refinement_steps", self.latent_refinement_steps),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")


def _tokenize(tokenizer: Any, texts: Sequence[str]) -> tuple[Tensor, Tensor]:
    if not texts:
        raise ValueError("semantic text list cannot be empty")
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"].bool()


def _tokenize_nested(
    tokenizer: Any,
    rows: Sequence[Sequence[str]],
    *,
    pad_text: str = "",
) -> tuple[Tensor, Tensor, Tensor]:
    if not rows:
        raise ValueError("nested semantic rows cannot be empty")
    width = max(len(row) for row in rows)
    if width <= 0:
        raise ValueError("every nested semantic batch needs at least one item")
    valid = torch.zeros(len(rows), width, dtype=torch.bool)
    flat: list[str] = []
    for b, row in enumerate(rows):
        for i in range(width):
            if i < len(row):
                flat.append(str(row[i]))
                valid[b, i] = True
            else:
                flat.append(pad_text)
    ids, mask = _tokenize(tokenizer, flat)
    return (
        ids.reshape(len(rows), width, ids.size(1)),
        mask.reshape(len(rows), width, mask.size(1)),
        valid,
    )


def _canonical_definitions(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    key_field: str = "key",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    ordered: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        for item_raw in row[field]:
            item = dict(item_raw)
            key = str(item[key_field])
            prior = by_key.get(key)
            if prior is None:
                by_key[key] = item
                ordered.append(item)
            elif prior != item:
                raise ValueError(f"inconsistent {field} definition for {key!r}")
    if not ordered:
        raise ValueError(f"{field} cannot be empty")
    return ordered, {str(item[key_field]): i for i, item in enumerate(ordered)}


def _factor_schema(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, int]],
    dict[str, list[str]],
]:
    first = {
        str(name): [dict(x) for x in items]
        for name, items in rows[0]["factor_schemas"].items()
    }
    canonical = repr(first)
    for row in rows[1:]:
        value = {
            str(name): [dict(x) for x in items]
            for name, items in row["factor_schemas"].items()
        }
        if repr(value) != canonical:
            raise ValueError("factor schema banks must be identical inside one compiled batch")

    key_index: dict[str, dict[str, int]] = {}
    factor_opcodes: dict[str, list[str]] = {}
    for name, items in first.items():
        keys = [str(x["key"]) for x in items]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate factor candidate key in {name!r}")
        key_index[name] = {key: i for i, key in enumerate(keys)}
        opcodes = [x.get("opcode") for x in items]
        non_null = [x is not None for x in opcodes]
        if any(non_null) and not all(non_null):
            raise ValueError(
                f"factor bank {name!r} mixes executable and semantic-only candidates"
            )
        if all(non_null):
            factor_opcodes[name] = [str(x) for x in opcodes]
    return first, key_index, factor_opcodes


def _descriptor_banks(
    rows: Sequence[Mapping[str, Any]],
    type_items: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
    banks: dict[str, list[str]] = {
        "type": [str(x["text"]) for x in type_items],
        "provenance": [],
        "temporal_scope": [],
    }
    seen = {name: set(values) for name, values in banks.items()}
    for row in rows:
        for item in row["fields"]:
            descriptors = item["descriptors"]
            for name in ("provenance", "temporal_scope"):
                text = str(descriptors[name])
                if text not in seen[name]:
                    banks[name].append(text)
                    seen[name].add(text)
    index = {
        name: {text: i for i, text in enumerate(values)}
        for name, values in banks.items()
    }
    return banks, index


def _structural_type_compatibility(
    *,
    rows: Sequence[Mapping[str, Any]],
    relation_index: Mapping[str, int],
    type_index: Mapping[str, int],
    max_edges: int,
) -> Tensor:
    result = torch.zeros(len(rows), max_edges, dtype=torch.bool)
    relation_defs: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        for relation in row["relation_candidates"]:
            relation_defs[str(relation["key"])] = relation
    for b, row in enumerate(rows):
        fields = row["fields"]
        for e, edge in enumerate(row["edges"]):
            if not bool(edge.get("valid", True)):
                continue
            relation = relation_defs[str(edge["relation_key"])]
            source_type = str(fields[int(edge["source_field"])]["type_key"])
            target_type = str(fields[int(edge["target_field"])]["type_key"])
            forward = (
                source_type in set(map(str, relation["domain"]))
                and target_type in set(map(str, relation["range"]))
            )
            reverse = (
                bool(relation["symmetric"])
                and target_type in set(map(str, relation["domain"]))
                and source_type in set(map(str, relation["range"]))
            )
            result[b, e] = forward or reverse
    return result


def _clone_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            out[key] = value.clone()
        elif isinstance(value, dict):
            out[key] = {
                k: (v.clone() if isinstance(v, Tensor) else copy.deepcopy(v))
                for k, v in value.items()
            }
        else:
            out[key] = copy.deepcopy(value)
    return out


def _remove_fields_variant(
    *,
    primary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    field_index_key: str,
) -> dict[str, Any]:
    batch = _clone_batch(primary)
    field_valid = batch["field_valid_mask"]
    edges = batch["edge_index"]
    edge_valid = batch["edge_valid_mask"]
    for b, row in enumerate(rows):
        removed = {int(x) for x in row[field_index_key]}
        for field_index in removed:
            field_valid[b, field_index] = False
            batch["field_confidence"][b, field_index] = 0.0
            batch["field_missing"][b, field_index] = 1.0
            batch["field_reliability"][b, field_index] = 0.0
            batch["field_metadata"][b, field_index] = 0.0
            batch["field_type_index"][b, field_index] = -1
            for values in batch["descriptor_indices"].values():
                values[b, field_index] = -1
        for e in range(edge_valid.size(1)):
            if not bool(edge_valid[b, e]):
                continue
            source = int(edges[b, e, 0])
            target = int(edges[b, e, 1])
            if source in removed or target in removed:
                edge_valid[b, e] = False
                batch["edge_reliability"][b, e] = 0.0
                batch["edge_recency"][b, e] = 0.0
                batch["edge_temporal_match"][b, e] = 0.0
                batch["edge_provenance_match"][b, e] = 0.0
                batch["edge_metadata"][b, e] = 0.0
    return batch


def _permuted_batch(
    *,
    primary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    batch = _clone_batch(primary)
    batch_size, fields = primary["field_valid_mask"].shape
    field_tensor_keys = (
        "field_input_ids",
        "field_attention_mask",
        "field_valid_mask",
        "field_confidence",
        "field_missing",
        "field_reliability",
        "field_type_index",
        "field_metadata",
    )
    for b, row in enumerate(rows):
        order = torch.tensor(row["field_permutation"], dtype=torch.long)
        if order.shape != (fields,):
            raise ValueError(
                "compiled rows in one batch must have field permutations over padded field geometry"
            )
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(fields)
        for key in field_tensor_keys:
            value = batch[key]
            value[b] = primary[key][b].index_select(0, order)
        for name, value in batch["descriptor_indices"].items():
            value[b] = primary["descriptor_indices"][name][b].index_select(
                0, order
            )
        valid_edges = primary["edge_valid_mask"][b]
        edge_value = primary["edge_index"][b].clone()
        if bool(valid_edges.any()):
            edge_value[valid_edges] = inverse[edge_value[valid_edges]]
        batch["edge_index"][b] = edge_value
    return batch


def compile_behavioral_batch(
    *,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    config: BehavioralBatchCompileConfig | None = None,
) -> dict[str, Any]:
    cfg = config or BehavioralBatchCompileConfig()
    cfg.validate()
    if not rows:
        raise ValueError("behavioral batch cannot be empty")
    batch_size = len(rows)

    type_items, type_index = _canonical_definitions(rows, "type_schema")
    relation_items, relation_index = _canonical_definitions(
        rows, "relation_candidates"
    )
    factor_banks, factor_key_index, factor_opcodes = _factor_schema(rows)
    descriptor_banks, descriptor_text_index = _descriptor_banks(
        rows, type_items
    )

    query_ids, query_mask = _tokenize(
        tokenizer, [str(row["query"]) for row in rows]
    )
    relation_ids, relation_mask = _tokenize(
        tokenizer, [str(x["text"]) for x in relation_items]
    )
    factor_input_ids: dict[str, Tensor] = {}
    factor_attention_mask: dict[str, Tensor] = {}
    factor_candidate_masks: dict[str, Tensor] = {}
    for name, items in factor_banks.items():
        ids, mask = _tokenize(tokenizer, [str(x["text"]) for x in items])
        factor_input_ids[name] = ids
        factor_attention_mask[name] = mask
        factor_candidate_masks[name] = torch.ones(
            batch_size, len(items), dtype=torch.bool
        )

    relation_candidate_mask = torch.zeros(
        batch_size, len(relation_items), dtype=torch.bool
    )
    for b, row in enumerate(rows):
        for item in row["relation_candidates"]:
            relation_candidate_mask[b, relation_index[str(item["key"])]] = True
        if not bool(relation_candidate_mask[b].any()):
            raise ValueError("every row requires one active relation candidate")

    relation_domain = torch.zeros(
        len(relation_items), len(type_items), dtype=torch.bool
    )
    relation_range = torch.zeros_like(relation_domain)
    relation_symmetric = torch.zeros(len(relation_items), dtype=torch.bool)
    for r, item in enumerate(relation_items):
        for key in item["domain"]:
            relation_domain[r, type_index[str(key)]] = True
        for key in item["range"]:
            relation_range[r, type_index[str(key)]] = True
        relation_symmetric[r] = bool(item["symmetric"])

    field_text_rows = [
        [str(x["text"]) for x in row["fields"]] for row in rows
    ]
    field_ids, field_attention, field_valid = _tokenize_nested(
        tokenizer, field_text_rows
    )
    max_fields = field_ids.size(1)
    field_confidence = torch.zeros(batch_size, max_fields)
    field_missing = torch.ones(batch_size, max_fields)
    field_reliability = torch.zeros(batch_size, max_fields)
    field_metadata = torch.zeros(batch_size, max_fields, 3)
    field_type_index = torch.full(
        (batch_size, max_fields), -1, dtype=torch.long
    )
    descriptor_indices = {
        name: torch.full(
            (batch_size, max_fields), -1, dtype=torch.long
        )
        for name in descriptor_banks
    }
    for b, row in enumerate(rows):
        for i, item in enumerate(row["fields"]):
            field_confidence[b, i] = float(item["confidence"])
            field_missing[b, i] = float(item["missing"])
            field_reliability[b, i] = float(item["reliability"])
            field_metadata[b, i] = torch.tensor(
                item["metadata"], dtype=torch.float32
            )
            field_type_index[b, i] = type_index[str(item["type_key"])]
            descriptor_indices["type"][b, i] = type_index[
                str(item["type_key"])
            ]
            for name in ("provenance", "temporal_scope"):
                descriptor_indices[name][b, i] = descriptor_text_index[
                    name
                ][str(item["descriptors"][name])]

    descriptor_input_ids: dict[str, Tensor] = {}
    descriptor_attention_mask: dict[str, Tensor] = {}
    for name, texts in descriptor_banks.items():
        ids, mask = _tokenize(tokenizer, texts)
        descriptor_input_ids[name] = ids
        descriptor_attention_mask[name] = mask

    max_edges = max(len(row["edges"]) for row in rows)
    edge_index = torch.zeros(batch_size, max_edges, 2, dtype=torch.long)
    edge_relation_index = torch.zeros(
        batch_size, max_edges, dtype=torch.long
    )
    edge_valid = torch.zeros(batch_size, max_edges, dtype=torch.bool)
    edge_metadata = torch.zeros(batch_size, max_edges, 4)
    edge_reliability = torch.zeros(batch_size, max_edges)
    edge_recency = torch.zeros(batch_size, max_edges)
    edge_temporal = torch.zeros(batch_size, max_edges)
    edge_provenance = torch.zeros(batch_size, max_edges)
    support_target = torch.zeros(batch_size, max_edges)
    for b, row in enumerate(rows):
        for e, item in enumerate(row["edges"]):
            edge_index[b, e] = torch.tensor(
                [item["source_field"], item["target_field"]],
                dtype=torch.long,
            )
            edge_relation_index[b, e] = relation_index[
                str(item["relation_key"])
            ]
            edge_valid[b, e] = bool(item.get("valid", True))
            edge_metadata[b, e] = torch.tensor(
                item["metadata"], dtype=torch.float32
            )
            edge_reliability[b, e] = float(item["reliability"])
            edge_recency[b, e] = float(item["recency"])
            edge_temporal[b, e] = float(item["temporal_match"])
            edge_provenance[b, e] = float(item["provenance_match"])
            support_target[b, e] = float(bool(item["support_target"]))

    candidate_rows = [
        [str(x) for x in row["candidate_answers"]] for row in rows
    ]
    candidate_ids, candidate_attention, candidate_valid = _tokenize_nested(
        tokenizer, candidate_rows
    )

    row_internal_descriptions=[
        row.get("internal_view_descriptions")
        for row in rows
    ]
    if any(value is not None for value in row_internal_descriptions):
        if not all(value is not None for value in row_internal_descriptions):
            raise ValueError(
                "internal view descriptor override must be supplied for every row in the compiled batch"
            )
        internal_descriptions=[
            str(x) for x in row_internal_descriptions[0]
        ]
        if len(internal_descriptions)!=len(INTERNAL_VIEW_NAMES):
            raise ValueError("internal view descriptor override count drift")
        for value in row_internal_descriptions[1:]:
            if [str(x) for x in value] != internal_descriptions:
                raise ValueError(
                    "internal view descriptor overrides must be identical inside one compiled batch"
                )
    else:
        internal_descriptions=list(INTERNAL_VIEW_DESCRIPTIONS)
    internal_ids, internal_attention = _tokenize(
        tokenizer, internal_descriptions
    )

    max_slots = max(len(row["event_sequence_target"]) for row in rows)
    if max_slots <= 0:
        raise ValueError("every behavioral row requires one terminal event")
    relation_targets = torch.zeros(
        batch_size, max_slots, dtype=torch.long
    )
    relation_counter = torch.full(
        (batch_size, max_slots), -1, dtype=torch.long
    )
    relation_step_mask = torch.zeros(
        batch_size, max_slots, dtype=torch.bool
    )
    event_targets = torch.zeros(
        batch_size, max_slots, dtype=torch.long
    )
    event_mask = torch.zeros(batch_size, max_slots, dtype=torch.bool)

    factor_targets = {
        name: torch.zeros(batch_size, dtype=torch.long)
        for name in factor_banks
    }
    factor_counter = {
        name: torch.full((batch_size,), -1, dtype=torch.long)
        for name in factor_banks
    }
    step_factor_targets = {
        name: torch.zeros(
            batch_size, max_slots, dtype=torch.long
        )
        for name in factor_banks
    }
    step_factor_mask = torch.zeros(
        batch_size, max_slots, dtype=torch.bool
    )

    public_target = torch.zeros(batch_size, dtype=torch.long)
    source_target = torch.full((batch_size,), -1, dtype=torch.long)
    target_target = torch.full((batch_size,), -1, dtype=torch.long)
    endpoint_active = torch.zeros(batch_size, dtype=torch.bool)
    decisive_active = torch.zeros(batch_size, dtype=torch.bool)
    irrelevant_active = torch.zeros(batch_size, dtype=torch.bool)
    recoverable = torch.zeros(
        batch_size, len(INTERNAL_VIEW_NAMES), dtype=torch.bool
    )
    view_index = {name: i for i, name in enumerate(INTERNAL_VIEW_NAMES)}
    applicability_target = torch.zeros(batch_size)
    uncertainty_target = torch.zeros(batch_size)

    for b, row in enumerate(rows):
        row_relation_keys = [
            str(x["key"]) for x in row["relation_candidates"]
        ]
        for s, local_target in enumerate(
            row["relation_sequence_target"]
        ):
            key = row_relation_keys[int(local_target)]
            relation_targets[b, s] = relation_index[key]
            relation_step_mask[b, s] = True
        for s, local_counter in enumerate(
            row["counterfactual_relation_sequence_target"]
        ):
            if int(local_counter) >= 0:
                key = row_relation_keys[int(local_counter)]
                relation_counter[b, s] = relation_index[key]

        events = list(row["event_sequence_target"])
        for s, event in enumerate(events):
            event_targets[b, s] = EVENT_INDEX[str(event)]
            event_mask[b, s] = True

        for name in factor_banks:
            target_key = str(row["factor_target_keys"][name])
            factor_targets[name][b] = factor_key_index[name][target_key]
            counter_key = row["counterfactual_factor_keys"][name]
            if counter_key is not None:
                factor_counter[name][b] = factor_key_index[name][
                    str(counter_key)
                ]
        for s, values in enumerate(row["step_factor_target_keys"]):
            for name in factor_banks:
                step_factor_targets[name][b, s] = factor_key_index[name][
                    str(values[name])
                ]
            step_factor_mask[b, s] = True

        public_target[b] = int(row["public_target_index"])
        endpoint = row["endpoint_target"]
        endpoint_active[b] = bool(endpoint["active"])
        if endpoint_active[b]:
            source_target[b] = int(endpoint["source_field"])
            target_target[b] = int(endpoint["target_field"])
        decisive_active[b] = bool(row["decisive_view_active"])
        irrelevant_active[b] = bool(row["irrelevant_view_active"])
        for name in row["recoverable_view_names"]:
            recoverable[b, view_index[str(name)]] = True
        applicability_target[b] = float(row["applicability_target"])
        uncertainty_target[b] = float(row["uncertainty_target"])

    support_valid_mask = _structural_type_compatibility(
        rows=rows,
        relation_index=relation_index,
        type_index=type_index,
        max_edges=max_edges,
    ) & edge_valid
    if bool((support_target.bool() & ~support_valid_mask).any()):
        raise ValueError(
            "behavioral positive support target is structurally type-incompatible"
        )

    primary = {
        "query_input_ids": query_ids,
        "query_attention_mask": query_mask,
        "relation_input_ids": relation_ids,
        "relation_attention_mask": relation_mask,
        "relation_domain_type_mask": relation_domain,
        "relation_range_type_mask": relation_range,
        "relation_symmetric": relation_symmetric,
        "relation_candidate_mask": relation_candidate_mask,
        "factor_input_ids": factor_input_ids,
        "factor_attention_mask": factor_attention_mask,
        "factor_candidate_masks": factor_candidate_masks,
        "factor_opcodes": factor_opcodes,
        "field_input_ids": field_ids,
        "field_attention_mask": field_attention,
        "field_valid_mask": field_valid,
        "field_confidence": field_confidence,
        "field_missing": field_missing,
        "field_reliability": field_reliability,
        "descriptor_input_ids": descriptor_input_ids,
        "descriptor_attention_mask": descriptor_attention_mask,
        "descriptor_indices": descriptor_indices,
        "field_type_index": field_type_index,
        "field_metadata": field_metadata,
        "edge_index": edge_index,
        "edge_relation_index": edge_relation_index,
        "edge_valid_mask": edge_valid,
        "edge_metadata": edge_metadata,
        "edge_reliability": edge_reliability,
        "edge_recency": edge_recency,
        "edge_temporal_match": edge_temporal,
        "edge_provenance_match": edge_provenance,
        "internal_view_descriptor_input_ids": internal_ids,
        "internal_view_descriptor_attention_mask": internal_attention,
        "internal_view_reliability": torch.ones(
            batch_size, len(INTERNAL_VIEW_NAMES)
        ),
        "candidate_input_ids": candidate_ids,
        "candidate_attention_mask": candidate_attention,
        "candidate_valid_mask": candidate_valid,
        "max_reasoning_steps": max_slots,
        "graph_message_steps": cfg.graph_message_steps,
        "fusion_refinement_steps": cfg.fusion_refinement_steps,
        "latent_slot_count": cfg.latent_slot_count,
        "latent_refinement_steps": cfg.latent_refinement_steps,
    }

    # Behavioral rows intentionally do not own token-evidence targets. The
    # dedicated operator intervention lane supplies exact char->token evidence
    # supervision. False evidence masks keep this lane from inventing labels.
    query_tokens = query_ids.size(1)
    relation_tokens = relation_ids.size(1)
    query_evidence_shape = (
        batch_size,
        max_slots,
        len(relation_items),
        query_tokens,
    )
    relation_evidence_shape = (
        batch_size,
        max_slots,
        len(relation_items),
        relation_tokens,
    )
    factor_evidence_target: dict[str, Tensor] = {}
    factor_evidence_valid: dict[str, Tensor] = {}
    step_factor_evidence_target: dict[str, Tensor] = {}
    step_factor_evidence_valid: dict[str, Tensor] = {}
    for name, ids in factor_input_ids.items():
        candidates = ids.size(0)
        tokens = ids.size(1)
        factor_evidence_target[name] = torch.zeros(
            batch_size, candidates, tokens
        )
        factor_evidence_valid[name] = torch.zeros(
            batch_size, candidates, tokens, dtype=torch.bool
        )
        step_factor_evidence_target[name] = torch.zeros(
            batch_size, max_slots, candidates, tokens
        )
        step_factor_evidence_valid[name] = torch.zeros(
            batch_size,
            max_slots,
            candidates,
            tokens,
            dtype=torch.bool,
        )

    operator_targets = {
        "relation_targets": relation_targets,
        "counterfactual_relation_targets": relation_counter,
        "relation_step_mask": relation_step_mask,
        "factor_targets": factor_targets,
        "counterfactual_factor_targets": factor_counter,
        "step_factor_targets": step_factor_targets,
        "step_factor_mask": step_factor_mask,
        "event_targets": event_targets,
        "event_mask": event_mask,
        "applicability_target": applicability_target,
        "query_evidence_target": torch.zeros(query_evidence_shape),
        "query_evidence_valid_mask": torch.zeros(
            query_evidence_shape, dtype=torch.bool
        ),
        "relation_schema_evidence_target": torch.zeros(
            relation_evidence_shape
        ),
        "relation_schema_evidence_valid_mask": torch.zeros(
            relation_evidence_shape, dtype=torch.bool
        ),
        "factor_schema_evidence_target": factor_evidence_target,
        "factor_schema_evidence_valid_mask": factor_evidence_valid,
        "step_factor_schema_evidence_target": step_factor_evidence_target,
        "step_factor_schema_evidence_valid_mask": step_factor_evidence_valid,
        "uncertainty_target": uncertainty_target,
    }
    behavioral_targets = {
        "public_target_index": public_target,
        "support_target": support_target,
        "support_valid_mask": support_valid_mask,
        "source_target_index": source_target,
        "target_target_index": target_target,
        "endpoint_active_mask": endpoint_active,
        "decisive_view_active_mask": decisive_active,
        "irrelevant_view_active_mask": irrelevant_active,
        "recoverable_view_mask": recoverable,
    }

    decisive = _remove_fields_variant(
        primary=primary,
        rows=rows,
        field_index_key="decisive_field_indices",
    )
    irrelevant = _remove_fields_variant(
        primary=primary,
        rows=rows,
        field_index_key="irrelevant_field_indices",
    )

    # Pad each row's semantic permutation to the compiled field width with all
    # padded indices appended. This preserves invalid padding as invalid while
    # still testing real field order invariance.
    padded_rows = []
    for row in rows:
        row_copy = dict(row)
        actual = len(row["fields"])
        order = list(map(int, row["field_permutation"]))
        if sorted(order) != list(range(actual)):
            raise ValueError("row field permutation is not a bijection")
        row_copy["field_permutation"] = (
            order + list(range(actual, max_fields))
        )
        padded_rows.append(row_copy)
    permuted = _permuted_batch(
        primary=primary,
        rows=padded_rows,
    )

    return {
        "primary_batch": primary,
        "decisive_ablated_batch": decisive,
        "irrelevant_removed_batch": irrelevant,
        "permuted_batch": permuted,
        "operator_targets": operator_targets,
        "behavioral_targets": behavioral_targets,
        "metadata": {
            "batch_size": batch_size,
            "runtime_relation_count": len(relation_items),
            "runtime_type_count": len(type_items),
            "runtime_factor_banks": len(factor_banks),
            "max_fields": max_fields,
            "max_edges": max_edges,
            "max_reasoning_steps": max_slots,
            "max_candidates": candidate_ids.size(1),
            "evidence_targets_owned_by_this_lane": False,
            "private_identity_data": False,
        },
    }
