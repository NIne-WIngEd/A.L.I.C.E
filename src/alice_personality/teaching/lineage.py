"""Separate fixed identity references from supervised episodes and derivatives.

This supports conditional development cases, not source-independent release
evaluation. Allocation is an externally pinned declaration, not proof that an
author honestly disclosed all ancestry. Model checkpoint dependencies are bound
separately; they are not new supervision episodes.
"""
from dataclasses import asdict, dataclass, field
from typing import Mapping

from ..identity.contracts import SYNTHETIC_KINDS

LINEAGE_SCHEMA = "alice-personality-construction-lineage-v1"
ALLOCATION_SCHEMA = "alice-personality-teaching-family-splits-v2"
SPLITS = {"TRAIN", "DEV", "CALIBRATION_TRAIN", "CALIBRATION_DEV", "FINAL"}
REFERENCE_KINDS = {"E0", "EINF", "UNKNOWN", *SYNTHETIC_KINDS}


def _require(condition, message):
    if not condition:
        from .contracts import TeachingError
        raise TeachingError(message)


def _names(value, *, nonempty=False):
    return type(value) is tuple and (bool(value) or not nonempty) \
        and all(isinstance(v, str) and v for v in value) and len(set(value)) == len(value)


@dataclass(frozen=True)
class ConstructionLineage:
    reference_family_ids: tuple[str, ...]
    context_family_ids: tuple[str, ...]
    supervision_family_ids: tuple[str, ...]
    schema: str = field(default=LINEAGE_SCHEMA, init=False)

    def validate(self, records):
        _require(self.schema == LINEAGE_SCHEMA and _names(self.reference_family_ids)
                 and _names(self.context_family_ids) and _names(self.supervision_family_ids, nonempty=True),
                 "construction lineage needs explicit unique reference/context/supervision families")
        refs, contexts = set(self.reference_family_ids), set(self.context_family_ids)
        actual = {record.source_family_id for record in records if record is not None}
        _require(not refs & contexts and refs | contexts == actual,
                 "reference/context lineage must partition all original source families")
        _require(all(record is None or record.source_family_id not in refs or record.source_kind in REFERENCE_KINDS
                     for record in records), "host, relationship and lived experience cannot become fixed identity references")

    def to_mapping(self):
        return {name: list(value) if isinstance(value, tuple) else value for name, value in asdict(self).items()}

    @classmethod
    def from_mapping(cls, value):
        _require(type(value) is dict and set(value) == set(cls.__dataclass_fields__)
                 and value.get("schema") == LINEAGE_SCHEMA, "construction lineage schema differs")
        names = ("reference_family_ids", "context_family_ids", "supervision_family_ids")
        _require(all(type(value[name]) is list for name in names), "serialized lineage families must be lists")
        return cls(*(tuple(value[name]) for name in names))


def validate_allocation(allocation):
    keys = {"schema", "family_splits", "reference_family_ids", "supervision_family_splits",
            "supervision_parents", "qualification_scope"}
    _require(isinstance(allocation, Mapping) and set(allocation) == keys
             and allocation["schema"] == ALLOCATION_SCHEMA
             and allocation["qualification_scope"] == "DEVELOPMENT_CONDITIONAL_CASES",
             "new lineage needs the exact conditional-development allocation schema")
    original, supervision, parents = (allocation[name] for name in
                                      ("family_splits", "supervision_family_splits", "supervision_parents"))
    for splits in (original, supervision):
        _require(type(splits) is dict and all(isinstance(name, str) and name and isinstance(split, str) and split in SPLITS
                                           for name, split in splits.items()), "lineage split declarations are malformed")
    refs = allocation["reference_family_ids"]
    _require(type(refs) is list and _names(tuple(refs))
             and all(original.get(name) == "TRAIN" for name in refs),
             "fixed reference bank must preserve its original TRAIN allocation")
    _require(type(parents) is dict and set(parents) == set(supervision),
             "every supervision family requires explicit disclosed parent ancestry")
    children = {name: set() for name in supervision}
    degree = {}
    for name, ancestors in parents.items():
        _require(type(ancestors) is list and _names(tuple(ancestors)), "supervision ancestors must be explicit unique names")
        _require(all(parent in supervision for parent in ancestors), "supervision ancestor is unresolved")
        _require(all(supervision[parent] == supervision[name] for parent in ancestors),
                 "a supervision derivative cannot cross its ancestor's held-out split")
        degree[name] = len(ancestors)
        for parent in ancestors:
            children[parent].add(name)
    ready = [name for name, count in degree.items() if not count]
    seen = 0
    while ready:
        parent = ready.pop(); seen += 1
        for child in children[parent]:
            degree[child] -= 1
            if not degree[child]:
                ready.append(child)
    _require(seen == len(supervision), "supervision ancestor graph contains a cycle")


def verify_allocation(lineage, records, split, allocation):
    _require(type(lineage) is ConstructionLineage, "construction cases require typed lineage")
    lineage.validate(records)
    validate_allocation(allocation)
    _require(split in SPLITS and split != "FINAL", "construction cannot train on FINAL outcomes")
    refs = set(allocation["reference_family_ids"])
    _require(set(lineage.reference_family_ids) <= refs, "case reference families lack explicit allocation")
    _require(all(allocation["family_splits"].get(name) == split for name in lineage.context_family_ids),
             "dynamic context ancestry crossed its source-family split")
    _require(all(allocation["supervision_family_splits"].get(name) == split for name in lineage.supervision_family_ids),
             "supervision episode crossed its teaching/calibration split")
