from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

from alice_personality.n0.full_envelope_stage_policy_v1 import J1,J2,J3


SEMANTIC_LANES=(
    "semantic_operator_intervention",
    "long_context_semantic",
)
NATURAL_LANES=("natural_relation",)
FULL_FABRIC_BY_STAGE={
    J1:(),
    J2:(
        "full_envelope_behavioral",
        "long_context_fabric",
    ),
    J3:(
        "full_envelope_behavioral",
        "runtime_view_supplement",
        "long_context_fabric",
    ),
}
LONG_SEMANTIC_SURFACES=frozenset({
    "query",
    "relation_schema",
    "factor_schema",
})
LONG_J2_FABRIC_SURFACES=frozenset({
    "type_schema",
    "field_text",
    "field_descriptor",
})
LONG_J3_FABRIC_SURFACES=frozenset({
    "query",
    "relation_schema",
    "factor_schema",
    "type_schema",
    "field_text",
    "field_descriptor",
    "candidate_text",
    "internal_view_descriptor",
    "additional_view_descriptor",
    "additional_view_source",
})


class FullEnvelopeTrainingBatchSchedulerV1:
    """Deterministic stage-aware lane scheduler for the public N0 successor.

    Lane order is an optimization operating policy, not a capability definition.
    Each microbatch comes from one semantically coherent lane so runtime schema
    banks need not be forced into a fake common shape. J1 receives long
    query/relation/factor examples through the semantic-operator path while
    downstream full-fabric lanes remain inactive. J2 adds only structural
    full-fabric surfaces. J3 activates the complete behavioral/runtime-view/
    long-context fabric.

    The scheduler does not define a permanent sampling ratio or row ceiling.
    Failure-driven public data can be appended and the same lane policy reused.
    """

    REQUIRED_LANES=(
        "semantic_operator_intervention",
        "long_context_semantic",
        "full_envelope_behavioral",
        "runtime_view_supplement",
        "long_context_fabric",
        "natural_relation",
    )

    def __init__(
        self,
        *,
        lanes: Mapping[str,Sequence[Mapping[str,Any]]],
        seed: int,
    ) -> None:
        unknown=set(lanes)-set(self.REQUIRED_LANES)
        if unknown:
            raise ValueError(
                "unknown full-envelope training lanes: "
                +repr(sorted(unknown))
            )
        missing=set(self.REQUIRED_LANES)-set(lanes)
        if missing:
            raise ValueError(
                "missing full-envelope training lanes: "
                +repr(sorted(missing))
            )
        self.seed=int(seed)
        self._rows={
            name:[dict(row) for row in values]
            for name,values in lanes.items()
        }
        for name,rows in self._rows.items():
            if not rows:
                raise ValueError(f"training lane is empty: {name}")
            if any(row.get("private_identity_data") is not False for row in rows):
                raise ValueError(f"private identity row in public lane: {name}")
            if any(row.get("final_validation_only") is True for row in rows):
                raise ValueError(f"FINAL row in optimizer-facing lane: {name}")

        self._lane_cursor=defaultdict(int)
        self._row_cursor=defaultdict(int)
        self._orders: dict[tuple[str,int],list[int]]={}

    @classmethod
    def split_long_context_rows(
        cls,
        rows: Sequence[Mapping[str,Any]],
    ) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
        semantic=[]
        fabric=[]
        for raw in rows:
            row=dict(raw)
            surface=str(row.get("long_context_surface",""))
            if surface in LONG_SEMANTIC_SURFACES:
                semantic.append(row)
            if surface in LONG_J3_FABRIC_SURFACES:
                fabric.append(row)
        if not semantic:
            raise ValueError("long-context rows lack J1 semantic surfaces")
        if not fabric:
            raise ValueError("long-context rows lack full-fabric surfaces")
        return semantic,fabric

    def active_semantic_lanes(self, *, stage: str) -> tuple[str,...]:
        if stage not in FULL_FABRIC_BY_STAGE:
            raise ValueError(f"unknown N0 training stage: {stage!r}")
        return SEMANTIC_LANES

    def active_natural_lanes(self, *, stage: str) -> tuple[str,...]:
        if stage not in FULL_FABRIC_BY_STAGE:
            raise ValueError(f"unknown N0 training stage: {stage!r}")
        return NATURAL_LANES

    def active_full_fabric_lanes(self, *, stage: str) -> tuple[str,...]:
        try:
            return tuple(FULL_FABRIC_BY_STAGE[str(stage)])
        except KeyError as exc:
            raise ValueError(f"unknown N0 training stage: {stage!r}") from exc

    def _active_lanes(self, *, stage: str, kind: str) -> tuple[str,...]:
        if kind=="semantic":
            return self.active_semantic_lanes(stage=stage)
        if kind=="natural":
            return self.active_natural_lanes(stage=stage)
        if kind=="full_fabric":
            return self.active_full_fabric_lanes(stage=stage)
        raise ValueError(f"unknown training lane kind: {kind!r}")

    def next_lane(self, *, stage: str, kind: str) -> str:
        lanes=self._active_lanes(stage=stage,kind=kind)
        if not lanes:
            raise ValueError(
                f"stage {stage!r} has no active lanes for kind {kind!r}"
            )
        key=(str(stage),str(kind))
        index=self._lane_cursor[key] % len(lanes)
        self._lane_cursor[key]+=1
        return lanes[index]

    def _eligible_rows(self, *, stage: str, lane: str) -> list[dict[str,Any]]:
        rows=self._rows[lane]
        if lane=="long_context_semantic":
            rows=[
                row for row in rows
                if str(row.get("long_context_surface",""))
                in LONG_SEMANTIC_SURFACES
            ]
        elif lane=="long_context_fabric":
            allowed=(
                LONG_J2_FABRIC_SURFACES
                if stage==J2
                else LONG_J3_FABRIC_SURFACES
            )
            rows=[
                row for row in rows
                if str(row.get("long_context_surface","")) in allowed
            ]
        if not rows:
            raise ValueError(
                f"no eligible rows for stage={stage!r} lane={lane!r}"
            )
        return rows

    def _order(self, *, stage: str, lane: str, epoch: int, size: int) -> list[int]:
        key=(f"{stage}:{lane}",int(epoch))
        order=self._orders.get(key)
        if order is None:
            order=list(range(size))
            rng=random.Random(
                f"alice-n0-full-envelope:{self.seed}:{stage}:{lane}:{epoch}"
            )
            rng.shuffle(order)
            self._orders[key]=order
        if len(order)!=size:
            raise RuntimeError(
                "lane row cardinality changed during one scheduler lifetime"
            )
        return order

    def next_rows(
        self,
        *,
        stage: str,
        kind: str,
        batch_size: int,
    ) -> tuple[str,list[dict[str,Any]]]:
        if int(batch_size)<=0:
            raise ValueError("batch_size must be positive")
        lane=self.next_lane(stage=stage,kind=kind)
        rows=self._eligible_rows(stage=stage,lane=lane)
        cursor_key=(str(stage),lane)
        cursor=self._row_cursor[cursor_key]
        out=[]
        for _ in range(int(batch_size)):
            epoch=cursor//len(rows)
            offset=cursor%len(rows)
            index=self._order(
                stage=stage,
                lane=lane,
                epoch=epoch,
                size=len(rows),
            )[offset]
            out.append(dict(rows[index]))
            cursor+=1
        self._row_cursor[cursor_key]=cursor
        return lane,out

    def report(self) -> dict[str,Any]:
        return {
            "schema":"alice.eipm.n0.full-envelope-training-batch-scheduler-report.v1",
            "lane_counts":{
                name:len(rows) for name,rows in sorted(self._rows.items())
            },
            "semantic_lanes":SEMANTIC_LANES,
            "natural_lanes":NATURAL_LANES,
            "full_fabric_by_stage":{
                stage:list(lanes)
                for stage,lanes in FULL_FABRIC_BY_STAGE.items()
            },
            "long_semantic_surfaces":sorted(LONG_SEMANTIC_SURFACES),
            "long_j2_fabric_surfaces":sorted(LONG_J2_FABRIC_SURFACES),
            "long_j3_fabric_surfaces":sorted(LONG_J3_FABRIC_SURFACES),
            "fixed_lane_sampling_ratio_as_capability_definition":False,
            "row_count_is_capability_ceiling":False,
            "private_identity_data":False,
        }
