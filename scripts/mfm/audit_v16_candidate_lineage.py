#!/usr/bin/env python3
"""Audit declared lineage of unreviewed MFM 1.6 annotation candidates.

This is a source-custody and split-feasibility diagnostic. It assigns no split,
creates no target or FINAL payload, and does not authenticate rights or review.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

from scripts.mfm.build_v16_source_review_packets import _load
from scripts.mfm.prepare_v16_annotation_drafts import prepare_drafts


SCHEMA = "mfm-v16-candidate-lineage-audit-v1"


def audit_candidate_lineage(drafts_path: Path, *, drafts_sha256: str,
                            packet_path: Path, packet_sha256: str,
                            dataset_dir: Path, inventory_path: Path,
                            inventory_sha256: str) -> dict[str, object]:
    """Replay pinned source intake, then group all *declared* shared ancestry."""
    raw = drafts_path.read_bytes()
    if len(drafts_sha256) != 64 or sha256(raw).hexdigest() != drafts_sha256:
        raise ValueError("annotation drafts differ from pinned SHA-256")
    regenerated = prepare_drafts(
        packet_path, packet_sha256=packet_sha256, dataset_dir=dataset_dir,
        inventory_path=inventory_path, inventory_sha256=inventory_sha256)
    if raw != regenerated:
        raise ValueError("annotation drafts differ from pinned source reconstruction")

    rows = [_load(line) for line in raw.splitlines()]
    return {"schema": SCHEMA, "drafts_sha256": drafts_sha256,
            "packets_sha256": packet_sha256,
            "inventory_sha256": inventory_sha256,
            **summarize_declared_lineage(rows)}


def summarize_declared_lineage(rows: list[dict]) -> dict[str, object]:
    """Group candidates; shared declared ancestors cannot cross partitions."""
    if not rows:
        raise ValueError("empty candidate set")
    parents = list(range(len(rows)))
    sizes = [1] * len(rows)

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(left: int, right: int) -> None:
        left, right = root(left), root(right)
        if left != right:
            if sizes[left] < sizes[right]:
                left, right = right, left
            parents[right] = left
            sizes[left] += sizes[right]

    first_by_lineage: dict[tuple[str, str], int] = {}
    source_id_digests: dict[str, str] = {}
    generator_families: set[str] = set()
    host_families: set[str] = set()
    anchor_count = 0
    for index, row in enumerate(rows):
        lineage = row["provenance"]["lineage"]
        generator = lineage["generator_family"]
        host = lineage["host_family"]
        generator_families.add(generator)
        host_families.add(host)
        keys = {("generator_family", generator), ("host_family", host),
                ("lineage_group", lineage["lineage_group"]),
                ("upstream_commit", lineage["upstream_commit"])}
        for source in row["source_anchor_inventory"]:
            source_id = source["source_id"]
            digest = source["source_file_sha256"]
            prior = source_id_digests.setdefault(source_id, digest)
            if prior != digest:
                raise ValueError("source ID has conflicting original-file digests")
            keys.add(("source_id", source_id))
            keys.add(("source_file_sha256", digest))
            anchor_count += 1
        for key in keys:
            prior_index = first_by_lineage.setdefault(key, index)
            join(index, prior_index)

    members: dict[int, list[dict]] = defaultdict(list)
    for index, row in enumerate(rows):
        members[root(index)].append(row)
    components = []
    for cases in members.values():
        ids = sorted(row["packet_id"] for row in cases)
        components.append({
            "id_sha256": sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
            "candidate_rows": len(cases),
            "host_families": len({row["provenance"]["lineage"]["host_family"]
                                  for row in cases}),
            "declared_generator_families": sorted({
                row["provenance"]["lineage"]["generator_family"] for row in cases}),
        })
    components.sort(key=lambda component: component["id_sha256"])
    return {
        "candidate_rows": len(rows),
        "host_families": len(host_families),
        "source_anchor_inventory_entries": anchor_count,
        "declared_generator_families": sorted(generator_families),
        "connected_declared_lineage_components": components,
        "three_way_split_declared_lineage_blocked": len(components) < 3,
        "independent_family_provenance_established": False,
        "rights_and_independent_review_established": False,
        "training_admitted": False,
        "sealed_final_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drafts", required=True, type=Path)
    parser.add_argument("--drafts-sha256", required=True)
    parser.add_argument("--packets", required=True, type=Path)
    parser.add_argument("--packets-sha256", required=True)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--inventory-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(audit_candidate_lineage(
        args.drafts, drafts_sha256=args.drafts_sha256,
        packet_path=args.packets, packet_sha256=args.packets_sha256,
        dataset_dir=args.dataset_dir, inventory_path=args.inventory,
        inventory_sha256=args.inventory_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
