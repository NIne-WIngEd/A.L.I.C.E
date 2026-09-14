from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from alice_personality.n0.curriculum import (
    validate_curriculum_manifest,
    validate_curriculum_rows,
)
from alice_personality.n0.curriculum_data import CurriculumDataset


ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "training/eipm/n0/sol_curriculum_seed_v0.1.jsonl"
SEED_MANIFEST = ROOT / "training/eipm/n0/sol_curriculum_seed_v0.1.origin.json"
COVERAGE = ROOT / "training/eipm/n0/sol_curriculum_coverage_v0.2.jsonl"
COVERAGE_MANIFEST = ROOT / "training/eipm/n0/sol_curriculum_coverage_v0.2.origin.json"
REGISTRY = ROOT / "docs/eipm/n0/competency_registry_v0.1.tsv"


def read_rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def registry_ids() -> set[str]:
    lines = REGISTRY.read_text(encoding="utf-8").splitlines()
    return {line.split("\t", 1)[0] for line in lines[1:] if line.strip()}


def test_teacher_shards_are_individually_valid_and_authorized() -> None:
    seed_summary = validate_curriculum_rows(SEED)
    coverage_summary = validate_curriculum_rows(COVERAGE)
    assert seed_summary["row_count"] == 60
    assert coverage_summary["row_count"] == 26

    seed_manifest = validate_curriculum_manifest(SEED, SEED_MANIFEST)
    coverage_manifest = validate_curriculum_manifest(COVERAGE, COVERAGE_MANIFEST)
    assert seed_manifest["private_identity_data"] is False
    assert coverage_manifest["private_identity_data"] is False
    assert seed_manifest["private_identity_gradient_authorized"] is False
    assert coverage_manifest["private_identity_gradient_authorized"] is False


def test_seed_plus_coverage_has_train_and_dev_for_every_registry_competency() -> None:
    rows = read_rows(SEED) + read_rows(COVERAGE)
    seen_ids: set[str] = set()
    splits: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        row_id = str(row["id"])
        assert row_id not in seen_ids
        seen_ids.add(row_id)
        splits[str(row["competency"])].add(str(row["split"]))

    expected = registry_ids()
    assert set(splits) == expected
    for competency in sorted(expected):
        assert "train" in splits[competency], competency
        assert "dev" in splits[competency], competency


def test_curriculum_dataset_loads_additive_shards_without_duplicate_ids() -> None:
    train = CurriculumDataset([SEED, COVERAGE], "train")
    dev = CurriculumDataset([SEED, COVERAGE], "dev")
    assert len(train) > 0
    assert len(dev) > 0
    assert len({str(row["id"]) for row in train.rows + dev.rows}) == len(train) + len(dev)
