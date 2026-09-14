from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
ACTIVE = ROOT / "configs/eipm/n0/public_corpus_v0.2.activated.json"


def test_v02_activated_manifest_is_exact_revision_and_balanced() -> None:
    raw = json.loads(ACTIVE.read_text(encoding="utf-8"))
    assert raw["status"] == "activated_public_n0_v02"
    assert raw["global_requirements"]["private_identity_data"] is False
    assert raw["global_requirements"]["private_identity_gradient"] is False
    sources = raw["sources"]
    assert len(sources) == 22
    assert len({row["source_id"] for row in sources}) == 22
    assert len({row["repo_id"] for row in sources}) == 22
    assert sum(float(row["target_share"]) for row in sources) == pytest.approx(1.0)
    assert max(float(row["target_share"]) for row in sources) <= 0.125
    assert all(len(row["revision"]) == 40 for row in sources)
    assert all(all(ch in "0123456789abcdef" for ch in row["revision"].lower()) for row in sources)
    assert all(row["require_row_license"] is True for row in sources)
    assert all(row["allowed_license_values"] for row in sources)
    assert all(row["provenance_field"] == "metadata.provenance" for row in sources)


def test_v02_activation_records_probe_as_non_authorizing() -> None:
    raw = json.loads(ACTIVE.read_text(encoding="utf-8"))
    probe = raw["source_probe"]
    assert probe["schema_passed_sources"] == 22
    assert probe["schema_failed_sources"] == 0
    assert probe["training_authorized_by_probe"] is False
    assert probe["manual_license_review_completed"] is True
