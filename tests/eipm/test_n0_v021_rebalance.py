from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ACTIVE = ROOT / "configs/eipm/n0/public_corpus_v0.2.1.activated.json"
MATERIALIZER = ROOT / "scripts/eipm/n0/materialize_public_corpus_v02.py"


def load_materializer_module():
    name = "alice_n0_v02_materializer"
    spec = importlib.util.spec_from_file_location(name, MATERIALIZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_v021_manifest_removes_underfilled_oer_without_widening_license_gate() -> None:
    raw = json.loads(ACTIVE.read_text(encoding="utf-8"))
    sources = raw["sources"]
    assert len(sources) == 21
    assert sum(float(row["target_share"]) for row in sources) == pytest.approx(1.0)
    assert max(float(row["target_share"]) for row in sources) <= 0.125
    assert "common-pile/oercommons_filtered" not in {row["repo_id"] for row in sources}
    edu = {
        row["repo_id"]: float(row["target_share"])
        for row in sources
        if row["category"] == "educational_qa_explanation"
    }
    assert edu == {
        "common-pile/stackexchange_filtered": pytest.approx(0.10),
        "common-pile/libretexts_filtered": pytest.approx(0.06),
        "common-pile/pressbooks_filtered": pytest.approx(0.06),
    }
    assert raw["global_requirements"]["private_identity_gradient"] is False


def test_long_document_chunking_bounds_sampling_granularity() -> None:
    module = load_materializer_module()
    text = ("alpha beta gamma delta\n" * 5000).strip()
    chunks = list(module.chunk_text(text, max_chars=16_000, min_chars=80))
    assert len(chunks) > 1
    assert all(80 <= len(chunk) <= 16_000 for chunk in chunks)
    assert "".join(chunks).replace("\n", "").replace(" ", "")


def test_budget_trim_never_exceeds_remaining_chars() -> None:
    module = load_materializer_module()
    text = ("word " * 1000).strip()
    trimmed = module.trim_to_budget(text, remaining=1000, min_chars=80)
    assert 80 <= len(trimmed) <= 1000
    assert module.trim_to_budget(text, remaining=40, min_chars=80) == ""
