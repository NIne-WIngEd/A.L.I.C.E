from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/eipm/n0/derive_tokenizer_corpus_v021_from_v01.py"
CONFIG = ROOT / "configs/eipm/n0/public_corpus_v0.2.1.activated.json"


def load_module():
    spec = importlib.util.spec_from_file_location("alice_n0_v021_offline_derivation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_offline_derivation_chunks_long_documents_and_preserves_budget() -> None:
    module = load_module()
    text = ("alpha beta gamma delta\n" * 5000).strip()
    chunks = list(module.split_chunks(text, max_chars=16_000, min_chars=80))
    assert len(chunks) > 1
    assert all(80 <= len(chunk) <= 16_000 for chunk in chunks)
    trimmed = module.trim_to_budget(chunks[0], remaining=1000, min_chars=80)
    assert 80 <= len(trimmed) <= 1000
    assert module.trim_to_budget(chunks[0], remaining=40, min_chars=80) == ""


def test_v021_offline_target_is_feasible_from_first_materialization_counts() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    shares = {row["source_id"]: float(row["target_share"]) for row in config["sources"]}
    assert len(shares) == 21
    assert sum(shares.values()) == pytest.approx(1.0)
    assert "common_pile_oercommons_filtered" not in shares

    # The failed 120M-char materialization still produced enough governed text
    # for every retained source to support a 100M-char tokenizer-only rebalance.
    observed_parent_chars = {
        "common_pile_wikimedia_filtered": 12015030,
        "common_pile_wikiteam_filtered": 4800209,
        "common_pile_arxiv_abstracts_filtered": 6000216,
        "common_pile_arxiv_papers_filtered": 6022592,
        "common_pile_pubmed_filtered": 7208171,
        "common_pile_stackexchange_filtered": 10800872,
        "common_pile_libretexts_filtered": 6006359,
        "common_pile_pressbooks_filtered": 6000176,
        "common_pile_ubuntu_irc_filtered": 8605163,
        "common_pile_youtube_filtered": 9622361,
        "common_pile_project_gutenberg_filtered": 9789007,
        "common_pile_pre_1929_books_filtered": 3775684,
        "common_pile_doab_filtered": 5783691,
        "common_pile_public_domain_review_filtered": 1200368,
        "common_pile_stackv2_edu_filtered": 3607263,
        "common_pile_github_archive_filtered": 2405612,
        "common_pile_python_peps_filtered": 1213282,
        "common_pile_usgpo_filtered": 3836055,
        "common_pile_regulations_filtered": 1802761,
        "common_pile_uk_hansard_filtered": 2484325,
        "common_pile_news_filtered": 10801292,
    }
    target_total = 100_000_000
    assert set(observed_parent_chars) == set(shares)
    for source_id, share in shares.items():
        assert observed_parent_chars[source_id] >= round(target_total * share)
