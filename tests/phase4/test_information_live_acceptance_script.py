from pathlib import Path


def test_acceptance_script_pins_all_p410_deterministic_tests_and_private_boundaries():
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts/run_phase4_live_information_acceptance.py").read_text(encoding="utf-8")
    assert text.count('"tests/phase4/test_information_') >= 17
    assert "build_phase4_live_acceptance_runtime" in text
    assert "validate_repository_release_state" in text
    assert "repository_snapshot_sha256" in text
    assert "run_repository_regression_tests" in text
    assert "private_root=vault" in text
    assert "live_fetch_attempts" in text
    assert "live_fetch_failures" in text
    assert "return 0 if record.approved else 2" in text


def test_acceptance_module_revalidates_exact_live_network_components():
    root = Path(__file__).resolve().parents[2]
    text = (root / "src/alice_information/live_acceptance.py").read_text(encoding="utf-8")
    assert "transport.validate_live_boundary()" in text
    assert "retriever._validate_runtime_components()" in text
    assert "rollback commit is not an ancestor" in text.casefold()
    assert "ALICE_BRAVE_SEARCH_API_KEY" in text
