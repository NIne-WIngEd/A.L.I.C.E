from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "phase4_information_release_script",
    ROOT / "scripts" / "run_phase4_information_release_audit.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _repository(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "alice@example.invalid")
    _git(repo, "config", "user.name", "A.L.I.C.E. Test")
    (repo / "state.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "state.txt")
    _git(repo, "commit", "-m", "baseline")
    rollback = _git(repo, "rev-parse", "HEAD")
    (repo / "state.txt").write_text("two\n", encoding="utf-8")
    _git(repo, "add", "state.txt")
    _git(repo, "commit", "-m", "release")
    head = _git(repo, "rev-parse", "HEAD")
    return repo, head, rollback


def test_repository_verifier_accepts_clean_exact_commit(tmp_path):
    repo, head, rollback = _repository(tmp_path)
    assert MODULE._verify_repository_state(
        repo,
        expected_commit=head,
        rollback_ref=rollback,
    ) == (head, rollback)


def test_repository_verifier_rejects_commit_mismatch(tmp_path):
    repo, _head, rollback = _repository(tmp_path)
    with pytest.raises(RuntimeError, match="does not match"):
        MODULE._verify_repository_state(
            repo,
            expected_commit="0" * 40,
            rollback_ref=rollback,
        )


def test_repository_verifier_rejects_dirty_tree(tmp_path):
    repo, head, rollback = _repository(tmp_path)
    (repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean"):
        MODULE._verify_repository_state(
            repo,
            expected_commit=head,
            rollback_ref=rollback,
        )


def test_repository_verifier_rejects_same_rollback(tmp_path):
    repo, head, _rollback = _repository(tmp_path)
    with pytest.raises(RuntimeError, match="differ"):
        MODULE._verify_repository_state(
            repo,
            expected_commit=head,
            rollback_ref=head,
        )


def test_repository_verifier_rejects_nonancestor_rollback(tmp_path):
    repo, head, _rollback = _repository(tmp_path)
    branch = _git(repo, "branch", "--show-current")
    _git(repo, "checkout", "--orphan", "other")
    (repo / "state.txt").write_text("other\n", encoding="utf-8")
    _git(repo, "add", "state.txt")
    _git(repo, "commit", "-m", "other")
    unrelated = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", branch)
    assert _git(repo, "rev-parse", "HEAD") == head
    with pytest.raises(RuntimeError, match="ancestor"):
        MODULE._verify_repository_state(
            repo,
            expected_commit=head,
            rollback_ref=unrelated,
        )


def test_release_parser_generates_runtime_evidence_internally():
    destinations = {action.dest for action in MODULE._parser()._actions}
    assert "runtime_manifest" in destinations
    assert "submissions" not in destinations
    assert "evaluation_report" not in destinations
