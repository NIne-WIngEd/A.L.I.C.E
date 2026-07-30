from __future__ import annotations

import json
import subprocess
import sys

from _information_final_evaluation_helpers import ROOT


def test_script_runs_runtime_backed_suite_and_writes_private_report(tmp_path) -> None:
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_phase4_information_evaluation.py"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.exists()
    assert "passed=true" in result.stdout
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["passed"] is True
    assert value["runtime_evidence"]["passed"] is True
    assert value["runtime_evidence"]["collected_test_count"] >= 640
    assert value["runtime_evidence"]["passed_test_count"] == value[
        "runtime_evidence"
    ]["collected_test_count"]
    assert "submissions" not in json.dumps(value)


def test_script_refuses_repository_output(tmp_path) -> None:
    output = ROOT / f".p48-forbidden-{tmp_path.name}.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_phase4_information_evaluation.py"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert not output.exists()


def test_script_refuses_overwrite(tmp_path) -> None:
    output = tmp_path / "report.json"
    output.write_text("existing")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_phase4_information_evaluation.py"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert output.read_text() == "existing"


def test_script_rejects_external_submission_bundle(tmp_path) -> None:
    output = tmp_path / "report.json"
    submissions = tmp_path / "submissions.json"
    submissions.write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_phase4_information_evaluation.py"),
            "--submissions",
            str(submissions),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert not output.exists()
