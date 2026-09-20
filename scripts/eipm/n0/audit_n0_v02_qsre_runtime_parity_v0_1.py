from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def run_bash(script: str, *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        ["bash", "-lc", script],
        cwd=str(cwd),
        env=merged,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    workdir = Path(args.workdir).resolve()

    env_root = Path(os.environ.get("ALICE_N0_REPO_ROOT", "")).resolve()
    env_workdir = Path(os.environ.get("ALICE_N0_WORKDIR", "")).resolve()
    if env_root != root:
        raise SystemExit(f"ALICE_N0_REPO_ROOT transfer drift: {env_root} != {root}")
    if env_workdir != workdir:
        raise SystemExit(f"ALICE_N0_WORKDIR transfer drift: {env_workdir} != {workdir}")

    pythonpath = os.environ.get("PYTHONPATH", "")
    required_python = [str(root / "src"), str(root / "scripts/eipm/n0")]
    missing = [value for value in required_python if value not in pythonpath.split(":")]
    if missing:
        raise SystemExit(f"container PYTHONPATH transfer drift: {missing}")

    import alice_personality.n0.qsre_t1_executor  # noqa: F401
    import alice_personality.n0.qsre_t2_operator  # noqa: F401

    submit_expr = 'ROOT="$' + '{SLURM_SUBMIT_DIR:-$PWD}"; printf "%s" "$ROOT"'
    with_submit = run_bash(
        submit_expr,
        cwd=root,
        env={"SLURM_SUBMIT_DIR": str(root)},
    )
    if with_submit.returncode != 0 or with_submit.stdout != str(root):
        raise SystemExit(
            "Bash submit-dir expansion failed: "
            f"rc={with_submit.returncode} out={with_submit.stdout!r} err={with_submit.stderr!r}"
        )

    fallback_expr = 'unset SLURM_SUBMIT_DIR; ROOT="$' + '{SLURM_SUBMIT_DIR:-$PWD}"; printf "%s" "$ROOT"'
    without_submit = run_bash(
        fallback_expr,
        cwd=root,
    )
    if without_submit.returncode != 0 or Path(without_submit.stdout).resolve() != root:
        raise SystemExit(
            "Bash PWD fallback expansion failed: "
            f"rc={without_submit.returncode} out={without_submit.stdout!r} err={without_submit.stderr!r}"
        )

    recovery_scripts = [
        root / "scripts/eipm/n0/run_n0_v02_qsre_t2_operator_recovery_v0_2.sh",
        root / "scripts/eipm/n0/magnolia_p100_n0_v02_qsre_t2_operator_recovery_v0_2.sbatch",
    ]
    guard_results = {}
    for script in recovery_scripts:
        proc = subprocess.run(
            ["bash", str(script)],
            cwd=str(root),
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        guard_results[script.name] = {
            "returncode": proc.returncode,
            "stderr": proc.stderr.strip(),
        }
        if proc.returncode != 90:
            raise SystemExit(
                f"superseded recovery script is not fail-closed: {script} rc={proc.returncode}"
            )

    payload = {
        "schema": "alice.eipm.n0.qsre-runtime-parity-audit.v0.1",
        "status": "PASS_QSRE_RUNTIME_BEHAVIORAL_PARITY_AUDIT",
        "repo_root": str(root),
        "workdir": str(workdir),
        "container_repo_root_transferred": True,
        "container_workdir_transferred": True,
        "container_pythonpath_transferred": True,
        "alice_personality_import": True,
        "bash_submit_dir_behavior": True,
        "bash_pwd_fallback_behavior": True,
        "superseded_recovery_scripts_fail_closed": guard_results,
        "source_literal_grep_used_as_oracle": False,
        "optimizer": False,
        "gradient": False,
        "gpu": False,
        "private_identity_data": False,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
