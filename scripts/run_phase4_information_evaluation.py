from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_information.final_evaluation_contract import (  # noqa: E402
    load_information_final_evaluation_benchmark,
    load_information_final_evaluation_policy,
)
from alice_information.final_evaluation_runtime import (  # noqa: E402
    load_information_final_evaluation_runtime_manifest,
    run_runtime_backed_information_final_evaluation,
    runtime_backed_report_to_dict,
)


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the offline runtime-backed P4.8 information evaluation."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--runtime-manifest", type=Path)
    args = parser.parse_args(argv)

    output = args.output.expanduser().resolve()
    if _within(output, ROOT.resolve()):
        raise SystemExit("Evaluation output must remain outside the repository.")
    if output.exists():
        raise SystemExit("Refusing to overwrite an existing evaluation report.")

    policy = load_information_final_evaluation_policy(args.policy)
    benchmark = load_information_final_evaluation_benchmark(
        args.benchmark,
        policy=policy,
    )
    manifest = load_information_final_evaluation_runtime_manifest(
        args.runtime_manifest,
        benchmark=benchmark,
        policy=policy,
    )
    report = run_runtime_backed_information_final_evaluation(
        repository_root=ROOT,
        benchmark=benchmark,
        policy=policy,
        manifest=manifest,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(runtime_backed_report_to_dict(report), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print("passed=" + str(report.passed).lower())
    print("evaluation_report_digest=" + report.evaluation_report.report_digest)
    print("runtime_evidence_digest=" + report.runtime_evidence.evidence_digest)
    print("report_digest=" + report.report_digest)
    print("output=" + str(output))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
