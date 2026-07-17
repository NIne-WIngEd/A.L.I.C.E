from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.grounded_response_evaluation import evaluate_grounded_responses


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument(
        "--benchmark",
        required=True,
        type=Path,
    )
    parser.add_argument("--pilot-name", default="pilot-v1")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    result = evaluate_grounded_responses(
        vault_root=args.vault,
        benchmark_path=args.benchmark,
        pilot_name=args.pilot_name,
        device=args.device,
    )
    print(json.dumps(result, indent=2))
    return (
        0
        if result["passes_all_thresholds"]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
