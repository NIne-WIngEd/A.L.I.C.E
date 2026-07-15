from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.semantic_benchmark_review import review_semantic_benchmark


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--pilot-name", default="pilot-v1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--snippet-characters", type=int, default=700)
    args = parser.parse_args()

    result = review_semantic_benchmark(
        vault_root=args.vault,
        benchmark_path=args.benchmark,
        pilot_name=args.pilot_name,
        device=args.device,
        candidate_limit=args.candidates,
        snippet_characters=args.snippet_characters,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
