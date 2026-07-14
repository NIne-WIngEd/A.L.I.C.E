from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.review_calibration import recalibrate_auto_review


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recalibrate an existing auto-review, reuse successful semantic "
            "results, and retry only previously unclassified items."
        )
    )
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
    )
    parser.add_argument("--no-presidio", action="store_true")
    parser.add_argument(
        "--approve-threshold",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--reject-threshold",
        type=float,
        default=0.85,
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--timeout-seconds", type=int, default=210)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    result = recalibrate_auto_review(
        vault_root=args.vault,
        model=args.model,
        base_url=args.ollama_url,
        use_presidio=not args.no_presidio,
        approve_threshold=args.approve_threshold,
        reject_threshold=args.reject_threshold,
        batch_size=args.batch_size,
        max_chars=args.max_chars,
        timeout_seconds=args.timeout_seconds,
        resume=not args.no_resume,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
