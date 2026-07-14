from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.auto_review import auto_review_pilot


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fast, resumable private pilot review using local extraction, "
            "privacy rules, and batched local Ollama."
        )
    )
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
    )
    parser.add_argument("--no-ollama", action="store_true")
    parser.add_argument("--use-presidio", action="store_true")
    parser.add_argument("--profile", type=Path)
    parser.add_argument(
        "--approve-threshold",
        type=float,
        default=0.92,
    )
    parser.add_argument(
        "--reject-threshold",
        type=float,
        default=0.92,
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=3000,
        help="Maximum extracted characters sent per item.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=6,
        help="Documents per Ollama request.",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=1200,
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Discard the compatible checkpoint and start again.",
    )
    args = parser.parse_args()

    result = auto_review_pilot(
        vault_root=args.vault,
        model=args.model,
        base_url=args.ollama_url,
        use_ollama=not args.no_ollama,
        use_presidio=args.use_presidio,
        approve_threshold=args.approve_threshold,
        reject_threshold=args.reject_threshold,
        profile_path=args.profile,
        max_chars=args.max_chars,
        batch_size=args.batch_size,
        resume=not args.no_resume,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
