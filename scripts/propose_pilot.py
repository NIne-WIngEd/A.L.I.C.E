from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.pilot import propose_pilot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a private, review-only pilot dataset proposal. "
            "No source file is copied, parsed, moved, or deleted."
        )
    )
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--target", type=int, default=120)
    parser.add_argument("--duplicate-groups", type=int, default=5)
    parser.add_argument("--seed", default="alice-pilot-v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = propose_pilot(
        vault_root=args.vault,
        target_total=args.target,
        duplicate_groups=args.duplicate_groups,
        selection_seed=args.seed,
    )
    print("\nPilot proposal complete:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
