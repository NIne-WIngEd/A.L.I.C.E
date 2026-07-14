from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from alice_vault.analysis import analyze_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the latest completed SHA-256 inventory without "
            "extracting, executing, moving, or deleting source files."
        )
    )
    parser.add_argument("--vault", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = analyze_inventory(vault_root=args.vault)
    print("\nInventory analysis complete:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
