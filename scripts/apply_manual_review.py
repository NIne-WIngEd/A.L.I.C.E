from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.auto_review import apply_manual_review


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply edits from the small manual-review CSV to the full private review CSV.")
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--manual-csv", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(apply_manual_review(vault_root=args.vault, manual_csv=args.manual_csv), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
