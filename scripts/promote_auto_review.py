from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.auto_review import promote_auto_review


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Promote a successful run-specific auto-review CSV to the "
            "canonical pilot-review CSV after closing Excel."
        )
    )
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--review-csv", type=Path)
    args = parser.parse_args()

    result = promote_auto_review(
        vault_root=args.vault,
        review_csv=args.review_csv,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
