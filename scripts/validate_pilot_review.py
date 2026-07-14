from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.pilot_review import validate_review


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate decisions in a private pilot-review CSV."
    )
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--proposal-id")
    parser.add_argument("--minimum-approved", type=int, default=100)
    parser.add_argument(
        "--minimum-contradiction-groups",
        type=int,
        default=2,
    )
    args = parser.parse_args()

    result = validate_review(
        vault_root=args.vault,
        proposal_id=args.proposal_id,
        minimum_approved=args.minimum_approved,
        minimum_contradiction_groups=(
            args.minimum_contradiction_groups
        ),
    ).to_dict()
    print(json.dumps(result, indent=2))
    return 0 if result["ready_to_finalize"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
