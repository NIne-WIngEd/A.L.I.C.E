from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.pilot_review import prepare_review


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a private human-review CSV for a pilot proposal."
    )
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--proposal-id")
    args = parser.parse_args()

    result = prepare_review(
        vault_root=args.vault,
        proposal_id=args.proposal_id,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
