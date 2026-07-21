from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.owner_attribution import initialize_owner_identity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument(
        "--primary-name",
        required=True,
    )
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--replace",
        action="store_true",
    )
    args = parser.parse_args()

    result = initialize_owner_identity(
        vault_root=args.vault,
        primary_name=args.primary_name,
        aliases=args.alias,
        replace=args.replace,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
