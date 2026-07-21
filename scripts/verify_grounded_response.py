from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.grounded_response import verify_grounded_response_package


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--response",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--context-package",
        required=True,
        type=Path,
    )
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args()

    result = verify_grounded_response_package(
        response_path=args.response,
        context_package_path=args.context_package,
        policy_path=args.policy,
    )
    print(json.dumps(result, indent=2))
    return (
        0
        if result["ready_for_conversation"]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
