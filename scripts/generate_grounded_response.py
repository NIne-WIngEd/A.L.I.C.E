from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.grounded_response import generate_grounded_response


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument(
        "--context-package",
        required=True,
        type=Path,
    )
    parser.add_argument("--policy", type=Path)
    parser.add_argument(
        "--show-answer",
        action="store_true",
    )
    args = parser.parse_args()

    result = generate_grounded_response(
        vault_root=args.vault,
        context_package_path=args.context_package,
        policy_path=args.policy,
    )
    output = dict(result["summary"])
    if args.show_answer:
        output["answer"] = result[
            "response_package"
        ]["model_output"]["answer"]

    print(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if output["verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
