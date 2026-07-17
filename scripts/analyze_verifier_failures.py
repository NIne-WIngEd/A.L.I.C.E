from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.verifier_failure_forensics import analyze_verifier_failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", required=True, type=Path)
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--pilot-name", default="pilot-v1")
    args = parser.parse_args()

    result = analyze_verifier_failures(
        holdout_details_path=args.details,
        holdout_bundle_path=args.holdout,
        vault_root=args.vault,
        pilot_name=args.pilot_name,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
