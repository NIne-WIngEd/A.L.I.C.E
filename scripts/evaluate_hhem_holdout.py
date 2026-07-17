from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.hhem_holdout import evaluate_hhem_holdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--pilot-name", default="pilot-v1")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--hhem-policy", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()

    result = evaluate_hhem_holdout(
        vault_root=args.vault,
        holdout_bundle_path=args.holdout,
        pilot_name=args.pilot_name,
        holdout_policy_path=args.policy,
        hhem_policy_path=args.hhem_policy,
        device=args.device,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
