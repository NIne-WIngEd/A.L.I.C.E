from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.hhem_holdout import prepare_hhem_holdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--audit-details", type=Path)
    parser.add_argument("--pilot-name", default="pilot-v1")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--judge-policy", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    result = prepare_hhem_holdout(
        vault_root=args.vault,
        benchmark_path=args.benchmark,
        calibration_bundle_path=args.calibration,
        audit_details_path=args.audit_details,
        pilot_name=args.pilot_name,
        sample_size=args.sample_size,
        holdout_policy_path=args.policy,
        judge_policy_path=args.judge_policy,
        device=args.device,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
