from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.hhem_calibration import evaluate_hhem_calibration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--pilot-name", default="pilot-v1")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()

    result = evaluate_hhem_calibration(
        vault_root=args.vault,
        calibration_bundle_path=args.calibration,
        pilot_name=args.pilot_name,
        policy_path=args.policy,
        device=args.device,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
