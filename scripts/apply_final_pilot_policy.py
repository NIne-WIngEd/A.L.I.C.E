from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.final_pilot_policy import apply_final_pilot_policy


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a conservative no-model final policy to the latest "
            "calibrated pilot review."
        )
    )
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument(
        "--approve-threshold",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--truncated-approve-threshold",
        type=float,
        default=0.92,
    )
    parser.add_argument(
        "--reject-threshold",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--audit-approved",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--audit-rejected",
        type=int,
        default=5,
    )
    args = parser.parse_args()

    result = apply_final_pilot_policy(
        vault_root=args.vault,
        approve_threshold=args.approve_threshold,
        truncated_approve_threshold=(
            args.truncated_approve_threshold
        ),
        reject_threshold=args.reject_threshold,
        audit_approved=args.audit_approved,
        audit_rejected=args.audit_rejected,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
