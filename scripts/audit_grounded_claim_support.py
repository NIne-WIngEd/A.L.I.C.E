from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.claim_support_audit import (
    audit_benchmark_claim_support,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vault",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--pilot-name",
        default="pilot-v1",
    )
    parser.add_argument(
        "--scope",
        choices=[
            "all",
            "expected-source-misses",
        ],
        default="all",
    )
    parser.add_argument(
        "--response-evaluation-details",
        type=Path,
    )
    parser.add_argument(
        "--audit-policy",
        type=Path,
    )
    parser.add_argument(
        "--response-policy",
        type=Path,
    )
    parser.add_argument(
        "--device",
        default="auto",
    )
    args = parser.parse_args()

    result = audit_benchmark_claim_support(
        vault_root=args.vault,
        benchmark_path=args.benchmark,
        pilot_name=args.pilot_name,
        scope=args.scope,
        response_evaluation_details_path=(
            args.response_evaluation_details
        ),
        audit_policy_path=args.audit_policy,
        response_policy_path=(
            args.response_policy
        ),
        device=args.device,
    )
    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    # Mismatch-only is diagnostic, not the final P1.11 gate.
    if not result[
        "passes_all_audit_thresholds"
    ]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
