from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.safe_extraction import extract_pilot


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract verified text from an approved private pilot snapshot "
            "using the machine-readable parser registry."
        )
    )
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--pilot-name", default="pilot-v1")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    summary = extract_pilot(
        vault_root=args.vault,
        pilot_name=args.pilot_name,
        registry_path=args.registry,
        resume=not args.no_resume,
        fail_on_error=args.fail_on_error,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["failed_extractions"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
