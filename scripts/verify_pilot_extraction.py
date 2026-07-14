from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.safe_extraction import verify_pilot_extraction


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify private pilot extraction hashes and provenance."
    )
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--pilot-name", default="pilot-v1")
    parser.add_argument("--registry", type=Path)
    args = parser.parse_args()

    result = verify_pilot_extraction(
        vault_root=args.vault,
        pilot_name=args.pilot_name,
        registry_path=args.registry,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["ready_for_chunking"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
