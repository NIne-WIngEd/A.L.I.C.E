from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.inventory import inventory

parser = argparse.ArgumentParser(description="Read-only dataset inventory")
parser.add_argument("--source", required=True, type=Path)
parser.add_argument("--vault", required=True, type=Path)
parser.add_argument("--mode", choices=["metadata", "sha256"], default="metadata")
parser.add_argument(
    "--classification",
    choices=["PUBLIC", "INTERNAL", "PRIVATE", "HIGHLY_SENSITIVE", "SECRETS"],
    default="HIGHLY_SENSITIVE",
)
parser.add_argument("--exclude-dir", action="append", default=[])
args = parser.parse_args()

print(
    json.dumps(
        inventory(
            args.source,
            args.vault,
            args.mode,
            args.classification,
            args.exclude_dir,
        ),
        indent=2,
    )
)
