from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.inventory import contains, ensure_layout

parser = argparse.ArgumentParser()
parser.add_argument("--vault", required=True, type=Path)
args = parser.parse_args()

vault = args.vault.expanduser().resolve()
if contains(vault, ROOT):
    raise SystemExit("Refusing to create private vault inside Git repository")

vault.mkdir(parents=True, exist_ok=True)
ensure_layout(vault)
(vault / ".alice_vault").write_text("A.L.I.C.E. private vault\n", encoding="utf-8")
(vault / "vault.json").write_text(
    json.dumps(
        {
            "vault_format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "root": str(vault),
            "default_data_classification": "HIGHLY_SENSITIVE",
            "raw_originals_immutable": True,
            "public_repository_allowed": False,
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(f"Vault initialized: {vault}")
print("No source data was copied or modified.")
