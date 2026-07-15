from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alice_vault.grounded_context import build_grounded_context

p = argparse.ArgumentParser()
p.add_argument("--vault", required=True, type=Path)
p.add_argument("--query", required=True)
p.add_argument("--pilot-name", default="pilot-v1")
p.add_argument("--max-sources", type=int)
p.add_argument("--device", default="auto")
a = p.parse_args()
result = build_grounded_context(
    vault_root=a.vault, query=a.query, pilot_name=a.pilot_name,
    max_sources=a.max_sources, device=a.device
)
print(json.dumps(result["summary"], indent=2))
