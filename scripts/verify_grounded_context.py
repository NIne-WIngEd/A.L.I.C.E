from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alice_vault.grounded_context import verify_grounded_context_package

p = argparse.ArgumentParser()
p.add_argument("--package", required=True, type=Path)
a = p.parse_args()
result = verify_grounded_context_package(package_path=a.package)
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["ready_for_llm_context"] else 2)
