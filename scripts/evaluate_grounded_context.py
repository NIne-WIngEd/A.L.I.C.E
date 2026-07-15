from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alice_vault.grounded_context_evaluation import evaluate_grounded_context

p = argparse.ArgumentParser()
p.add_argument("--vault", required=True, type=Path)
p.add_argument("--benchmark", required=True, type=Path)
p.add_argument("--pilot-name", default="pilot-v1")
p.add_argument("--device", default="auto")
a = p.parse_args()
result = evaluate_grounded_context(
    vault_root=a.vault, benchmark_path=a.benchmark,
    pilot_name=a.pilot_name, device=a.device
)
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["passes_coverage_threshold"] else 2)
