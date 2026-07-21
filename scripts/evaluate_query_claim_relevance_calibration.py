from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alice_vault.query_claim_relevance_calibration import evaluate_query_claim_relevance_calibration
p=argparse.ArgumentParser(); p.add_argument("--vault", required=True, type=Path); p.add_argument("--calibration", required=True, type=Path); p.add_argument("--policy", type=Path)
a=p.parse_args(); print(json.dumps(evaluate_query_claim_relevance_calibration(vault_root=a.vault, bundle_path=a.calibration, policy_path=a.policy), indent=2))
