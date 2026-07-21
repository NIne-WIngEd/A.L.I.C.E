from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alice_vault.query_claim_relevance_calibration import prepare_query_claim_relevance_calibration
p = argparse.ArgumentParser()
p.add_argument("--vault", required=True, type=Path); p.add_argument("--benchmark", required=True, type=Path); p.add_argument("--audit-details", type=Path)
p.add_argument("--pilot-name", default="pilot-v1"); p.add_argument("--sample-size", type=int); p.add_argument("--policy", type=Path); p.add_argument("--judge-policy", type=Path); p.add_argument("--device", default="auto")
a = p.parse_args()
print(json.dumps(prepare_query_claim_relevance_calibration(vault_root=a.vault, benchmark_path=a.benchmark, audit_details_path=a.audit_details, pilot_name=a.pilot_name, sample_size=a.sample_size, policy_path=a.policy, judge_policy_path=a.judge_policy, device=a.device), indent=2))
