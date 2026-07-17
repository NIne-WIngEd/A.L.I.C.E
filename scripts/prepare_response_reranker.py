from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from alice_vault.response_reranker import prepare_response_reranker
p=argparse.ArgumentParser()
p.add_argument("--vault",required=True,type=Path)
p.add_argument("--policy",type=Path)
a=p.parse_args()
print(json.dumps(prepare_response_reranker(vault_root=a.vault,policy_path=a.policy),indent=2))
