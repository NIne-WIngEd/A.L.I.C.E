from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from alice_vault.retrieval import verify_index

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--vault', required=True, type=Path)
    p.add_argument('--pilot-name', default='pilot-v1')
    p.add_argument('--chunk-set-id')
    p.add_argument('--policy', type=Path)
    a = p.parse_args()
    result = verify_index(vault_root=a.vault, pilot_name=a.pilot_name,
                          chunk_set_id=a.chunk_set_id, policy_path=a.policy)
    print(json.dumps(result, indent=2))
    return 0 if result['ready_for_evaluation'] else 2

if __name__ == '__main__':
    raise SystemExit(main())
