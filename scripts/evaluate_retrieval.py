from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from alice_vault.retrieval import evaluate

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--vault', required=True, type=Path)
    p.add_argument('--pilot-name', default='pilot-v1')
    p.add_argument('--benchmark', type=Path)
    a = p.parse_args()
    result = evaluate(vault_root=a.vault, pilot_name=a.pilot_name,
                      benchmark_path=a.benchmark)
    print(json.dumps(result, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
