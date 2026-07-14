from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from alice_vault.retrieval import create_lexical_benchmark

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--vault', required=True, type=Path)
    p.add_argument('--pilot-name', default='pilot-v1')
    p.add_argument('--cases', type=int)
    a = p.parse_args()
    result = create_lexical_benchmark(vault_root=a.vault,
                                      pilot_name=a.pilot_name,
                                      case_count=a.cases)
    print(json.dumps(result, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
