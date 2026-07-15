from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alice_vault.semantic_evaluation import validate_semantic_benchmark

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vault", required=True, type=Path)
    p.add_argument("--benchmark", required=True, type=Path)
    a = p.parse_args()
    result = validate_semantic_benchmark(
        vault_root=a.vault,
        benchmark_path=a.benchmark,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["ready_for_evaluation"] else 2

if __name__ == "__main__":
    raise SystemExit(main())
