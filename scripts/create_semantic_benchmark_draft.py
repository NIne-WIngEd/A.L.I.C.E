from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alice_vault.semantic_evaluation import create_semantic_benchmark_draft

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vault", required=True, type=Path)
    p.add_argument("--questions", required=True, type=Path)
    p.add_argument("--pilot-name", default="pilot-v1")
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    result = create_semantic_benchmark_draft(
        vault_root=a.vault,
        questions_path=a.questions,
        pilot_name=a.pilot_name,
        device=a.device,
    )
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
