from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alice_vault.semantic_retrieval import prepare_embedding_model

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vault", required=True, type=Path)
    p.add_argument("--policy", type=Path)
    p.add_argument("--device", default="auto")
    p.add_argument("--replace", action="store_true")
    a = p.parse_args()
    result = prepare_embedding_model(
        vault_root=a.vault,
        policy_path=a.policy,
        device=a.device,
        replace=a.replace,
    )
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
