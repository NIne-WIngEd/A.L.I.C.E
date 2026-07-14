from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alice_vault.chunk_catalog import build_pilot_chunks

def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic private chunks from verified pilot extraction outputs.")
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--pilot-name", default="pilot-v1")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_pilot_chunks(args.vault, args.pilot_name, args.policy, args.replace), indent=2))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
