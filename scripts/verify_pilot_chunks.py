from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alice_vault.chunk_catalog import verify_pilot_chunks

def main() -> int:
    parser = argparse.ArgumentParser(description="Verify chunk hashes, offsets, provenance, SQLite counts, and deterministic rebuild equivalence.")
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--pilot-name", default="pilot-v1")
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args()
    result = verify_pilot_chunks(args.vault, args.pilot_name, args.policy)
    print(json.dumps(result, indent=2))
    return 0 if result["ready_for_indexing"] else 2
if __name__ == "__main__":
    raise SystemExit(main())
