from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alice_vault.retrieval import SearchFilters
from alice_vault.semantic_retrieval import hybrid_search

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vault", required=True, type=Path)
    p.add_argument("--query", required=True)
    p.add_argument("--pilot-name", default="pilot-v1")
    p.add_argument("--family", action="append", default=[])
    p.add_argument("--year", action="append", default=[])
    p.add_argument("--source-bucket", action="append", default=[])
    p.add_argument("--contradiction", action="append", default=[])
    p.add_argument("--exclude-truncated", action="store_true")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    result = hybrid_search(
        vault_root=a.vault,
        query=a.query,
        pilot_name=a.pilot_name,
        filters=SearchFilters(
            families=tuple(a.family),
            years=tuple(a.year),
            source_buckets=tuple(a.source_bucket),
            contradiction_labels=tuple(a.contradiction),
            include_truncated=not a.exclude_truncated,
        ),
        limit=a.limit,
        device=a.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
