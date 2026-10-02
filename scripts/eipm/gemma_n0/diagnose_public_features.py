"""Reproduce one externally pinned closed PUBLIC semantic cache and audit sources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.alice_personality.gemma_n0.public_feature_diagnostics import DiagnosticError, diagnose_public_features
from src.alice_personality.gemma_n0.public_feature_handoff import HandoffError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-directory", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--closure-sha256", required=True)
    parser.add_argument("--semantic-export", type=Path, required=True)
    parser.add_argument("--export-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        receipt = diagnose_public_features(args.package_directory,
            expected_manifest_sha256=args.manifest_sha256, expected_closure_sha256=args.closure_sha256,
            semantic_export_path=args.semantic_export, expected_export_sha256=args.export_sha256,
            output_directory=args.output_directory, cpu_threads=args.cpu_threads)
    except (DiagnosticError, HandoffError, OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: receipt[key] for key in ("schema", "state", "status", "receipt_sha256",
                                                 "n0_approved", "personality_qualified")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
