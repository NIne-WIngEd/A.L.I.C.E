"""Close/export or admit one externally pinned PUBLIC expanded feature package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.alice_personality.gemma_n0.public_feature_handoff import (
    HandoffError, export_public_features, import_public_features, write_import_receipt,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--experiment", type=Path, required=True)
    export.add_argument("--experiment-sha256", required=True)
    export.add_argument("--plan", type=Path, required=True)
    export.add_argument("--cache-directory", type=Path, required=True)
    export.add_argument("--output-directory", type=Path, required=True)
    admit = commands.add_parser("import")
    admit.add_argument("--package-directory", type=Path, required=True)
    admit.add_argument("--manifest-sha256", required=True)
    admit.add_argument("--closure-sha256", required=True)
    admit.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            summary = export_public_features(experiment_path=args.experiment,
                expected_experiment_sha256=args.experiment_sha256, plan_path=args.plan,
                cache_directory=args.cache_directory, output_directory=args.output_directory)
        else:
            imported = import_public_features(args.package_directory,
                expected_manifest_sha256=args.manifest_sha256, expected_closure_sha256=args.closure_sha256)
            write_import_receipt(imported, args.output_receipt)
            summary = imported.binding
    except (HandoffError, OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
