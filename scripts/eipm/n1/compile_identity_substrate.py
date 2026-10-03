"""Local, explicitly pinned source compilation; no model load or training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.alice_personality.n1.compiler import (  # noqa: E402
    IdentitySubstrateError, compile_package, curated_frontier_v2_pin,
    load_package_pin, verify_compiled,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("compile")
    create.add_argument("package")
    create.add_argument("output_dir")
    pins = create.add_mutually_exclusive_group(required=True)
    pins.add_argument("--curated-frontier-v2", action="store_true")
    pins.add_argument("--pin-json")
    create.add_argument("--raw-lineage-json")
    create.add_argument("--split-seed", default="alice-eipm-evidence-families-v1")
    verify = commands.add_parser("verify")
    verify.add_argument("output_dir")
    verify.add_argument("--expected-source-sha256")
    verify.add_argument("--expected-receipt-sha256")
    args = parser.parse_args(argv)
    try:
        if args.command == "compile":
            pin = curated_frontier_v2_pin() if args.curated_frontier_v2 else load_package_pin(args.pin_json)
            registry = json.loads(Path(args.raw_lineage_json).read_text(encoding="utf-8")) \
                if args.raw_lineage_json else None
            receipt = compile_package(args.package, args.output_dir, pin=pin,
                                      split_seed=args.split_seed, raw_lineage_registry=registry)
        else:
            receipt = verify_compiled(args.output_dir,
                                      expected_source_archive_sha256=args.expected_source_sha256,
                                      expected_receipt_sha256=args.expected_receipt_sha256)
        print(json.dumps({"schema": receipt["schema"], "state": receipt["state"],
                          "receipt_sha256": receipt["receipt_sha256"],
                          "active_record_count": receipt["active_record_count"],
                          "source_family_count": receipt["source_family_count"],
                          "private_gradient_authorized": receipt["private_gradient_authorized"],
                          "acceptance_authority": receipt["acceptance_authority"]}, sort_keys=True))
        return 0
    except (IdentitySubstrateError, OSError, ValueError, TypeError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
