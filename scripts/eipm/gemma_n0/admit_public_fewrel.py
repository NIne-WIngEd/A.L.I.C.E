"""Admit the pinned public FewRel TRAIN/DEV pair; never training approval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.alice_personality.gemma_n0.public_fewrel import (
    PublicSourceError, admit_fewrel_train_dev, verify_fewrel_admission,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    admit = commands.add_parser("admit")
    admit.add_argument("--rows", type=Path, required=True)
    admit.add_argument("--bank", type=Path, required=True)
    admit.add_argument("--manifest", type=Path, required=True)
    admit.add_argument("--audit", type=Path, required=True)
    admit.add_argument("--output", type=Path, required=True)
    admit.add_argument("--declared-public-train-dev", action="store_true", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "admit":
            receipt = admit_fewrel_train_dev(
                args.rows, args.bank, args.manifest, args.audit, args.output,
                declared_public=args.declared_public_train_dev)
        else:
            receipt = verify_fewrel_admission(args.receipt)
    except (PublicSourceError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"state": receipt["state"], "receipt_sha256": receipt["receipt_sha256"],
                      "statistics": receipt["statistics"], "training_authorized": False,
                      "n0_approved": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
