from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.judge_calibration import evaluate_judge_calibration


def latest_bundle(vault: Path, pilot_name: str) -> Path:
    directory = vault / "manifests" / "calibration" / pilot_name
    candidates = sorted(
        directory.glob("judge-calibration-*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No judge-calibration bundle was found")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--pilot-name", default="pilot-v1")
    args = parser.parse_args()

    vault = args.vault.expanduser().resolve(strict=True)
    bundle = (
        args.calibration.expanduser().resolve(strict=True)
        if args.calibration
        else latest_bundle(vault, args.pilot_name)
    )

    result = evaluate_judge_calibration(
        vault_root=vault,
        bundle_path=bundle,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
