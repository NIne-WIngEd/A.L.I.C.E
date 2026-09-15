#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

import torch

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.v02_model import AliceN0V02Model
from alice_personality.n0.v02_training import export_ranker_state, sha256_file


EXACT_PARAMETERS = 136_594_435


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the exact random-init baseline for alice-n0-semantic-v0.2."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()

    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise SystemExit("Install requirements-n0.txt before baseline export") from exc

    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty baseline directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    config_path = Path(args.config).resolve()
    config = load_n0_config(config_path)
    model = AliceN0V02Model(config)
    report = model.parameter_report()
    if int(report["total_parameters"]) != EXACT_PARAMETERS:
        raise SystemExit(
            f"parameter gate failed: expected={EXACT_PARAMETERS} observed={report['total_parameters']}"
        )

    mlm_dir = output / "mlm"
    mlm_dir.mkdir(parents=True, exist_ok=True)
    model.mlm.save_pretrained(mlm_dir, safe_serialization=True)
    ranker_path = output / "ranker.safetensors"
    save_file(export_ranker_state(model), str(ranker_path))

    receipt = {
        "schema": "alice.eipm.n0.v02-random-baseline-receipt.v0.1",
        "model_id": config.model_id,
        "seed": args.seed,
        "exact_parameter_count": report["total_parameters"],
        "auxiliary_head_parameters": report["auxiliary_head_parameters"],
        "config_sha256": sha256_file(config_path),
        "ranker_sha256": sha256_file(ranker_path),
        "git_revision": git_revision(),
        "random_initialization_only": True,
        "v01_weights_used": False,
        "model_training_performed": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "private_identity_gradient_authorized": False,
    }
    receipt_path = output / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "parameters": report,
                "seed": args.seed,
                "receipt_sha256": sha256_file(receipt_path),
                "private_identity_gradient": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
