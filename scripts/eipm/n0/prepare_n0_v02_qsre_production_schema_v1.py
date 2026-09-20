from __future__ import annotations

import argparse
from pathlib import Path

import torch

from qsre_production_runtime import materialize_dynamic_relation_schema


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--schema",required=True)
    p.add_argument("--semantic-config",required=True)
    p.add_argument("--semantic-checkpoint",required=True)
    p.add_argument("--tokenizer-dir",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--receipt",required=True)
    p.add_argument("--device",default="cpu")
    args=p.parse_args()

    receipt=materialize_dynamic_relation_schema(
        schema_path=Path(args.schema),
        semantic_config_path=Path(args.semantic_config),
        semantic_checkpoint=Path(args.semantic_checkpoint),
        tokenizer_dir=Path(args.tokenizer_dir),
        output_path=Path(args.output),
        receipt_path=Path(args.receipt),
        device=torch.device(args.device),
    )
    print("status="+receipt["status"])
    print("schema_cache_sha256="+receipt["schema_cache_sha256"])
    print("relations="+str(receipt["relations"]))


if __name__=="__main__":
    main()
