#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from alice_personality.n0.corpus_receipt import verify_corpus_receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify an N0 public corpus receipt and every manifested shard before training."
    )
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--source-config", required=True)
    args = parser.parse_args()
    result = verify_corpus_receipt(args.corpus_dir, args.source_config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
