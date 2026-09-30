"""Turn owner-authorized fictional cases into train-only MFM curriculum JSONL.

Nominal renderer partitions in the candidate source are never called independent
evaluation. This conversion preserves their lineage but assigns training-only
status; model qualification must use a different source/generator family.
"""

from __future__ import annotations

import argparse
import base64
from hashlib import sha256
import json
import os
from pathlib import Path

from cognitive_kernel.formation_gold import compile_formation_case

from assemble_formation_curriculum import AUTHORIZATION, SCHEMA, _bytes


def convert(source: Path, output: Path, manifest: Path, *, expected_sha256: str) -> dict:
    raw = source.read_bytes()
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("fictional candidate bytes differ from frozen digest")
    rows = json.loads(raw)
    if not isinstance(rows, list) or not rows:
        raise ValueError("fictional candidate corpus is empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_suffix(output.suffix + ".tmp")
    digest = sha256()
    cases: set[str] = set()
    hosts: set[str] = set()
    scenarios: dict[str, int] = {}
    with staging.open("wb") as writer:
        for row in rows:
            if row.get("admission") != "candidate_only_unreviewed" or row.get(
                    "training_rights") != "owner_authorized_chatgpt_codex_teaching_output":
                raise ValueError("case origin or owner teaching authorization changed")
            compiled = compile_formation_case(row)
            if compiled.gold.case_id in cases:
                raise ValueError("duplicate candidate case")
            cases.add(compiled.gold.case_id)
            hosts.add(compiled.host_family)
            scenarios[ row["scenario_family"] ] = scenarios.get(row["scenario_family"], 0) + 1
            values = dict(compiled.texts)
            record = {
                "schema": SCHEMA, "case_id": compiled.gold.case_id,
                "split": "train", "origin_nominal_split": compiled.split,
                "authorization_id": AUTHORIZATION,
                "origin_corpus_sha256": expected_sha256,
                "source_family": compiled.source_family,
                "generator_family": compiled.generator_family,
                "scenario_family": row["scenario_family"],
                "context": compiled.gold.context.metadata_record(),
                "sources": [{"ref_id": ref.ref_id,
                             "content_b64": base64.b64encode(
                                 values[ref.ref_id].encode("utf-8")).decode("ascii")}
                            for ref in compiled.gold.context.evidence],
                "target": {"schema": "mfm-formation-target-v1",
                           "proposals": [p.record() for p in compiled.gold.expected],
                           "dispositions": [d.record() for d in
                                            compiled.gold.expected_dispositions]},
            }
            b = _bytes(record)
            digest.update(b)
            writer.write(b)
    os.replace(staging, output)
    receipt = {
        "schema": "mfm-synthetic-curriculum-receipt-v1",
        "status": "owner_authorized_synthetic_training_only_no_independent_final",
        "authorization_id": AUTHORIZATION,
        "origin_corpus_sha256": expected_sha256,
        "curriculum_sha256": digest.hexdigest(),
        "curriculum_path": output.name,
        "cases": len(cases), "host_families": len(hosts),
        "scenario_counts": scenarios,
        "generator_family": "chatgpt-templated-diagnostic-v1",
    }
    manifest.write_bytes(_bytes(receipt))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output, args.manifest,
                             expected_sha256=args.source_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
