"""Freeze newly generated Multi-Source source metadata without QA or labels.

This inventory is an annotation intake, not a formation training corpus. It
contains source paths/digests and no simulator truth, model API output, or
reviewer claims. The upstream generator and dataset licenses remain attached.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess


SOURCE_TYPES = ("daily_self_report", "device_log", "objective_log",
                "planner", "profile_ltm")
GENERATOR_FILES = ("generate_personas.py", "generate_events.py",
                   "generate_sources.py", "source_projector.py")


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(("git", *args), cwd=cwd, text=True).strip()


def inventory(dataset: Path, upstream: Path, *, seed: int) -> dict[str, object]:
    dataset = dataset.resolve()
    upstream = upstream.resolve()
    if _git("status", "--porcelain", cwd=upstream):
        raise ValueError("upstream generator worktree is not clean")
    version = _git("rev-parse", "HEAD", cwd=upstream)
    license_file = upstream / "data/DATA_LICENSE"
    license_text = license_file.read_text()
    if "Attribution 4.0 International" not in license_text or "Cached LLM outputs" not in license_text:
        raise ValueError("expected upstream data license and provider carveout are missing")
    manifest = dataset / "config/personas.json"
    config = json.loads(manifest.read_bytes())
    meta = config["benchmark_metadata"]
    if meta["seed"] != seed:
        raise ValueError("upstream seed does not match the requested fresh run")
    personas = config["personas"]
    records = []
    for persona in sorted(personas, key=lambda p: p["id"]):
        persona_id = persona["id"]
        if not isinstance(persona_id, str) or not persona_id.startswith("bench_"):
            raise ValueError("unexpected persona identity")
        home = dataset / persona_id
        ancestor = home / "event_table.json"
        source_records = []
        for kind in SOURCE_TYPES:
            path = home / "structural_sources" / f"{kind}.json"
            payload = json.loads(path.read_bytes())
            if payload["persona_id"] != persona_id or payload["source_type"] != kind:
                raise ValueError("structural source identity or type mismatch")
            units = payload.get("records", payload.get("facts"))
            if not isinstance(units, (list, dict)) or not units:
                raise ValueError("structural source has no observed units")
            source_records.append({"source_type": kind,
                                   "path": path.relative_to(dataset).as_posix(),
                                   "sha256": _hash(path),
                                   "units": len(units)})
        records.append({"persona_id": persona_id, "host_family": persona_id,
                        "generator_family": f"multisource-membench-{version}",
                        "ancestor_event_table_sha256": _hash(ancestor),
                        "sources": source_records})
    generator_root = upstream / "src/survey2agent/data_generation"
    return {
        "schema": "mfm-source-annotation-inventory-v1",
        "status": "source_candidate_only_no_formation_labels_or_review",
        "training_admitted": False,
        "upstream_repo": "https://github.com/TianchengY/multisource-membench",
        "upstream_commit": version,
        "seed": seed,
        "generator_sha256s": {name: _hash(generator_root / name) for name in GENERATOR_FILES},
        "upstream_personas_sha256": _hash(manifest),
        "upstream_code_license": {"spdx": "Apache-2.0", "sha256": _hash(upstream / "LICENSE")},
        "upstream_data_license": {"spdx": "CC-BY-4.0", "sha256": _hash(license_file),
                                  "source": "data/DATA_LICENSE"},
        "attribution": "Multi-Source Memory Benchmark, Tiancheng Yin et al.; see upstream CITATION.cff",
        "excluded_from_formation_input": ["ground_truth.json", "benchmark/results/",
                                          "extracted_atoms/", "method_outputs/",
                                          "event_table.json (simulator ancestor, not an observed source)"],
        "source_families": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--generator-repo", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = inventory(args.dataset_dir, args.generator_repo, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
    args.output.write_bytes(raw)
    print(json.dumps({"output_sha256": sha256(raw).hexdigest(),
                      "source_families": len(result["source_families"]),
                      "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
