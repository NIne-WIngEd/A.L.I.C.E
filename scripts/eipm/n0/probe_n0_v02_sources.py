#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested_get(row: dict[str, Any], dotted: str) -> Any:
    current: Any = row
    for key in dotted.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def flatten_sources(plan: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for group in plan["mixture"]:
        category = str(group["category"])
        for source in group["sources"]:
            sources.append(
                {
                    "category": category,
                    "repo_id": str(source["repo_id"]),
                    "target_share": float(source["target_share"]),
                }
            )
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve exact Hugging Face revisions and inspect the row-level rights/provenance "
            "schema for every planned N0 v0.2 public source. This probe never authorizes training."
        )
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-rows", type=int, default=64)
    parser.add_argument("--requested-revision", default="main")
    args = parser.parse_args()

    if args.sample_rows < 1:
        raise SystemExit("--sample-rows must be positive")

    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install requirements-n0.txt before probing N0 v0.2 sources") from exc

    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "planned_not_activated":
        raise SystemExit("source probe expects a non-activated corpus plan")

    sources = flatten_sources(plan)
    if not sources:
        raise SystemExit("corpus plan contains no sources")

    category_share = sum(float(group["target_share"]) for group in plan["mixture"])
    source_share = sum(source["target_share"] for source in sources)
    if abs(category_share - 1.0) > 1e-9 or abs(source_share - 1.0) > 1e-9:
        raise SystemExit("corpus plan mixture shares must sum to 1.0")

    api = HfApi()
    reports: list[dict[str, Any]] = []
    failures = 0

    for planned in sources:
        repo_id = planned["repo_id"]
        report: dict[str, Any] = {
            **planned,
            "requested_revision": args.requested_revision,
            "split": "train",
            "text_field": "text",
            "license_field": "metadata.license",
            "provenance_field": "metadata.provenance",
            "url_field": "metadata.url",
            "id_field": "id",
            "training_authorized": False,
            "private_identity_data": False,
        }
        try:
            info = api.dataset_info(repo_id=repo_id, revision=args.requested_revision)
            resolved_revision = str(info.sha)
            report["resolved_revision"] = resolved_revision

            dataset = load_dataset(
                repo_id,
                split="train",
                streaming=True,
                revision=resolved_revision,
            )
            rows = list(itertools.islice(iter(dataset), args.sample_rows))
            if not rows:
                raise RuntimeError("streaming dataset returned zero rows")

            top_level_fields: set[str] = set()
            metadata_fields: set[str] = set()
            license_counts: Counter[str] = Counter()
            presence = Counter()
            text_nonempty = 0

            for row in rows:
                top_level_fields.update(str(key) for key in row.keys())
                metadata = row.get("metadata")
                if isinstance(metadata, dict):
                    metadata_fields.update(str(key) for key in metadata.keys())

                if "text" in row:
                    presence["text"] += 1
                    if str(row.get("text") or "").strip():
                        text_nonempty += 1
                if row.get("id") not in (None, ""):
                    presence["id"] += 1

                row_license = nested_get(row, "metadata.license")
                if row_license not in (None, ""):
                    presence["license"] += 1
                    license_counts[str(row_license)] += 1

                provenance = nested_get(row, "metadata.provenance")
                if provenance not in (None, ""):
                    presence["provenance"] += 1

                url = nested_get(row, "metadata.url")
                if url not in (None, ""):
                    presence["url"] += 1

            sampled = len(rows)
            schema_pass = all(
                presence[name] == sampled
                for name in ("text", "id", "license", "provenance")
            )
            report.update(
                {
                    "sampled_rows": sampled,
                    "top_level_fields": sorted(top_level_fields),
                    "metadata_fields": sorted(metadata_fields),
                    "required_presence": {
                        name: int(presence[name])
                        for name in ("text", "id", "license", "provenance", "url")
                    },
                    "text_nonempty_rows": text_nonempty,
                    "observed_license_counts": dict(sorted(license_counts.items())),
                    "schema_pass": schema_pass,
                    "manual_license_review_required": True,
                    "candidate_allowed_license_values": [],
                }
            )
            if not schema_pass:
                failures += 1
        except Exception as exc:  # keep the full source matrix visible in one run
            failures += 1
            report.update(
                {
                    "schema_pass": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "manual_license_review_required": True,
                    "candidate_allowed_license_values": [],
                }
            )

        reports.append(report)
        print(
            json.dumps(
                {
                    "repo_id": repo_id,
                    "resolved_revision": report.get("resolved_revision"),
                    "sampled_rows": report.get("sampled_rows", 0),
                    "schema_pass": report.get("schema_pass", False),
                    "observed_license_values": sorted(
                        (report.get("observed_license_counts") or {}).keys()
                    ),
                    "error": report.get("error"),
                },
                sort_keys=True,
            )
        )

    result = {
        "schema": "alice.eipm.n0.public-source-schema-probe.v0.2",
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "requested_revision": args.requested_revision,
        "sample_rows_per_source": args.sample_rows,
        "source_count": len(reports),
        "schema_pass_count": sum(bool(row.get("schema_pass")) for row in reports),
        "schema_failure_count": failures,
        "all_schema_passed": failures == 0,
        "sources": reports,
        "manual_license_review_required": True,
        "training_authorized": False,
        "tokenizer_authorized": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "source_count": result["source_count"],
                "schema_pass_count": result["schema_pass_count"],
                "schema_failure_count": result["schema_failure_count"],
                "all_schema_passed": result["all_schema_passed"],
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )

    if failures:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
