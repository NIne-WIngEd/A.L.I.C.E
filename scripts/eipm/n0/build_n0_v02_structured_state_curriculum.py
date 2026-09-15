#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from alice_personality.n0.curriculum import sha256_file
from alice_personality.n0.curriculum_data import CurriculumDataset
from alice_personality.n0.v02_training import verify_teacher_registry


FIELD_TYPE_IDS = {"padding": 0, "context": 1, "candidate": 2}
PROVENANCE_IDS = {"padding": 0, "governed_public_teacher": 1}
RELATION_ROLE_IDS = {"padding": 0, "context_for_candidate": 1, "candidate_response": 2}
TEMPORAL_SCOPE_IDS = {"padding": 0, "unspecified": 1}


def compile_teacher_row(row: dict[str, Any], split: str) -> list[dict[str, Any]]:
    row_id = str(row.get("id", "")).strip()
    prompt = str(row.get("prompt", "")).strip()
    rationale = str(row.get("rationale", "")).strip()
    competency = str(row.get("competency", "")).strip()
    principle_tag = str(row.get("principle_tag", "")).strip()
    candidates = [str(value).strip() for value in row.get("candidates", [])]
    preferred = {int(value) for value in row.get("preferred_indices", [])}

    if not row_id or not prompt or not rationale or not competency:
        raise ValueError(f"teacher row {row_id or '<missing>'} lacks required text")
    if len(candidates) < 2 or not preferred:
        raise ValueError(f"teacher row {row_id} requires candidates and preferred indices")
    if min(preferred) < 0 or max(preferred) >= len(candidates):
        raise ValueError(f"teacher row {row_id} has preferred index outside candidate range")

    examples: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not candidate:
            raise ValueError(f"teacher row {row_id} contains an empty candidate")
        examples.append(
            {
                "id": f"{row_id}.structured.c{index}",
                "source_teacher_id": row_id,
                "split": split,
                "competency": competency,
                "principle_tag": principle_tag,
                "fields": [
                    {
                        "name": "context",
                        "text": prompt,
                        "field_type_id": 1,
                        "provenance_id": 1,
                        "relation_role_id": 1,
                        "temporal_scope_id": 1,
                        "confidence": 1.0,
                        "missing": False,
                    },
                    {
                        "name": "candidate",
                        "text": candidate,
                        "field_type_id": 2,
                        "provenance_id": 1,
                        "relation_role_id": 2,
                        "temporal_scope_id": 1,
                        "confidence": 1.0,
                        "missing": False,
                    },
                ],
                "target_text": rationale,
                "compatibility_label": 1 if index in preferred else 0,
                "text_generated_by_compiler": False,
            }
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-registry", required=True)
    parser.add_argument("--teacher-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    registry = Path(args.teacher_registry).resolve()
    audit = Path(args.teacher_audit).resolve()
    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()

    curriculum_paths, teacher_report = verify_teacher_registry(repo_root, registry, audit)
    examples: list[dict[str, Any]] = []
    source_split_rows: dict[str, int] = {}
    for split in ("train", "dev"):
        dataset = CurriculumDataset(curriculum_paths, split)
        source_split_rows[split] = len(dataset)
        for row in dataset.rows:
            examples.extend(compile_teacher_row(row, split))

    ids = [str(row["id"]) for row in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("compiled examples contain duplicate ids")

    split_counts = Counter(str(row["split"]) for row in examples)
    label_counts = Counter(int(row["compatibility_label"]) for row in examples)
    competency_counts = Counter(str(row["competency"]) for row in examples)
    if set(label_counts) != {0, 1}:
        raise ValueError("compiled examples must include positive and negative labels")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in examples:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "schema": "alice.eipm.n0.v02-structured-state-curriculum.v0.1",
        "status": "COMPILED_NOT_ACTIVATED",
        "source_teacher_registry": str(registry),
        "source_teacher_registry_sha256": sha256_file(registry),
        "source_teacher_audit": str(audit),
        "source_teacher_audit_sha256": sha256_file(audit),
        "source_registered_rows": int(teacher_report["registered_rows"]),
        "source_split_rows": source_split_rows,
        "compiled_file": str(output),
        "compiled_sha256": sha256_file(output),
        "compiled_rows": len(examples),
        "compiled_split_counts": dict(sorted(split_counts.items())),
        "compatibility_label_counts": {str(k): v for k, v in sorted(label_counts.items())},
        "competency_count": len(competency_counts),
        "text_generated_by_compiler": False,
        "field_type_ids": FIELD_TYPE_IDS,
        "provenance_ids": PROVENANCE_IDS,
        "relation_role_ids": RELATION_ROLE_IDS,
        "temporal_scope_ids": TEMPORAL_SCOPE_IDS,
        "activation_authorized": False
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
