from __future__ import annotations

import json
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))


def _bools(detail: dict[str, Any]) -> tuple[bool, bool, bool, bool]:
    human = str(detail.get("human_label", "")) == "supported"
    qwen = str(detail.get("qwen_verdict", "")).lower() == "supported"
    hhem = bool(detail.get("hhem_predicted_supported", False))
    fever = str(detail.get("fever_decision", "")) == "keep_entailment"
    return human, qwen, hhem, fever


def analyze_verifier_failures(
    *,
    holdout_details_path: Path,
    holdout_bundle_path: Path,
    vault_root: Path,
    pilot_name: str = "pilot-v1",
) -> dict[str, Any]:
    details = _load(holdout_details_path)
    bundle = _load(holdout_bundle_path)
    vault_root = vault_root.expanduser().resolve(strict=True)

    if int(details.get("hhem_holdout_evaluation_schema_version", -1)) != 1:
        raise ValueError("Expected HHEM holdout evaluation details")
    if int(bundle.get("holdout_schema_version", -1)) != 1:
        raise ValueError("Expected HHEM holdout bundle")

    bundle_by_id = {
        str(item.get("item_id", "")): item
        for item in bundle.get("items", [])
        if str(item.get("item_id", ""))
    }

    rows: list[dict[str, Any]] = []
    vector_counts: Counter[str] = Counter()
    vector_label_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for detail in details.get("items", []):
        item_id = str(detail.get("item_id", ""))
        source = bundle_by_id.get(item_id, {})
        human, qwen, hhem, fever = _bools(detail)
        vector = f"Q{int(qwen)}H{int(hhem)}F{int(fever)}"
        vector_counts[vector] += 1
        vector_label_counts[vector][str(detail.get("human_label", ""))] += 1

        evidence_windows = source.get("evidence_windows", [])
        evidence_texts = []
        for window in evidence_windows:
            if isinstance(window, dict):
                text = str(window.get("text", "")).strip()
            else:
                text = str(window).strip()
            if text and text not in evidence_texts:
                evidence_texts.append(text)

        rows.append({
            "item_id": item_id,
            "query_id": str(detail.get("query_id", "")),
            "human_label": str(detail.get("human_label", "")),
            "claim_text": str(source.get("claim_text", "")),
            "evidence_texts": evidence_texts,
            "qwen_supported": qwen,
            "hhem_supported": hhem,
            "hhem_score": detail.get("hhem_score"),
            "fever_supported": fever,
            "decision_vector": vector,
            "qwen_false_positive": (not human) and qwen,
            "hhem_false_positive": (not human) and hhem,
            "fever_false_positive": (not human) and fever,
            "unanimous_false_positive": (not human) and qwen and hhem and fever,
            "qwen_false_negative": human and (not qwen),
            "hhem_false_negative": human and (not hhem),
            "fever_false_negative": human and (not fever),
        })

    qwen_fp = [row for row in rows if row["qwen_false_positive"]]
    hhem_fp = [row for row in rows if row["hhem_false_positive"]]
    unanimous_fp = [row for row in rows if row["unanimous_false_positive"]]

    public_summary = {
        "verifier_failure_forensics_schema_version": 1,
        "run_id": str(uuid.uuid4()),
        "created_at": _now(),
        "pilot_name": pilot_name,
        "source_holdout_run_id": str(details.get("run_id", "")),
        "source_holdout_id": str(details.get("holdout_id", "")),
        "item_count": len(rows),
        "decision_vector_counts": dict(sorted(vector_counts.items())),
        "decision_vector_human_labels": {
            key: dict(value)
            for key, value in sorted(vector_label_counts.items())
        },
        "qwen_false_positive_count": len(qwen_fp),
        "hhem_false_positive_count": len(hhem_fp),
        "fever_false_positive_count": sum(row["fever_false_positive"] for row in rows),
        "unanimous_false_positive_count": len(unanimous_fp),
        "qwen_hhem_shared_false_positive_count": sum(
            row["qwen_false_positive"] and row["hhem_false_positive"]
            for row in rows
        ),
        "qwen_false_positive_query_ids": sorted({
            row["query_id"] for row in qwen_fp if row["query_id"]
        }),
        "hhem_false_positive_query_ids": sorted({
            row["query_id"] for row in hhem_fp if row["query_id"]
        }),
        "diagnostic_conclusion": (
            "inspect_private_failure_cases_before_selecting_or_validating_a_new_gate"
        ),
        "production_gate_changed": False,
        "private_text_uploaded": False,
    }

    run_id = public_summary["run_id"]
    private_path = (
        vault_root / "manifests" / "calibration" / pilot_name
        / f"verifier-failure-forensics-{run_id}.json"
    )
    summary_path = (
        vault_root / "manifests" / "exports"
        / f"verifier-failure-forensics-summary-{run_id}.json"
    )

    private = {
        **public_summary,
        "private_failure_cases": [
            row for row in rows
            if (
                row["qwen_false_positive"]
                or row["hhem_false_positive"]
                or row["unanimous_false_positive"]
                or row["qwen_false_negative"]
                or row["hhem_false_negative"]
            )
        ],
    }
    _atomic_json(private_path, private)

    public_summary["private_details_path"] = str(private_path)
    _atomic_json(summary_path, public_summary)
    public_summary["summary_path"] = str(summary_path)
    return public_summary
