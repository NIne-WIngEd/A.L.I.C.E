from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HHEMCalibrationPolicy:
    policy_id: str
    model_id: str
    revision: str
    batch_size: int
    default_device: str
    minimum_labeled_items: int
    high_precision_target: float
    minimum_recall_for_promising_gate: float
    minimum_roc_auc_for_promising_gate: float
    private_text_uploaded: bool
    memory_write_allowed: bool
    external_action_allowed: bool
    tool_calling_allowed: bool
    web_access_allowed: bool
    source_path: Path


_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_policy_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "policies"
        / "hhem_calibration_policy.json"
    )


def load_hhem_calibration_policy(
    path: Path | None = None,
) -> HHEMCalibrationPolicy:
    source = (
        path or default_policy_path()
    ).expanduser().resolve(strict=True)
    data = json.loads(
        source.read_text(encoding="utf-8")
    )
    if int(
        data.get(
            "hhem_calibration_policy_schema_version",
            -1,
        )
    ) != 1:
        raise ValueError(
            "Unsupported HHEM calibration policy schema"
        )

    policy = HHEMCalibrationPolicy(
        policy_id=str(data["policy_id"]),
        model_id=str(data["model_id"]),
        revision=str(data.get("revision", "main")),
        batch_size=int(data["batch_size"]),
        default_device=str(data["default_device"]),
        minimum_labeled_items=int(
            data["minimum_labeled_items"]
        ),
        high_precision_target=float(
            data["high_precision_target"]
        ),
        minimum_recall_for_promising_gate=float(
            data["minimum_recall_for_promising_gate"]
        ),
        minimum_roc_auc_for_promising_gate=float(
            data["minimum_roc_auc_for_promising_gate"]
        ),
        private_text_uploaded=bool(
            data["private_text_uploaded"]
        ),
        memory_write_allowed=bool(
            data["memory_write_allowed"]
        ),
        external_action_allowed=bool(
            data["external_action_allowed"]
        ),
        tool_calling_allowed=bool(
            data["tool_calling_allowed"]
        ),
        web_access_allowed=bool(
            data["web_access_allowed"]
        ),
        source_path=source,
    )

    if policy.batch_size < 1:
        raise ValueError(
            "batch_size must be positive"
        )
    if policy.minimum_labeled_items < 1:
        raise ValueError(
            "minimum_labeled_items must be positive"
        )
    for value in (
        policy.high_precision_target,
        policy.minimum_recall_for_promising_gate,
        policy.minimum_roc_auc_for_promising_gate,
    ):
        if not 0 <= value <= 1:
            raise ValueError(
                "Calibration thresholds must be between 0 and 1"
            )
    if policy.private_text_uploaded:
        raise ValueError(
            "Private text may not be uploaded"
        )
    if any(
        (
            policy.memory_write_allowed,
            policy.external_action_allowed,
            policy.tool_calling_allowed,
            policy.web_access_allowed,
        )
    ):
        raise ValueError(
            "HHEM calibration must remain read-only and offline"
        )
    return policy


def model_directory(
    *,
    vault_root: Path,
    policy: HHEMCalibrationPolicy,
) -> Path:
    safe_model = policy.model_id.replace(
        "/",
        "__",
    )
    safe_revision = policy.revision.replace(
        "/",
        "_",
    ).replace(
        ":",
        "_",
    )
    return (
        vault_root
        / "models"
        / "hhem"
        / f"{safe_model}__{safe_revision}"
    )


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file()
    ):
        digest.update(
            path.relative_to(root)
            .as_posix()
            .encode("utf-8")
        )
        digest.update(b"\0")
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(
                    1024 * 1024
                )
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def prepare_hhem_model(
    *,
    vault_root: Path,
    policy_path: Path | None = None,
) -> dict[str, Any]:
    from huggingface_hub import (
        HfApi,
        snapshot_download,
    )

    vault_root = (
        vault_root.expanduser()
        .resolve(strict=True)
    )
    policy = load_hhem_calibration_policy(
        policy_path
    )
    root = model_directory(
        vault_root=vault_root,
        policy=policy,
    )
    root.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    info = HfApi().model_info(
        policy.model_id,
        revision=policy.revision,
    )
    resolved_revision = str(
        info.sha
    )

    snapshot_download(
        repo_id=policy.model_id,
        revision=resolved_revision,
        local_dir=str(root),
    )

    result = {
        "hhem_model_manifest_schema_version": 1,
        "model_id": policy.model_id,
        "requested_revision": (
            policy.revision
        ),
        "resolved_revision": (
            resolved_revision
        ),
        "prepared_at": _now(),
        "trust_remote_code_required": True,
        "private_data_used_during_download": False,
        "model_tree_digest": _tree_digest(
            root
        ),
        "model_path": str(root),
    }

    summary_path = (
        vault_root
        / "manifests"
        / "exports"
        / (
            "hhem-model-summary-"
            f"{uuid.uuid4().hex}.json"
        )
    )
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )
    result[
        "summary_path"
    ] = str(
        summary_path
    )
    return result


def _transformers_major_version() -> int:
    import transformers

    raw = str(
        transformers.__version__
    ).split(
        ".",
        1,
    )[0]

    try:
        return int(
            raw
        )
    except ValueError as exc:
        raise RuntimeError(
            "Could not determine the installed Transformers major version"
        ) from exc


def load_local_hhem_model(
    *,
    vault_root: Path,
    policy_path: Path | None = None,
    device: str | None = None,
):
    transformers_major = (
        _transformers_major_version()
    )

    if transformers_major >= 5:
        raise RuntimeError(
            "HHEM-2.1-Open custom model code is not compatible with "
            "Transformers 5.x in this A.L.I.C.E. integration. "
            "Install a 4.x release with: "
            'py -m pip install --upgrade --force-reinstall '
            '"transformers>=4.45,<5.0"'
        )

    policy = load_hhem_calibration_policy(
        policy_path
    )
    root = model_directory(
        vault_root=(
            vault_root.expanduser()
            .resolve(strict=True)
        ),
        policy=policy,
    )
    if not root.is_dir():
        raise FileNotFoundError(
            "Local HHEM model is missing. "
            "Run scripts/prepare_hhem_verifier.py first."
        )

    resolved_device = (
        device
        or policy.default_device
    )
    key = (
        str(root),
        resolved_device,
    )
    if key in _MODEL_CACHE:
        return (
            _MODEL_CACHE[key],
            policy,
        )

    from transformers import (
        AutoModelForSequenceClassification,
    )

    kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
    }
    try:
        model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                str(root),
                dtype="auto",
                **kwargs,
            )
        )
    except TypeError:
        model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                str(root),
                torch_dtype="auto",
                **kwargs,
            )
        )

    if hasattr(
        model,
        "eval",
    ):
        model.eval()

    if (
        resolved_device
        not in {
            "",
            "auto",
            "cpu",
        }
        and hasattr(
            model,
            "to",
        )
    ):
        model.to(
            resolved_device
        )

    _MODEL_CACHE[
        key
    ] = model
    return (
        model,
        policy,
    )


def _is_human_calibration_bundle(
    path: Path,
) -> bool:
    """Return True only for the original blind human-review bundle.

    Files such as judge-calibration-evaluation-details-*.json share the same
    filename prefix but do not contain claim_text/evidence_windows. Selecting
    one of those newer derivative files causes HHEM pair construction to see
    no usable premise-hypothesis pairs.
    """
    if path.name.startswith(
        "judge-calibration-evaluation-"
    ):
        return False

    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return False

    if int(
        data.get(
            "judge_calibration_bundle_schema_version",
            -1,
        )
    ) != 1:
        return False

    items = data.get(
        "items",
        [],
    )
    if not isinstance(
        items,
        list,
    ):
        return False

    return any(
        isinstance(
            item,
            dict,
        )
        and str(
            item.get(
                "claim_text",
                "",
            )
        ).strip()
        and isinstance(
            item.get(
                "evidence_windows",
                [],
            ),
            list,
        )
        for item in items
    )


def latest_calibration_bundle(
    *,
    vault_root: Path,
    pilot_name: str,
) -> Path:
    directory = (
        vault_root
        / "manifests"
        / "calibration"
        / pilot_name
    )

    candidates = [
        path
        for path in directory.glob(
            "judge-calibration-*.json"
        )
        if _is_human_calibration_bundle(
            path
        )
    ]

    candidates.sort(
        key=lambda path: (
            path.stat().st_mtime
        ),
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            "No original human judge-calibration bundle "
            "with claim_text and evidence_windows was found"
        )

    return candidates[0]


def build_hhem_premise(
    item: dict[str, Any],
) -> str:
    """Use exactly the evidence windows shown during blind human review."""
    seen: set[str] = set()
    parts: list[str] = []

    for window in item.get(
        "evidence_windows",
        [],
    ):
        text = str(
            window.get(
                "text",
                "",
            )
        ).strip()
        if (
            text
            and text not in seen
        ):
            seen.add(
                text
            )
            parts.append(
                text
            )

    return "\n\n".join(
        parts
    )


def score_hhem_pairs(
    *,
    model,
    pairs: list[
        tuple[str, str]
    ],
    batch_size: int,
) -> list[float]:
    scores: list[
        float
    ] = []

    for start in range(
        0,
        len(pairs),
        batch_size,
    ):
        batch = pairs[
            start : start + batch_size
        ]
        raw = model.predict(
            batch
        )

        if hasattr(
            raw,
            "detach",
        ):
            raw = (
                raw.detach()
                .cpu()
                .tolist()
            )
        elif hasattr(
            raw,
            "tolist",
        ):
            raw = raw.tolist()

        if not isinstance(
            raw,
            list,
        ):
            raw = [
                raw
            ]

        for value in raw:
            if isinstance(
                value,
                list,
            ):
                if len(value) != 1:
                    raise ValueError(
                        "Unexpected HHEM score shape"
                    )
                value = value[
                    0
                ]
            score = float(
                value
            )
            scores.append(
                max(
                    0.0,
                    min(
                        1.0,
                        score,
                    ),
                )
            )

    if len(
        scores
    ) != len(
        pairs
    ):
        raise RuntimeError(
            "HHEM returned an unexpected number of scores"
        )

    return scores


def _binary_labels(
    human_labels: list[str],
) -> list[int]:
    return [
        1
        if label == "supported"
        else 0
        for label in human_labels
    ]


def _binary_metrics(
    *,
    expected: list[int],
    predicted: list[int],
) -> dict[str, float | int]:
    tp = sum(
        left == 1
        and right == 1
        for left, right in zip(
            expected,
            predicted,
        )
    )
    tn = sum(
        left == 0
        and right == 0
        for left, right in zip(
            expected,
            predicted,
        )
    )
    fp = sum(
        left == 0
        and right == 1
        for left, right in zip(
            expected,
            predicted,
        )
    )
    fn = sum(
        left == 1
        and right == 0
        for left, right in zip(
            expected,
            predicted,
        )
    )

    total = len(
        expected
    )
    accuracy = (
        (
            tp
            + tn
        )
        / total
        if total
        else 0.0
    )
    precision = (
        tp
        / (
            tp
            + fp
        )
        if (
            tp
            + fp
        )
        else 0.0
    )
    recall = (
        tp
        / (
            tp
            + fn
        )
        if (
            tp
            + fn
        )
        else 0.0
    )
    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        )
        else 0.0
    )

    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": round(
            accuracy,
            6,
        ),
        "support_precision": round(
            precision,
            6,
        ),
        "support_recall": round(
            recall,
            6,
        ),
        "support_f1": round(
            f1,
            6,
        ),
    }


def _roc_auc(
    *,
    expected: list[int],
    scores: list[float],
) -> float | None:
    positives = [
        score
        for label, score in zip(
            expected,
            scores,
        )
        if label == 1
    ]
    negatives = [
        score
        for label, score in zip(
            expected,
            scores,
        )
        if label == 0
    ]

    if (
        not positives
        or not negatives
    ):
        return None

    wins = 0.0
    comparisons = 0
    for positive in positives:
        for negative in negatives:
            comparisons += 1
            if positive > negative:
                wins += 1.0
            elif math.isclose(
                positive,
                negative,
            ):
                wins += 0.5

    return round(
        wins
        / comparisons,
        6,
    )


def _average_precision(
    *,
    expected: list[int],
    scores: list[float],
) -> float | None:
    positive_count = sum(
        expected
    )
    if positive_count == 0:
        return None

    ranked = sorted(
        zip(
            scores,
            expected,
        ),
        key=lambda item: (
            -item[
                0
            ]
        ),
    )

    true_positive = 0
    precision_sum = 0.0

    for rank, (
        _score,
        label,
    ) in enumerate(
        ranked,
        start=1,
    ):
        if label == 1:
            true_positive += 1
            precision_sum += (
                true_positive
                / rank
            )

    return round(
        precision_sum
        / positive_count,
        6,
    )


def _threshold_candidates(
    scores: list[float],
) -> list[float]:
    candidates = {
        0.0,
        1.0,
        *(
            round(
                index
                / 100,
                2,
            )
            for index in range(
                1,
                100,
            )
        ),
        *(
            round(
                score,
                6,
            )
            for score in scores
        ),
    }
    return sorted(
        candidates
    )


def _threshold_sweep(
    *,
    expected: list[int],
    scores: list[float],
    high_precision_target: float,
) -> dict[str, Any]:
    rows = []

    for threshold in _threshold_candidates(
        scores
    ):
        predicted = [
            1
            if score
            >= threshold
            else 0
            for score in scores
        ]
        metrics = _binary_metrics(
            expected=expected,
            predicted=predicted,
        )
        rows.append(
            {
                "threshold": round(
                    float(
                        threshold
                    ),
                    6,
                ),
                **metrics,
            }
        )

    best_f1 = max(
        rows,
        key=lambda row: (
            float(
                row[
                    "support_f1"
                ]
            ),
            float(
                row[
                    "support_precision"
                ]
            ),
            float(
                row[
                    "accuracy"
                ]
            ),
        ),
    )

    best_accuracy = max(
        rows,
        key=lambda row: (
            float(
                row[
                    "accuracy"
                ]
            ),
            float(
                row[
                    "support_precision"
                ]
            ),
            float(
                row[
                    "support_recall"
                ]
            ),
        ),
    )

    high_precision = [
        row
        for row in rows
        if (
            float(
                row[
                    "support_precision"
                ]
            )
            >= high_precision_target
            and int(
                row[
                    "true_positive"
                ]
            )
            > 0
        )
    ]

    best_high_precision = (
        max(
            high_precision,
            key=lambda row: (
                float(
                    row[
                        "support_recall"
                    ]
                ),
                float(
                    row[
                        "support_precision"
                    ]
                ),
                float(
                    row[
                        "accuracy"
                    ]
                ),
            ),
        )
        if high_precision
        else None
    )

    return {
        "best_f1": best_f1,
        "best_accuracy": (
            best_accuracy
        ),
        "best_high_precision": (
            best_high_precision
        ),
        "threshold_count": len(
            rows
        ),
        "rows": rows,
    }


def _existing_judge_metrics(
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    human = [
        str(
            item[
                "human_label"
            ]
        )
        for item in items
    ]
    expected = _binary_labels(
        human
    )

    qwen = [
        1
        if str(
            item.get(
                "qwen_auditor",
                {},
            ).get(
                "verdict",
                "",
            )
        )
        == "supported"
        else 0
        for item in items
    ]

    fever = [
        1
        if str(
            item.get(
                "fever_nli",
                {},
            ).get(
                "decision",
                "",
            )
        )
        == "keep_entailment"
        else 0
        for item in items
    ]

    return {
        "qwen_auditor": (
            _binary_metrics(
                expected=expected,
                predicted=qwen,
            )
        ),
        "fever_nli": (
            _binary_metrics(
                expected=expected,
                predicted=fever,
            )
        ),
    }


def evaluate_hhem_calibration(
    *,
    vault_root: Path,
    calibration_bundle_path: Path | None = None,
    pilot_name: str = "pilot-v1",
    policy_path: Path | None = None,
    device: str | None = None,
    model_loader=None,
) -> dict[str, Any]:
    vault_root = (
        vault_root.expanduser()
        .resolve(strict=True)
    )
    policy = load_hhem_calibration_policy(
        policy_path
    )

    if calibration_bundle_path is None:
        calibration_bundle_path = (
            latest_calibration_bundle(
                vault_root=vault_root,
                pilot_name=pilot_name,
            )
        )
    else:
        calibration_bundle_path = (
            calibration_bundle_path
            .expanduser()
            .resolve(strict=True)
        )

    bundle = json.loads(
        calibration_bundle_path.read_text(
            encoding="utf-8"
        )
    )

    labeled_items = [
        item
        for item in bundle.get(
            "items",
            [],
        )
        if str(
            item.get(
                "human_label",
                "",
            )
        )
        in {
            "supported",
            "partially_supported",
            "unsupported",
        }
    ]

    if len(
        labeled_items
    ) < policy.minimum_labeled_items:
        raise ValueError(
            "Not enough human-labeled calibration items"
        )

    pairs: list[
        tuple[str, str]
    ] = []
    scored_items: list[
        dict[str, Any]
    ] = []

    for item in labeled_items:
        premise = build_hhem_premise(
            item
        )
        hypothesis = str(
            item.get(
                "claim_text",
                "",
            )
        ).strip()

        if (
            not premise
            or not hypothesis
        ):
            continue

        pairs.append(
            (
                premise,
                hypothesis,
            )
        )
        scored_items.append(
            item
        )

    if not pairs:
        raise ValueError(
            "No calibration items had usable premise-hypothesis pairs"
        )

    if model_loader is None:
        model, _ = load_local_hhem_model(
            vault_root=vault_root,
            policy_path=policy_path,
            device=device,
        )
    else:
        loaded = model_loader(
            vault_root=vault_root,
            policy_path=policy_path,
            device=device,
        )
        model = (
            loaded[
                0
            ]
            if isinstance(
                loaded,
                tuple,
            )
            else loaded
        )

    scores = score_hhem_pairs(
        model=model,
        pairs=pairs,
        batch_size=policy.batch_size,
    )

    human_labels = [
        str(
            item[
                "human_label"
            ]
        )
        for item in scored_items
    ]
    expected = _binary_labels(
        human_labels
    )

    roc_auc = _roc_auc(
        expected=expected,
        scores=scores,
    )
    average_precision = (
        _average_precision(
            expected=expected,
            scores=scores,
        )
    )

    sweep = _threshold_sweep(
        expected=expected,
        scores=scores,
        high_precision_target=(
            policy
            .high_precision_target
        ),
    )

    existing = _existing_judge_metrics(
        scored_items
    )

    best_high_precision = (
        sweep[
            "best_high_precision"
        ]
    )

    diagnostic_recommendation = (
        "hhem_mixed_requires_more_labels"
    )

    if (
        roc_auc is not None
        and roc_auc
        < 0.55
    ):
        diagnostic_recommendation = (
            "hhem_not_discriminative_on_calibration_sample"
        )
    elif (
        roc_auc is not None
        and roc_auc
        >= policy
        .minimum_roc_auc_for_promising_gate
        and best_high_precision
        is not None
        and float(
            best_high_precision[
                "support_recall"
            ]
        )
        >= policy
        .minimum_recall_for_promising_gate
    ):
        diagnostic_recommendation = (
            "hhem_promising_requires_holdout_validation"
        )

    run_id = str(
        uuid.uuid4()
    )
    private_root = (
        vault_root
        / "manifests"
        / "calibration"
        / pilot_name
    )
    export_root = (
        vault_root
        / "manifests"
        / "exports"
    )
    private_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    export_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    details_path = (
        private_root
        / (
            "hhem-calibration-details-"
            f"{run_id}.json"
        )
    )
    summary_path = (
        export_root
        / (
            "hhem-calibration-summary-"
            f"{run_id}.json"
        )
    )

    details = {
        "hhem_calibration_schema_version": 1,
        "run_id": run_id,
        "calibration_id": str(
            bundle.get(
                "calibration_id",
                "",
            )
        ),
        "pilot_name": pilot_name,
        "model_id": policy.model_id,
        "model_revision": (
            policy.revision
        ),
        "calibration_bundle_path": str(
            calibration_bundle_path
        ),
        "labeled_item_count": len(
            scored_items
        ),
        "human_label_counts": dict(
            Counter(
                human_labels
            )
        ),
        "human_binary_definition": (
            "supported=positive; "
            "partially_supported and unsupported=negative"
        ),
        "hhem": {
            "roc_auc": (
                roc_auc
            ),
            "average_precision": (
                average_precision
            ),
            "best_f1_threshold": (
                sweep[
                    "best_f1"
                ]
            ),
            "best_accuracy_threshold": (
                sweep[
                    "best_accuracy"
                ]
            ),
            "best_high_precision_threshold": (
                best_high_precision
            ),
        },
        "existing_judges": (
            existing
        ),
        "diagnostic_recommendation": (
            diagnostic_recommendation
        ),
        "production_gate_changed": False,
        "holdout_validation_required_before_production": True,
        "private_text_uploaded": False,
        "items": [
            {
                "item_id": str(
                    item[
                        "item_id"
                    ]
                ),
                "query_id": str(
                    item[
                        "query_id"
                    ]
                ),
                "human_label": str(
                    item[
                        "human_label"
                    ]
                ),
                "hhem_score": round(
                    float(
                        score
                    ),
                    6,
                ),
                "qwen_verdict": str(
                    item.get(
                        "qwen_auditor",
                        {},
                    ).get(
                        "verdict",
                        "",
                    )
                ),
                "fever_entailment_probability": float(
                    item.get(
                        "fever_nli",
                        {},
                    ).get(
                        "best_entailment_probability",
                        0.0,
                    )
                    or 0.0
                ),
            }
            for item, score in zip(
                scored_items,
                scores,
            )
        ],
        "threshold_sweep": (
            sweep[
                "rows"
            ]
        ),
    }

    details_path.write_text(
        json.dumps(
            details,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        key: value
        for key, value in details.items()
        if key
        not in {
            "items",
            "threshold_sweep",
        }
    }
    summary[
        "private_details_path"
    ] = str(
        details_path
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary[
        "summary_path"
    ] = str(
        summary_path
    )
    return summary
