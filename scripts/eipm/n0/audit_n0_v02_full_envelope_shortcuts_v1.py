#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Callable


PASS = "PASS_N0_FULL_ENVELOPE_SHORTCUT_PREFLIGHT_V1"
WORD_RE = re.compile(r"[a-z0-9]+")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def tokens(text: str) -> list[str]:
    return WORD_RE.findall(str(text).lower())


def majority(values: list[Any]) -> Any:
    if not values:
        raise ValueError("majority() requires values")
    counts = collections.Counter(values)
    return min(counts, key=lambda x: (-counts[x], repr(x)))


def relation_target_positions(row: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(x) for x in row.get("relation_sequence_target") or [])


def relation_keys_from_positions(
    row: dict[str, Any],
    positions: tuple[int, ...] | list[int],
) -> tuple[str, ...] | None:
    bank = list(row.get("relation_candidates") or [])
    out: list[str] = []
    for index in positions:
        i = int(index)
        if not 0 <= i < len(bank):
            return None
        out.append(str(bank[i].get("key", "")))
    return tuple(out)


def target_relation_keys(row: dict[str, Any]) -> tuple[str, ...]:
    result = relation_keys_from_positions(row, relation_target_positions(row))
    if result is None:
        raise ValueError(f"{row.get('id')}: target relation index outside bank")
    return result


def factor_target_tuple(row: dict[str, Any]) -> tuple[tuple[str, int], ...]:
    targets = row.get("factor_targets") or {}
    return tuple(sorted((str(k), int(v)) for k, v in targets.items()))


def event_target(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(x) for x in row.get("event_sequence_target") or [])


class MultinomialNB:
    def __init__(self) -> None:
        self.label_counts: collections.Counter[str] = collections.Counter()
        self.token_counts: dict[str, collections.Counter[str]] = {}
        self.total_tokens: collections.Counter[str] = collections.Counter()
        self.vocab: set[str] = set()

    def fit(self, texts: list[str], labels: list[str]) -> "MultinomialNB":
        if len(texts) != len(labels) or not texts:
            raise ValueError("NB fit geometry drift")
        for text, label in zip(texts, labels):
            label = str(label)
            self.label_counts[label] += 1
            counter = self.token_counts.setdefault(label, collections.Counter())
            for token in tokens(text):
                counter[token] += 1
                self.total_tokens[label] += 1
                self.vocab.add(token)
        return self

    def score(self, text: str, label: str) -> float:
        total_docs = sum(self.label_counts.values())
        labels = max(len(self.label_counts), 1)
        score = math.log((self.label_counts[label] + 1.0) / (total_docs + labels))
        vocab = max(len(self.vocab), 1)
        denom = self.total_tokens[label] + vocab
        counter = self.token_counts[label]
        for token in tokens(text):
            score += math.log((counter[token] + 1.0) / denom)
        return score

    def predict(self, text: str) -> str:
        if not self.label_counts:
            raise ValueError("NB not fitted")
        labels = sorted(self.label_counts)
        return max(labels, key=lambda label: (self.score(text, label), -labels.index(label)))


def encode_tuple(values: tuple[int, ...]) -> str:
    return ",".join(str(x) for x in values)


def decode_tuple(value: str) -> tuple[int, ...]:
    if value == "":
        return ()
    return tuple(int(x) for x in value.split(","))


def build_feature_majority(
    train: list[dict[str, Any]],
    feature_fn: Callable[[dict[str, Any]], Any],
    target_fn: Callable[[dict[str, Any]], Any],
) -> tuple[dict[Any, Any], Any]:
    buckets: dict[Any, list[Any]] = collections.defaultdict(list)
    values: list[Any] = []
    for row in train:
        target = target_fn(row)
        buckets[feature_fn(row)].append(target)
        values.append(target)
    return ({key: majority(v) for key, v in buckets.items()}, majority(values))


def template_baseline(train: list[dict[str, Any]], dev: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pos_map, pos_default = build_feature_majority(
        train,
        lambda r: (str(r.get("intervention", "")), int(r.get("runtime_relation_count", 0))),
        relation_target_positions,
    )
    factor_map, factor_default = build_feature_majority(
        train,
        lambda r: str(r.get("intervention", "")),
        factor_target_tuple,
    )
    event_map, event_default = build_feature_majority(
        train,
        lambda r: str(r.get("intervention", "")),
        event_target,
    )
    applicability_map, applicability_default = build_feature_majority(
        train,
        lambda r: str(r.get("intervention", "")),
        lambda r: int(r.get("applicability_target", 0)),
    )
    out = []
    for row in dev:
        intervention = str(row.get("intervention", ""))
        out.append({
            "relation_positions": pos_map.get(
                (intervention, int(row.get("runtime_relation_count", 0))),
                pos_default,
            ),
            "factor_targets": dict(factor_map.get(intervention, factor_default)),
            "event_sequence": event_map.get(intervention, event_default),
            "applicability": applicability_map.get(intervention, applicability_default),
        })
    return out


def metadata_baseline(train: list[dict[str, Any]], dev: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feature = lambda r: (
        int(r.get("runtime_relation_count", 0)),
        int(r.get("runtime_reasoning_steps", 0)),
        int(r.get("runtime_operator_slots", 0)),
    )
    pos_map, pos_default = build_feature_majority(train, feature, relation_target_positions)
    factor_map, factor_default = build_feature_majority(train, feature, factor_target_tuple)
    event_map, event_default = build_feature_majority(train, feature, event_target)
    applicability_map, applicability_default = build_feature_majority(
        train, feature, lambda r: int(r.get("applicability_target", 0))
    )
    return [
        {
            "relation_positions": pos_map.get(feature(row), pos_default),
            "factor_targets": dict(factor_map.get(feature(row), factor_default)),
            "event_sequence": event_map.get(feature(row), event_default),
            "applicability": applicability_map.get(feature(row), applicability_default),
        }
        for row in dev
    ]


def query_only_baseline(train: list[dict[str, Any]], dev: list[dict[str, Any]]) -> list[dict[str, Any]]:
    texts = [str(r.get("query", "")) for r in train]
    position_model = MultinomialNB().fit(
        texts,
        [encode_tuple(relation_target_positions(r)) for r in train],
    )
    factor_names = sorted((train[0].get("factor_targets") or {}).keys())
    factor_models = {
        name: MultinomialNB().fit(
            texts,
            [str(int(r["factor_targets"][name])) for r in train],
        )
        for name in factor_names
    }
    event_model = MultinomialNB().fit(
        texts,
        [json.dumps(list(event_target(r)), separators=(",", ":")) for r in train],
    )
    applicability_model = MultinomialNB().fit(
        texts,
        [str(int(r.get("applicability_target", 0))) for r in train],
    )
    out = []
    for row in dev:
        query = str(row.get("query", ""))
        out.append({
            "relation_positions": decode_tuple(position_model.predict(query)),
            "factor_targets": {
                name: int(model.predict(query)) for name, model in factor_models.items()
            },
            "event_sequence": tuple(json.loads(event_model.predict(query))),
            "applicability": int(applicability_model.predict(query)),
        })
    return out


def schema_only_baseline(train: list[dict[str, Any]], dev: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate_texts: list[str] = []
    candidate_labels: list[str] = []
    for row in train:
        positive = set(relation_target_positions(row))
        for i, candidate in enumerate(row.get("relation_candidates") or []):
            candidate_texts.append(str(candidate.get("text", "")))
            candidate_labels.append("positive" if i in positive else "negative")
    candidate_model = MultinomialNB().fit(candidate_texts, candidate_labels)
    length_default = majority([len(relation_target_positions(r)) for r in train])
    factor_default = dict(majority([factor_target_tuple(r) for r in train]))
    event_default = majority([event_target(r) for r in train])
    applicability_default = majority([int(r.get("applicability_target", 0)) for r in train])
    out = []
    for row in dev:
        bank = list(row.get("relation_candidates") or [])
        ranked = sorted(
            range(len(bank)),
            key=lambda i: (
                candidate_model.score(str(bank[i].get("text", "")), "positive")
                - candidate_model.score(str(bank[i].get("text", "")), "negative"),
                -i,
            ),
            reverse=True,
        )
        positions = tuple(ranked[: min(int(length_default), len(ranked))])
        out.append({
            "relation_positions": positions,
            "factor_targets": dict(factor_default),
            "event_sequence": event_default,
            "applicability": applicability_default,
        })
    return out


def lexical_overlap_baseline(
    train: list[dict[str, Any]],
    dev: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    query_texts = [str(r.get("query", "")) for r in train]
    factor_names = sorted((train[0].get("factor_targets") or {}).keys())
    factor_models = {
        name: MultinomialNB().fit(
            query_texts,
            [str(int(r["factor_targets"][name])) for r in train],
        )
        for name in factor_names
    }
    event_model = MultinomialNB().fit(
        query_texts,
        [json.dumps(list(event_target(r)), separators=(",", ":")) for r in train],
    )
    applicability_model = MultinomialNB().fit(
        query_texts,
        [str(int(r.get("applicability_target", 0))) for r in train],
    )

    def guess_steps(query: str) -> int:
        lower = query.lower()
        if "not represented by any supplied schema" in lower:
            return 0
        if (
            ("first " in lower and "then " in lower)
            or ("trace backward first" in lower and "and then" in lower)
            or ("traverse the first relation" in lower and "then traverse" in lower)
        ):
            return 2
        return 1

    out = []
    for row in dev:
        query = str(row.get("query", ""))
        q = set(tokens(query))
        bank = list(row.get("relation_candidates") or [])
        scored = []
        for i, candidate in enumerate(bank):
            c = set(tokens(candidate.get("text", "")))
            overlap = len(q & c) / max(len(q | c), 1)
            scored.append((overlap, -i, i))
        scored.sort(reverse=True)
        k = min(guess_steps(query), len(bank))
        positions = tuple(item[2] for item in scored[:k])
        out.append({
            "relation_positions": positions,
            "factor_targets": {
                name: int(model.predict(query)) for name, model in factor_models.items()
            },
            "event_sequence": tuple(json.loads(event_model.predict(query))),
            "applicability": int(applicability_model.predict(query)),
        })
    return out


def evaluate_operator_baseline(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    gate: dict[str, float],
) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError("baseline row/prediction count drift")
    relation_hits = 0
    event_hits = 0
    applicability_hits = 0
    full_hits = 0
    factor_names = sorted((rows[0].get("factor_targets") or {}).keys())
    factor_hits = collections.Counter()
    for row, pred in zip(rows, predictions):
        predicted_keys = relation_keys_from_positions(
            row, tuple(pred["relation_positions"])
        )
        relation_ok = predicted_keys == target_relation_keys(row)
        relation_hits += int(relation_ok)
        event_ok = tuple(pred["event_sequence"]) == event_target(row)
        event_hits += int(event_ok)
        applicability_ok = int(pred["applicability"]) == int(
            row.get("applicability_target", 0)
        )
        applicability_hits += int(applicability_ok)
        factor_ok_all = True
        for name in factor_names:
            ok = int(pred["factor_targets"].get(name, -999)) == int(
                row["factor_targets"][name]
            )
            factor_hits[name] += int(ok)
            factor_ok_all = factor_ok_all and ok
        full_hits += int(
            relation_ok and event_ok and applicability_ok and factor_ok_all
        )
    n = max(len(rows), 1)
    per_bank = {name: factor_hits[name] / n for name in factor_names}
    factor_macro = sum(per_bank.values()) / max(len(per_bank), 1)
    result = {
        "rows": len(rows),
        "relation_sequence_exact": relation_hits / n,
        "factor_macro_accuracy": factor_macro,
        "factor_bank_accuracy": per_bank,
        "event_sequence_exact": event_hits / n,
        "applicability_accuracy": applicability_hits / n,
        "full_operator_exact": full_hits / n,
    }
    result["clears_operator_gate"] = bool(
        result["relation_sequence_exact"] >= float(gate["relation_sequence_exact_clear_min"])
        and result["factor_macro_accuracy"] >= float(gate["factor_macro_accuracy_clear_min"])
        and result["event_sequence_exact"] >= float(gate["event_sequence_exact_clear_min"])
        and result["full_operator_exact"] >= float(gate["full_operator_exact_clear_min"])
    )
    return result


def permutation_probe(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    checked = 0
    for row in rows:
        bank = list(row.get("relation_candidates") or [])
        if len(bank) <= 1:
            continue
        perm = list(range(len(bank)))
        rng.shuffle(perm)
        inverse = {old: new for new, old in enumerate(perm)}
        original_keys = target_relation_keys(row)
        remapped_positions = tuple(inverse[int(i)] for i in relation_target_positions(row))
        permuted_row = dict(row)
        permuted_row["relation_candidates"] = [bank[i] for i in perm]
        remapped_keys = relation_keys_from_positions(permuted_row, remapped_positions)
        if remapped_keys != original_keys:
            raise SystemExit(
                f"{row.get('id')}: candidate permutation changed semantic target"
            )
        checked += 1
    if checked <= 0:
        raise SystemExit("candidate permutation probe had no multi-candidate rows")
    return {"rows_checked": checked, "semantic_target_invariant": True}


def hard_negative_probe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    coverage: dict[str, bool] = {}
    for row in rows:
        keys = target_relation_keys(row)
        bank_count = len(row.get("relation_candidates") or [])
        negatives = bank_count - len(set(relation_target_positions(row)))
        for key in keys:
            coverage.setdefault(key, False)
            if negatives >= 1:
                coverage[key] = True
    missing = sorted(key for key, ok in coverage.items() if not ok)
    if missing:
        raise SystemExit(
            "target relations missing hard-negative coverage: " + repr(missing)
        )
    return {
        "target_relation_count": len(coverage),
        "all_target_relations_have_hard_negative_coverage": True,
    }


def counterfactual_pair_probe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        pair_id = row.get("counterfactual_pair_id")
        if pair_id:
            groups[str(pair_id)].append(row)
    if not groups:
        raise SystemExit(
            "counterfactual_pairs_required but semantic/operator rows have no "
            "explicit auditable counterfactual_pair_id"
        )

    valid = 0
    relations: set[tuple[str, str]] = set()
    for pair_id, pair in groups.items():
        if len(pair) != 2:
            raise SystemExit(f"{pair_id}: counterfactual pair must contain exactly two rows")
        a, b = pair
        if a.get("split") != b.get("split"):
            raise SystemExit(f"{pair_id}: counterfactual pair crosses split")
        if a.get("entities") != b.get("entities"):
            raise SystemExit(f"{pair_id}: counterfactual pair changed entities")
        if [x.get("key") for x in a.get("relation_candidates") or []] != [
            x.get("key") for x in b.get("relation_candidates") or []
        ]:
            raise SystemExit(f"{pair_id}: counterfactual pair changed candidate bank/order")
        if target_relation_keys(a) != target_relation_keys(b):
            raise SystemExit(f"{pair_id}: counterfactual pair changed relation target")
        if a.get("factor_targets", {}).get("role") == b.get("factor_targets", {}).get("role"):
            raise SystemExit(f"{pair_id}: role counterfactual did not change role target")
        if str(a.get("query")) == str(b.get("query")):
            raise SystemExit(f"{pair_id}: counterfactual query did not change")
        relations.add((str(a.get("split")), str(a.get("relation_family"))))
        valid += 1

    required = {
        (str(row.get("split")), str(row.get("relation_family")))
        for row in rows
    }
    missing = sorted(required - relations)
    if missing:
        raise SystemExit(
            "counterfactual pair coverage missing split/relation families: "
            + repr(missing[:20])
        )
    return {
        "pair_count": valid,
        "split_relation_family_coverage_complete": True,
    }


def behavioral_candidate_only_probe(
    train: list[dict[str, Any]],
    dev: list[dict[str, Any]],
    clear_min: float,
) -> dict[str, Any]:
    texts: list[str] = []
    labels: list[str] = []
    for row in train:
        target = int(row["public_target_index"])
        for i, text in enumerate(row.get("candidate_answers") or []):
            texts.append(str(text))
            labels.append("positive" if i == target else "negative")
    model = MultinomialNB().fit(texts, labels)
    hits = 0
    for row in dev:
        candidates = [str(x) for x in row.get("candidate_answers") or []]
        scores = [
            model.score(text, "positive") - model.score(text, "negative")
            for text in candidates
        ]
        predicted = max(range(len(scores)), key=lambda i: (scores[i], -i))
        hits += int(predicted == int(row["public_target_index"]))
    accuracy = hits / max(len(dev), 1)
    return {
        "candidate_only_lexical_top1": accuracy,
        "clear_threshold": float(clear_min),
        "clears_judgment_gate": bool(accuracy >= float(clear_min)),
    }


def behavioral_template_position_probe(
    train: list[dict[str, Any]],
    dev: list[dict[str, Any]],
    clear_min: float,
) -> dict[str, Any]:
    mapping, default = build_feature_majority(
        train,
        lambda r: str(r.get("scenario_family", "")),
        lambda r: int(r["public_target_index"]),
    )
    hits = sum(
        int(
            int(mapping.get(str(row.get("scenario_family", "")), default))
            == int(row["public_target_index"])
        )
        for row in dev
    )
    accuracy = hits / max(len(dev), 1)
    return {
        "template_only_target_position_top1": accuracy,
        "clear_threshold": float(clear_min),
        "clears_judgment_gate": bool(accuracy >= float(clear_min)),
    }


def behavioral_factor_majority_probe(
    train: list[dict[str, Any]],
    dev: list[dict[str, Any]],
    clear_min: float,
) -> dict[str, Any]:
    names = sorted((train[0].get("factor_target_keys") or {}).keys())
    majority_by_bank = {
        name: majority([str(r["factor_target_keys"][name]) for r in train])
        for name in names
    }
    per_bank = {
        name: sum(
            int(str(row["factor_target_keys"][name]) == majority_by_bank[name])
            for row in dev
        ) / max(len(dev), 1)
        for name in names
    }
    macro = sum(per_bank.values()) / max(len(per_bank), 1)
    return {
        "majority_factor_macro_accuracy": macro,
        "factor_bank_accuracy": per_bank,
        "clear_threshold": float(clear_min),
        "clears_factor_gate": bool(macro >= float(clear_min)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--semantic-rows", required=True)
    p.add_argument("--semantic-manifest", required=True)
    p.add_argument("--behavioral-rows", required=True)
    p.add_argument("--behavioral-manifest", required=True)
    p.add_argument("--semantic-contract", required=True)
    p.add_argument("--behavioral-contract", required=True)
    p.add_argument("--final-contract", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    semantic_rows = read_jsonl(Path(args.semantic_rows))
    behavioral_rows = read_jsonl(Path(args.behavioral_rows))
    semantic_manifest = json.loads(Path(args.semantic_manifest).read_text(encoding="utf-8"))
    behavioral_manifest = json.loads(Path(args.behavioral_manifest).read_text(encoding="utf-8"))
    semantic_contract = json.loads(Path(args.semantic_contract).read_text(encoding="utf-8"))
    behavioral_contract = json.loads(Path(args.behavioral_contract).read_text(encoding="utf-8"))
    final_contract = json.loads(Path(args.final_contract).read_text(encoding="utf-8"))

    controls = semantic_contract["shortcut_controls"]
    required_controls = (
        "template_only_prediction_probe_required",
        "lexical_overlap_baseline_required",
        "metadata_only_baseline_required",
        "schema_only_without_query_baseline_required",
        "query_only_without_schema_baseline_required",
        "random_candidate_permutation_consistency_required",
        "hard_negative_minimum_per_positive",
        "counterfactual_pairs_required",
        "no_single_shortcut_baseline_may_clear_operator_gate",
    )
    missing_controls = [name for name in required_controls if controls.get(name) is not True]
    if missing_controls:
        raise SystemExit("shortcut control disabled or missing: " + repr(missing_controls))

    train = [row for row in semantic_rows if row.get("split") == "train"]
    dev = [row for row in semantic_rows if row.get("split") == "dev"]
    btrain = [row for row in behavioral_rows if row.get("split") == "train"]
    bdev = [row for row in behavioral_rows if row.get("split") == "dev"]
    if not train or not dev or not btrain or not bdev:
        raise SystemExit("shortcut preflight requires nonempty TRAIN/DEV in both lanes")

    if int(semantic_manifest.get("rows", -1)) != len(semantic_rows):
        raise SystemExit("semantic manifest row count drift")
    if int(behavioral_manifest.get("rows", -1)) != len(behavioral_rows):
        raise SystemExit("behavioral manifest row count drift")

    gate = semantic_contract["shortcut_probe_gate"]
    final_det = final_contract["deterministic_behavior_gate"]
    if float(gate["relation_sequence_exact_clear_min"]) != float(
        final_det["ordered_relation_sequence_exact_min"]
    ):
        raise SystemExit("shortcut relation gate drift from final-v2 precommit")
    if float(gate["factor_macro_accuracy_clear_min"]) != float(
        final_det["factor_macro_accuracy_min"]
    ):
        raise SystemExit("shortcut factor gate drift from final-v2 precommit")

    baselines = {
        "template_only": evaluate_operator_baseline(
            dev, template_baseline(train, dev), gate
        ),
        "lexical_overlap": evaluate_operator_baseline(
            dev, lexical_overlap_baseline(train, dev), gate
        ),
        "metadata_only": evaluate_operator_baseline(
            dev, metadata_baseline(train, dev), gate
        ),
        "schema_only_without_query": evaluate_operator_baseline(
            dev, schema_only_baseline(train, dev), gate
        ),
        "query_only_without_schema": evaluate_operator_baseline(
            dev, query_only_baseline(train, dev), gate
        ),
    }
    cleared = sorted(
        name for name, result in baselines.items()
        if result["clears_operator_gate"]
    )
    if cleared:
        raise SystemExit(
            "shortcut baseline cleared operator gate: " + repr(cleared)
        )

    permutation = permutation_probe(dev, seed=20260922)
    hard_negative = hard_negative_probe(semantic_rows)
    counterfactual = counterfactual_pair_probe(semantic_rows)

    behavioral_gate = float(gate["behavioral_shortcut_clear_min"])
    behavioral = {
        "candidate_only_lexical": behavioral_candidate_only_probe(
            btrain, bdev, behavioral_gate
        ),
        "template_target_position": behavioral_template_position_probe(
            btrain, bdev, behavioral_gate
        ),
        "factor_majority": behavioral_factor_majority_probe(
            btrain,
            bdev,
            float(gate["factor_macro_accuracy_clear_min"]),
        ),
    }
    behavioral_cleared = sorted(
        name
        for name, result in behavioral.items()
        if result.get("clears_judgment_gate") is True
        or result.get("clears_factor_gate") is True
    )
    if behavioral_cleared:
        raise SystemExit(
            "behavioral shortcut baseline cleared governed gate: "
            + repr(behavioral_cleared)
        )

    payload = {
        "schema": "alice.eipm.n0.full-envelope-shortcut-preflight.v1",
        "status": PASS,
        "semantic_rows": len(semantic_rows),
        "semantic_train_rows": len(train),
        "semantic_dev_rows": len(dev),
        "behavioral_rows": len(behavioral_rows),
        "behavioral_train_rows": len(btrain),
        "behavioral_dev_rows": len(bdev),
        "operator_gate": gate,
        "operator_baselines": baselines,
        "operator_baselines_clearing_gate": cleared,
        "candidate_permutation": permutation,
        "hard_negative": hard_negative,
        "counterfactual": counterfactual,
        "behavioral_shortcuts": behavioral,
        "behavioral_shortcuts_clearing_gate": behavioral_cleared,
        "legacy_t2_shortcut_receipt_used_as_authority": False,
        "private_identity_data": False,
        "gradient": False,
        "optimizer": False,
        "gpu_training_authorized": False,
        "final_validation_opened": False,
        "n0_complete": False,
    }
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(PASS)


if __name__ == "__main__":
    main()
