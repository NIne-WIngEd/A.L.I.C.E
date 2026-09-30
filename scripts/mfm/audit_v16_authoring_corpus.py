#!/usr/bin/env python3
"""Check exact v1.6 diagnostic authoring rows and report admission blockers.

This checks structure and grounded references, not semantic truth, consent,
reviewer identity, independent splits, a sealed FINAL or model capability.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.formation_learning_v16 import (
    FULL_ROLE_DIMENSIONS, learning_example_v16_from_record,
    model_input_sha256_v16, supervised_output_record_v16,
)


def _pair_isolation(pair: str, left: dict, right: dict) -> str:
    """Require exact same input aside from one decisive source byte digest.

    A changed `case_id`, lineage, expected answer or generated proposal ID is
    outside the model input. A changed host/scope, authority, target registry,
    role, timestamp, or irrelevant source invalidates the contrast.
    """
    if left["split"] != right["split"]:
        raise CognitiveKernelContractError(f"counterfactual {pair} crosses splits")
    if left["lineage"]["host_family"] != right["lineage"]["host_family"]:
        raise CognitiveKernelContractError(f"counterfactual {pair} changes host family")
    l_sources, r_sources = left["sources"], right["sources"]
    if len(l_sources) != len(r_sources):
        raise CognitiveKernelContractError(f"counterfactual {pair} changes source count")
    changed = [l["ref_id"] for l, r in zip(l_sources, r_sources, strict=True)
               if l["content_b64"] != r["content_b64"]]
    if len(changed) != 1 or [s["ref_id"] for s in l_sources] != [s["ref_id"] for s in r_sources]:
        raise CognitiveKernelContractError(f"counterfactual {pair} must flip one source")
    left_context = deepcopy(left["context"])
    right_context = deepcopy(right["context"])
    for context in (left_context, right_context):
        # Normalize exactly one expected difference. Everything else in the
        # model-visible metadata must match, including registered target links.
        matches = [e for e in context["base_context"]["evidence"]
                   if e["ref_id"] == changed[0]]
        if len(matches) != 1:
            raise CognitiveKernelContractError(f"counterfactual {pair} lost decisive source")
        matches[0]["content_digest"] = "0" * 64
    if left_context != right_context:
        raise CognitiveKernelContractError(f"counterfactual {pair} changed other input metadata")
    for left_source, right_source in zip(l_sources, r_sources, strict=True):
        if left_source["ref_id"] != changed[0] and left_source != right_source:
            raise CognitiveKernelContractError(f"counterfactual {pair} changed another source")
    return changed[0]


def audit(path: Path, expected_sha256: str) -> dict:
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    if digest != expected_sha256 or len(expected_sha256) != 64:
        raise CognitiveKernelContractError("v1.6 authoring corpus differs from pinned SHA-256")
    cases: dict[str, dict] = {}
    pairs: dict[str, list[dict]] = defaultdict(list)
    by_split: Counter[str] = Counter()
    observed: dict[str, Counter[str]] = defaultdict(Counter)
    inputs: dict[str, str] = {}
    family_splits: dict[str, set[str]] = defaultdict(set)
    source_splits: dict[str, set[str]] = defaultdict(set)
    proposal_kinds: dict[str, Counter[str]] = defaultdict(Counter)
    disposition_actions: dict[str, Counter[str]] = defaultdict(Counter)
    for lineno, line in enumerate(raw.splitlines(), 1):
        if not line:
            raise CognitiveKernelContractError(f"empty corpus row at line {lineno}")
        row = json.loads(line)
        case_id = row["case_id"]
        if case_id in cases:
            raise CognitiveKernelContractError("duplicate case ID")
        if row["split"] not in {"train", "development"}:
            raise CognitiveKernelContractError("diagnostic authoring cannot contain FINAL")
        if row.get("authorization_id") != "owner-directed-mfm-v16-synthetic":
            raise CognitiveKernelContractError("synthetic row lacks pinned owner direction")
        if row.get("status", {}).get("qualification") != "diagnostic-only-unadmitted" or (
            row["status"].get("target_review") != "single-author-assertion-only" or
            row["status"].get("rights") != "pending-independent-authentication"):
            raise CognitiveKernelContractError("seed falsely claims admission/review/rights")
        example = learning_example_v16_from_record(row, split=row["split"])
        supervised_output_record_v16(example)
        input_digest = model_input_sha256_v16(example)
        if input_digest in inputs:
            raise CognitiveKernelContractError(f"duplicate input in {inputs[input_digest]} and {case_id}")
        inputs[input_digest] = case_id
        by_split[row["split"]] += 1
        for dimension, state in example.adjudications:
            if state not in {"present", "negative"}:
                raise CognitiveKernelContractError("unknown label in supervised diagnostic seed")
            if state == "present":
                observed[row["split"]][dimension] += 1
        proposal_kinds[row["split"]].update(p.base.kind for p in example.target.proposals)
        disposition_actions[row["split"]].update(
            d.action for d in example.target.base.dispositions)
        lineage = row["lineage"]
        for key in ("host_family", "generator_family"):
            family_splits[f"{key}:{lineage[key]}"].add(row["split"])
        for _, source in example.opened_sources:
            source_splits[sha256(source).hexdigest()].add(row["split"])
        if lineage.get("counterfactual_pair"):
            pairs[lineage["counterfactual_pair"]].append(row)
        cases[case_id] = row
    if not cases:
        raise CognitiveKernelContractError("empty v1.6 authoring corpus")
    matched_pairs = {}
    for pair, rows in sorted(pairs.items()):
        if len(rows) != 2:
            raise CognitiveKernelContractError(f"counterfactual {pair} needs exactly two rows")
        matched_pairs[pair] = _pair_isolation(pair, *rows)
    histories: dict[str, list[str]] = defaultdict(list)
    for case_id, row in cases.items():
        lineage = row["lineage"]
        if lineage.get("history_id"):
            histories[lineage["history_id"]].append(case_id)
        for parent_id in lineage.get("parent_case_ids", []):
            if parent_id not in cases or cases[parent_id]["split"] != row["split"]:
                raise CognitiveKernelContractError("history parent is absent or crosses split")
            parent = cases[parent_id]
            if (parent["context"]["base_context"]["scope"] !=
                    row["context"]["base_context"]["scope"] or
                    parent["context"]["base_context"]["authority_namespace_id"] !=
                    row["context"]["base_context"]["authority_namespace_id"]):
                raise CognitiveKernelContractError("history parent changes host or authority")
            parent_sources = {s["ref_id"]: s["content_b64"] for s in parent["sources"]}
            child_sources = {s["ref_id"]: s["content_b64"] for s in row["sources"]}
            if not parent_sources.items() <= child_sources.items():
                raise CognitiveKernelContractError("history child rewrites parent evidence")
    for history_id, members in histories.items():
        if len({cases[name]["split"] for name in members}) != 1:
            raise CognitiveKernelContractError(f"history {history_id} crosses splits")
    same_source = sorted(k for k, splits in source_splits.items() if len(splits) > 1)
    shared_family = sorted(k for k, splits in family_splits.items() if len(splits) > 1)
    return {
        "schema": "mfm-v16-authoring-audit-v1", "corpus_sha256": digest,
        "state": "synthetic-training-and-development-diagnostics-only",
        "case_counts": {split: by_split[split] for split in ("train", "development")},
        "positive_case_counts": {
            split: {name: observed[split][name] for name in sorted(FULL_ROLE_DIMENSIONS)}
            for split in ("train", "development")},
        "controlled_input_pairs": matched_pairs,
        "linked_history_cases": {key: sorted(value) for key, value in sorted(histories.items())},
        "proposal_kind_counts": {
            split: dict(sorted(proposal_kinds[split].items()))
            for split in ("train", "development")},
        "disposition_action_counts": {
            split: dict(sorted(disposition_actions[split].items()))
            for split in ("train", "development")},
        "cross_split_source_digest_count": len(same_source),
        "cross_split_family_keys": shared_family,
        "full_role_admission": "blocked",
        "blocking_reasons": [
            "single assistant author; no independent target reviews or authenticated rights receipts",
            "development shares synthetic generator lineage with training",
            "no sealed independently adjudicated FINAL; no real owner histories in this seed",
            "contract grounding does not prove semantic entailment or later judgment",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = audit(args.corpus, args.expected_sha256)
    output = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(output)
    print(output.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
