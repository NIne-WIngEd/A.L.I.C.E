from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.grounded_context import build_grounded_context
from alice_vault.grounded_response import (
    atomic_json,
    generate_grounded_response,
    load_grounded_response_policy,
)
from alice_vault.owner_attribution import annotate_context_owner_relation
from alice_vault.query_claim_relevance_holdout import (
    calibration_item_ids,
    load_query_claim_relevance_holdout_policy,
    select_with_query_cap,
    stable_item_id,
)
from alice_vault.response_context_enrichment import enrich_response_context
from alice_vault.response_reranker import load_local_response_reranker


def cached_embedding_loader(vault_root: Path, device: str):
    from alice_vault.semantic_retrieval import _load_local_model, load_semantic_policy

    policy = load_semantic_policy()
    model, _ = _load_local_model(
        vault_root=vault_root,
        policy=policy,
        device=device,
    )

    def loader(*args, **kwargs):
        return model

    return loader


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--pilot-name", default="pilot-v1")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--max-per-query", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args()

    vault = args.vault.expanduser().resolve(strict=True)
    benchmark_path = args.benchmark.expanduser().resolve(strict=True)
    calibration_path = args.calibration.expanduser().resolve(strict=True)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    policy = load_query_claim_relevance_holdout_policy(args.policy)
    response_policy = load_grounded_response_policy()

    excluded_item_ids = calibration_item_ids(calibration)
    excluded_queries = set(policy.excluded_regression_query_ids)
    calibration_query_ids = {
        str(item.get("query_id", ""))
        for item in calibration.get("items", [])
        if str(item.get("query_id", ""))
    }

    cases = [
        case for case in benchmark.get("cases", [])
        if str(case.get("status", "")).casefold() == "approved"
        and str(case.get("query_id", "")) not in excluded_queries
    ]

    model_loader = cached_embedding_loader(vault, args.device)
    reranker, reranker_policy = load_local_response_reranker(
        vault_root=vault,
        device=args.device,
    )

    candidates = []
    temp_root = vault / "temporary"
    temp_root.mkdir(parents=True, exist_ok=True)

    for index, case in enumerate(cases, start=1):
        query_id = str(case.get("query_id", ""))
        question = str(case.get("question", "")).strip()
        temp_context = (
            temp_root / f"p1-11-relevance-holdout-{uuid.uuid4().hex}.json"
        )
        try:
            context_result = build_grounded_context(
                vault_root=vault,
                query=question,
                pilot_name=args.pilot_name,
                device=args.device,
                save=False,
                model_loader=model_loader,
            )
            context_package = context_result["package"]

            expansion = response_policy.evidence_expansion
            if expansion.get("enabled") is True:
                context_package = enrich_response_context(
                    vault_root=vault,
                    context_package=context_package,
                    passages_per_source=int(expansion["passages_per_source"]),
                    maximum_characters_per_source=int(
                        expansion["maximum_characters_per_source"]
                    ),
                    lexical_overlap_weight=float(
                        expansion["lexical_overlap_weight"]
                    ),
                    minimum_passage_characters=int(
                        expansion["minimum_passage_characters"]
                    ),
                    device=args.device,
                    model_loader=model_loader,
                )

            context_package = annotate_context_owner_relation(
                vault_root=vault,
                context_package=context_package,
                require_identity=True,
            )
            atomic_json(temp_context, context_package)

            response = generate_grounded_response(
                vault_root=vault,
                context_package_path=temp_context,
                save=False,
            )
            claims = (
                response["response_package"]
                .get("model_output", {})
                .get("claims", [])
            )

            kept = 0
            for claim in claims:
                if not isinstance(claim, dict):
                    continue
                text = str(claim.get("text", "")).strip()
                if not text:
                    continue
                item_id = stable_item_id(query_id, text)
                if item_id in excluded_item_ids:
                    continue
                candidates.append(
                    {
                        "item_id": item_id,
                        "query_id": query_id,
                        "question": question,
                        "claim_text": text,
                        "citations": list(claim.get("citations", [])),
                        "relevance_human_label": "",
                        "relevance_human_labeled_at": "",
                    }
                )
                kept += 1

            print(
                f"Generated holdout candidates {index}/{len(cases)} "
                f"({query_id}): {kept} new claims",
                file=sys.stderr,
            )
        finally:
            temp_context.unlink(missing_ok=True)

    if not candidates:
        raise ValueError(
            "No fresh holdout candidates remain after calibration exclusion"
        )

    pairs = [[item["question"], item["claim_text"]] for item in candidates]
    scores = reranker.predict(
        pairs,
        batch_size=reranker_policy.batch_size,
        show_progress_bar=False,
    )
    for item, score in zip(candidates, scores):
        item["relevance_score"] = float(score)

    sample_size = args.sample_size or policy.default_sample_size
    max_per_query = args.max_per_query or policy.max_per_query
    selected = select_with_query_cap(
        candidates,
        sample_size=min(sample_size, len(candidates)),
        max_per_query=max_per_query,
    )

    holdout_id = str(uuid.uuid4())
    private_root = vault / "manifests" / "calibration" / args.pilot_name
    export_root = vault / "manifests" / "exports"
    bundle_path = (
        private_root / f"query-claim-relevance-holdout-{holdout_id}.json"
    )
    summary_path = (
        export_root / f"query-claim-relevance-holdout-prepare-summary-{holdout_id}.json"
    )
    selected_query_counts = Counter(item["query_id"] for item in selected)
    selected_query_ids = set(selected_query_counts)

    bundle = {
        # Keep compatibility with run_query_claim_relevance_review.py.
        "query_claim_relevance_calibration_bundle_schema_version": 1,
        "query_claim_relevance_holdout_bundle_schema_version": 1,
        "holdout_id": holdout_id,
        "source_calibration_id": str(calibration.get("calibration_id", "")),
        "source_calibration_path": str(calibration_path),
        "pilot_name": args.pilot_name,
        "benchmark_path": str(benchmark_path),
        "source_kind": "fresh_current_p1_11_post_fix_outputs",
        "objective": policy.objective,
        "frozen_threshold": policy.frozen_threshold,
        "threshold_frozen_before_human_review": True,
        "threshold_sweep_allowed": False,
        "candidate_count_before_selection": len(candidates),
        "selected_sample_size": len(selected),
        "selected_query_count": len(selected_query_counts),
        "selected_query_counts": dict(selected_query_counts),
        "max_per_query": max_per_query,
        "excluded_exact_calibration_item_count": len(excluded_item_ids),
        "excluded_regression_query_ids": sorted(excluded_queries),
        "query_overlap_with_calibration_count": len(
            selected_query_ids.intersection(calibration_query_ids)
        ),
        "independence_scope": (
            "fresh-generation claim-level holdout; exact calibration item_ids "
            "and known regression queries excluded; query_ids may overlap"
        ),
        "reranker": {
            "model_id": reranker_policy.model_id,
            "revision": reranker_policy.revision,
            "policy_id": reranker_policy.policy_id,
        },
        "items": selected,
    }
    atomic_json(bundle_path, bundle)

    summary = {
        "query_claim_relevance_holdout_prepare_summary_schema_version": 1,
        "holdout_id": holdout_id,
        "source_calibration_id": str(calibration.get("calibration_id", "")),
        "candidate_count_before_selection": len(candidates),
        "selected_sample_size": len(selected),
        "selected_query_count": len(selected_query_counts),
        "selected_query_counts": dict(selected_query_counts),
        "query_overlap_with_calibration_count": bundle[
            "query_overlap_with_calibration_count"
        ],
        "frozen_threshold": policy.frozen_threshold,
        "threshold_frozen_before_human_review": True,
        "private_bundle_path": str(bundle_path),
        "private_text_uploaded": False,
    }
    atomic_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
