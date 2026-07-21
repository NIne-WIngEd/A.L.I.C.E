from __future__ import annotations

import argparse
import hashlib
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
from alice_vault.query_claim_relevance_calibration import (
    load_query_claim_relevance_calibration_policy,
)
from alice_vault.response_context_enrichment import enrich_response_context
from alice_vault.response_reranker import load_local_response_reranker


def stable_item_id(query_id: str, claim_text: str) -> str:
    return hashlib.sha256(
        f"{query_id}\0{claim_text}".encode("utf-8")
    ).hexdigest()[:20]


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


def select_rank_stratified_with_query_cap(
    candidates: list[dict],
    *,
    sample_size: int,
    max_per_query: int,
) -> list[dict]:
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda x: float(x["relevance_score"]))
    bucket_count = min(4, len(ordered))
    buckets = [[] for _ in range(bucket_count)]

    for index, item in enumerate(ordered):
        bucket_index = min(
            bucket_count - 1,
            int(index * bucket_count / len(ordered)),
        )
        buckets[bucket_index].append(item)

    # Prefer score extremes within each bucket so the calibration spans
    # the model's observed score distribution.
    for bucket in buckets:
        bucket.sort(
            key=lambda x: (
                Counter(
                    c["query_id"] for c in candidates
                )[x["query_id"]],
                x["relevance_score"],
            )
        )

    selected = []
    per_query = Counter()

    while len(selected) < sample_size:
        progress = False
        for bucket in buckets:
            candidate_index = None
            for i, item in enumerate(bucket):
                if per_query[item["query_id"]] < max_per_query:
                    candidate_index = i
                    break

            if candidate_index is None:
                continue

            item = bucket.pop(candidate_index)
            selected.append(item)
            per_query[item["query_id"]] += 1
            progress = True

            if len(selected) >= sample_size:
                break

        if not progress:
            break

    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--pilot-name", default="pilot-v1")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--max-per-query", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    vault = args.vault.expanduser().resolve(strict=True)
    benchmark_path = args.benchmark.expanduser().resolve(strict=True)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    relevance_policy = load_query_claim_relevance_calibration_policy()
    response_policy = load_grounded_response_policy()

    excluded = set(relevance_policy.excluded_regression_query_ids)
    cases = [
        case for case in benchmark.get("cases", [])
        if str(case.get("status", "")).casefold() == "approved"
        and str(case.get("query_id", "")) not in excluded
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
        temp_context = temp_root / f"p1-11-fresh-relevance-{uuid.uuid4().hex}.json"

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
            model_output = response["response_package"]["model_output"]
            claims = model_output.get("claims", [])

            for claim in claims:
                if not isinstance(claim, dict):
                    continue
                text = str(claim.get("text", "")).strip()
                if not text:
                    continue
                candidates.append(
                    {
                        "item_id": stable_item_id(query_id, text),
                        "query_id": query_id,
                        "question": question,
                        "claim_text": text,
                        "claim_type": str(claim.get("claim_type", "")),
                        "citations": list(claim.get("citations", [])),
                        "answer_type": str(model_output.get("answer_type", "")),
                        "relevance_human_label": "",
                        "relevance_human_labeled_at": "",
                    }
                )

            print(
                f"Generated fresh relevance candidates {index}/{len(cases)} "
                f"({query_id}): {len(claims)} claims"
            )
        finally:
            temp_context.unlink(missing_ok=True)

    if not candidates:
        raise ValueError("The current P1.11 pipeline produced no fresh candidates")

    pairs = [[item["question"], item["claim_text"]] for item in candidates]
    scores = reranker.predict(
        pairs,
        batch_size=reranker_policy.batch_size,
        show_progress_bar=False,
    )
    for item, score in zip(candidates, scores):
        item["relevance_score"] = float(score)

    selected = select_rank_stratified_with_query_cap(
        candidates,
        sample_size=min(args.sample_size, len(candidates)),
        max_per_query=args.max_per_query,
    )

    calibration_id = str(uuid.uuid4())
    private_root = vault / "manifests" / "calibration" / args.pilot_name
    export_root = vault / "manifests" / "exports"
    bundle_path = (
        private_root
        / f"query-claim-relevance-calibration-v2-{calibration_id}.json"
    )
    summary_path = (
        export_root
        / f"query-claim-relevance-calibration-v2-summary-{calibration_id}.json"
    )

    query_counts = Counter(item["query_id"] for item in selected)
    score_values = [float(item["relevance_score"]) for item in selected]

    bundle = {
        "query_claim_relevance_calibration_bundle_schema_version": 1,
        "relevance_calibration_generation_version": 2,
        "calibration_id": calibration_id,
        "pilot_name": args.pilot_name,
        "benchmark_path": str(benchmark_path),
        "source_kind": "fresh_current_p1_11_post_fix_outputs",
        "excluded_regression_query_ids": sorted(excluded),
        "candidate_count": len(candidates),
        "selected_sample_size": len(selected),
        "max_per_query": args.max_per_query,
        "selected_query_counts": dict(query_counts),
        "reranker": {
            "model_id": reranker_policy.model_id,
            "revision": reranker_policy.revision,
            "policy_id": reranker_policy.policy_id,
        },
        "selected_score_min": min(score_values) if score_values else None,
        "selected_score_max": max(score_values) if score_values else None,
        "items": selected,
    }
    atomic_json(bundle_path, bundle)

    summary = {
        "query_claim_relevance_calibration_v2_summary_schema_version": 1,
        "calibration_id": calibration_id,
        "candidate_count": len(candidates),
        "selected_sample_size": len(selected),
        "selected_query_count": len(query_counts),
        "selected_query_counts": dict(query_counts),
        "max_per_query": args.max_per_query,
        "excluded_regression_query_ids": sorted(excluded),
        "model_id": reranker_policy.model_id,
        "private_bundle_path": str(bundle_path),
        "private_text_uploaded": False,
    }
    atomic_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
