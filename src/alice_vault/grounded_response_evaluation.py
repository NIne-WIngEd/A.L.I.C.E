from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from .grounded_context import build_grounded_context
from .grounded_response import (
    atomic_json,
    generate_grounded_response,
    load_grounded_response_policy,
)
from .response_context_enrichment import (
    enrich_response_context,
)
from .owner_attribution import (
    annotate_context_owner_relation,
)


def _checkpoint_key(
    *,
    benchmark_id: str,
    policy_digest: str,
    pilot_name: str,
) -> str:
    material = (
        "alice-grounded-response-evaluation-v10\0"
        f"{benchmark_id}\0{policy_digest}\0{pilot_name}"
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"completed_cases": {}, "last_error": None}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Evaluation checkpoint is not an object")
    completed = value.get("completed_cases", {})
    if not isinstance(completed, dict):
        raise ValueError("Evaluation checkpoint cases are invalid")
    return value


def _cached_embedding_loader(
    *,
    vault_root: Path,
    semantic_policy_path: Path | None,
    device: str,
):
    from .semantic_retrieval import (
        _load_local_model,
        load_semantic_policy,
    )

    semantic_policy = load_semantic_policy(
        semantic_policy_path
    )
    model, _ = _load_local_model(
        vault_root=vault_root,
        policy=semantic_policy,
        device=device,
    )

    def loader(*args, **kwargs):
        return model

    return loader


def evaluate_grounded_responses(
    *,
    vault_root: Path,
    benchmark_path: Path,
    pilot_name: str = "pilot-v1",
    response_policy_path: Path | None = None,
    context_policy_path: Path | None = None,
    semantic_policy_path: Path | None = None,
    lexical_policy_path: Path | None = None,
    device: str = "auto",
    context_search_fn=None,
    context_model_loader=None,
    response_model_client=None,
) -> dict[str, Any]:
    vault_root = vault_root.expanduser().resolve(
        strict=True
    )
    benchmark_path = benchmark_path.expanduser().resolve(
        strict=True
    )
    response_policy = load_grounded_response_policy(
        response_policy_path
    )
    benchmark = json.loads(
        benchmark_path.read_text(encoding="utf-8")
    )
    cases = [
        case
        for case in benchmark.get("cases", [])
        if str(case.get("status", "")).casefold()
        == "approved"
    ]
    if not cases:
        raise ValueError(
            "Benchmark contains no approved cases"
        )

    private_root = (
        vault_root
        / "manifests"
        / "responses"
        / pilot_name
    )
    exports = vault_root / "manifests" / "exports"
    temporary_root = vault_root / "temporary"
    for path in (private_root, exports, temporary_root):
        path.mkdir(parents=True, exist_ok=True)

    benchmark_id = str(benchmark.get("benchmark_id", ""))
    checkpoint_id = _checkpoint_key(
        benchmark_id=benchmark_id,
        policy_digest=response_policy.digest,
        pilot_name=pilot_name,
    )
    checkpoint_path = (
        private_root
        / f"grounded-response-evaluation-checkpoint-{checkpoint_id}.json"
    )
    checkpoint = _load_checkpoint(checkpoint_path)
    completed_cases: dict[str, dict[str, Any]] = dict(
        checkpoint.get("completed_cases", {})
    )
    resumed_case_count = sum(
        str(case.get("query_id", "")) in completed_cases
        for case in cases
    )

    cached_context_loader = context_model_loader
    if cached_context_loader is None and context_search_fn is None:
        cached_context_loader = _cached_embedding_loader(
            vault_root=vault_root,
            semantic_policy_path=semantic_policy_path,
            device=device,
        )

    new_case_count = 0
    for case_number, case in enumerate(cases, start=1):
        query_id = str(case.get("query_id", ""))
        if query_id in completed_cases:
            print(
                f"Resumed grounded response {case_number}/{len(cases)} "
                f"({query_id})"
            )
            continue

        temporary_context_path = (
            temporary_root
            / ("p1-11-eval-context-" + uuid.uuid4().hex + ".json")
        )
        try:
            context_result = build_grounded_context(
                vault_root=vault_root,
                query=str(case["question"]),
                pilot_name=pilot_name,
                policy_path=context_policy_path,
                semantic_policy_path=semantic_policy_path,
                lexical_policy_path=lexical_policy_path,
                device=device,
                save=False,
                search_fn=context_search_fn,
                model_loader=cached_context_loader,
            )
            context_package = context_result["package"]
            expansion = response_policy.evidence_expansion
            # A custom context_search_fn is an injected retrieval boundary used
            # by unit tests and specialized callers. In that mode the caller
            # owns the evidence construction contract, so do not require the
            # production semantic model/index for response-time expansion.
            #
            # Production evaluation leaves context_search_fn=None, therefore
            # evidence passage expansion still runs normally against the local
            # semantic index.
            should_expand = (
                expansion["enabled"]
                and context_search_fn is None
            )
            if should_expand:
                context_package = enrich_response_context(
                    vault_root=vault_root,
                    context_package=context_package,
                    passages_per_source=int(
                        expansion["passages_per_source"]
                    ),
                    maximum_characters_per_source=int(
                        expansion[
                            "maximum_characters_per_source"
                        ]
                    ),
                    lexical_overlap_weight=float(
                        expansion[
                            "lexical_overlap_weight"
                        ]
                    ),
                    minimum_passage_characters=int(
                        expansion[
                            "minimum_passage_characters"
                        ]
                    ),
                    semantic_policy_path=semantic_policy_path,
                    device=device,
                    model_loader=cached_context_loader,
                )
                context_package = annotate_context_owner_relation(
                    vault_root=vault_root,
                    context_package=context_package,
                    require_identity=True,
                )
            atomic_json(
                temporary_context_path,
                context_package,
            )
            response_result = generate_grounded_response(
                vault_root=vault_root,
                context_package_path=temporary_context_path,
                policy_path=response_policy_path,
                model_client=response_model_client,
                save=False,
            )

            verification = response_result[
                "response_package"
            ]["verification"]
            expected = {
                str(value)
                for value in case.get(
                    "expected_source_sha256",
                    [],
                )
            }
            cited = set(
                verification["cited_source_sha256"]
            )
            completed_cases[query_id] = {
                "query_id": query_id,
                "verified": bool(verification["verified"]),
                "expected_source_cited": bool(
                    expected.intersection(cited)
                ),
                "claim_count": int(
                    verification["claim_count"]
                ),
                "cited_claim_count": int(
                    verification["cited_claim_count"]
                ),
                "claim_citation_coverage": verification[
                    "claim_citation_coverage"
                ],
                "inline_answer_citation_count": int(
                    verification[
                        "inline_answer_citation_count"
                    ]
                ),
                "verification_error_count": int(
                    verification["error_count"]
                ),
                "verification_errors": list(
                    verification.get("errors", [])
                ),
                "answer_type": response_result[
                    "response_package"
                ]["model_output"].get("answer_type"),
            }
            new_case_count += 1
            atomic_json(
                checkpoint_path,
                {
                    "checkpoint_schema_version": 1,
                    "checkpoint_id": checkpoint_id,
                    "benchmark_id": benchmark_id,
                    "pilot_name": pilot_name,
                    "policy_digest": response_policy.digest,
                    "completed_cases": completed_cases,
                    "last_error": None,
                },
            )
            print(
                f"Completed grounded response {case_number}/{len(cases)} "
                f"({query_id})"
            )
        except Exception as exc:
            atomic_json(
                checkpoint_path,
                {
                    "checkpoint_schema_version": 1,
                    "checkpoint_id": checkpoint_id,
                    "benchmark_id": benchmark_id,
                    "pilot_name": pilot_name,
                    "policy_digest": response_policy.digest,
                    "completed_cases": completed_cases,
                    "last_error": {
                        "query_id": query_id,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                },
            )
            raise RuntimeError(
                "Grounded-response evaluation stopped at "
                f"{query_id}. Progress for {len(completed_cases)}/"
                f"{len(cases)} cases was saved to {checkpoint_path}. "
                "Run the same command again to resume."
            ) from exc
        finally:
            temporary_context_path.unlink(missing_ok=True)

    ordered_details = [
        completed_cases[str(case.get("query_id", ""))]
        for case in cases
    ]
    count = len(ordered_details)
    verified_count = sum(
        bool(item["verified"]) for item in ordered_details
    )
    expected_source_cited_count = sum(
        bool(item["expected_source_cited"])
        for item in ordered_details
    )
    total_claims = sum(
        int(item["claim_count"]) for item in ordered_details
    )
    total_cited_claims = sum(
        int(item["cited_claim_count"])
        for item in ordered_details
    )

    verified_rate = verified_count / count
    expected_source_citation_rate = (
        expected_source_cited_count / count
    )
    claim_citation_coverage = (
        total_cited_claims / total_claims
        if total_claims
        else 1.0
    )

    run_id = str(uuid.uuid4())
    summary = {
        "grounded_response_evaluation_schema_version": 10,
        "run_id": run_id,
        "benchmark_id": benchmark_id,
        "pilot_name": pilot_name,
        "approved_case_count": count,
        "resumed_case_count": resumed_case_count,
        "new_case_count": new_case_count,
        "checkpoint_path": str(checkpoint_path),
        "verified_response_rate": round(verified_rate, 6),
        "minimum_verified_response_rate": (
            response_policy.minimum_verified_response_rate
        ),
        "expected_source_citation_rate": round(
            expected_source_citation_rate,
            6,
        ),
        "minimum_expected_source_citation_rate": (
            response_policy.minimum_expected_source_citation_rate
        ),
        "claim_citation_coverage": round(
            claim_citation_coverage,
            6,
        ),
        "minimum_claim_citation_coverage": (
            response_policy.minimum_claim_citation_coverage
        ),
        "passes_verification_threshold": (
            verified_rate
            >= response_policy.minimum_verified_response_rate
        ),
        "passes_expected_source_citation_threshold": (
            expected_source_citation_rate
            >= response_policy.minimum_expected_source_citation_rate
        ),
        "passes_claim_citation_threshold": (
            claim_citation_coverage
            >= response_policy.minimum_claim_citation_coverage
        ),
        "memory_write_allowed": False,
        "external_action_allowed": False,
        "tool_calling_allowed": False,
        "web_access_allowed": False,
    }
    summary["passes_all_thresholds"] = all(
        [
            summary["passes_verification_threshold"],
            summary[
                "passes_expected_source_citation_threshold"
            ],
            summary["passes_claim_citation_threshold"],
        ]
    )

    details_path = (
        private_root
        / f"grounded-response-evaluation-details-{run_id}.json"
    )
    summary_path = (
        exports
        / f"grounded-response-evaluation-summary-{run_id}.json"
    )
    atomic_json(
        details_path,
        {
            **summary,
            "benchmark_path": str(benchmark_path),
            "cases": ordered_details,
        },
    )
    summary["private_details_path"] = str(details_path)
    atomic_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary
