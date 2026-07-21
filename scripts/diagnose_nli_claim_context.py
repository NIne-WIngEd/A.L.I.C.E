from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.claim_entailment_gate import (
    cited_passages_for_claim,
    load_claim_entailment_policy,
    load_local_claim_entailment_model,
)


def _softmax(values):
    values = [float(value) for value in values]
    peak = max(values)
    exponentials = [
        math.exp(value - peak)
        for value in values
    ]
    total = sum(exponentials)
    return [
        value / total
        for value in exponentials
    ]


def _score(model, policy, premise: str, hypothesis: str) -> dict:
    logits = model.predict(
        [(premise, hypothesis)],
        show_progress_bar=False,
    )[0]
    probabilities = {
        label: probability
        for label, probability in zip(
            policy.label_order,
            _softmax(logits),
        )
    }
    return {
        label: round(float(probabilities[label]), 6)
        for label in ("contradiction", "entailment", "neutral")
    }


def _latest_response_summary(vault: Path) -> Path:
    candidates = sorted(
        (
            vault
            / "manifests"
            / "exports"
        ).glob(
            "grounded-response-summary-*.json"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No grounded-response summary was found"
        )
    return candidates[0]


def _context_path(
    *,
    vault: Path,
    response: dict,
) -> Path:
    package_id = str(
        response.get(
            "context_package_id",
            "",
        )
    )
    if not package_id:
        raise ValueError(
            "Response has no context_package_id"
        )

    summary_path = (
        vault
        / "manifests"
        / "exports"
        / f"context-summary-{package_id}.json"
    )
    if not summary_path.is_file():
        raise FileNotFoundError(
            "Matching context summary was not found: "
            + str(summary_path)
        )
    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )
    return Path(
        summary["package_path"]
    ).expanduser().resolve(strict=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vault",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--response",
        type=Path,
    )
    parser.add_argument(
        "--claim-index",
        type=int,
        default=1,
        help="1-based pre-gate claim index",
    )
    parser.add_argument(
        "--device",
        default="auto",
    )
    args = parser.parse_args()

    vault = args.vault.expanduser().resolve(
        strict=True
    )

    if args.response is None:
        summary_path = _latest_response_summary(
            vault
        )
        summary = json.loads(
            summary_path.read_text(
                encoding="utf-8"
            )
        )
        response_path = Path(
            summary["response_path"]
        ).expanduser().resolve(strict=True)
    else:
        response_path = (
            args.response.expanduser()
            .resolve(strict=True)
        )

    response = json.loads(
        response_path.read_text(
            encoding="utf-8"
        )
    )
    pre_gate = response.get(
        "pre_gate_model_output"
    )
    if not isinstance(pre_gate, dict):
        print(
            json.dumps(
                {
                    "diagnostic_schema_version": 1,
                    "pre_gate_claim_stored": False,
                    "message": (
                        "This response predates pre-gate "
                        "claim preservation. Generate a new "
                        "response after applying the patch."
                    ),
                },
                indent=2,
            )
        )
        return 2

    claims = list(
        pre_gate.get(
            "claims",
            [],
        )
    )
    claim_position = args.claim_index - 1
    if not (
        0
        <= claim_position
        < len(claims)
    ):
        raise IndexError(
            "claim-index is outside the "
            "pre-gate claim list"
        )
    claim = claims[
        claim_position
    ]

    context_path = _context_path(
        vault=vault,
        response=response,
    )
    context = json.loads(
        context_path.read_text(
            encoding="utf-8"
        )
    )

    policy = load_claim_entailment_policy()
    model, _ = (
        load_local_claim_entailment_model(
            vault_root=vault,
            device=args.device,
        )
    )

    passages = cited_passages_for_claim(
        claim=claim,
        context_package=context,
        limit=(
            policy
            .maximum_evidence_passages_per_claim
        ),
    )
    if not passages:
        raise ValueError(
            "The pre-gate claim has no usable "
            "cited evidence passages"
        )

    hypothesis = str(
        claim.get(
            "text",
            "",
        )
    ).strip()

    individual = []
    for index, passage in enumerate(
        passages,
        start=1,
    ):
        individual.append(
            {
                "passage_index": index,
                "citation": passage[
                    "citation"
                ],
                "premise_character_count": len(
                    passage["premise"]
                ),
                **_score(
                    model,
                    policy,
                    passage["premise"],
                    hypothesis,
                ),
            }
        )

    combined = "\n\n".join(
        passage["premise"]
        for passage in passages
    )

    cited_ids = {
        str(value)
        for value in claim.get(
            "citations",
            [],
        )
    }
    evidence_by_citation = {
        str(item.get("citation")): item
        for item in context.get(
            "evidence",
            [],
        )
    }
    cited_records = [
        evidence_by_citation[citation]
        for citation in cited_ids
        if citation in evidence_by_citation
    ]
    all_owner_self_records = bool(
        cited_records
    ) and all(
        str(
            item.get(
                "owner_relation",
                "",
            )
        )
        == "owner_self_record"
        for item in cited_records
    )

    explicit_owner_premise = combined
    if all_owner_self_records:
        explicit_owner_premise = (
            "Trusted metadata: the following "
            "evidence comes from the user's own "
            "self-records and describes the user's "
            "personal roles, projects, education, "
            "experience, or achievements where "
            "those statements appear.\n\n"
            + combined
        )

    output = {
        "diagnostic_schema_version": 1,
        "pre_gate_claim_stored": True,
        "claim_index": args.claim_index,
        "claim_type": str(
            claim.get(
                "claim_type",
                "",
            )
        ),
        "claim_citation_count": len(
            cited_ids
        ),
        "passage_count": len(
            passages
        ),
        "all_cited_sources_owner_self_records": (
            all_owner_self_records
        ),
        "individual_passages": individual,
        "best_individual_entailment": max(
            item["entailment"]
            for item in individual
        ),
        "combined_owner_aware": {
            "premise_character_count": len(
                combined
            ),
            **_score(
                model,
                policy,
                combined,
                hypothesis,
            ),
        },
        "combined_explicit_owner_attribution": {
            "premise_character_count": len(
                explicit_owner_premise
            ),
            **_score(
                model,
                policy,
                explicit_owner_premise,
                hypothesis,
            ),
        },
        "nli_model_id": policy.model_id,
        "nli_label_order": list(policy.label_order),
        "configured_entailment_threshold": (
            policy.entailment_threshold
        ),
        "configured_contradiction_threshold": (
            policy.contradiction_threshold
        ),
        "private_text_printed": False,
    }

    print(
        json.dumps(
            output,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
