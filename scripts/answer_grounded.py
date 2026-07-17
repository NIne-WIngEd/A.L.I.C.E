from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alice_vault.grounded_context import build_grounded_context
from alice_vault.grounded_response import (
    atomic_json,
    generate_grounded_response,
    load_grounded_response_policy,
)
from alice_vault.response_context_enrichment import (
    enrich_response_context,
)
from alice_vault.owner_attribution import (
    annotate_context_owner_relation,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--pilot-name", default="pilot-v1")
    parser.add_argument("--show-answer", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    context = build_grounded_context(
        vault_root=args.vault,
        query=args.query,
        pilot_name=args.pilot_name,
        device=args.device,
    )
    response_policy = load_grounded_response_policy()
    context_package_path = Path(
        context["summary"]["package_path"]
    )
    context_package = context["package"]
    expansion = response_policy.evidence_expansion
    if expansion["enabled"]:
        context_package = enrich_response_context(
            vault_root=args.vault,
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
            device=args.device,
        )
        atomic_json(
            context_package_path,
            context_package,
        )

        context_summary_path = Path(
            context["summary"]["summary_path"]
        )
        context["summary"][
            "package_fingerprint"
        ] = context_package[
            "package_fingerprint"
        ]
        context["summary"][
            "response_context_enriched"
        ] = True
        context["summary"][
            "response_selected_passage_count"
        ] = context_package[
            "response_context_enrichment"
        ]["selected_passage_count"]
        atomic_json(
            context_summary_path,
            context["summary"],
        )

    context_package = annotate_context_owner_relation(
        vault_root=args.vault,
        context_package=context_package,
        require_identity=True,
    )
    atomic_json(
        context_package_path,
        context_package,
    )
    context["summary"][
        "package_fingerprint"
    ] = context_package[
        "package_fingerprint"
    ]
    context["summary"][
        "owner_attribution_available"
    ] = True
    context["summary"][
        "owner_relation_counts"
    ] = context_package[
        "owner_identity_context"
    ]["relation_counts"]
    atomic_json(
        Path(context["summary"]["summary_path"]),
        context["summary"],
    )

    response = generate_grounded_response(
        vault_root=args.vault,
        context_package_path=context_package_path,
    )

    if args.show_answer:
        output = {
            **response["summary"],
            "answer": response[
                "response_package"
            ]["model_output"]["answer"],
        }
    else:
        output = response["summary"]

    print(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        )
    )
    return (
        0
        if response["summary"]["verified"]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
