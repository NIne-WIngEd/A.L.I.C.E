#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


PASS = "PASS_N0_FULL_ENVELOPE_PROOF_OBLIGATIONS_STATIC_V1"
FAIL = "FAIL_N0_FULL_ENVELOPE_PROOF_OBLIGATIONS_STATIC_V1"


def test_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--contract", required=True)
    p.add_argument("--repo-root", default=".")
    p.add_argument("--supersession", required=True)
    p.add_argument("--retrospective", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    root = Path(args.repo_root).resolve()
    contract_path = Path(args.contract).resolve()
    supersession_path = Path(args.supersession).resolve()
    retrospective_path = Path(args.retrospective).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite {output_path}")

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    supersession = json.loads(supersession_path.read_text(encoding="utf-8"))
    retrospective = json.loads(retrospective_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    if contract.get("schema") != "alice.eipm.n0.full-envelope-proof-obligations.v1":
        errors.append("proof-obligation schema drift")
    if contract.get("authorization", {}).get("optimizer") is not False:
        errors.append("optimizer unexpectedly authorized")
    if contract.get("authorization", {}).get("gradient") is not False:
        errors.append("gradient unexpectedly authorized")
    if contract.get("authorization", {}).get("gpu_training") is not False:
        errors.append("GPU training unexpectedly authorized")
    if contract.get("authorization", {}).get("n0_complete") is not False:
        errors.append("N0 unexpectedly marked complete")

    known_tests: dict[str, set[str]] = {}
    for relative in contract.get("static_test_files", []):
        path = root / str(relative)
        if not path.is_file():
            errors.append(f"static test file missing: {relative}")
            continue
        known_tests[str(relative)] = test_functions(path)

    all_test_names: dict[str, str] = {}
    duplicate_tests: list[str] = []
    for relative, names in known_tests.items():
        for name in names:
            if name in all_test_names:
                duplicate_tests.append(name)
            else:
                all_test_names[name] = relative
    if duplicate_tests:
        errors.append(
            "duplicate proof test function names: " + repr(sorted(duplicate_tests))
        )

    ids: set[str] = set()
    static_count = 0
    runtime_blocking = []
    training_blocking = []
    final_blocking = []
    for obligation in contract.get("obligations", []):
        oid = str(obligation.get("id", ""))
        if not oid:
            errors.append("obligation missing id")
            continue
        if oid in ids:
            errors.append(f"duplicate obligation id: {oid}")
        ids.add(oid)
        kind = str(obligation.get("kind", ""))
        requirement = str(obligation.get("requirement", "")).strip()
        if not requirement:
            errors.append(f"{oid}: empty requirement")

        if kind == "STATIC_REQUIRED":
            static_count += 1
            test = str(obligation.get("test", ""))
            if test not in all_test_names:
                errors.append(f"{oid}: required static test missing: {test}")
        elif kind == "CI_REQUIRED":
            receipt = str(obligation.get("receipt", "")).strip()
            if not receipt.startswith("PASS_N0_"):
                errors.append(f"{oid}: invalid CI receipt name: {receipt}")
        elif kind == "RUNTIME_REQUIRED":
            if obligation.get("status") != "UNRESOLVED_BLOCKING":
                errors.append(f"{oid}: runtime obligation must remain blocking before runtime receipt")
            runtime_blocking.append(oid)
        elif kind == "TRAINING_REQUIRED":
            if obligation.get("status") != "UNRESOLVED_BLOCKING":
                errors.append(f"{oid}: training obligation must remain blocking before training")
            training_blocking.append(oid)
        elif kind == "FINAL_REQUIRED":
            if obligation.get("status") != "SEALED_UNOPENED_BLOCKING":
                errors.append(f"{oid}: final obligation must remain sealed/unopened")
            final_blocking.append(oid)
        else:
            errors.append(f"{oid}: unknown obligation kind {kind!r}")

    if static_count < 30:
        errors.append("proof matrix unexpectedly narrow: fewer than 30 static obligations")
    if not runtime_blocking:
        errors.append("no unresolved runtime blockers declared")
    if not training_blocking:
        errors.append("no unresolved training blockers declared")
    if not final_blocking:
        errors.append("no sealed final blocker declared")

    for relative in contract.get("critical_source_files", []):
        path = root / str(relative)
        if not path.is_file():
            errors.append(f"critical successor source missing: {relative}")
            continue
        content = path.read_text(encoding="utf-8")
        for pattern in contract.get("forbidden_critical_source_patterns", []):
            if str(pattern) in content:
                errors.append(
                    f"critical successor source contains forbidden pattern {pattern!r}: {relative}"
                )

    for constraint in contract.get("targeted_source_constraints", []):
        relative = str(constraint.get("path", ""))
        path = root / relative
        if not path.is_file():
            errors.append(f"targeted constrained source missing: {relative}")
            continue
        content = path.read_text(encoding="utf-8")
        for pattern in constraint.get("forbidden_patterns", []):
            if str(pattern) in content:
                errors.append(
                    f"targeted source contains forbidden pattern {pattern!r}: {relative}"
                )
        for pattern in constraint.get("required_patterns", []):
            if str(pattern) not in content:
                errors.append(
                    f"targeted source missing required pattern {pattern!r}: {relative}"
                )

    retrospective_stages = set(
        str(x) for x in retrospective.get("stages", {}).keys()
    )
    mapped = set(
        str(x) for x in supersession.get("historical_to_successor", {}).keys()
    )
    aliases = {
        "semantic_base": "semantic_v02",
        "structured_state": "structured_state_v01",
        "evidence_adapter": "evidence_view_adapter_v01",
        "evidence_graph": "dual_endpoint_evidence_graph",
        "fusion": "cross_context_fusion_v02",
        "latent_pool": "adaptive_latent_pool_v02",
        "causal_arbitration": "downstream_causal_arbitration",
        "query_edge_repair_chain": "query_edge_repair_chain",
        "qsre_t1": "qsre_t1",
        "qsre_t2": "qsre_t2",
        "production_core": "production_core_v1",
        "binder_v2": "binder_v2",
        "final_self_validation_v1": "final_validation_v1",
        "tokenizer_v021": "tokenizer_v021",
        "public_corpus_v021": "public_corpus_v021",
        "governed_teacher_bank_v02": "governed_teacher_bank_v02",
        "semantic_auxiliary_heads_v02": "semantic_auxiliary_heads_v02",
        "fixed_readiness_suites_v02": "fixed_readiness_and_novel_cross_suites_v02",
    }
    for stage in sorted(retrospective_stages):
        mapped_key = aliases.get(stage)
        if mapped_key is None or mapped_key not in mapped:
            errors.append(
                f"retrospective stage lacks explicit supersession entry: {stage}"
            )

    required_residual_entries = {
        "tokenizer_v021",
        "public_corpus_v021",
        "governed_teacher_bank_v02",
        "semantic_auxiliary_heads_v02",
        "fixed_readiness_and_novel_cross_suites_v02",
        "downstream_causal_arbitration",
        "query_edge_repair_chain",
    }
    missing_residual = sorted(required_residual_entries - mapped)
    if missing_residual:
        errors.append(
            "historical residual authority lacks supersession mapping: "
            + repr(missing_residual)
        )

    result: dict[str, Any] = {
        "schema": "alice.eipm.n0.full-envelope-proof-obligations-static-audit.v1",
        "status": PASS if not errors else FAIL,
        "errors": errors,
        "obligation_count": len(ids),
        "static_obligation_count": static_count,
        "runtime_blocking": runtime_blocking,
        "training_blocking": training_blocking,
        "final_blocking": final_blocking,
        "retrospective_stage_count": len(retrospective_stages),
        "supersession_entry_count": len(mapped),
        "historical_pass_can_close_unresolved_obligation": False,
        "optimizer_authorized": False,
        "gradient_authorized": False,
        "gpu_training_authorized": False,
        "n0_complete": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
