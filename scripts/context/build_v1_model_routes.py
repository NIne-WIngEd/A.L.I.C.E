#!/usr/bin/env python3
"""Build or query exact-source V1 model routes beside the Graphify N0 graph.

This is a navigation index only. It neither extracts a V1 code graph nor
promotes branch documents, generated summaries, or test output to authority.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


OUTPUT = Path("research/graphify-context/V1_MODEL_ROUTES.json")
ROUTES = {
    "foundation": {
        "branch": "research/gemma4-v1-source-derivation-20260930",
        "files": {
            "docs/GEMMA4_V1_SOURCE_AND_DERIVATIVE_CONTRACT.md": "source custody and derivative contract",
            "src/alice_foundation/gemma4_v1.py": "clone, mutation, and derivative verification",
            "src/alice_foundation/gemma4_inventory.py": "source tensor inventory",
            "tests/foundation/test_gemma4_v1.py": "foundation contract tests",
        },
    },
    "mfm": {
        "branch": "research/mfm-v1-licensed-base-20260930",
        "files": {
            "docs/MFM_V1_LICENSED_BASE_DIRECTION_2026-09-30.md": "V1 source direction and qualification limits",
            "docs/MFM_V1_SPECIALIST_TRAINING_PATH_2026-09-30.md": "MFM specialist training path",
            "docs/MFM_V16_BUILD_CONTINUATION_2026-10-02.md": "MFM recovered corpus, compute decisions and remaining full-capability roadmap",
            "docs/MFM_COMPUTE_RECOVERY_RECEIPTS_2026-10-02.json": "actual failed export audit, full synthetic capacity, Kaggle isolation evidence and selected protected CPU route",
            "docs/MFM_ROOT_CAUSE_INVESTIGATION_2026-10-04.json": "completed teacher fit, failed generated formation, actual causal analysis and corrected measurement handoff",
            "docs/receipts/MFM_V16_ROOT_ANALYSIS_576702.json": "sealed original full-target source-ablation analysis; initial value grouping covers quoted values only",
            "docs/receipts/MFM_V16_ROOT_ANALYSIS_576703.json": "sealed corrected all-scalar causal analysis; does not establish semantic ability",
            "docs/receipts/MFM_V16_COMPLETE_DIAGNOSTIC_576702.json": "complete matched nine trained and nine seeded invalid outputs; preserved original failure",
            "docs/receipts/MFM_V16_SOURCE_VIEW_576714_SUMMARY.json": "public aggregate projection of completed corrected441case source view; protected original remains authoritative",
            "docs/receipts/MFM_V16_DECISION_PROBE_576715.json": "sealed actual16case calibrated stress and exact next-step restart result",
            "docs/receipts/MFM_V16_DECISION_FIT_576715.json": "sealed completed corrected54update teacher fit; generated capability unproved",
            "docs/receipts/MFM_V16_DECISION_RUN_576715.json": "exact corrected fit code/objective/runtime/source and stress ancestry",
            "docs/receipts/MFM_V16_DECISION_DIAGNOSTIC_576715.json": "all9trained and9seeded corrected outputs invalid; actualFP32 reference-prefix parity",
            "docs/receipts/MFM_V16_LANGUAGE_ANALYSIS_576724.json": "sealed whole-corpus exposure and field-frequency diagnosis; common-field underlearning plus separate unseen-positive identity gap",
            "docs/MFM_V16_OUTPUT_INTERFACE_SUCCESSOR_2026-10-04.md": "root input/objective/evaluation audit, actual corpus boundary and full decision construction; byte proposal on hold",
            "docs/MFM_V1_ROLE_BOUNDARY_DIAGNOSTIC_2026-09-30.md": "paired behavior and role boundary diagnostic",
            "docs/MFM_FORMATION_CONTRACT_V1_6_ADMISSION_2026-09-30.md": "full-role formation contract and admission gap",
            "docs/MFM_V16_CORPUS_AUTHORING_AND_ADMISSION_2026-09-30.md": "fictional v1.6 authoring pack, audit, and missing gold",
            "docs/MFM_V16_INTERNAL_BLIND_REVIEW_2026-09-30.md": "source-only synthetic target QA and correction limits",
            "docs/MFM_V16_AUTHENTICATED_GOLD_2026-09-30.md": "external signer roster and signed rights/review boundary",
            "docs/MFM_V16_P2_CANDIDATE_BRIDGE_2026-09-30.md": "narrow existing P2 gate bridge and unsupported-field limits",
            "src/cognitive_kernel/formation_v1_specialist.py": "separate formation model architecture",
            "src/cognitive_kernel/formation_semantics_v16.py": "versioned sensitivity, episode, relationship and mission semantics",
            "src/cognitive_kernel/formation_learning_v16.py": "v1.6 source-bound target codec and admitted learning rows",
            "src/cognitive_kernel/formation_decision_v16.py": "complete semantic decision compiler; allocates new local IDs while preserving existing and shared uncertainty identities",
            "src/cognitive_kernel/formation_decision_specialist_v16.py": "calibrated fresh lexical embeddings with selected full specialist geometry unchanged",
            "scripts/mfm/v16_prior_memory_inputs.py": "explicit fictional read snapshots for incomplete teaching inputs; no expected actions or write authority",
            "scripts/mfm/prepare_v16_contract_successor.py": "actual admitted source-view preparation; immutable unchanged-state reuse and separately bound source patches",
            "scripts/mfm/formation_source_view.py": "external source-view pin and every full input/target/runtime/stress check before gradients",
            "scripts/mfm/train_v16_decision_specialist.py": "full decision teacher construction gated on actual complete stress/restart proof",
            "scripts/mfm/run_v16_decision_diagnostic.py": "target-free matched generated controls with explicit FP32 cache parity and honest exact agreement scope",
            "scripts/mfm/analyze_v16_decision_fit.py": "read-only full corrected corpus token-frequency and source membership plus same-forward field-stratified prefix measurements",
            "scripts/mfm/magnolia_v16_language_analysis_20261004.sbatch": "protected owner-account bounded language/root-interface diagnosis without optimization",
            "scripts/mfm/continue_v16_decision_learning.py": "bounded full-corpus child learning with parent checkpoint/Adam/RNG lineage and equal-case train stopping; not open-identity repair",
            "scripts/mfm/run_v16_budget_diagnostic.py": "truthful plateau/cap completion and raw child/seed generation against byte-retained parent control",
            "scripts/mfm/analyze_v16_budget_failure.py": "all441 precise-field cohort measurements, fixed29 train generations and later retained parse/EOS cache parity; read-only",
            "scripts/mfm/magnolia_v16_budget_failure_20261006.sbatch": "protected owner-account complete-cohort diagnostic with immutable child and original raw controls",
            "tests/governance/test_mfm_v16_budget_failure_analysis.py": "cohort masking, precise source/scope fields, actual greedy/EOS replay and sealed objective remeasurement",
            "docs/MFM_BUDGET_FAILURE_DISCRIMINATOR_2026-10-06.md": "mandatory diagnosis evidence and full-roster discriminator plan before architecture or further fit",
            "docs/receipts/MFM_V16_BUDGET_FAILURE_576915_PUBLIC.json": "verified public projection of completed full441/29 diagnosis; original private seal separate, no positive formation established",
            "docs/MFM_SOURCE_SEQUENCE_CONSTRUCTION_2026-10-07.md": "researched balanced full-role and source sequence construction after complete diagnosis; full Alice/Fable destination and limits",
            "src/cognitive_kernel/formation_sequence_specialist.py": "explicit learned formation/source task vectors, exact parent migration and shared fresh head; caches require task binding",
            "scripts/mfm/formation_sequence_curriculum.py": "complete source-only UTF8 sequence targets and full432case29stratum objective; no teacher target or truncation",
            "scripts/mfm/train_v16_sequence_specialist.py": "externally pinned parent weights, reset Adam, new task seed and actual lexical stress/restart before bounded full-roster teaching",
            "scripts/mfm/run_v16_sequence_diagnostic.py": "target-free task-bound matched38formation controls and exact source stress with separate metrics/raw failures",
            "scripts/mfm/magnolia_v16_sequence_learning_20261007.sbatch": "protected owner-account CPU source/formation stress, four-epoch teaching and matched generated diagnostics",
            "docs/receipts/MFM_V16_SEQUENCE_576945_PUBLIC.json": "verified completed sequence projection; capacity and fit pass, zero positive formation; original private seals separate",
            "docs/MFM_SEQUENCE_FAILURE_DISCRIMINATOR_2026-10-07.md": "completed source/formation failure diagnosis and bounded fixed-prefix sensitivity interpretation",
            "scripts/mfm/analyze_v16_sequence_failure.py": "read-only full441x2 precise fields and frequency, full432 objective, original41late task-bound parity and source sensitivity",
            "scripts/mfm/formation_sequence_failure_metrics.py": "exact source field targets, paired source interventions and FP32 teacher-prefix token ranks; no learning or authority",
            "scripts/mfm/magnolia_v16_sequence_failure_20261007.sbatch": "protected owner-account single read-only completed576945 mechanism discriminator with external fit/raw pins",
            "tests/governance/test_mfm_sequence_failure_analysis.py": "complete per-case/task objective reconciliation, target-independent foreign mapping, dual frequency and stopping-stage guards",
            "tests/governance/test_mfm_sequence_failure_metrics.py": "real task-bound rank projection, precise source targets and source-state/mask intervention controls",
            "tests/governance/test_mfm_sequence_failure_custody.py": "real external pins, seals/raw hashes, complete rosters and correctly resealed task/control/weight tamper rejection",
            "scripts/mfm/formation_reference_catalog.py": "unintegrated explicit caller reference catalog and lossless typed codec; checksums outside learning, no learned pointer evidence",
            "tests/governance/test_mfm_task_specialist.py": "zero-task migration, task gradients, late cache/reference and actual task-bound routes; component evidence only",
            "tests/governance/test_mfm_sequence_curriculum.py": "source-only custody, exact roundtrip, complete29strata/432roster weights and differentiable two-task objective",
            "tests/governance/test_mfm_sequence_training.py": "global minibatch weights, lexical/prior-memory stress and canonical progress order regression",
            "tests/governance/test_mfm_sequence_diagnostic.py": "complete candidate/checkpoint/progress/seed ancestry, target-free task routing and separate outcomes",
            "tests/governance/test_mfm_reference_catalog.py": "typed caller inventory gaps, lossless decision semantics and caller-only catalog binding; no model qualification",
            "docs/receipts/MFM_V16_BUDGET_COMPONENT_576741.json": "completed8epoch216step bounded construction component; unqualified",
            "docs/receipts/MFM_V16_BUDGET_RUN_576741.json": "immutable parent/optimizer/source/code/recipe ancestry of bounded child",
            "docs/receipts/MFM_V16_BUDGET_PROGRESS_576741.json": "all432 train equal-case versus token-weighted progress and diverse train field failure aggregates",
            "docs/receipts/MFM_V16_BUDGET_DIAGNOSTIC_576741.json": "raw parent/child/seed control seals; all9child EOS but malformed JSON, zero grounded outputs",
            "scripts/mfm/magnolia_v16_budget_learning_20261004.sbatch": "protected6additional-epoch maximum construction with train-only progress and no automatic budget extension",
            "tests/governance/test_mfm_v16_language_analysis.py": "field provenance, positive corpus frequency and weighted same-forward metric aggregation",
            "tests/governance/test_mfm_v16_learning_continuation.py": "parent transfer and interrupted child next-state identity, equal-case objective and honest terminal claims",
            "scripts/mfm/magnolia_v16_decision_learning_20261004.sbatch": "protected owner-account source-view to stress/restart to teacher fit and matched generation workflow",
            "tests/governance/test_mfm_v16_decision_interface.py": "complete semantic roundtrip, local-ID invariance and existing/shared uncertainty counterexamples",
            "tests/governance/test_mfm_v16_contract_successor.py": "actual admitted subset boundary, reference availability, runtime ancestry and native binary input checks",
            "tests/governance/test_mfm_v16_decision_model.py": "construction reader rejection, optimizer restart and generated cache/wording scope checks; no capability claim",
            "src/cognitive_kernel/formation_adjudication_v16.py": "externally pinned reviewer and rights signature verification",
            "src/cognitive_kernel/formation_incremental_decoder.py": "experimental specialist inference cache requiring parity receipts",
            "src/cognitive_kernel/formation_context_planner.py": "registered-source retrieval and permission-bound context assembly",
            "src/alice_memory/formation_v16_p2_adapter.py": "loss-aware v1.6 proposal staging to P2 candidate gate",
            "scripts/mfm/build_v16_authoring_seed.py": "fictional text-only authoring pack generator",
            "scripts/mfm/audit_v16_authoring_corpus.py": "v1.6 synthetic authoring coverage and admission-blocker audit",
            "scripts/mfm/train_v1_formation_specialist.py": "training and processor preflight",
            "scripts/mfm/train_v16_formation_specialist.py": "versioned v1.6 specialist training and admission",
            "scripts/mfm/formation_feature_bank.py": "lossless source-bound frozen representation cache and external digest verification",
            "scripts/mfm/export_v16_frozen_features.py": "complete admitted source-only CPU BF16 frozen feature export",
            "scripts/mfm/train_v16_cached_specialist.py": "FP32 cache-only teacher training, stress probe and actual optimizer restart",
            "scripts/mfm/run_v16_cached_diagnostic.py": "matched generated FP32 teacher development diagnostics and raw failure preservation",
            "scripts/mfm/analyze_v16_cached_fit.py": "read-only target alignment, scalar field loss, source interventions and unfitted initial-scale diagnosis",
            "scripts/mfm/magnolia_v16_root_analysis_20261004.sbatch": "protected actual-runtime causal diagnosis and externally pinned partial-control recovery",
            "scripts/mfm/magnolia_v16_corrected_analysis_20261004.sbatch": "separate corrected scalar-value measurement without training or model export",
            "scripts/mfm/magnolia_v16_feature_resume_20261002.sbatch": "bounded TIMEOUT-only source export continuation with preserved interrupted writes",
            "scripts/mfm/magnolia_v16_cached_pipeline_20261002.sbatch": "dependent full-size stress, teacher fit and generated-control compute handoff",
            "scripts/mfm/magnolia_v16_dedicated_export_20261002.sbatch": "dedicated locked CPU export runtime and verified saved-case recovery",
            "scripts/mfm/probe_v16_cached_capacity.py": "public random-tensor full decoder accumulation and exact restart resource measurement",
            "scripts/mfm/magnolia_v16_cached_cpu_pipeline_20261002.sbatch": "protected CPU full corpus stress, teacher learning and generated-control fallback",
            "scripts/mfm/kaggle_nodata_gpu_isolation_probe_20261002.py": "owner-private input-free Kaggle actual accelerator and two isolation capability probes",
            "scripts/mfm/benchmark_v16_cpu_precision.py": "measured CPU FP32 source arithmetic versus committed BF16 states",
            "scripts/mfm/magnolia_v16_frozen_feature_export.sbatch": "protected same-owner Magnolia allocated CPU feature job",
            "scripts/mfm/issue_v16_longitudinal_synthetic_expansion.py": "pinned synthetic corpus reissuance from original owner-custody archive",
            "tests/governance/test_mfm_v16_frozen_features.py": "frozen cache live-loss/gradient parity, custody and multistep restart tests",
            "tests/governance/test_mfm_v16_cached_diagnostic.py": "cached generated-control lineage, EOS, FP32 and unqualified diagnostic tests",
            "scripts/mfm/run_v1_formation_specialist.py": "trained and seeded control inference",
            "scripts/mfm/run_v16_formation_specialist.py": "versioned v1.6 specialist inference",
            "scripts/mfm/qualify_v16_formation_specialist.py": "v1.6 specialist assessment and diagnostic boundary",
            "scripts/mfm/verify_gemma4_pretrained.py": "historical MFM source receipt schema",
            "scripts/mfm/run_gemma4_base_behavior.py": "historical baseline requiring MFM source receipt",
            "scripts/mfm/run_gemma4_prepared_base_behavior.py": "prepared source public diagnostic",
            "scripts/mfm/qualify_v1_role_boundary.py": "role boundary comparison",
            "tests/governance/test_mfm_v16_downstream_intervention.py": "synthetic proposal-to-gate-to-retrieval integration regression",
            "tests/governance/test_formation_incremental_decoder.py": "experimental cache parity regression",
            "tests/governance/test_mfm_v16_specialist_assessment.py": "v1.6 specialist assessment contract tests",
            "tests/governance/test_mfm_v16_signed_fit_gate.py": "externally pinned signed corpus optimizer gate tests",
            "tests/phase2/test_formation_v16_p2_adapter.py": "synthetic narrow P2 bridge and rejection regressions",
        },
    },
    "fbm": {
        "branch": "research/fbm-v1-licensed-base-20260930",
        "files": {
            "training/fbm/base_assembly/gemma4_12b_mfm_v1.seed.json": "FBM ordered replay seed",
            "training/fbm/base_assembly/mfm_v16_teacher_and_source_slices_20261001.seed.json": "FBM recovered teacher CPU evidence and frozen-feature build continuation",
            "docs/fable-builder/traces/FBM_TRACE_20261002_MFM_RECOVERY_AND_FROZEN_FEATURE_ROUTE.jsonl": "FBM artifact recovery, measured tests and compute-route procedure traces",
            "training/fbm/base_assembly/mfm_v1_training_role_coverage_20260930.json": "hash-bound training-role coverage audit",
            "docs/fable-builder/FBM_V1_GEMMA4_BASE_DISSECTION_AND_ROLE_RECIPE_2026-09-30.md": "FBM source and role recipe",
            "docs/fable-builder/traces/FBM_TRACE_20260930_GEMMA4_MFM_REPLAY_SEED.jsonl": "FBM process trace",
            "docs/fable-builder/traces/FBM_TRACE_20260930_GEMMA4_MFM_PUBLICATION_REBIND.jsonl": "published cross-branch lineage",
            "docs/fable-builder/traces/FBM_TRACE_20260930_GEMMA4_CLONE_FIRST_FORMATION_SEED.jsonl": "clone-first decision trace",
            "docs/fable-builder/traces/FBM_TRACE_20260930_MFM_FULL_ROLE_READINESS_AUDIT.jsonl": "full-fit stop and corpus audit trace",
            "docs/fable-builder/traces/FBM_TRACE_20260930_MFM_V15_PROBE_ONLY_CONTRACT.jsonl": "v1.5 trainer probe-only correction",
            "docs/fable-builder/traces/FBM_TRACE_20260930_MFM_V16_PUBLIC_CONTRACT_PIN.jsonl": "published v1.6 contract pin",
            "docs/fable-builder/traces/FBM_TRACE_20260930_MFM_PUBLIC_DOCS_CORRECTION_PIN.jsonl": "matched control and exact public MFM pin",
            "docs/fable-builder/traces/FBM_TRACE_20260930_MFM_V16_AUTONOMOUS_REPLAY_LADDER.jsonl": "v1.6 admission and autonomous replay seed",
            "docs/fable-builder/traces/FBM_TRACE_20260930_MFM_AMENDED_V16_AUTHORING_DIAGNOSTIC.jsonl": "amended synthetic target hash and diagnostic-only replay boundary",
            "docs/fable-builder/traces/FBM_TRACE_20260930_MFM_SIGNED_FULL_FIT_AND_P2_BRIDGE.jsonl": "signed fit gate and narrow P2 bridge replay pin",
            "docs/fable-builder/traces/FBM_TRACE_20261004_MFM_FIT_FAILURE_ROOT_ANALYSIS.jsonl": "completed fit versus failed formation, causal diagnosis and metric review methods; procedure only",
        },
    },
}


def git(*args: str) -> str:
    p = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    return p.stdout.strip()


def live_heads() -> dict[str, str]:
    branches = [route["branch"] for route in ROUTES.values()]
    raw = git("ls-remote", "--heads", "origin", *branches)
    heads = {ref.removeprefix("refs/heads/"): oid for oid, ref in
             (line.split("\t") for line in raw.splitlines())}
    missing = sorted(set(branches) - set(heads))
    if missing:
        raise RuntimeError(f"V1 branch missing on origin: {missing}")
    return heads


def build() -> dict:
    live = live_heads()
    areas = {}
    for area, route in ROUTES.items():
        branch = route["branch"]
        local = git("rev-parse", f"refs/remotes/origin/{branch}")
        if local != live[branch]:
            raise RuntimeError(f"STALE_LOCAL_REF {branch}: local={local} live={live[branch]}; fetch first")
        files = []
        for path, purpose in route["files"].items():
            row = git("ls-tree", local, "--", path)
            if not row:
                raise RuntimeError(f"MISSING_SOURCE_FILE {branch}@{local}:{path}")
            metadata, name = row.split("\t", 1)
            mode, kind, blob = metadata.split()
            if kind != "blob" or name != path:
                raise RuntimeError(f"INVALID_SOURCE_FILE {branch}@{local}:{path}")
            files.append({"path": path, "purpose": purpose, "blob_sha": blob})
        areas[area] = {"branch": branch, "commit": local, "files": files}
    return {
        "schema": "alice.graphify.v1-model-routes.v1",
        "authority": "navigation-only",
        "graph_status": "existing Graphify code graph is N0-only; no V1 graph built",
        "qualification_status": "no model qualification can be inferred from this index",
        "areas": areas,
    }


def encoded(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    group.add_argument("--query", metavar="TEXT")
    args = ap.parse_args()
    expected = build()
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(encoded(expected), encoding="utf-8")
        print(f"WROTE {OUTPUT}")
        return
    if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != encoded(expected):
        raise SystemExit("STALE_V1_ROUTES: regenerate from fetched exact live branch heads")
    if args.check:
        print("V1_ROUTES_FRESH: all three live source branches and file blobs match")
        return
    words = args.query.lower().split()
    matches = [
        {"area": area, "branch": route["branch"], "commit": route["commit"], **file}
        for area, route in expected["areas"].items()
        for file in route["files"]
        if all(word in f'{area} {file["purpose"]} {file["path"]}'.lower() for word in words)
    ]
    print(json.dumps(matches[:12], indent=2))


if __name__ == "__main__":
    main()
