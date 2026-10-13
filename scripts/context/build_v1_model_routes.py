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
            "docs/receipts/MFM_CONDITIONAL_CENSUS_576995_PUBLIC.json": "externally sealed actual complete40,868-pair Gemma census,4,862 exact source inputs and measured encoding/resource geometry; no model forward or fit",
            "docs/receipts/MFM_CONDITIONAL_FEATURE_PILOT_576999_FAILURE_PUBLIC.json": "preserved actual runtime failure before Gemma; fixture rejected at earlier receipt binding and intended tensor guard was not reached",
            "docs/receipts/MFM_CONDITIONAL_FEATURE_FIXTURE_DIAGNOSIS_577003_PUBLIC.json": "externally pinned unchanged-runtime mapped/released/owned tensor controls establish NFS fixture lifetime cause; no Gemma forward or learning claim",
            "scripts/mfm/magnolia_public_conditional_features_v2_20261007.sbatch": "historical first source export block with actual91-test gate before frozen Gemma source construction; full-roster target retained",
            "scripts/mfm/magnolia_public_conditional_source_export_20261007.sbatch": "full4,862-source same-binding continuation, successful parent/external progress custody, verified bank resume and scheduler blocks; no reduced model scope",
            "scripts/mfm/magnolia_public_conditional_source_export_parent577007_20261007.sbatch": "immediate577007 child continuing the same complete source bank with exact parent terminal/progress custody",
            "docs/receipts/MFM_SOURCE_EXPORT_BLOCK_577006_PUBLIC.json": "actual91-check closure,144-source stopped block, process-tree resource evidence and verified child custody; not complete bank or learned capability",
            "docs/MFM_PUBLIC_FOUNDATION_PRODUCER_CONTRACT_2026-10-07.md": "fresh full-geometry conditional foundation producer, Gemma source-only boundary and subsequent full formation dependencies",
            "docs/superpowers/plans/2026-10-07-mfm-public-foundation-training.md": "full-roster reader, parameter-free task binding, global group/reference objective and exact optimizer-boundary restart construction",
            "scripts/mfm/public_conditional_training.py": "complete externally pinned public bank and exact source/target ancestry joins, global semantic-group weights and prescribed deterministic epoch order",
            "tests/governance/test_mfm_public_conditional_training.py": "real source/tensor custody, changing verified-bank pin, exact sampler and unequal-reference/length objective gradients; fixtures only",
            "src/cognitive_kernel/formation_conditional_task.py": "explicit parameter-free public task view preserving original fresh producer parameters, positions and checkpoint tensor names",
            "tests/governance/test_mfm_conditional_task.py": "exact producer logits/loss/gradients, actual reference/cache generation and direct unchanged checkpoint reload",
            "scripts/mfm/train_public_conditional_foundation.py": "complete E20 conditional foundation and zero-update final/seed observation; actual exported runtime closure is separate from checkout imports",
            "src/cognitive_kernel/formation_reference_specialist.py": "owned typed reference head, strict full lexical parent migration and actual lexical generation through predicted references to grounding; untrained mechanics",
            "src/cognitive_kernel/formation_reference_decision.py": "exact caller/global-role binding and real predicted-identity resolution through original compiler plus explicit grounding",
            "scripts/mfm/formation_reference_target_alignment.py": "distinct complete original-pair reference view, explicit ordinal teaching and exact actual generated whole-wire geometry",
            "tests/governance/test_mfm_reference_specialist.py": "strict migration, owned gradients, typed masks and complete inference/raw failure/RNG/no-retry mechanics; no learned behavior",
            "tests/governance/test_mfm_reference_decision.py": "complete bound target/reference/consumer controls and registered-but-unsupported grounding refusal",
            "scripts/mfm/public_checkpoint_history.py": "same-run source, cursor, optimizer, RNG and historical checkpoint lineage; actual provider read-only history mounting remains separate",
            "tests/governance/test_mfm_public_checkpoint_history.py": "real small-model history restore and tamper controls; no full learned capability claim",
            "scripts/mfm/kaggle_public_foundation_fit_20261008.py": "tiny exact-source full-fit capsule, fixed complete-roster learning program and storage forecast; no provider submission",
            "scripts/mfm/kaggle_public_foundation_fit_20261008_runtime.py": "offline capsule/input guards and unchanged full-fit invocation; no custom duration timeout",
            "tests/governance/test_mfm_kaggle_public_foundation_fit_20261008.py": "actual tiny archive/import/argument and package boundary controls; hosted optimizer results remain separate",
            "docs/MFM_FULL_PUBLIC_FIT_PROGRAM_2026-10-09.md": "fixed E20/B16/C2288 fresh-producer program, complete epoch continuation, source-only Gemma and full-role destination",
            "docs/receipts/MFM_FULL_PUBLIC_FOUNDATION_FIT_LAUNCHED_20261009_TRACE91_PUBLIC.json": "sealed actual full-fit launch and provider running observation, verified bank duplicate cleanup and pending learned result",
            "docs/receipts/MFM_PUBLIC_DEVELOPMENT_OBSERVATION_STORAGE_20261009_TRACE92_PUBLIC.json": "frozen all506/4271 prospective observations,183 existing literal-value contrasts and semantic interpretation limits; backup and semantic scoring pending",
            "docs/receipts/MFM_STREAMED_PRIMARY_STORAGE_ROUTE_20261009_TRACE93_PUBLIC.json": "actual streamed fixture and recovery catalog restoration, fresh20-file backup upload pending complete custody, recovered status access with same full-fit run still running",
            "docs/receipts/MFM_CONTINUATION_AND_STORAGE_RECOVERY_20261009_TRACE94_PUBLIC.json": "reviewed full-history continuation and exact cache mechanics, closed child cleanup failure, repeated provider502 reconciled with21 accepted metadata objects; actual checkpoint custody, native recovery activation and learning remain pending",
            "docs/receipts/MFM_FULL_FIT_SESSION0_AND_STORAGE_RECOVERY_20261009_TRACE95_PUBLIC.json": "actual first3-epoch boundary and safe bounded metadata; completed public storage control and finite legacy progress; deployed checkpoint relay launch correction, whole custody, E20 and readonly resume pending",
            "docs/receipts/MFM_TLS_CUSTODY_CONTINUATION_20261009_TRACE96_PUBLIC.json": "reviewed native trust/source roundtrip and actual41/83 custody launch; legacy48 uploads complete, whole restore/catalog, resumed state and fullMFM pending",
            "docs/receipts/MFM_CUSTODY_RECOVERY_20261009_TRACE97_PUBLIC.json": "preserved source EOF and SSH failures; reviewed atomic recovery, detached verification and actual replacement launch; whole custody and full role remain pending",
            "docs/receipts/MFM_ALIGNMENT_AND_DURABLE_LEGACY_CUSTODY_20261009_TRACE98_PUBLIC.json": "caller-bound text geometry, independent regressions and actual whole20/catalog custody; preserved restore-consumer publication failures and full learned role pending",
            "docs/receipts/MFM_GEMMA_GEOMETRY_AND_LEGACY_CLEANUP_20261009_TRACE99_PUBLIC.json": "actual repeatable Gemma source geometry, preserved processor failures, deployed Drive restore and exact20 native cache cleanup; full formation learning pending",
            "docs/receipts/MFM_FULL_TARGET_AND_S1_CONTINUATION_20261010_TRACE100_PUBLIC.json": "complete original target geometry, original41 catalog custody, immutable S1 running; full caller features and full MFM pending",
            "docs/receipts/MFM_ROOTS_ALIGNMENT_20261010_PUBLIC.json": "original purpose, source-pinned roles and honest full MFM/FBM gaps; unresolved physical policy conflict",
            "docs/MFM_ROOTS_ALIGNMENT_AUDIT_2026-10-10.md": "comic and entity roots audit, full MFM shipment and remaining governed outcome and builder transfer sequence",
            "docs/receipts/MFM_CALLER_MEANING_AND_JOINT_INPUT_20261010_TRACE102_PUBLIC.json": "admitted target-free caller meaning and causal whole text; independent80 controls and actual geometry; learned full-role remains unqualified",
            "docs/MFM_CALLER_MEANING_AND_JOINT_INPUT_2026-10-10.md": "complete original prior Claim and source byte joins, input boundaries and causal world construction next",
            "docs/receipts/MFM_CURRENT_STACK_ALIGNMENT_20261010_TRACE103_PUBLIC.json": "latest owner-ratified stack alignment; older Fable prose is lag and FloRA implements a compatible slice",
            "docs/MFM_CURRENT_STACK_ALIGNMENT_2026-10-10.md": "current Kurrent/JetStream transport/XTDB/LadybugDB/NebulaGraph selection; bounded FloRA engines and no model change",
            "docs/receipts/MFM_CAUSAL_WORLD_INPUT_20261010_TRACE105_PUBLIC.json": "root-admitted original causal input or exact licensing clarification; full role and deployment qualification distinct",
            "docs/receipts/MFM_KURRENT_LICENSE_CLARIFICATION_20261010_TRACE106_PUBLIC.json": "root-admitted original causal input or exact licensing clarification; full role and deployment qualification distinct",
            "docs/MFM_CAUSAL_WORLD_INPUT_2026-10-10.md": "original Experience/Claim causal cuts and current-boundary delivery; separate full-family extension next",
            "docs/MFM_EPISODE_ORIGINAL_PRIOR_INPUT_2026-10-10.md": "original Episode metadata/full/summary, historical/current role separation and honest actual-failure saved closure",
            "docs/research/MFM_TASK_CONTENT_INPUT_NEXT_CONSTRUCTION_2026-10-10.md": "reviewed next original Mission/Result/Workspace and distinct uncertainty input direction; exact schemas and implementation outstanding",
            "docs/research/MFM_TASK_CONTENT_INPUT_SCHEMA_REVISION2_2026-10-10.md": "root-admitted exact original task and uncertainty schemas; implementation and learned use separate",
            "docs/research/MFM_TASK_CONTENT_INPUT_PLAN_REVISION2_2026-10-10.md": "connected real-producer input plan with current per-part permissions and separately held actual processor control",
            "docs/research/MFM_TASK_CONTENT_DEPENDENCY_READ_MATRIX_2026-10-10.md": "complete typed causal dependency and verification-versus-delivery read closure matrix",
            "docs/receipts/MFM_TASK_CONTENT_SCHEMA_ADMISSION_20261010_TRACE108_PUBLIC.json": "TRACE108 failed schema admission, ordered coherent closure and static scope-only root admission method",
            "docs/MFM_S1_LINEAGE_AND_CONNECTED_INPUT_PROGRESS_2026-10-10.md": "actual S1 epoch6 metadata lineage, original-observation identity correction and full-role continuation gaps",
            "docs/receipts/MFM_S1_LINEAGE_AND_INPUT_METHOD_20261010_TRACE109_PUBLIC.json": "TRACE109 actual original metadata and real custody controls; no tensor custody or learning qualification",
            "docs/MFM_ORIGINAL_CUSTODY_AND_EVIDENCE_METHOD_2026-10-10.md": "faithful custody and evidence lifecycle source; actual complete input budget refusal preserved",
            "docs/receipts/MFM_ORIGINAL_CUSTODY_AND_EVIDENCE_20261010_TRACE110_PUBLIC.json": "TRACE110 faithful custody and returned evidence; preserved v3/v4 complete-budget failures, completed exact143 cleanup; full MFM and S1 whole custody pending",
            "docs/MFM_S2_RETAINED_PARENT_CONTINUATION_METHOD_2026-10-10.md": "unchanged E20 retained-parent continuation alongside explicit pending backup; restored S2 state remains unverified",
            "docs/receipts/MFM_S2_RETAINED_PARENT_CONTINUATION_20261010_TRACE111_PUBLIC.json": "TRACE111 accepted S2 launch and bracketed RUNNING, preserved non-Drive lineage and credential/runtime claim boundaries",
            "docs/MFM_S3_COMPLETE_PARENT_CONTINUATION_METHOD_2026-10-11.md": "unchanged S3 continuation from genuine S2 originals with pending custody and first CPU allocation failure locus",
            "docs/receipts/MFM_S3_COMPLETE_PARENT_CONTINUATION_20261011_TRACE112_PUBLIC.json": "TRACE112 original S2 accepted resume and exact S3 source or launch status; no full-role learning or custody promotion",
            "docs/MFM_COMPLETE_JOINT_RUNTIME_IDENTITY_METHOD_2026-10-11.md": "reviewed two-command numeric owner correction and actual front/inner boundary; whole source study unproved",
            "docs/receipts/MFM_COMPLETE_JOINT_RUNTIME_IDENTITY_20261011_TRACE113_PUBLIC.json": "TRACE113 safe node-scoped boundary evidence and preserved original failures; no model capacity or learning qualification",
            "docs/MFM_IMPORT_PACKAGE_AND_PAIRED_CONSTRUCTION_METHOD_2026-10-11.md": "complete package before imports, supplied paired promotion and durable original replay; finite source proofs",
            "docs/receipts/MFM_IMPORT_PACKAGE_AND_PAIRED_CONSTRUCTION_20261011_TRACE114_PUBLIC.json": "TRACE114 exact safe construction snapshot; no full source capacity, corpus or learning qualification",
            "docs/MFM_PAIRED_GEOMETRY_AND_MATERIAL_READ_METHOD_2026-10-11.md": "actual saved pair, exact21 published source and admitted active material reads; explicit unfinished full-role limits",
            "docs/receipts/MFM_PAIRED_GEOMETRY_AND_MATERIAL_READ_20261011_TRACE115_PUBLIC.json": "TRACE115 genuine source and paired proofs; original styleFAILED retained, no corpus, capacity or learning claim",
            "docs/MFM_MATERIAL_READ_AND_SAVED_EXPORT_METHOD_2026-10-11.md": "completed private material protocol; failed whole plus separate saved continuation, awaiting annotations and supplied-rights limits",
            "docs/receipts/MFM_MATERIAL_READ_AND_SAVED_EXPORT_20261011_TRACE116_PUBLIC.json": "TRACE116 exact current reads, saved FileRef export and composed reviews; procedure only, no corpus or learning authority",
            "docs/MFM_NEW_WORLD_SOURCE_OBSERVATION_METHOD_2026-10-11.md": "new-world original source method; two complete cuts plus whole default capacity refusal, exact restoration and provenance",
            "docs/receipts/MFM_NEW_WORLD_SOURCE_OBSERVATION_20261011_TRACE117_PUBLIC.json": "TRACE117 preserved setup/origin failures and actual38564-over32768 refusal; procedure only, no six-cut or learning success",
            "docs/MFM_S3_COMPLETION_AND_S4_PREPARATION_METHOD_2026-10-11.md": "genuine S3 original checkpoint acceptance and unchanged S4 preparation; exact restoration, quota and CPU-study boundaries",
            "docs/receipts/MFM_S3_COMPLETION_AND_S4_PREPARATION_20261011_TRACE118_PUBLIC.json": "TRACE118 S3 epoch12 step27456 and preserved history; preparation only, no S4 launch or full learning qualification",
            "docs/MFM_S4_ACCEPTANCE_AND_FINAL_S5_CONTINUATION_METHOD_2026-10-11.md": "accepted original S4 and exact final-five S5 continuation; compact complete history, distinct launcher profile and honest lifecycle scope",
            "docs/receipts/MFM_S4_ACCEPTANCE_AND_FINAL_S5_CONTINUATION_20261011_TRACE119_PUBLIC.json": "TRACE119 S4 epoch15 step34320 and final S5 profile evidence; no final E20 or full-MFM qualification by inference",
            "docs/MFM_S5_FOUNDATION_COMPLETION_2026-10-12.md": "genuine E20 conditional foundation fit; owner parked full-system stress draft and returned to actual model construction",
            "docs/receipts/MFM_S5_FOUNDATION_COMPLETION_20261012_TRACE120_PUBLIC.json": "TRACE120 complete21-entry six-session lineage and full split likelihood reports; full formation, new reload and Drive custody remain unqualified",
            "docs/MFM_COMPLETED_FOUNDATION_OBSERVATION_2026-10-12.md": "direct zero-update final/seed source-only observation and independently continuing missing native capability construction",
            "docs/receipts/MFM_COMPLETED_FOUNDATION_OBSERVATION_20261012_TRACE121_PUBLIC.json": "TRACE121 reviewed source/package and initial provider RUNNING only; no successful generated behavior by implication",
            "tests/governance/test_mfm_public_completed_observation.py": "canonical full-prefix FP32 observation, whole frozen cohort, exact checkpoint bytes and retained cache-failure boundaries",
            "docs/MFM_COMPLETED_LANGUAGE_FOUNDATION_OBSERVATION_METHOD_2026-10-13.md": "researched explicit reference method, actual complete v3 reload/generation and unfinished semantic/full-MFM qualification",
            "docs/receipts/MFM_CANONICAL_REFERENCE_OBSERVATION_20261013_TRACE124_PUBLIC.json": "TRACE124 unchanged fitting functions,32 source controls, strict327-file closure and actual package admission",
            "docs/receipts/MFM_S1_WHOLE_DRIVE_CUSTODY_20261013_TRACE125_PUBLIC.json": "TRACE125 all55 original native actions,27 whole restores and final catalog; storage custody only",
            "docs/receipts/MFM_COMPLETED_REFERENCE_OBSERVATION_20261013_TRACE126_PUBLIC.json": "TRACE126 genuine private v3 COMPLETE and1012 exact original rows; source contrasts descriptive and semantic review separate",
            "docs/receipts/PROJECT_EXACT_TWO_ARCHIVE_RELEASE_20261013_TRACE127_PUBLIC.json": "TRACE127 exact obsolete shared archive release after usable whole Drive backup and current identity/use checks; scope only",
            "docs/MFM_FIXED_FOUNDATION_REVIEW_2026-10-13.md": "complete150-calibration/862-audit original review under unchanged policy, post-finalization mapping and durable Drive originals; descriptive only",
            "docs/receipts/MFM_FIXED_FOUNDATION_REVIEW_20261013_TRACE128_PUBLIC.json": "TRACE128 exact fixed review, corrected mandatory whole-custody check and three complete Drive readbacks; no independent gold or full-MFM claim",
            "docs/MFM_NATIVE_JOINT_LEARNING_2026-10-13.md": "actual one-decoder lexical/reference/whole-copy objective, strict complete-state reload and full material/migration dependencies",
            "docs/receipts/MFM_NATIVE_JOINT_LEARNING_20261013_TRACE129_PUBLIC.json": "TRACE129 final12 source controls and independent original type/provenance/input refusals; untrained mechanics",
            "src/cognitive_kernel/formation_joint_learning.py": "owned joint objective and complete finiteFP32 export/reload with canonical actual target/processor/provenance and physical consumer source closure",
            "tests/governance/test_mfm_joint_learning.py": "actual one-decoder gradients, complete whole-state generated consumer and refusal-before-decode/load controls; tiny untrained fixtures",
            "docs/MFM_FULL_SIZE_NATIVE_COMPONENT_BUILD_2026-10-13.md": "genuine S5 lexical-to-reference-to-complete-joint assembly and full-state reload; source/packet admission separate from runtime",
            "docs/receipts/MFM_FULL_SIZE_NATIVE_COMPONENT_BUILD_20261013_TRACE130_PUBLIC.json": "TRACE130 original S5 binding,12 tiny builder and9 package controls plus metadata-only shape forecasts; not full-size result",
            "scripts/mfm/build_native_joint_component.py": "direct parent assembly plus explicit complete-state source-lineage migration with original isolated strict cold load and current strict whole reload, complete source/parent/RNG brackets; zero updates",
            "docs/receipts/MFM_COMPLETE_STATE_SOURCE_MIGRATION_20261013_TRACE139_PUBLIC.json": "TRACE139 actual103-file source import and genuine48-tensor original/current migration, reviewed full138-state CPU packet and original pre-dataset upload failure; full-size migration pending",
            "docs/MFM_BROADER_FOUNDATION_MATERIAL_2026-10-13.md": "three source-first chronological cases extending the original six, exact caller/registry grounding and source-use declarations; full native geometry and learning remain pending",
            "docs/receipts/MFM_BROADER_GROUNDED_FOUNDATION_20261013_TRACE140_PUBLIC.json": "TRACE140 nine-case material covers all ten dimensions and six critical kinds; actual source/codec/reference construction with separate diagnostic development, no learning or independent-gold claim",
            "docs/receipts/MFM_ACTUAL_NATIVE_JOINT_COMPLETION_20261013_TRACE131_PUBLIC.json": "TRACE131 actual full-size private offline CPU completion; complete115-to122-to138 state preserved and strict full reload, zero joint updates and large Drive custody pending",
            "docs/MFM_NATIVE_JOINT_SOURCE_FEATURES_2026-10-13.md": "complete frozen native text source construction and coherent preflight/output/native-field diagnoses; actual12B features and full admitted material remain pending",
            "docs/receipts/MFM_NATIVE_JOINT_SOURCE_FEATURES_20261013_TRACE132_PUBLIC.json": "TRACE132 final native-field root3 controls and independent unchanged actual saved-field readback review; fixture mechanics only",
            "scripts/mfm/export_joint_source_features.py": "complete native text fields and detached BF16 frozen backbone source states, physical clone/bank and whole original profile bindings; no targets or answer head",
            "tests/governance/test_mfm_joint_source_features.py": "whole native field forward/save/readback and original budget/path/preload/tensor controls with explicit fixture/host limits",
            "docs/MFM_NATIVE_JOINT_TRAINING_2026-10-13.md": "direct owned J4/J5/J6 joint learner, complete authored physical joins and bundle-derived full proof/delivery roster, connected world mass and exact owned restart; production fit remains pending",
            "docs/receipts/MFM_NATIVE_JOINT_TRAINING_20261013_TRACE133_PUBLIC.json": "TRACE133 root8 final controls and independent original-path exact owned restart, invalid RNG/named-order refusal and separately scoped bank-lifetime witness; P1 mechanics and failclosed J5 candidate",
            "scripts/mfm/train_native_joint_formation.py": "direct complete lexical/reference/copy fit over independently admitted material, new named AdamW, one-bank consumption, complete boundary cursor/state restart and raw generated consumer; no default production schedule",
            "tests/governance/test_mfm_native_joint_training.py": "complete author-conversion/world-cut-variant/native-field/owned-update/restart controls and supplied P2 refusal; fixture and real-material scope kept separate",
            "docs/MFM_FULL_SOURCE_TEXT_CONSTRUCTION_2026-10-13.md": "actual complete six-cut default census and separate full-source J6 geometry construction; unchanged J5 refusals, no host capacity or rights inferred",
            "docs/receipts/MFM_FULL_SOURCE_CENSUS_20261013_TRACE134_PUBLIC.json": "TRACE134 actual22475/28814/38564/42784/52466/52488 complete native text batches and independent full physical typed readback; source observation only",
            "docs/receipts/MFM_J6_FULL_SOURCE_CONSTRUCTION_20261013_TRACE135_PUBLIC.json": "TRACE135 actual full6 default/offset geometry, final typed Mapping/saved JSON correction and complete code-binding migration preserving original native/semantic data; no processor rerun/capacity/rights/fit",
            "docs/receipts/MFM_MODEL_AUTHOR_CONSTRUCTION_20261013_TRACE136_PUBLIC.json": "TRACE136 proof-safe current-cut author views and explicit stochastic model-author provenance; raw output retained, unqualified route controls only",
            "docs/MFM_COMPLETE_TEXT_JOINT_MATERIAL_2026-10-13.md": "complete six-cut authored targets and canonical/mixed-wire native geometry, material15 and direct learner joins; seven positive dimensions, broader foundation and admitted production features/capacity remain pending",
            "docs/receipts/MFM_J6_DIRECT_LEARNER_AND_MIXED_WIRE_20261013_TRACE138_PUBLIC.json": "TRACE138 actual six mixed-wire observations, final learner six root controls and unchanged original omitted-roster refusal; no production joint fit or full-role qualification",
            "docs/receipts/MFM_COMPLETE_J6_MATERIAL_AND_CONSUMERS_20261013_TRACE137_PUBLIC.json": "TRACE137 nine source pins, six actual source/target/material joins, retained failures and finite independent review; no production fit or learned capability claim",
            "scripts/mfm/formation_paired_teaching_v6.py": "source-first full J6 external FILE join and separately authored original decision, annotations and explicit reference/copy packet; genuine material lineage retained",
            "scripts/mfm/formation_joint_target_alignment_v6.py": "distinct complete original target native alignment and exact whole pair validation; canonical original target geometry separate from mixed reference/copy wire",
            "tests/governance/test_mfm_paired_teaching_v6.py": "original source-before-author, explicit choice and strict complete native paired geometry controls; fixture bindings are not actual Gemma observations",
            "src/cognitive_kernel/formation_dataset_admission.py": "original corpus rights/source/split admission and versioned unqualified model-author origin; assertions are not cryptographic identity or gold",
            "scripts/mfm/assemble_v16_owner_teacher_corpus.py": "existing complete target assembly with raw model input/output/converter lineage and explicit intake-v3 origin",
            "docs/MFM_V16_OWNER_TEACHER_TRAINING_LANE_2026-10-01.md": "owner-authorized unqualified teaching with versioned stochastic model-author provenance and preserved old routes",
            "scripts/mfm/formation_joint_input_v6.py": "explicit pinned original J5 render/joins, complete J6 native text default/offset field identity and strict saved full-source freeze under52488 proposed construction bound; existing consumers require explicit interface expansion",
            "tests/governance/test_mfm_joint_input_v6.py": "three strict live/saved native field and actual BatchFeature mapping controls; preserved numeric-alias and wrong-container failures",
            "tests/governance/test_mfm_native_joint_component_build.py": "genuine tiny migration chain and coherent metadata/readonly/import/state refusal controls",
            "docs/MFM_NATIVE_REFERENCE_PRODUCER_2026-10-13.md": "concrete native generated-reference producer, independent grounding and unfinished original-copy/full-formation roadmap",
            "docs/receipts/MFM_NATIVE_REFERENCE_PRODUCER_20261013_TRACE122_PUBLIC.json": "TRACE122 root37 and independent4 mechanics, retained original failures and failed observation runtime closure; no learned reference/full MFM claim",
            "src/cognitive_kernel/formation_original_copy_specialist.py": "native source/start/conditional UTF8 end head, complete FP32 parent migration and exact whole-tuple supervision",
            "src/cognitive_kernel/formation_original_copy_decision.py": "same own generated mixed-wire reference/copy queries, exact original byte resolution and real compiler/grounding",
            "scripts/mfm/formation_original_copy_target_alignment.py": "complete reference/copy marker roster, authored ordinal and exact declaration supervision admission",
            "tests/governance/test_mfm_original_copy_specialist.py": "17 owned assembly/source-head/whole-tuple/migration controls; untrained mechanics only",
            "tests/governance/test_mfm_original_copy_decision.py": "actual mixed-wire consumer and original grounding with retained teacher/provenance refusals",
            "docs/MFM_NATIVE_ORIGINAL_COPY_PRODUCER_2026-10-13.md": "native copy construction, independently caught omitted reference labels and pending full-role material/learning",
            "docs/receipts/MFM_NATIVE_ORIGINAL_COPY_PRODUCER_20261013_TRACE123_PUBLIC.json": "TRACE123 composed17 and48 final-pin controls and independent correction review; no learned copy or full MFM claim",
            "docs/superpowers/specs/2026-10-12-mfm-native-reference-producer-design.md": "adopted actual own-generation/reference/compiler/grounding contract with teacher-free inference and raw first failures",
            "docs/superpowers/plans/2026-10-12-mfm-native-reference-producer.md": "native model plus complete target/consumer source construction and actual assembly plan",
            "docs/superpowers/specs/2026-10-12-mfm-native-original-copy-design.md": "adopted next native source/start/conditional UTF8 end and byte/context representation; exact tuple supervision and citation/anchor coverage",
            "docs/superpowers/plans/2026-10-12-mfm-native-original-copy-producer.md": "next owned original-copy construction; complete source/target/model path rather than a stress wrapper",
            "docs/receipts/MFM_EPISODE_ORIGINAL_PRIOR_INPUT_20261010_TRACE107_PUBLIC.json": "admitted input construction; original exit1/72 and separate36 saved checks; no fresh tokenization or full-role qualification",
            "scripts/mfm/formation_caller_meaning_v3.py": "target-free complete permitted original task/uncertainty meanings and typed historical joins",
            "scripts/mfm/formation_input_content_prototype.py": "strict original RawBuffer capture/reference custody and finite immutable input commits",
            "scripts/mfm/formation_joint_input_v3.py": "original complete task-input contextual geometry; preserved custody-heavy profile",
            "scripts/mfm/formation_joint_input_v4.py": "complete semantic model input with exact custody companion, field/part geometry and cache bindings",
            "scripts/mfm/formation_joint_target_alignment.py": "whole shifted-target geometry, typed candidate ordinal and explicit literal alternatives",
            "scripts/mfm/formation_paired_teaching.py": "versioned saved-J4 supplied-record freeze before independently authored complete v1.6 targets",
            "scripts/mfm/formation_task_content_input.py": "typed Mission/Result/Workspace and attempt/outcome original validation; supplied input scope",
            "scripts/mfm/formation_uncertainty_input.py": "immutable uncertainty identity and genuinely new original resolution history; input-only",
            "scripts/mfm/formation_world_caller_v3.py": "current per-part delivery and independent verification boundaries with authorized audience projection",
            "scripts/mfm/formation_world_replay_v3.py": "exact causal task-content/material bindings, original dependencies and proof-only verification responsibility",
            "tests/governance/mfm_task_content_fixtures.py": "real finite RawBuffer/Experience/projection producers with fictional current-policy adapter",
            "tests/governance/test_mfm_caller_meaning_v3.py": "Connected fictional controls for caller meaning v3; component evidence only",
            "tests/governance/test_mfm_input_content_prototype.py": "Connected fictional controls for input content prototype; component evidence only",
            "tests/governance/test_mfm_joint_input_v3.py": "Connected fictional controls for joint input v3; component evidence only",
            "tests/governance/test_mfm_joint_input_v4.py": "Connected fictional controls for joint input v4; component evidence only",
            "tests/governance/test_mfm_joint_target_alignment.py": "Five portable literal/role controls; full original saved-evidence suite retained privately and explicitly",
            "tests/governance/test_mfm_paired_teaching.py": "Connected fictional controls for paired teaching; component evidence only",
            "tests/governance/test_mfm_task_content_input.py": "Connected fictional controls for task content input; component evidence only",
            "tests/governance/test_mfm_uncertainty_input.py": "Connected fictional controls for uncertainty input; component evidence only",
            "tests/governance/test_mfm_world_caller_v3.py": "Connected fictional controls for world caller v3; component evidence only",
            "tests/governance/test_mfm_world_replay_v3.py": "Connected fictional controls for world replay v3; component evidence only",
            "scripts/mfm/formation_episode_input.py": "genuine Episode metadata/full/summary and append-invocation/profile originals",
            "scripts/mfm/formation_caller_meaning_v2.py": "Claim/Episode target-free meaning and historically bound member registrations",
            "scripts/mfm/formation_world_replay_v2.py": "exact causal original history and commit-time Episode registration proof",
            "scripts/mfm/formation_world_caller_v2.py": "current per-part read-boundary responsibility and complete permitted originals",
            "scripts/mfm/formation_joint_input_v2.py": "whole original-part field/token/candidate/cache joins; no prior observation promotion",
            "tests/governance/mfm_episode_fixtures.py": "real reversible original producers and fictional each-part permission adapter",
            "tests/governance/test_mfm_episode_input.py": "complete original Episode parts and genuine contract/receipt refusal controls",
            "tests/governance/test_mfm_caller_meaning_v2.py": "original family meanings, lifecycle and historical role provenance",
            "tests/governance/test_mfm_world_replay_v2.py": "causal commits, independent counters, future and changed-original refusal",
            "tests/governance/test_mfm_world_caller_v2.py": "historical-current separation and stop at first changed part permission",
            "tests/governance/test_mfm_joint_input_v2.py": "complete original field joins and source ineligibility with saved-cache responsibilities",
            "docs/receipts/MFM_FABLE_STACK_DOC_CORRECTION_20261010_TRACE104_PUBLIC.json": "exact published Fable doc branch correction; capability/release predicates preserved and main unchanged",
            "docs/MFM_FULL_TARGET_FIELD_BUILD_2026-10-10.md": "complete canonical target fields, actual Gemma token positions and original shifted decoder construction",
            "docs/research/MFM_CALLER_CANDIDATE_INPUT_RESEARCH_2026-10-10.md": "missing complete caller semantic features and primary joint versus separate encoding mechanisms",
            "scripts/mfm/formation_target_alignment.py": "whole target field and syntax geometry with independent caller ancestry before supervision",
            "scripts/mfm/formation_field_responsibility.py": "normative typed field roles and independently authored reference and original literal declarations",
            "docs/MFM_ORIGINAL_SOURCE_ALIGNMENT_BUILD_2026-10-09.md": "original UTF8, actual rendered tokenizer positions and canonical literal alignment; real Gemma export, learned selection/copying and native/outcome formation remain explicit",
            "scripts/mfm/formation_source_alignment.py": "caller and processor bound original UTF8, escaped source-token geometry and all literal occurrences; separate native and learned responsibilities",
            "tests/governance/test_mfm_source_alignment.py": "connected original-byte, escape, repetition and externally pinned rejection controls for the text representation",
            "docs/MFM_DEVELOPMENT_CONTINUATION_2026-10-09.md": "unchanged E20 continuation, source interventions, caller reference selection, original-byte copy alignment and full governed formation roadmap",
            "tests/governance/test_mfm_public_conditional_foundation.py": "real weighted gradients, full sampler boundaries, actual checkpoint and different-next-batch restart controls; component fixtures do not establish full learned MFM",
            "docs/MFM_SOURCE_FEATURE_FIXTURE_DIAGNOSIS_2026-10-07.md": "ordered eight-step fixture diagnosis, primary file-lifecycle research, exact guard-reaching verification and foundation/storage continuation",
            "docs/receipts/MFM_DRIVE_SCOPED_STORAGE_GATE_20261007_PUBLIC.json": "repeated exact-scope metadata inventory and prospective archive envelope; no provider quota claim and original upload-before-gate gap retained",
            "docs/receipts/MFM_TOKENIZER_METADATA_DRIVE_20261007_PUBLIC.json": "exact original seven-asset metadata subset, fresh prewrite storage gate, completed Drive upload and full archive restore; no weights or processor compatibility claim",
            "docs/receipts/MFM_PUBLIC_FOUNDATION_IMPLEMENTATION_20261007_PUBLIC.json": "reviewed full-roster implementation, real component interruption/objective evidence and precise pending bank/resource/tokenizer/learning/full-role work",
            "scripts/mfm/archive_public_foundation_inputs.py": "complete externally pinned bank tensor and legal/census custody, content-addressed archive and full readback; durable restore remains separate",
            "scripts/mfm/magnolia_public_foundation_archive_20261008.sbatch": "original protected owner-account allocated CPU recipe for complete bank tensor verification and archive readback; execution is not established here",
            "scripts/mfm/magnolia_public_foundation_archive_v2_20261008.sbatch": "successor exact-commit protected owner-account CPU archive recipe using separately prepared v2 checkout; complete bank unchanged",
            "tests/governance/test_mfm_public_foundation_archive.py": "real reduced-bank byte/reader roundtrip, quarantine, custody tamper and retained failed scratch controls; no full production custody proof",
            "scripts/mfm/kaggle_public_foundation_20261008.py": "private exact-commit public material/code/processor package preparation with durable-restore pins and explicit full-producer measure-only recipe; no upload",
            "scripts/mfm/kaggle_public_foundation_20261008_runtime.py": "verified public archive transport and exact offline Transformers5.17.0-only overlay preserving Torch/CUDA/media; unchanged full producer measure-only invocation",
            "tests/governance/test_mfm_kaggle_public_foundation_20261008.py": "real archive/Git packaging and trainer-subprocess overlay, exact offline dependency and full-pinned measure-only controls; no actual processor/CUDA/learning proof",
            "docs/MFM_COMPLETE_BANK_CONSTRUCTION_2026-10-08.md": "full Alice/Fable foundation-before-personalization continuation from complete source-export metadata through tensor/archive/restore custody, actual producer measurement and later learning/full semantic formation",
            "docs/receipts/MFM_COMPLETE_PUBLIC_SOURCE_METADATA_20261008_PUBLIC.json": "verified complete4,862-source/40,868-pair export metadata and all4,720 progress snapshots; full tensor custody, archive, durable restore and learning remain pending",
            "docs/receipts/MFM_KAGGLE_OFFLINE_DEPENDENCY_20261008_PUBLIC.json": "hash-verified original Transformers5.17.0-only offline wheel preparation and local transport controls; preserves Torch/media, actual Kaggle processor/CUDA compatibility pending",
            "docs/receipts/MFM_FULL_BANK_CONSTRUCTION_SUBMISSION_20261008_PUBLIC.json": "reviewed complete-bank archive job577090, ordered oldGit source-transfer diagnosis and corrected exact source guards; Kaggle/Drive/learning remain separate execution results",
            "docs/receipts/MFM_V16_SEQUENCE_FAILURE_576984_PUBLIC.json": "externally pinned completed576984 method aggregate; all882 task rows and41 retained controls; diagnostic pass is not capability",
            "docs/MFM_FULL_ROLE_SEQUENCE_SUCCESSOR_2026-10-07.md": "full semantic route and researched preparation boundaries, independent caller input and conditional teaching before learned producer successor",
            "docs/research/MFM_SEQUENCE_ROUTE_COMPARISON_2026-10-07.md": "primary research and transfer limits for fresh conditional decoder, lexical reuse and larger pretrained language alternatives",
            "docs/research/MFM_NEXT_ROUTE_DESIGN_REVIEW_2026-10-07.md": "independent full-role/data/compute design review before construction; no blind lexical bridge or pointer-only qualification",
            "docs/research/MFM_CONDITIONAL_TEACHING_MATERIALS_2026-10-07.md": "original-source grants, actual human pair custody and scope limits; conflicting source rights remain held",
            "docs/superpowers/plans/2026-10-07-mfm-sequence-material-input.md": "ordered implementation and verification plan for material, caller boundary, actual processor census and source feature preparation",
            "configs/mfm/e2e_conditional_material_v1.json": "exact original E2E training revision, CSV/grant/legal hashes, semantic grouped splits and conditional-only use",
            "scripts/mfm/prepare_public_conditional_material.py": "pinned original human source-to-text worker with notices, ancestry, exact pair dedupe and alternative-reference split custody",
            "scripts/mfm/formation_caller_input.py": "teacher-independent byte-complete caller input and explicit scoped registry, path eligibility and source-subject/registered-target distinction",
            "scripts/mfm/preflight_public_conditional_material.py": "complete actual Gemma processor census, exact target roundtrip, role/notice parent joins and separate target-free source roster",
            "scripts/mfm/export_public_conditional_features.py": "externally pinned target-free census and same frozen clone source-state export; complete tensor custody and bounded interruption recovery",
            "scripts/mfm/magnolia_public_conditional_census_20261007.sbatch": "protected owner-account60-test CPU census of complete public teaching material, no pretrained model forward or optimizer",
            "scripts/mfm/magnolia_public_conditional_features_20261007.sbatch": "bounded protected owner-account source-only frozen feature preparation after externally published actual census pin",
            "tests/governance/test_mfm_public_conditional_material.py": "source byte/notice/role custody, semantic alternatives and grouped split controls",
            "tests/governance/test_mfm_caller_input.py": "independent opened-source/scoped-registry path eligibility, source subject roles, digest tamper and no teacher-input tests",
            "tests/governance/test_mfm_public_conditional_preflight.py": "actual worker join, target losslessness, early EOS, exact source reuse, scope rejection and original-notice parent pin tests",
            "tests/governance/test_mfm_public_conditional_features.py": "real tiny frozen safetensor source forwards, complete census roster and actual interrupted child-process resume custody tests",
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
            "docs/fable-builder/traces/FBM_TRACE_20261004_MFM_FIT_FAILURE_ROOT_ANALYSIS.jsonl": "completed fit versus failed formation, actual native reference assembly and original-copy construction methods; procedure only",
            "docs/fable-builder/MFM_NATIVE_ORIGINAL_COPY_SOURCE_CAPTURE_2026-10-13.md": "reusable actual original-copy construction and omitted whole-wire reference-supervision diagnosis",
            "docs/fable-builder/MFM_REFERENCE_ROUTE_AND_DRIVE_CUSTODY_CAPTURE_2026-10-13.md": "original reference observation,55-action custody and exact shared archive release methods; procedure only",
            "docs/fable-builder/MFM_JOINT_LEARNING_AND_FIXED_REVIEW_CAPTURE_2026-10-13.md": "complete original review/custody and actual joint objective/full-component construction with coherent canonical binding diagnosis; procedure only",
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
