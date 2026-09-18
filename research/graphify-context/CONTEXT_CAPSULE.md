# A.L.I.C.E. Sol Context Capsule

- Request: coverage-g7-comic-continuous-memory-001
- Question: What does the A.L.I.C.E. comic require from continuous memory across years, especially remembering why an old decision worked then, why it later stopped working, and what changed?
- Source: alice-eipm-v1-build @ 021c5021a98104b35f9c8e94b19e48d21f25f132
- Catalog source: 021c5021a98104b35f9c8e94b19e48d21f25f132

## Active mission

- Mission schema: alice-context-active-mission-state-v1
- Source commit: 021c5021a98104b35f9c8e94b19e48d21f25f132
- Implementation status: SELECTOR_PRIOR_HEAD_REPAIR_FAILED_ENDPOINT_ROLE_DEFECT_LOCALIZED_ENDPOINT_READ_REPAIR_IMPLEMENTED_PENDING_PREP_AND_GPU
- Implementation source: configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.9.json
- Continuity overlay: docs/chat-context/2026-09-17/sol/N0_ENDPOINT_REPAIR_AND_DOWNSTREAM_ARBITRATION_HANDOFF.md
- Latest observed Magnolia job: 575718
- Execution rule: Implementation config governs committed model/build state. A newer continuity overlay may supersede its observed runtime status and next-action wording. Open the original handoff before execution.

## Source pointers

- [50] alice-eipm-v1-build:docs/MEMORY_ARCHITECTURE_HOLD.md — A.L.I.C.E. Memory Architecture Hold — Supersession Record
- [41] alice-eipm-v1-build:docs/MEMORY_M1_DECISION_REGISTER.md — Memory M1 Decision Register
  - status:  M1-DX0 through M1-D9 owner-ratified on 2026-08-05
- [37] alice-eipm-v1-build:docs/MEMORY_IDENTITY_FORMATION_AND_HOST_LEARNING_ARCHITECTURE.md — A.L.I.C.E. Memory Identity, Formation, Host Learning, and Repository Lifecycle
  - status:  Owner-ratified architecture decision
- [35] alice-context:docs/chat-context/2026-08-31/src/S02-memory-handoff.md — A.L.I.C.E. Stage G / G+ Memory Architecture Handoff
- [35] alice-context:docs/chat-context/2026-09-13/sol/N0_FRONTIER_RESEARCH_DECISION.md — A.L.I.C.E. N0 Frontier Research Decision — 2026-09-13
  - status:  continuity handoff. No permanent EIPM weights created.
- [35] alice-eipm-v1-build:docs/MEMORY_POLICY.md — A.L.I.C.E. Memory and Knowledge Policy
- [35] docs/memory-identity-host-learning-ratification-v2:docs/MEMORY_IDENTITY_FORMATION_AND_HOST_LEARNING_ARCHITECTURE.md — A.L.I.C.E. Memory Identity, Formation, Host Learning, and Repository Lifecycle
  - status:  Owner-ratified architecture decision
- [30] alice-eipm-v1-build:docs/MEMORY_EXTERNAL_SYSTEMS_REVIEW.md — External Memory Systems Code and Architecture Review
- [30] alice-eipm-v1-build:docs/MEMORY_RECORD_AND_PROVENANCE_STANDARD.md — Memory Record and Provenance Standard — Polyglot Cognitive Fabric
  - status:  Owner-ratified under Memory M1 on 2026-08-05
- [30] alice-eipm-v1-build:docs/PHASE_2_MEMORY_CORE_ARCHITECTURE.md — Phase 2 — Memory Core Architecture
  - status:  P2.0–P2.9 implemented; Phase 2 complete
- [28] alice-context:docs/chat-context/2026-08-31/src/S03-alice-memory.txt — S03-alice-memory.txt
- [28] alice-context:docs/chat-context/2026-09-13/sol/MAINSTREAMED_EIPM_AND_FABLE_BUILDER_DECISION.md — Mainstreamed EIPM + Fable Builder Decision

## Graphify navigation hints

- decision -> policies/friday_release_attestation_schema.json:L145
- decision() -> tests/phase5/attention_workspace_helpers.py:L74
- require() -> scripts/eipm/n0/preflight_n0_v02_teacher_dev_challenge.py:L36
- test_response_wrapper_rejects_split_verified_source_set_across_sentences() -> tests/phase4/test_information_conversation_bridge.py:L628
- _changed_lines() -> scripts/audit_capability_barriers.py:L488
- test_same_key_memory_requires_later_transition_aware_promotion() -> tests/phase2/test_memory_candidate_promotion.py:L322
- MemoryUnitEnvelope -> src/cognitive_kernel/memory_contracts.py:L377
- hhem_holdout.py -> src/alice_vault/hhem_holdout.py:L1
- InterruptThenRespondModel -> tests/phase3/test_conversation_orchestration_resume.py:L24
- pathlib -> :
- json -> :
- cognitive_kernel/__init__.py -> src/cognitive_kernel/__init__.py:L1
- dataclasses -> :
- typing -> :
- collections -> :
- m2_closeout_evaluation.py -> src/cognitive_kernel/m2_closeout_evaluation.py:L1

## External/private routing

- Recommended: True
- Reason: question references a private/external source class

## Trust contract

- This capsule routes retrieval; it is not canonical A.L.I.C.E. truth.
- Open consequential claims in the original branch/path source before acting.
- Keep branch-specific, experimental, superseded and canonical states distinct.
- Graphify nodes are navigation hints only.
- If private external routing is recommended, search the private Library manifest/source corpus before declaring context complete.
