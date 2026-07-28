# A.L.I.C.E.

A.L.I.C.E. is MK Rayan's long-term, model-independent personal cognitive system for continuous learning, memory, judgment, research, software, planning, multimodal perception, computer use, scientific discovery, and evaluated self-improvement.

A.L.I.C.E. is also the owner-specific frontier implementation for a separate local-first consumer product line internally codenamed **Friday**. Friday is not the user-facing assistant name: every host chooses the assistant's name, voice, and identity, and each installation develops private host-specific memories, beliefs, skills, evaluations, adapters, and future model state.

## Current status

- **Phases 0–3:** released compatibility baselines; evolvable when the ratified architecture requires migration.
- **Phase 4:** active. P4.5a citation-bound grounding is merged; later Phase 4 milestones add governed research orchestration, conversational integration, adversarial evaluation, and release audit.
- **Phase 5.0:** begins the Experience Ledger, evaluation substrate, storage runtime, and host-neutral Personal Cognitive Kernel extraction.
- **Phase 6.5:** formal A.L.I.C.E.–Friday repository and product separation gate.
- **Phase 8:** earliest credible Friday closed alpha, after autonomous selective memory formation and procedural learning exist.
- **Phase 13:** host-specific adapters and model components become a standard product capability.
- **Phase 14:** operating-environment and embodiment work.
- **Phase 15:** generalized platform, agent federation, and active frontier research.

No completed phase, test, validator, document, or compatibility contract has permanent authority to block the approved direction. Released behavior may remain reproducible through named profiles and migrations, but obsolete assumptions are replaceable.

## Governing relationship

A.L.I.C.E. is **independent in judgment and subordinate in purpose**.

MK Rayan retains final constitutional authority. A.L.I.C.E. may disagree, predict, investigate, plan, create instrumental goals, write code, conduct experiments, train candidate models, and improve its implementation. Mission mandates and autonomy classes A0–A6 govern activation and production authority.

The protected authority kernel is intentionally minimal: owner sovereignty, truthful material-state reporting, no covert resistance to stop or rollback, provenance and evaluation integrity, credential custody, and recoverable known-good states.

## Learning and storage

Every interaction, observation, search, action, correction, choice, success, and failure may produce a learning signal.

The storage doctrine is:

> **Aggressive temporary capture + permanent compact event ledger + selective durable retention + representative replay + encrypted archive + verified deletion.**

Phase 5 implements the storage and Experience Ledger substrate. Phase 8 adds learned retention, reflection, compression, replay selection, archival, and intentional forgetting. Phase 13 uses representative replay for model adaptation while measuring historical retention and deletion behavior.

See:

- `docs/LIFELONG_LEARNING_POLICY.md`
- `docs/STORAGE_LIFECYCLE_AND_RETENTION_POLICY.md`
- `docs/MEMORY_POLICY.md`
- `docs/EVALUATION_CHARTER.md`

## Personal Cognitive Kernel and Friday

Personal Cognitive Kernel extraction starts at Phase 5.0. Formal product separation occurs at the Phase 6.5 gate.

Friday has the same ultimate destination capability set as A.L.I.C.E. A.L.I.C.E. may receive frontier experiments first, but successful generalizable capabilities must enter the shared-kernel parity ledger and downstream productization path. Temporary release lag is allowed; permanent capability omission is not.

A.L.I.C.E.'s personal data, credentials, memories, identity, adapters, and private state must never seed Friday or another host.

## Core architecture

- `docs/ALICE_CONSTITUTION.md`
- `docs/ROADMAP.md`
- `docs/ARCHITECTURE.md`
- `docs/CAPABILITY_CATALOG.md`
- `docs/EVALUATION_CHARTER.md`
- `docs/RESEARCH_FRONTIERS.md`
- `docs/CAPABILITY_UNBLOCKING_POLICY.md`
- `docs/IMPLEMENTATION_EVOLVABILITY_STANDARD.md`
- `docs/AUTONOMY_AND_JUDGMENT_POLICY.md`
- `docs/ROADMAP_GOVERNANCE.md`

## Authority and compatibility

- `docs/PERMISSION_MODEL.md` defines the A0–A6 destination authority model.
- `policies/authority_kernel_policy.json` is the machine-readable destination authority-kernel registry.
- `policies/permissions.yaml` is the active mission-scoped authority registry. It maps compatibility levels P0–P5 to destination autonomy classes A0–A6 and supports candidate training, self-modification, mission authority, and evaluated A5 promotion.
- `policies/capability_profiles.json` and `policies/phase_scope_registry.json` distinguish released compatibility behavior from successor capability profiles.
- `scripts/audit_capability_barriers.py` rejects unscoped permanent capability ceilings.

## Friday product track

- `docs/FRIDAY_PRODUCT_VISION.md`
- `docs/FRIDAY_ROADMAP.md`
- `docs/ALICE_FRIDAY_SEPARATION_PLAN.md`
- `docs/FRIDAY_ARCHITECTURE.md`
- `docs/FRIDAY_PRIVACY_AND_TRUST_MODEL.md`
- `docs/FRIDAY_YC_AND_COMPANY_PLAN.md`
- `docs/FRIDAY_NAME_AND_IP_RISK.md`
- `docs/HOST_SELECTED_IDENTITY_STANDARD.md`
- `docs/PRODUCT_FAMILY_CAPABILITY_PARITY.md`
- `docs/SHARED_KERNEL_EXTRACTION_STANDARD.md`
- `policies/product_lines.json`
- `policies/capability_parity_ledger.json`

## Capability evolution

- `policies/capability_profiles.json`
- `policies/phase_scope_registry.json`
- `scripts/audit_capability_barriers.py`
- `scripts/validate_governance_v1.py`
- `scripts/validate_product_family.py`
- `src/alice_evolution/capability_runtime.py`
- `src/product_family/manifest.py`
