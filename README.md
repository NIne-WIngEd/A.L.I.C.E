# A.L.I.C.E.

A.L.I.C.E. is MK Rayan's long-term, model-independent personal cognitive system for continuous learning, memory, judgment, research, software, planning, multimodal perception, computer use, scientific discovery, and self-improvement.

A.L.I.C.E. is also the flagship research implementation for a separate general product line currently codenamed **Friday**. Friday will be a local-first personal cognitive system that each host installs, owns, and trains on their own machine without giving the developer access to raw host data.

## Current status

- Phases 0–3: released baselines and fully migratable when required by the ratified architecture.
- Phase 4: active; current work is around P4.5.
- Phase 5: begins the host-neutral Personal Cognitive Kernel extraction.
- Phase 6.5: formal A.L.I.C.E.–Friday repository separation gate.
- Phase 8: earliest credible Friday closed alpha, after autonomous memory formation exists.

No completed phase, test, validator, document, or compatibility contract has permanent authority to block the approved direction. Released behavior may be preserved through profiles and migrations, but obsolete assumptions are replaceable.

## Core documents

- `docs/ALICE_CONSTITUTION.md`
- `docs/ROADMAP.md`
- `docs/ARCHITECTURE.md`
- `docs/CAPABILITY_CATALOG.md`
- `docs/RESEARCH_FRONTIERS.md`
- `docs/CAPABILITY_UNBLOCKING_POLICY.md`
- `docs/IMPLEMENTATION_EVOLVABILITY_STANDARD.md`

## Friday product track

- `docs/FRIDAY_PRODUCT_VISION.md`
- `docs/FRIDAY_ROADMAP.md`
- `docs/ALICE_FRIDAY_SEPARATION_PLAN.md`
- `docs/FRIDAY_ARCHITECTURE.md`
- `docs/FRIDAY_PRIVACY_AND_TRUST_MODEL.md`
- `docs/FRIDAY_YC_AND_COMPANY_PLAN.md`
- `docs/FRIDAY_NAME_AND_IP_RISK.md`
- `docs/SHARED_KERNEL_EXTRACTION_STANDARD.md`
- `policies/product_lines.json`

## Capability evolution

- `policies/capability_profiles.json`
- `policies/phase_scope_registry.json`
- `scripts/audit_capability_barriers.py`
- `scripts/validate_product_family.py`
- `src/alice_evolution/capability_runtime.py`
- `src/product_family/manifest.py`

## Consumer product identity and parity

`Friday` is the internal codename for the general consumer distribution. Each host chooses the assistant's local name. The consumer distribution shares A.L.I.C.E.'s complete destination capability set through the Personal Cognitive Kernel; it is not planned as a permanently reduced edition. See `docs/HOST_SELECTED_IDENTITY_STANDARD.md` and `docs/PRODUCT_FAMILY_CAPABILITY_PARITY.md`.
