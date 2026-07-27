# Phase Scope Policy

**Version:** 1.1.0

A phase defines delivery order and the currently validated deployment profile. It does not define the maximum intelligence, agency, product shape, or future architecture of A.L.I.C.E. or Friday.

## Required metadata

Any phase-scoped restrictive file must declare:

- `scope_kind: phase_local` or equivalent;
- `capability_ceiling: false`;
- the profile or milestone it validates;
- the future profile, phase, migration, or condition that may supersede it.

## Historical material

Documents for completed milestones are retained for reproducibility. Their restrictions are historical unless repeated in an active policy or the minimal authority kernel.

## Compatibility modules

Released modules may retain strict behavior for reproducibility. They must be named compatibility profiles and may not prevent parallel or successor profiles from enabling broader capability.

## No phase immunity

Phase 0–4 documents, policies, schemas, source files, tests, fixtures, scripts, and workflows may be changed when necessary to implement the ratified architecture, remove capability barriers, support the A.L.I.C.E.–Friday product family, or correct architectural debt.

Released behavior is protected by evidence, migrations, and explicit compatibility profiles—not by making old implementation files untouchable.

## Stable roadmap, evolvable implementation

New ideas normally enter an existing capability domain as a module, experiment, product-track milestone, capability-catalog entry, ADR, or research-frontier item. Top-level domains remain stable unless a dependency becomes impossible or the owner changes the terminal purpose. Internal implementation and earlier phases remain evolvable.
