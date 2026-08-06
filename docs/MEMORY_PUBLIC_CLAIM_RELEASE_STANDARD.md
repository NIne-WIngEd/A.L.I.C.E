# Memory Public Claim Release Standard

**Version:** 1.0.0
**Status:** Owner-ratified under Memory M1 on 2026-08-05

## 1. Purpose

Public statements must communicate A.L.I.C.E.'s destination without representing research, prototypes, or plans as implemented production capability.

## 2. Required status vocabulary

```text
destination
research_active
prototype_operational
shadow_evaluated
canary_enabled
production_profile_enabled
degraded
superseded
retired
compatibility_only
```

Every material claim names the capability, profile, date or release, evidence reference, and limitations.

## 3. Claim rules

- `destination` describes intended architecture or capability.
- `research_active` means investigation, design, experiments, or benchmarks are underway.
- `prototype_operational` means code runs in an isolated or synthetic environment.
- `shadow_evaluated` means the capability observes or computes without production authority or influence.
- `canary_enabled` means bounded production influence exists under an exact profile.
- `production_profile_enabled` means the capability is active only in the named production profile.
- `degraded` names reduced or unreliable operation.
- `superseded` names a replaced design or implementation.
- `retired` names an inactive capability or artifact with retention and deletion disposition.
- `compatibility_only` describes preserved historical behavior that does not govern successors.

A proposal, schema, registry entry, benchmark harness, or documentation statement is not implementation evidence.

## 4. Experimental authority

Owner-authorized shadow authorities, private test stores, alternate backends, graph systems, event streams, workflow engines, distributed clusters, model training, and challenger serving are valid research states.

Their status, custody, authority role, isolation, and deletion behavior must be explicit.

## 5. Destination language

Public material may describe powerful future capability when it clearly uses destination or research language. It must not imply a technology, scale, context, model, graph, workflow, training, or deployment limit merely because the current release is smaller.

## 6. Invariants

Every public claim preserves:

- owner sovereignty;
- privacy and product isolation;
- provenance and epistemic distinctions;
- truthful material state;
- deletion and correction rights;
- rollback and recovery;
- clone-aware constitutional honesty.

## 7. Release evidence

A production claim links to:

- exact commit and artifact digests;
- profile and configuration;
- tests and evaluation;
- known limitations and degraded modes;
- data and model lineage when relevant;
- deletion and rollback evidence;
- authority and approval record.

## 8. Staleness

Status blocks must identify their release or date. A historical profile cannot be read as the global current state. Superseded text points to its successor.
