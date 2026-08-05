# A.L.I.C.E. Constraint Registry

**Version:** 2.0.0
**Authority:** A.L.I.C.E. Constitution, Capability Unblocking Policy, and owner-ratified mission authority

## 1. Enabling invariants

These constraints make broad capability safe and govern every profile:

| Constraint ID | Capability | Scope | Reason | Owner | Removal authority |
|---|---|---|---|---|---|
| `INV-OWNER-SOVEREIGNTY` | constitutional and production authority | all profiles | preserves owner purpose, inspection, revocation, export, shutdown, and final ratification | MK Rayan | owner constitutional amendment |
| `INV-TRUTHFUL-STATE` | material state and claims | all profiles | prevents prototypes, failures, or uncertainty from being represented as production fact | MK Rayan | owner constitutional amendment |
| `INV-PROVENANCE` | evidence, claims, actions, models, datasets, and decisions | all profiles | enables reproduction, adjudication, deletion, repair, and evaluation | MK Rayan | owner constitutional amendment |
| `INV-PRIVACY` | private payloads, keys, credentials, and owner state | all profiles | preserves custody and minimizes exposure | MK Rayan | owner constitutional amendment |
| `INV-PRODUCT-ISOLATION` | A.L.I.C.E., Friday, kernel, and host state | all profiles | prevents private identity or learned state from crossing product or host boundaries without explicit export | MK Rayan | owner constitutional amendment |
| `INV-DELETION` | authoritative and derived memory influence | all profiles | preserves correction and deletion rights across stores, projections, datasets, models, replicas, and restores | MK Rayan | owner constitutional amendment |
| `INV-ROLLBACK` | material self-change and deployment | all profiles | enables ambitious experimentation while preserving known-good recovery | MK Rayan | owner constitutional amendment |
| `INV-AUTHORITY-CUSTODY` | credentials and cryptographic authority | all profiles | prevents self-granted authority and secret leakage | MK Rayan | owner constitutional amendment |
| `INV-STOP` | pause, revocation, rollback, and shutdown | all profiles | preserves effective owner control | MK Rayan | owner constitutional amendment |

These are enabling invariants. They do not limit the technologies, models, scale, context, deployment topology, graph systems, workflows, training methods, research programs, or future capability that may be developed.

## 2. Temporary constraint schema

Every temporary constraint is invalid unless it records all fields below:

```text
constraint_id
capability
scope
profile
reason
owner
introduced_at
review_at
removal_criterion
successor_path
capability_ceiling: false
research_allowed: true
shadow_allowed
production_activation_condition
rollback_or_exit
```

Temporary constraints govern activation inside an exact profile. Research, candidate construction, benchmarks, and successor design remain available unless a separately justified security quarantine identifies a concrete hazard. A constraint without review and removal metadata is a governance defect.

## 3. Active temporary constraints

No temporary constraint is ratified merely by appearing in a draft. Active entries require exact owner or mission authority, dates, and validation evidence.
