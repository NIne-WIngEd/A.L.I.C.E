# Memory Shadow Migration Stage C+E Implementation

**Version:** 1.0.0
**Status:** reversible prototype operational
**Baseline:** PR #83 merge `2ad8ffcafa25ffc8bc5129ae32dbb201236950ca`
**Scope:** destination-candidate profiling and synthetic or separately owner-authorized shadow-read evaluation

## 1. Material state

Stage A inventory/registration and Stage B deterministic read-only adapters are prototype-operational.

This tranche begins the next admitted read-only work:

- Stage C — describe and compare destination candidates behind backend-neutral authority contracts;
- Stage E — compare Phase 2 and candidate observations on the same synthetic or separately owner-authorized workloads.

The released Phase 2 Memory Core remains the current authority for its released profile, compatibility baseline, test oracle, fallback, and export source.

## 2. Implemented artifacts

`src/cognitive_kernel/shadow_migration_stage_c_e.py` adds digest-bound contracts for:

- destination candidate profiles;
- shadow-read workloads;
- baseline and candidate observations;
- correctness, deletion, privacy, product-isolation, latency, and staleness comparison receipts;
- destination-candidate evaluations that can recommend only a later research gate, continued evaluation, or repair.

The existing reversible M2 prototypes are registered as one candidate composition across Claim Authority, shadow adjudication, projection, bounded serving, durable workflow, and deletion propagation. That descriptor is not a permanent backend choice. Embedded, relational, distributed, event, graph, vector, object, workflow, model, and future candidate compositions remain allowed.

`src/cognitive_kernel/shadow_migration_stage_c_e_evaluation.py` produces a deterministic synthetic report outside public Git. It exercises conflict, correction, deletion, explanation, product-isolation, privacy, latency, and staleness comparisons without reading private payloads or influencing production serving.

## 3. Workload authority

Synthetic workloads contain only digest-bound synthetic identifiers.

An `owner_authorized` workload must carry an explicit authorization reference. The contract does not itself open a private store, materialize a private query, or persist private paths. A later owner-authorized runner may use the same contract under separate custody, evidence, and deletion controls.

## 4. Activation boundary

This profile is reversible and nonproduction.

It performs no historical private backfill, controlled write mirroring, destructive live deletion, canonical authority transfer, canary authority, production influence, cutover, Phase 2 retirement, Friday state transfer, or P5.1e unblock.

Those are successor activation states rather than prohibited capabilities. Their research may proceed under ratified parallel tracks. Production activation requires the exact evidence and authority profile for the affected consequence class.

No database, topology, model, context size, graph engine, workflow engine, training method, deployment profile, or scale is selected as a permanent ceiling by this tranche.

## 5. Evaluation semantics

A comparison checks:

- expected result identity;
- authority namespace and product-host scope;
- conflict and correction handling;
- deletion exclusion;
- private-payload non-exposure;
- product isolation;
- latency and staleness deltas;
- explanation-trace binding.

A candidate with any correctness, deletion, privacy, or isolation regression is marked degraded. A clean candidate may be equivalent, improved, or inconclusive. Even a successful evaluation does not select production authority.

## 6. Exit and rollback

Rollback is file-level:

1. remove the Stage C+E source, policy, tests, and documentation;
2. restore the README, catalog, migration-plan, admission-policy, Stage A+B status, and package exports;
3. delete external synthetic reports from the selected vault report directory if no longer needed;
4. retain Phase 2 and the Stage A+B oracle unchanged.

## 7. Validation

Required before PR:

- Stage C+E targeted contract and deterministic-report tests;
- Stage A+B, M2 closeout, and relevant Phase 2 regressions;
- Phase 5 regression;
- governance and product validators;
- governance regression;
- full repository regression;
- changed-surface and repository-wide capability audits;
- staged file-by-file repository audit;
- `git diff --check`;
- signed candidate commit and external report hash.
