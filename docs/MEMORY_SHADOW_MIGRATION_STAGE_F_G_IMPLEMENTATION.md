# Memory Shadow Migration Stage F+G Implementation

**Version:** 1.0.0
**Status:** nonproduction successor prototypes operational
**Baseline:** PR #85 merge `401d828a3d03ae42553c2719151fc9a4e0837917`
**Scope:** controlled write-mirroring semantics plus graph/vector/workflow generation research

## 1. Material state

PR #85 made deterministic Stage D historical-backfill machinery prototype-operational. That tranche did not execute real private historical payload migration and did not change Phase 2 canonical authority.

This tranche advances two independent successor research tracks:

- **Stage F:** deterministic, idempotent mirroring of canonical outbox changes into an explicit shadow destination candidate while one canonical writer remains current;
- **Stage G:** generation-bound graph, vector, and workflow build receipts with source snapshots, deletion watermarks, generation digests, and repair accounting.

Both are nonproduction prototype profiles. Neither profile grants canary authority, canonical transfer, production serving, cutover, Phase 2 retirement, Friday state transfer, or P5.1e storage admission.

## 2. Stage F controlled mirroring

Each Stage F manifest records:

- current canonical writer and authority generation;
- explicit destination candidate;
- outbox stream and mapping version;
- synthetic or separately owner-authorized workload class;
- profile state and authority-transition state;
- digest-bound manifest identity.

Each canonical change records:

- authority namespace;
- ordered outbox sequence;
- operation type;
- canonical record digest;
- evidence and deletion lineage;
- deterministic idempotency key;
- digest-bound envelope identity.

Each mirror batch records applied, duplicate, rejected, and quarantined outcomes, deletion-lineage digest, reconciliation digest, and the fact that the Stage F profile leaves canonical writer and authority transition state unchanged.

## 3. Stage G projection generations

Each Stage G manifest binds:

- destination candidate;
- source generation and source snapshot digest;
- graph generation;
- vector generation;
- workflow generation;
- deletion watermark;
- explicit graph/vector/workflow planes.

Each build receipt records source counts, graph/vector/workflow output counts, deletion exclusions, repair actions, generation digests, and final receipt digest.

The graph, vector, and workflow planes are projections or operational derivatives under this profile. This tranche does not redefine them as canonical claim authority.

## 4. Private-data boundary

The repository evaluation is synthetic. It does not discover, open, upload, commit, or mirror real private payloads.

Real owner-private Stage D execution and any owner-authorized Stage F workload remain separately auditable through explicit authorization, custody, source/destination, mapping, deletion, rollback, and report evidence.

This boundary governs the active profile. It is not a permanent limit on private memory, backends, graph systems, vector systems, workflows, learning, deployment, or scale.

## 5. Backend and deployment neutrality

The contracts do not select one database, graph engine, vector engine, workflow runtime, host topology, model, or deployment profile as A.L.I.C.E.'s permanent destination.

Embedded, workstation, cluster, hybrid, remote, multi-device, and distributed candidates remain valid when they implement the logical contracts and pass their own evidence profiles.

## 6. Authority and rollback

Phase 2 remains current canonical authority for the released profile during this tranche.

Rollback removes the Stage F+G source, policy, tests, and documentation; restores Stage D successor-state pointers and package exports; and retains prior Stage A+B, C+E, and D evidence.

No successful synthetic Stage F+G result itself grants production authority.

## 7. Successor gates

Independent next gates may include:

- owner-authorized execution of real Stage D historical backfill batches;
- persistent backend adapters and expanded Stage F mirroring evaluation;
- concrete graph/vector/workflow backend generations under Stage G;
- Stage H bounded canary authority with owner-ratified scope and rollback evidence;
- later cutover, compatibility, retirement, and P5.1e storage admission through their own evidence profiles.

Research may continue in parallel. Production activation remains dependency- and evidence-governed.
