# Memory Performance and Reliability Standard

**Draft:** 0.2<br>
**Claim Store direction:** Accepted 2026-08-03<br>
**Status:** Provisional targets requiring hardware calibration

## 1. Objective

Memory must improve A.L.I.C.E. without making ordinary interaction unreliable,
slow, or computationally unbounded.

## 2. Hot-path prohibitions

The synchronous dialogue path must not perform:

- unbounded scans;
- per-candidate database round trips;
- full graph traversal;
- full index rebuild or verification;
- large-model curation;
- deep consolidation;
- automatic deletion;
- model training;
- blocking cold-archive restore.

## 3. Provisional local-host SLOs

Measured excluding base-model generation:

| Operation | Target |
|---|---:|
| Compact evidence append p95 | <= 25 ms |
| Protected-core and mission load p95 | <= 40 ms |
| Structured current-state lookup p95 | <= 75 ms |
| Hybrid candidate retrieval + batch hydration at 100K claims p95 | <= 150 ms |
| Memory Context Packet assembly p95 | <= 50 ms |
| Total ordinary memory overhead at 100K claims p95 | <= 350 ms |
| Total ordinary memory overhead at 1M evidence events p95 | <= 750 ms |
| Explicit correction available to serving plane p95 | <= 5 s |
| Explicit deletion hidden from ordinary retrieval p95 | <= 5 s |

Targets must be measured on defined hardware profiles and may be revised only
through evaluation evidence.

## 4. Claim Store serving rule

- Ordinary reads use the materialized `current_claims` projection.
- Ordinary responses never replay complete claim history.
- Appending a claim version and updating its current projection occur in one
  authoritative transaction.
- Historical reconstruction, provenance expansion, and rollback are explicit
  bounded operations.
- Claim write conflicts use expected-current-version or equivalent optimistic
  concurrency checks.
- Idempotency keys prevent duplicate claim-version creation after retries.

## 5. Context budgets

- Memory context is hard-capped.
- The protected core never grows with history.
- Default memory budget is the lower of 4096 tokens or 12% of usable context.
- Raw evidence enters only when required.
- Contradictions and uncertainty cannot be dropped merely to maximize relevance.
- Every packet reports used and rejected tokens.

## 6. Queue classes

1. `critical_correction_deletion`
2. `owner_explicit_memory`
3. `active_mission_outcome`
4. `normal_session_curation`
5. `projection_refresh`
6. `index_maintenance`
7. `deep_consolidation`
8. `training_candidate_analysis`

Critical work may preempt lower classes. Training analysis stops first under
resource pressure.

## 7. Worker requirements

Every worker must implement:

- durable workflow ID and ordered event history;
- deterministic replay of orchestration decisions;
- external side effects isolated as idempotent activities;
- bounded concurrency;
- bounded queue size;
- idempotency;
- retry limit and exponential backoff;
- heartbeat;
- lease expiration;
- checkpoint/resume;
- cancellation;
- supersession;
- dead-letter quarantine;
- metrics;
- graceful shutdown;
- recovery after process or host failure without duplicating semantic writes.

## 8. Curation-lag targets

| Class | Provisional p95 |
|---|---:|
| Owner correction/deletion | <= 30 s |
| Explicit owner memory request | <= 2 min |
| Active mission outcome | <= 5 min |
| Normal conversation curation | <= 30 min |
| Deep consolidation | <= 24 h |

Dialogue remains functional if these targets are missed.

## 9. Index rules

Each index generation records:

- source-store generation;
- embedding model and dimensionality;
- normalization version;
- schema version;
- build time;
- record count;
- digest;
- status.

Mixed embedding generations are prohibited.

Full verification runs asynchronously. Ordinary reads verify generation binding
and touched records.

## 10. Database access rules

- batch hydrate candidate IDs;
- batch load conflict and provenance edges;
- no global correction scan per query;
- use materialized current-state projections;
- use prepared statements;
- cap candidate pools;
- measure query plans;
- add indexes only with write-cost evaluation.

## 11. Backpressure

Under compute, memory, or disk pressure:

1. pause deep consolidation;
2. pause training-candidate work;
3. reduce embedding batch size;
4. defer low-value curation;
5. preserve corrections, deletion, evidence, and active-mission state;
6. never silently delete protected artifacts.

## 12. Reliability tests

Required tests include:

- worker crash at every checkpoint;
- duplicate event delivery;
- out-of-order events;
- stale lease recovery;
- partial secondary-index failure;
- embedding migration;
- database lock contention;
- corrupt projection rebuild;
- cold archive unavailable;
- graph store unavailable;
- deletion during curation;
- correction during response assembly;
- rollback after later exposure;
- host restart with backlog;
- deterministic workflow replay after code restart;
- duplicate activity delivery and idempotent result reuse;
- 10M-event ledger scan and sampled verification.

## 13. Observability

Expose:

- retrieval p50/p95/p99;
- candidate counts by stage;
- packet token use;
- cache hit rate;
- index age/generation;
- queue depth and oldest age;
- worker heartbeat;
- curation lag;
- promotion/rejection rates;
- stale-memory use;
- deletion propagation lag;
- projection rebuild time;
- storage value per byte.
