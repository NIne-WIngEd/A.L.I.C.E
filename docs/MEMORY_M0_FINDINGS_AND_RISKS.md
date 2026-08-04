# Memory M0 Findings and Risks

**Baseline commit:** `d5d311ec49f1e4a3e5a7cf688062c7dc2f46d4ec`<br>
**Generated:** `2026-08-04T05:07:30Z`<br>
**Decision status:** M0 evidence baseline; no runtime renovation

## Executive findings

1. Phase 2 and Phase 5 remain separate authorities with no production
   event-to-claim bridge.
2. Current retrieval correctness is conservative, but several safeguards
   perform work proportional to total stored memory.
3. Candidate and conflict hydration use repeated authoritative queries.
4. SQLite is appropriate for the current single-host design, but the current
   independent stores and immediate write transactions require a future
   coordinator before background curation is activated.
5. Full integrity verification is valuable but must leave the ordinary hot path.
6. Private-store metadata is fragmented across multiple store types and needs
   a canonical registry before correction/deletion propagation is possible.

## Measured baseline

At the largest completed synthetic Phase 2 scale (**5000 records**):

- lexical search p95: **473.563 ms**;
- lexical full verification: **300.945 ms**;
- authoritative SELECTs per lexical search p50: **23.000**;
- database size: **5.43 MiB**;
- lexical index size: **1.29 MiB**.

The sanitized private inventory found **6** SQLite candidate stores. This count does not imply that every file is an
active A.L.I.C.E. memory authority.

## Risk register

| ID | Severity | Risk | M-stage response |
|---|---|---|---|
| M0-R1 | Major | Full lexical-index verification and semantic artifact hashing can scale with the complete store during retrieval. | M3 generation-bound manifests and incremental verification |
| M0-R2 | Major | Global corrected-target scans and per-candidate loads create query growth and N+1 behavior. | M3 current-state projection and batch hydration |
| M0-R3 | Major | Phase 2, ledger, lifecycle, and tier stores lack one registered derivative/deletion graph. | M1 deletion model and M2 deletion receipts |
| M0-R4 | Major | There is no production event-to-candidate-to-claim pipeline. | M2 contracts, then M4 bridge |
| M0-R5 | Moderate | Independent BEGIN IMMEDIATE writers may contend once background workers are enabled. | M1 online/offline boundary and M2 write coordinator contract |
| M0-R6 | Moderate | Full ledger integrity verification scans every event. | Scheduled integrity passes and checkpointed verification |
| M0-R7 | Major | Context packet, token budget, and retrieval trace are not implemented. | M1 ratification and M3 serving renovation |
| M0-R8 | Critical if activated early | Parametric personal memory lacks acceptable deletion and rollback guarantees. | Remains blocked |

<!-- m0-private-schema-refinement-v1:start -->
## M0 private schema-classification refinement

Private report SHA-256: `0675CE5D67011CD7522824CAA0A8AA4FCFB827D23276FBF6700A6A1C16FAB9C2`

- Recognized A.L.I.C.E. memory/storage schemas: **0**
- Other readable SQLite schemas: **6**
- Metadata-unavailable SQLite files: **0**
- Private row values and payload content read: **No**
- Raw private paths recorded: **No**

The original inventory's `other_sqlite_store` result was not sufficient for
M0 closure because its prefix matcher did not recognize every published
schema name, including the Phase 2 root table `memories`. This refinement
uses complete public table signatures and SHA-256 fingerprints for unknown
schemas.

Remaining risk: M1 must create a canonical store registry. Files that do not
match a published memory/storage schema must not be treated as memory
authorities merely because they reside in the private vault.
<!-- m0-private-schema-refinement-v1:end -->
## M0 exit assessment

- Architecture Hold: **active**
- Repository inventory: **complete for the audited commit**
- Private-store inventory: **sanitized metadata baseline complete**
- Synthetic latency baseline: **complete at configured M0 scales**
- Production Claim Store: **not implemented**
- Phase 2 migration: **not started**
- P5.1e: **paused**

M0 may close after these documents are reviewed, committed, and merged.
The next work is the remaining M1 ratification decisions—not runtime code.
