# Memory Baseline Benchmark Plan and Initial Results

**Baseline commit:** `d5d311ec49f1e4a3e5a7cf688062c7dc2f46d4ec`<br>
**Generated:** `2026-08-04T05:07:30Z`<br>
**Data:** Synthetic only

## Benchmark rules

- No private store is used for performance measurement.
- The benchmark imports the repository's current production APIs.
- Each scale uses a fresh temporary store outside the repository and vault.
- The synthetic embedding model measures index and serving mechanics, not
  embedding quality or real-model inference time.
- Results are machine-specific and become comparison baselines, not universal
  release guarantees.

## Phase 2 Memory Core

| Records | Write p95 ms | DB | Lexical build ms | Lexical verify ms | Lexical search p95 ms | Auth SELECTs/search p50 | Semantic build ms | Semantic search p95 ms | Hybrid search p95 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1.979 | 312.00 KiB | 41.062 | 18.460 | 20.919 | 23.000 | 181.025 | 69.732 | 43.154 |
| 1000 | 1.716 | 1.25 MiB | 160.924 | 108.008 | 78.651 | 23.000 | 479.901 | 328.069 | 355.249 |
| 5000 | 1.965 | 5.43 MiB | 478.010 | 300.945 | 473.563 | 23.000 | 2646.416 | 1129.263 | 1827.567 |

## Experience Ledger

| Events | Append p95 ms | Integrity verify ms | Inspect ms | DB |
|---:|---:|---:|---:|---:|
| 100 | 2.274 | 27.085 | 1.716 | 188.00 KiB |
| 1000 | 3.673 | 707.455 | 38.579 | 1.54 MiB |
| 5000 | 3.226 | 4733.721 | 44.738 | 7.61 MiB |

## Existing test-suite wall time

| Suite | Exit | Elapsed ms | Summary |
|---|---:|---:|---|
| `phase2` | 0 | 23805.964 | ........................................................................ [ 88%]<br>................................................                         [100%]<br>408 passed in 22.93s |
| `phase5` | 0 | 5711.413 | ........................................................................ [ 88%]<br>...................                                                      [100%]<br>163 passed in 5.22s |

## Required later scale gates

M0 records the initial machine baseline. Later gates must add:

- 10K, 100K, 1M, and 10M evidence events;
- 1K, 10K, and 100K claim versions and episodes;
- concurrent readers and the single write coordinator;
- correction, deletion, rollback, stale-index, and worker-recovery load;
- actual local embedding models and hardware-specific model profiles;
- repeated cold-start and warm-cache distributions.

## Comparison rule

Every M3 or later optimization must compare against this exact baseline
commit and disclose data scale, hardware, cache state, and benchmark code.
