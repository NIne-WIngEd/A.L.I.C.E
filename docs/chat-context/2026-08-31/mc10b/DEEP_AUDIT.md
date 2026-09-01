# MC10B Full E-INF Frontier v1.1.0 — Deep Audit Report

## Release decision

v1.1.0 is a controller redesign, not a continuation of the v1.0.x hotfix chain. Generation methodology and frozen authority artifacts remain unchanged; the execution, transport, preflight, resume, telemetry, finalization, and recovery layers were re-audited as one system.

## Root causes recovered from failed full-frontier attempts

1. **Failure masking in v1.0.1.** CPU preflight returned failure, but PowerShell StrictMode immediately accessed a success-only field and threw `PropertyNotFoundStrict`, hiding the real probe failure.
2. **Cross-file private-blob contract drift.** v1.0.2 exposed the deterministic root failure: the frozen MC10B1 transport/worker requires `mc10b1-private-input.bin`, while the full-frontier package had uploaded a differently named blob. Retrying could never fix this.
3. **Duplicated orchestration contracts.** The old path repeated constants and state assumptions in PowerShell, the CPU probe, and the GPU worker. Fixing one layer could expose a stale contract in another.

Both failed full-frontier runs stopped before GPU generation. The public live state showed zero primary candidates generated.

## System-wide changes

- PowerShell reduced to a minimal parameter wrapper; Python is the sole controller.
- Private input filename derived directly from the frozen transport module.
- Exact local dataset file contract and SHA verification before upload.
- Exact remote file visibility check before preflight.
- CPU preflight validates private blob, extraction, source authority, 65→5+60 partition, pilot audit authorization, resume rows, pinned runtime, and GitHub read/write.
- Only explicitly classified transient preflight failures may receive one controlled retry. Deterministic invariant failures do not loop.
- Resume JSONL reader can recover only a torn final record, including a truncated UTF-8 code point. Corruption before the final record fails closed.
- Resume telemetry is recomputed from candidate rows. Persisted metrics/gates/acknowledgements must match fresh recomputation.
- Six 10-packet telemetry checkpoints remain. Acknowledgement resolves a review pause only; it does not accept any candidate.
- Soft generation deadline budgets the worst-case full three-attempt obligation before starting another obligation.
- Public GitHub telemetry strips candidate/prompt/E0/path/traceback/token material. Full exceptions remain private.
- Kernel logs are captured locally before remote cleanup when possible.
- If terminal output cannot be safely retrieved, remote GPU kernel and private dataset are preserved for recovery.
- Model cleanup and remote cleanup are best effort and cannot invalidate an already persisted valid checkpoint.
- Final result requires an exact file set, standalone validator, independent source-bound controller revalidation, and atomic canonical publication.
- Existing canonical output is idempotently reused only when its manifest is identical; differing output is never overwritten silently.

## Qualification matrix

The release self-test contains 16 checks:

1. Python compilation in memory.
2. Frozen generation and authority hashes.
3. Cross-file blob contract + shared resume telemetry validator.
4. Deterministic private transport round-trip.
5. Exact local Kaggle dataset contract.
6. Kaggle CSV/table file-list parsing.
7. Real MC10B1 pilot audit and telemetry baseline.
8. Crash-safe final JSONL tail recovery + middle-corruption rejection.
9. Six-block / 720-candidate telemetry scale test.
10. Resume telemetry recomputation and acknowledgement binding.
11. Synthetic 720-candidate standalone + independent final validation.
12. Exact final file-set fail-closed test.
13. Atomic canonical publication and idempotency.
14. Deterministic/transient preflight retry policy.
15. Deep failure, privacy, recovery, and soft-deadline guards.
16. Minimal PowerShell wrapper static sanity.

The outer package qualifier additionally verifies exact package membership, complete SHA256SUMS coverage, frozen byte hashes, JSON/JSONL parsing, Python syntax, the PowerShell wrapper contract, the 16-check self-test, and zero generated cache artifacts.

## Invariants that remain closed

- `E_INF_accepted_count = 0`
- `A_SYN_generated_count = 0`
- `model_training_performed = false`
- `MC10B_complete = false`
- `MC10C_start_allowed = false`
- `stage_g_closed = false`
- `phase2_replaced = false`
- generator has no identity/acceptance authority
- UNKNOWN remains a competitor
- broader-canon / held-out falsification remains mandatory before any promotion
