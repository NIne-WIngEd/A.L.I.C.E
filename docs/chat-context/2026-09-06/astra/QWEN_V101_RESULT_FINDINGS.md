# Qwen job 575089: verified infrastructure failure and v102 continuation

The complete uploaded result ZIP was checked against its exact SHA-256 and the original workload authority. Every manifest member verified. Recomputing the public summary produced `7248ad930b77e2a391c1c12a852728e9e6f122bce3d07e48f64415919306fbf7`, exactly matching the summary published by the user's launcher in context commit `e3f14b46db3ef4094d62536a2c4293ec769db507`.

[Structured review](QWEN_V101_RESULT_REVIEW.json) locates the complete original ZIP and all evidence hashes. The user reported 10 portable v101 tests passed and one intended POSIX-only skip in 1.243 seconds. Upload, Slurm submission, collection, download and context publication all worked. The path repair is now supported by actual execution.

## What stopped the worker

Job `575089` reached `RUNTIME_PREPARATION` and raised `URLError` with `CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`. No runtime/model lock, model service, probe or task attempt was reached. The application exited 76; the scheduler reported FAILED. All 16 matrix rows are NOT_ATTEMPTED. There is no measured Qwen accuracy or semantic failure. The raw false qualification flag means the prerequisite run did not complete.

The explicit CA setup was already documented in [v1.7.6's TLS successor](../../2026-09-04/MC10D_V1_7_6_RUNTIME_TLS_CA_SUCCESSOR.md). Astra's new worker omitted it. The repair restores that exact trust policy for Python HTTPS and subprocesses. It keeps certificate-chain and hostname verification and refuses insecure redirects. Python's [SSL documentation](https://docs.python.org/3.11/library/ssl.html#ssl.create_default_context) describes explicit CA loading and verification defaults.

## Why telemetry stayed pending

The compute ledger's original run directory contains the initial PREFLIGHT/RUNNING snapshot. Its later payloads returned exit 73. The prior [working handoff](../../2026-09-04/ALICE_MC10D_CONTINUATION_HANDOFF_20260904.md) explicitly declares telemetry run IDs immutable and records this collision exit. The new worker incorrectly treated the existing publisher as a mutable heartbeat updater. Its recovery loop repeated the same mismatch. Re-running v101 alone would not fix this.

The complete source ZIP retains its original telemetry failure receipts. A separate terminal FAILED snapshot was appended from the independently verified evidence under `alice-qwen38-a1-8e7a384a745496f4-terminal-c926d9e355dd`. The original run files were preserved. [Terminal ledger commit](https://github.com/NIne-WIngEd/Rayan-Compute-Ledger/commit/3f42de3531b396ee53c1aab797852ce18ac88f14) records that external recovery. This does not rewrite the historical summary or claim its original publisher succeeded.

## One bounded successor

[V102](tools/qwen-runtime-v102/README.md) creates only the explicit new execution `alice-qwen38-a2-c926d9e355dd`. It does not restart or rewrite job 575089. Before submission it checks exact parent evidence, absence of any inference intent and the live closed scheduler identity. It then verifies HTTPS against the pinned runtime endpoint and freezes the full public model manifest. Failure at these checks prevents sbatch.

The new package preserves the approved model/profile, six-hour Magnolia CPU envelope, frozen prompt, 16 task bytes and pass rule. The probe and all 16 task-request hashes match the original v100 requests exactly. The old controller state remains separate. The new state lives under `qwen-fallback-a2` and reconciles one scheduler job on later invocations. No automatic further attempt follows a terminal stop.

Telemetry now assigns each distinct payload its own immutable snapshot ID. A lost acknowledgement retries the exact persisted bytes. The global helper stays unchanged. The returned evidence includes source reconciliation, CA identity, pre-submission checks and exact telemetry payload hashes.

All 48 software tests passed from the final ZIP. The full simulated worker lifecycle uses an immutable ledger. Tests also cover the prior Windows path defect, explicit CA checks, unsafe redirects, failed and successful pre-submission gates, exact parent evidence, frozen manifest reuse, task non-repetition and publication recovery. Literal PowerShell/SSH Python verifier code and bash syntax passed. Actual v102 Windows/Magnolia execution is pending.

## Carry these lessons forward

Treat the existing provider's path, trust, receipt and immutable-identity contracts as implementation requirements when writing a new controller. Preserve their named regression cases in the executable preflight. Local tests must model the actual external contract at each boundary. Green selftests and a heartbeat cannot establish successful inference. Infrastructure failure before any attempt must remain distinct from a completed model judgment that fails semantic gates.

The broader accepted audit, 287 candidates, 63 replacements, one deferred item, valid family receipts and G → G+ → Fable ordering remain in force. No four-family successor, private pointwise eligibility, breadth readiness, acceptance, promotion, training or MC8 access is granted here.
