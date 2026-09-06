# Qwen infrastructure successor v1.0.2

Save this ZIP and `Start-ALICEAstraQwenQualificationV102.ps1` together in Downloads. Run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$HOME\Downloads\Start-ALICEAstraQwenQualificationV102.ps1"
```

The launcher verifies the ZIP, extracts to a fresh folder, verifies every package file, compiles Python and runs all 48 offline tests. It then checks Magnolia's source failure and HTTPS access before any `sbatch` call. After submission the same launcher attaches to the same new execution. Keep the transcript and return the result ZIP when produced.

## Why this is a new execution

The v101 Windows transport repair worked. Magnolia job `575089` started, stopped and returned a verified result ZIP. Its failure was `CERTIFICATE_VERIFY_FAILED` in `RUNTIME_PREPARATION`. No model identity, model service, throughput probe or calibration task was reached. All 16 task rows are `NOT_ATTEMPTED`; the false qualification flag is a stopped preflight, not a measured Qwen semantic failure.

The previous code also reused one immutable compute-ledger run ID for changing heartbeat bytes. Its first publication succeeded. Later publications stopped with exit 73. Repeating the v101 launcher cannot resolve that contract mismatch.

This release restores two earlier workflow safeguards: the explicit system CA setup documented for v1.7.6 and separate immutable telemetry snapshot identities. It preserves the complete original failed execution and uses one explicit infrastructure successor:

| Identity | Value |
|---|---|
| Approved calibration | `alice-qwen38-a1-8e7a384a745496f4` |
| Preserved source Slurm job | `575089` |
| Source result ZIP SHA-256 | `c926d9e355dd6ec83ad18fd78f35265c664e5ea6f317871c20d7af6b0f8d8f5f` |
| New execution | `alice-qwen38-a2-c926d9e355dd` |
| New local controller state | `C:\ALICE_Vault\tools\alice-astra\qwen-fallback-a2\controller-state.json` |

There is no source-state migration or deletion. The approved model/profile, public task bytes, prompt, probe request, all 16 task requests and scientific pass rule remain unchanged. A separate package hash identifies the repaired infrastructure. No new model approval is required for this bounded continuation of the already approved public calibration.

## Before submission

The remote controller checks exact critical source-file hashes, the original TLS failure and absence of any model/probe/task intent. It reconciles job `575089` as the same terminal `FAILED` job with exit `76:0`. Any changed source evidence or prior inference intent blocks this successor.

It selects a readable system CA bundle from the earlier three-path policy. Python HTTPS explicitly uses that bundle with certificate-chain and hostname verification. It exports the same CA location for subprocesses. The pinned decompressor installation receives an explicit `--cert` argument. An HTTPS-to-HTTP redirect is refused. There is no insecure download fallback.

The controller performs a bounded HTTPS HEAD request through the pinned runtime's redirect and fetches the public model manifest. It freezes the full manifest digest before submission. The worker reuses those exact manifest bytes. A failed preflight writes a diagnostic receipt and does not create a submission intent or scheduler job.

## Compute and evidence

The unchanged allocation uses Magnolia partition `node`, QOS `normal`, 20 CPUs, 48 GiB, an exclusive CPU allocation and a six-hour wall limit. It has no requeue. The separate 128-token public throughput probe checks the conservative remaining-time budget before the 16 tasks. It is not a calibration task. Each task can consume one request up to 6144 tokens. A recorded task intent is never retried automatically.

Each telemetry update gets a content-derived snapshot ID. Different bytes get a new ID. Lost acknowledgements retry the exact saved snapshot bytes. The existing global publisher and all prior ledger runs remain unchanged. The controller prints the actual latest snapshot URL. Terminal telemetry recovery does not perform inference.

The result bundle includes the source reconciliation, CA identity, pre-submission HTTPS/manifest preflight, runtime/decoder identity, full responses, finish metrics, exact contracts and latest telemetry payload. The independent verifier checks those hashes and recomputes scores against the frozen tasks. Raw model rationales are excluded from GitHub publication. Private candidates and MC8 remain outside this package.

The original terminal result remains failed and immutable. A separate terminal snapshot was appended to the compute ledger from the independently verified uploaded result. The historical ZIP correctly retains its original telemetry-failure receipts.

## Exit and recovery

- **0:** actual public calibration passed and its publication completed. This alone grants no successor binding or downstream eligibility.
- **74:** connection/monitoring pending. Rerun v102 to attach to this same new execution.
- **75:** publication pending. Rerun v102 to retry publication of saved evidence.
- **76:** deterministic stop or unsuccessful calibration. Return the transcript and result ZIP if produced. This release creates no further execution after a terminal stop.

All four exact judge-family receipts still need verification before a new binding. Breadth v104 remains blocked by its separate prerequisite migration. Preserve the 287-candidate pool, 63 replacements, one deferred item and prior family evidence. No Kaggle/GPU, private pointwise work, acceptance, promotion, training or hidden MC8 access is authorized by this infrastructure repair.

The offline tests include the full simulated worker lifecycle with an immutable ledger, correct/incorrect qualification cases, CA verification, redirect rejection, pre-submission failure, frozen manifest reuse, exact original request hashes, source-state protection and reconnect/publication recovery. These are software tests. Actual v102 Windows/Magnolia execution remains pending until the launcher is run.
