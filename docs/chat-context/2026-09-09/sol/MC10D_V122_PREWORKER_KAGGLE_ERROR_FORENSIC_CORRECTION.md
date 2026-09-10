# MC10D v1.2.2 pre-worker Kaggle ERROR forensic correction

Date: 2026-09-09

Canonical main remains frozen at `0abaed85873c3f8de04765847eb7700b0e20433f`.

## Observed v1.2.2 boundary

Exact kernel:
- ref: `mkrayanyan/alice-mc10d-sf-mis-abc67430d799`
- job ID: `ABC67430D799CD766BC97D78471580EC451528EA3C48A51720C3798C4F25DB1C`
- terminal status: `ERROR`

The v1.2.2 local controller successfully:
- verified the exact parent/input/hybrid artifacts;
- verified the exact simulation payload `7D217570C167575E24630E3BE150786B260C16518D0349EF2EF94E1A088A0925`;
- reverified MC8;
- migrated the v1.2.1 durable checkpoint;
- preserved the 23 superseded v1.2.1 technical attempts;
- created a new durable pre-dispatch checkpoint.

The remote v1.2.2 kernel produced **no worker evidence at all**:
- no `worker-started.json`;
- no `job-spec-receipt.json`;
- no `gpu.json`;
- no `runtime.json`;
- no `model-receipt.json`;
- no `capacity-canary.json`;
- no `job-failure.json`;
- no `job-result.json`;
- no `progress.json`;
- no `worker.log`;
- zero private candidate files.

Therefore the public context-capacity canary did **not** run. The terminal message `public capacity canary failed before private candidate execution` was a local controller misclassification caused by treating absence of the canary receipt after a provider ERROR as a canary failure.

## Quota accounting correction

v1.2.2 controller code used:

`find_remote_receipt(..., "job-result.json", ...) or {"elapsed_seconds": job.soft_seconds}`

before recording job spend.

Because the remote kernel produced no `job-result.json`, the local quota ledger charged the full 5400-second soft limit. This is a conservative internal charge, not measured provider runtime.

Current local ledger:
- v1.2.1 charge: 7200 seconds;
- v1.2.2 synthetic fallback charge: 5400 seconds;
- displayed total: 12600 seconds.

The second value is not authoritative Kaggle consumption and must not be used as actual quota spend until provider status/log authority is recovered.

## Static v1.2.2 canary design defects also found

Even if the kernel had reached the worker, the canary mixed context-capacity and structured-output qualification:
- request size was enforced in bytes, while the scientific capacity criterion was tokenizer count;
- the canary required a full 32-result structured generation rather than a minimal output;
- its Ollama service log was not copied into durable output;
- 40 private candidate tasks were embedded in the same job even though private execution was gated behind the canary.

Those issues must not be carried forward.

## Next action

No new GPU kernel is authorized yet.

First recover the existing kernel's provider-side failure authority, with zero GPU execution:
- raw Kaggle kernel status including `failureMessage` if exposed by the installed SDK;
- Kaggle kernel logs if the installed CLI supports the `kernels logs` command;
- read-only pull and SHA comparison of the remote source.

Only after the exact pre-worker failure is identified may a successor be released.

The successor architecture, if provider execution itself is viable, must separate:
1. public minimal-output >8192-token context probe;
2. public exact-schema structured-output probe;
3. one repaired private Mistral candidate pilot;
4. adaptive durable continuation only after a valid real pilot.

No automatic full-soft-limit quota charge is allowed merely because a remote result receipt is absent.
