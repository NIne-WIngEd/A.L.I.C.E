# A.L.I.C.E. — Sol Continuation Handoff (2026-09-07)

This is the newest continuity boundary after the owner returned the Magnolia runtime-route survey. It supplements the Astra master handoff.

## Mission

A.L.I.C.E. remains one continuing personal cognitive system. The current MC10D judge work is only one governed checkpoint in the larger Stage G program. Do not mistake package success, runtime success, or judge calibration for cognition, memory-fabric acceptance, A-SYN authority, training authority, or Stage G closure.

Canonical `main` remains frozen at `0abaed85873c3f8de04765847eb7700b0e20433f`.

## Repository/branch re-audit

All 25 public branches were enumerated again. Historical planning/feature/fix/docs branches are ancestors of frozen main. The deliberate divergences are the context/live evidence branches. `alice-telemetry` has no common ancestor with main.

Heads before this handoff commit:
- main: `0abaed85873c3f8de04765847eb7700b0e20433f`
- alice-context: `5c4f8c01895f4205c629fbb385d4f5e35fcaecfc`
- alice-mc10b-live: `86ae99ca11d5c9a5c72f4d8dd22c7ec90eb882ea`
- alice-mc10c-live: `53d2fb4060c681fea2097d5d15e9ce4f354e61f5`
- alice-telemetry: `956c973af3452706218e45256c0155c516c95d83`

## Scientific state that must not regress

- MC10B: 60/60 frontier packets; 720 raw E-INF; no historical acceptance; no training.
- MC10C: 24/24 strict eligible packets; 288 raw A-SYN; 24 UNKNOWN competitors; accepted/promoted/trained = 0.
- MC10D: 63 valid replacements + 1 governed defer + 224 retained originals = effective pool 287.
- Slot 63 stays deferred after exhaustion. No attempt 4.
- Slot 64 is complete. Never regenerate.
- MC8 remains sealed.
- Private pointwise screening has not started.
- A-SYN acceptance/promotion and model training remain blocked.
- Stage G remains open. Phase 2 remains canonical compatibility authority/test oracle.

Gemma evidence and Mistral/Granite bindings are preserved. GLM failed its semantic role and must not be rerun unchanged or rescued by gate weakening. Qwen fallback is owner-approved but **not scientifically evaluated**.

## Latest execution truth

V105 a3 run `alice-qwen38-a3-750881d854fa`, job `575182`, is closed.

It stopped before Ollama readiness. Probe attempts = 0. Task attempts = 0. Five telemetry snapshots were verified. Qwen remains `NOT_EVALUATED`.

Do not rerun V105. Do not reopen a1/a2/a3. Do not infer a Qwen failure from a runtime failure.

## Runtime-route truth

Trace SHA-256: `11DBC2BD6B7ED7A3BE0CD333986EFB72DBB403D03639DCD8EDFE714202218DFF`.

Observed facts:
- glibc 2.17 host.
- pinned prebuilt Ollama executable requires GLIBC_2.28.
- bundled libraries include newer requirements including GLIBC_2.27.
- no Apptainer/Singularity/Podman/bwrap/proot route observed.
- CMake 3.24.2 and GCC 11.4/12.3/13.2 modules exist.
- only old Go modules up to 1.13.4 were observed.
- survey was read-only and changed no execution/scientific state.

## Recovered operating rules

- Preserve failed state before diagnosis.
- Never infer failure from stderr alone; require exit codes and raw receipts.
- Do not infer physical file origin from exported ZIP layout.
- Keep Windows local paths and Magnolia POSIX remote paths as separate types.
- Use bounded SSH/SCP and exact absolute remote paths.
- `squeue` is active queue observation; `sacct` is historical terminal authority.
- Persist raw scheduler argv/exit/stdout/stderr before assertions.
- Never blind-resubmit after acknowledgement loss.
- Telemetry failure must not trigger inference repetition.
- Qualification profile must equal downstream deployment profile.
- Do not weaken provenance/hygiene gates to make a judge pass.
- Do not create recursive hotfix ancestry just because a fixture is green.
- Repeated failure class => stop patching and re-audit authority/state.
- On Windows prefer a small audited PowerShell launcher around Python; avoid giant here-strings/state machines and .pyz.
- Preserve untracked Vault work before Git cleanup; do not casually reset/clean canonical main.

## Immediate next boundary

The previous conversational idea, “build Ollama from source on Magnolia,” is a hypothesis until upstream requirements are verified.

Before creating another package:
1. verify the exact source-build requirements of the pinned Ollama version from authoritative upstream source;
2. determine whether the required Go/CMake/C++ toolchain can be reproducibly provisioned while targeting glibc 2.17;
3. prove the resulting runtime can preserve the exact frozen Qwen model/profile/request contract;
4. if source build is not cleanly supportable, choose a different compatible provider/runtime route without changing judge science;
5. only then produce one consolidated successor with preflight, ABI proof, empty-service readiness, the existing CPU throughput gate, and unchanged 16-task calibration.

No private candidate evaluation or downstream Stage G authority may be crossed by runtime work.
