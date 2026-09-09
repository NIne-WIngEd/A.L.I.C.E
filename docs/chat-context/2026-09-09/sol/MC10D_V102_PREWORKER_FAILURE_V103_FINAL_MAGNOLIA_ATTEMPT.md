# MC10D Magnolia v1.0.2 pre-worker failure — v1.0.3 final attempt ready

Date: 2026-09-09

Canonical main remains frozen at `0abaed85873c3f8de04765847eb7700b0e20433f`.

## Scientific authority unchanged

- hybrid pointwise freeze SHA-256: `8495B3C4B3207511D615BD0B7788EB064114B3F763E7E4299EDCC1EE869617B7`
- effective candidates: 287
- simulation eligible: 216
- pointwise reject owner hybrid: 71
- simulation payload SHA-256: `7D217570C167575E24630E3BE150786B260C16518D0349EF2EF94E1A088A0925`
- full private simulation has not started
- A-SYN acceptance/promotion remain false
- model training remains false

## Magnolia route authority

- A100 route remains unauthorized under owner QOS.
- Proven 2xP100 route remains job 575311.
- v1.0.2 provider-equivalence package SHA-256: `44F6CE8D6B22AF4E31739F106112298EC2CAB07153C7F55D85573AC2F01A5E3D`.
- actual v1.0.2 Slurm job: `575324`.
- user-observed sequence: RUNNING / AWAITING_WORKER -> FAILED.
- v1.0.2 result bundle SHA-256: `FBA6A039393319C711CEC7DDCC2A87FEEF3AE14A49574D41387817913651FC9E`.
- v1.0.2 provider status: `STOPPED_NOT_PROVIDER_EQUIVALENT`.
- Because no worker progress was observed, v1.0.2 collection classifies this boundary as pre-worker provider failure; exact local bundle must be verified before another submission.

## Sole final Magnolia attempt

Release:
- package: `ALICE_MC10D_MAGNOLIA_P100X2_PROVIDER_EQUIVALENCE_v1.0.3.zip`
- package SHA-256: `BCD7AA31853E73BC191EAB8CDC8D9F16EC93569223F3817993E700ADC3AA53B7`
- launcher SHA-256: `3151B515FC267B58C9D910A82AA9283FC4A60451E3761B8BEC161BBC8A7D2132`
- run ID: `alice-mc10d-magnolia-p100x2-provider-equivalence-v103-final`
- job name: `alice-mc10d-p100eq-v103f`

The launcher must verify the exact v1.0.2 result bundle and require:
- result status STOPPED_NOT_PROVIDER_EQUIVALENT;
- failure class PRE_WORKER_PROVIDER_FAILURE;
- no worker-started.json;
- no progress.json;
- slurm.err present.

Only then may v1.0.3 proceed.

## v1.0.3 structural changes

- Slurm wrapper executes no module command before provider_worker.py.
- Worker evidence starts before module/toolchain resolution.
- Read-only controller preflight verifies exact Python/rclone, GCC 11.4, CMake >=3.24, CUDA 11.8, >=100 GiB home free space, normal QOS authority, and the 2xP100 node before any GPU submission.
- Exact Ollama and llama.cpp sources use commit-addressed HTTPS archives rather than Magnolia git clone/fetch.
- Ollama's pinned llama compatibility patch is explicitly pre-applied to the raw llama.cpp source override and reverse-check verified before the root CMake build.
- Source build binds sm_60, CUDA 11.8, exact compilers, exact Go 1.26.0, exact Ollama 0.32.15, and exact llama.cpp full commit.
- Runtime/dependency/module lineage, Drive round-trip, four frozen model identities, exact v2.0.0 request semantics, public scenario/falsification geometry, and public capacity probes remain mandatory.

## Final routing rule

This is the last Magnolia provider attempt for MC10D.

- `PASS_PROVIDER_EQUIVALENT` -> build the private Magnolia simulation successor using the exact qualified runtime/dependency/module envelope and frozen early-definitive-failure contract.
- Any deterministic v1.0.3 failure -> permanently close Magnolia for MC10D and move directly to Kaggle durable checkpoint/resume.
- Exit 74 means transport/scheduler uncertainty only; reattach to the exact same v1.0.3 run identity. Never create a second job while unresolved.
