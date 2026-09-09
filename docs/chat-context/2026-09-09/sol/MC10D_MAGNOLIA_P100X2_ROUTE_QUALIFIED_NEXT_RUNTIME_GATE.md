# MC10D Magnolia 2xP100 route qualified — next runtime gate

Date: 2026-09-09

Canonical main remains frozen at `0abaed85873c3f8de04765847eb7700b0e20433f`.

## Scientific authority entering this boundary

- hybrid pointwise freeze SHA-256: `8495B3C4B3207511D615BD0B7788EB064114B3F763E7E4299EDCC1EE869617B7`
- effective candidates: 287
- simulation eligible: 216
- pointwise reject owner hybrid: 71
- post-freeze MC8 verification/open complete
- simulation payload SHA-256: `7D217570C167575E24630E3BE150786B260C16518D0349EF2EF94E1A088A0925`
- full simulation/falsification has not started
- A-SYN acceptance/promotion remain false
- model training remains false

## A100 route

Magnolia A100 remains unavailable to the owner account. Actual job 575309 did not establish admission; the partition/QOS policy requires `bxmarg`, while the owner is authorized for `normal` and cannot request `bxmarg`.

Do not resubmit A100 work unless the account/QOS assignment changes.

## 2xP100 route — actual allocation qualified

The bounded route-only qualification was run with:

- partition: `gpu`
- QOS: `normal`
- GRES: `gpu:p100:2`
- actual Slurm job: `575311`

Observed terminal authority:

- `575311|COMPLETED|0:0`
- two Tesla P100 devices visible
- `probe_completed=true`
- `route_qualification_pass=true`
- output SHA-256: `F152A2BF9D7EF9EB4AEA7FA767A09DCB5F5013C38FB7AECC625103A8BF157623`

The route probe uploaded no private A.L.I.C.E. payload, downloaded no model, performed no inference, and mutated no MC10D scientific state.

This qualifies **actual scheduler/allocation access only**. It does not qualify the P100 route as provider-equivalent for MC10D private simulation.

## Exact next gate

Before any private simulation can use Magnolia P100:

1. qualify exact Ollama `v0.32.15` source runtime at commit `b7871fc0d1d82fe109536efa3e0e8e411c766c75`;
2. verify exact `go 1.26.0` toolchain using the official Linux-amd64 archive SHA-256 `aac1b08a0fb0c4e0a7c1555beb7b59180b05dfc5a3d62e40e9de90cd42f88235`;
3. verify Ollama's pinned `LLAMA_CPP_VERSION=b10488` resolves to `9d77fa17254e1dee4b9e92504c91611a60b1359f`;
4. prove a supported CUDA 12 toolchain/backend can target P100 (sm_60), build against Magnolia's glibc 2.17 userland, and start an empty Ollama service;
5. inspect produced ELF GLIBC requirements and shared-library resolution;
6. only after runtime success, run public-only frozen model/profile equivalence for the exact Gemma/Qwen/Mistral/Granite profiles.

No private candidate/scenario payload is permitted before all provider-equivalence gates pass.

If the CUDA-12/toolchain/runtime gate fails, stop Magnolia qualification and use the Kaggle durable candidate-boundary checkpoint/resume fallback. Do not weaken the scientific gate and do not enter another recursive infrastructure hotfix chain.

## Preserved operating rule

The earlier source-runtime attempt failed partly because a Windows controller interpreted a POSIX remote result path with Windows path semantics. Future Magnolia controllers must keep remote paths as POSIX strings, use the dedicated ED25519 BatchMode SSH route, preserve failed remote state, and never blind-resubmit.
