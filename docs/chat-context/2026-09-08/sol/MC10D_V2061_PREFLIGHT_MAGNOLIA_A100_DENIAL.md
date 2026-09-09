# MC10D v2.0.6.1 simulation preflight + Magnolia A100 denial

Date: 2026-09-08

Canonical main remains frozen at `0abaed85873c3f8de04765847eb7700b0e20433f`.

## Hybrid pointwise freeze

- pointwise freeze SHA-256: `8495B3C4B3207511D615BD0B7788EB064114B3F763E7E4299EDCC1EE869617B7`
- candidates: 287
- simulation eligible: 216
- owner-hybrid pointwise rejects: 71

## v2.0.6.1 preflight success

The repaired local-only preflight completed successfully:
- exact parent v2.0.0 verified
- exact hybrid pointwise freeze verified
- local Vault freeze matched downloaded freeze
- exact scientific input verified
- MC8 opened only after the hybrid pointwise freeze
- MC8 hidden material was not sent to the scenario builder
- all 24 packets remain represented
- simulation payload SHA-256: `7D217570C167575E24630E3BE150786B260C16518D0349EF2EF94E1A088A0925`
- 13,824 base candidate-scenarios
- 41,472 paraphrase probe obligations
- 864 family-candidate obligations
- nominal 2,592 model requests
- simulation has not started
- A-SYN acceptance/promotion remain false
- model training remains false

## Magnolia A100 route — actual admission result

A100 Slurm test-only under:
- partition=`suliaoma`
- qos=`normal`
- gres=`gpu:a100:1`

reported a hypothetical start on gpu003, but the actual submitted job `575309` remained PENDING with:

`Job's QOS not permitted to use this partition (suliaoma allows bxmarg not normal)`

This resolves an old ambiguity. Magnolia's `sbatch --test-only` was not sufficient evidence of actual owner admission for this partition.

Preserved Slurm authority already showed:
- owner association QOS: `normal`
- `suliaoma` partition AllowQos: `bxmarg`
- requesting `--qos=bxmarg` for the owner returns `Invalid qos specification`

Therefore the A100 route is currently **not authorized** for the owner account. Do not submit further A100 jobs unless the account/QOS assignment changes.

## Magnolia fallback still under consideration

The authorized GPU partition remains:
- partition=`gpu`
- qos=`normal`
- node gpu001
- 2 x Tesla P100-PCIE-12GB

This route is not yet qualified for MC10D private simulation because the exact Kaggle worker was bound to 2x Tesla T4 >=15GB/device and the prebuilt Ollama v0.32.15 runtime is incompatible with Magnolia's glibc 2.17.

A prospective provider-equivalence qualification may still test:
- actual 2x P100 allocation
- exact Ollama v0.32.15 source build against Magnolia userland/CUDA
- exact frozen model loading
- public calibration equivalence
- durable checkpointing

Private simulation must not be routed to P100 before those gates pass.

If Magnolia P100 cannot be qualified, use Kaggle with durable candidate-boundary checkpoint/resume and stop before weekly quota exhaustion. Do not weaken scientific gates to fit provider limits.
