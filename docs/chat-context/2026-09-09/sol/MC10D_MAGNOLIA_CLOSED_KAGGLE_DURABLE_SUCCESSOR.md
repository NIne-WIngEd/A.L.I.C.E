# MC10D Magnolia closed — Kaggle durable simulation/falsification successor

Date: 2026-09-09

Canonical main remains frozen at `0abaed85873c3f8de04765847eb7700b0e20433f`.

## Final Magnolia boundary

The owner authorized one final Magnolia attempt and explicitly directed that failure should end the Magnolia investigation and move MC10D to Kaggle.

v1.0.3 final package:
- package SHA-256: `BCD7AA31853E73BC191EAB8CDC8D9F16EC93569223F3817993E700ADC3AA53B7`
- launcher SHA-256: `3151B515FC267B58C9D910A82AA9283FC4A60451E3761B8BEC161BBC8A7D2132`

The v1.0.3 launcher verified the exact v1.0.2 result bundle SHA-256 `FBA6A039393319C711CEC7DDCC2A87FEEF3AE14A49574D41387817913651FC9E`, then refused to submit a new Magnolia job because the actual v1.0.2 result was **not** classified as the assumed `PRE_WORKER_PROVIDER_FAILURE`.

Observed terminal:
- `v1.0.2 did not fail at the pre-worker boundary. Refusing a new Magnolia job.`

Therefore:
- no v1.0.3 Magnolia Slurm job was submitted;
- no additional private payload was sent to Magnolia;
- no v1.0.4 or further Magnolia provider hotfix is authorized;
- Magnolia is closed as an MC10D execution provider for this stage.

This supersedes the previous context note's provisional assumption that v1.0.2 was a pre-worker failure.

## Scientific authority unchanged

- hybrid pointwise freeze SHA-256: `8495B3C4B3207511D615BD0B7788EB064114B3F763E7E4299EDCC1EE869617B7`
- effective candidates: 287
- simulation eligible: 216
- pointwise rejects: 71
- simulation payload SHA-256: `7D217570C167575E24630E3BE150786B260C16518D0349EF2EF94E1A088A0925`
- full private simulation has not started
- A-SYN acceptance=false
- A-SYN promotion=false
- model training=false
- Stage G remains open

## Next exact route

Provider: Kaggle 2xT4.

Successor contract:
- exact v2.0.0 frozen model identities, prompts, schemas, request profiles and simulation/falsification semantics;
- one active GPU job at a time;
- provider-neutral candidate-family identity;
- candidate-boundary recoverable outputs;
- content-addressed Google Drive checkpoint;
- `CHECKPOINT_DURABLE=true` only after Drive upload + readback + exact SHA-256 verification;
- no blind repush on ambiguous Kaggle control-plane state;
- explicit quota stop preserves progress;
- technical/missing/invalid inference is never converted into semantic failure;
- completion modes are `FULL_FOUR_FAMILY_EVALUATION` or `EARLY_DEFINITIVE_FAIL_PROOF`;
- PASS requires all four families;
- early termination can only produce FAIL when remaining families are mathematically incapable of changing the final v2.0.0 result.

Execution-only family order is Mistral -> Granite -> Gemma -> Qwen, with Qwen last because its frozen thinking profile is known to be the expensive family. This ordering does not modify scientific scoring or model identities.

After all 216 survivors have a terminal scientific disposition, produce an early-stop-aware MC10D simulation/falsification freeze and the MC10E select/abstain proposal. No A-SYN acceptance, promotion or training occurs at that boundary.
