# Kaggle Provider Switch — 2026-09-07

## Owner direction

The owner explicitly directed the project to stop spending more time on Magnolia infrastructure if it continues to block progress and to use Kaggle instead.

This changes the **compute provider direction**. It does not weaken the MC10D scientific gates.

## Why Magnolia is paused

The read-only runtime survey was useful and established the native glibc incompatibility. The later source-runtime feasibility job `575231` then terminated `FAILED`.

The local controller did not download that result because of a controller bug:

`unexpected result path`

The controller compared the returned absolute POSIX path against a value constructed through local Windows `pathlib.Path` semantics. This is a client implementation defect. It is not evidence that Qwen passed or failed.

No more Magnolia hotfix chain is authorized merely to recover this infrastructure-only experiment. Preserve the remote state. Do not rerun it.

## Re-activate the proven Kaggle backend

Historical A.L.I.C.E. evidence already validated:

- private Kaggle Dataset + private script kernel
- internet enabled
- `alice-kaggle v0.2`
- 2 x NVIDIA Tesla T4
- result download to Windows
- remote cleanup
- usable failure logs

That backend is now the active route for this Qwen public calibration.

## New frozen Qwen compute profile

Profile:

`qwen38_thinking_on_public_calibration_kaggle_t4x2_v1`

Changed:

- provider: Magnolia -> Kaggle
- acceleration: CPU-only -> 2 x Tesla T4

Unchanged:

- family/model: Qwen `qwen3.8:27b-q4_K_M`
- quantization: `Q4_K_M`
- Ollama version: `0.32.15`
- thinking: on
- temperature: 1.0
- top_p: 0.95
- top_k: 20
- min_p: 0.0
- presence penalty: 0.0
- repeat penalty: 1.0
- context: 8192
- output ceiling: 6144
- seed formula
- structured output schema
- exact 16 public fictional tasks
- gold labels
- pass rule
- calibration-only status

The GPU execution profile is frozen before first output. If this exact profile later becomes the deployed Qwen judge profile, qualification and downstream use remain profile-consistent.

## Pending execution

Worker SHA-256:

`8BF89028BB9798FA60FEFA23F0CE1FAD3DA1E380CE6C7D6CCE33593D58011CB8`

Launcher SHA-256:

`C5228489F73585D94BB5F69E61213597841654A4A4B82DDE85B8AC6CB9B625E2`

The worker performs runtime/model identity checks, a 128-token throughput probe, then at most one request for each of the 16 public tasks. It records every request, response, timing, token count and score. It does not access the 287 private candidates or MC8.

## Progress rule

After this provider switch, do not spend days repairing peripheral infrastructure. Use the already proven compute backend. Escalate infrastructure work only when it is necessary to preserve scientific identity or recover unique evidence.

The next material progress target is a real Qwen public-calibration result.
