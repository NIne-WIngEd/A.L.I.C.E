# START HERE — Sol continuation, 2026-09-07

Read in this order:

1. `STATE.json`
2. `KAGGLE_PROVIDER_SWITCH.md`
3. `SOL_CONTINUATION_HANDOFF.md`
4. Prior authority: `../../2026-09-06/astra/ASTRA_MASTER_CONTINUATION_HANDOFF.md`

Current boundary: Qwen is still **NOT_EVALUATED**. a1/a2/a3 remain closed. V105 must not be rerun.

The Magnolia source-runtime feasibility job was executed as job `575231` and reached a terminal `FAILED` scheduler state. The local collector then hit a Windows/POSIX path-comparison bug and refused the result path. That collector defect is not a Qwen scientific result. The owner has now directed the project to stop spending time on Magnolia infrastructure and use Kaggle so the architecture can advance.

**Magnolia is paused. Do not rerun job 575231 and do not create another Magnolia repair package.**

The next action is one bounded Kaggle public-calibration execution using the already validated private `alice-kaggle v0.2` route with two Tesla T4 GPUs. The model, quantization, prompt, public 16 tasks, decoding settings, gold labels and pass rule remain unchanged. The compute profile is newly frozen as `qwen38_thinking_on_public_calibration_kaggle_t4x2_v1` before any Qwen response is observed.

No private pointwise candidates. No MC8. No A-SYN acceptance/promotion. No training. Canonical main remains frozen.
