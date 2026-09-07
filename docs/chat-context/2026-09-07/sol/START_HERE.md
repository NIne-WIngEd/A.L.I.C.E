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

## 2026-09-07 Kaggle v1.0.2 transport correction

Two local-only launch failures produced zero remote Qwen execution: V100 referenced a stale missing `alice-kaggle.ps1` wrapper, and V101 recreated the already-solved Windows PowerShell 5.1 UTF-8 BOM bug in `kernel-metadata.json`. Do not rerun either launcher.

The replacement is `ALICE_KAGGLE_QWEN_PUBLIC_v1.0.2.zip` with a tiny PowerShell launcher and a Python controller. The controller owns all JSON/state, writes UTF-8 without BOM, stages exactly `kernel-metadata.json` plus one self-contained `script.py`, persists one deterministic Kaggle identity before push, pushes at most once, reconciles 404/not-found without repushing, resumes the same identity after local interruption, preserves raw CLI exit/stdout/stderr receipts, retrieves output before cleanup, and preserves failed remote kernels for diagnosis.

The exact deterministic kernel ref is `mkrayanyan/alice-qwen-k1-v102-55b8e7acb4c1`. The previously validated shared runtime dataset `mkrayanyan/alice-tournament-runtime-50539c5fe9bf` supplies the pinned Ollama v0.32.15 archive so the job does not waste Kaggle ephemeral disk on another 1.42 GB runtime download. The worker still verifies the archive and binary hashes, requires two T4s, proves material model use on both GPUs, performs a pre-model disk gate, then runs the frozen throughput probe and 16 public tasks only.

No private pointwise data. No MC8. No A-SYN acceptance/promotion. No training. Canonical main remains frozen.
