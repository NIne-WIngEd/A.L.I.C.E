# N0 Import PASS and Model-Build Continuation — 2026-09-14

## Active immediate state

The real bounded public N0 corpus/tokenizer lineage is now imported and verified in the canonical Magnolia runtime:

`$HOME/rayan-compute/rayan-n0/n0-v01`

Import result: PASS.

Verified imported corpus:

- accepted characters: 412,144,896
- accepted rows: 5,933
- source count: 5
- verified shards: 5
- verified bytes: 422,476,606
- corpus receipt SHA-256: `6cf717095e0eea4e790d0c852014a8047fd5ffc9f30550b2445c8a992a0bc293`
- source config SHA-256: `b0d1fb42dd276e04a249f2d2cb9207dd3b841fb95950adbc2e16ae9fc9bc1a5c`
- private identity data: false

Imported transfer archive:

- archive SHA-256: `d8df5c0cab365cf55c66fc173d439f34c3d40a409ab45cada1c21cca5a02a182`
- lineage revision: `cb23ac483d0ec5bcc9107f895ada0a6a9c8c34ba`

The corrected import used the Bash `-lc` command boundary through `magnolia_udocker_exec.sh`. The earlier direct `python ...` invocation failed before import because the wrapper has `/bin/bash` as the container entrypoint. Do not repeat that invocation shape.

## Owner caution / compute-routing rule

Do not fall back into a Magnolia infrastructure loop.

From this point:

- Magnolia is for accelerator work that directly advances the model, primarily the proven 2×P100 route.
- Kaggle is only a network-capable acquisition fallback when network access is genuinely needed.
- Local/GitHub work is preferred for code, data contracts, evaluation logic, curriculum construction, and branch updates that do not require remote acceleration.
- Passed infrastructure gates stay closed: runtime smoke, DDP mechanics, CPU DNS diagnosis, Kaggle transport, and public-lineage import.
- Do not rerun a closed gate unless a new concrete failure makes its assumption invalid.

## Next model-building step

Run the prepared 200-step 2×P100 public MLM segment on the imported canonical lineage.

This is NOT a throwaway qualification run. It is the first durable real N0 semantic-backbone training segment. It writes resumable checkpoints under the canonical N0 checkpoint lineage. If healthy, continue from the resulting checkpoint rather than restarting.

After the first segment:

1. inspect actual loss trajectory, throughput, tokens seen, checkpoint receipt, and memory/runtime behavior;
2. choose the next real public-MLM training duration from evidence;
3. continue N0 training from the same checkpoint lineage;
4. move into the Sol curriculum ranker/evaluation path once the backbone is sufficiently learned;
5. then proceed toward N1 private identity supervision under the existing explicit private-gradient authorization boundary.

The objective is to build EIPM, not endlessly validate providers.

No private E0/E-INF/A-SYN gradient is authorized yet.
