# Magnolia Public N0 Lineage Import — 2026-09-14

## Result

The verified Kaggle-built public N0 corpus/tokenizer lineage was imported into the canonical Magnolia runtime target:

`$HOME/rayan-compute/rayan-n0/n0-v01`

The first import invocation attempted to pass `python ...` directly through `magnolia_udocker_exec.sh`. Because that wrapper uses `/bin/bash` as the container entrypoint, Bash attempted to execute the Python binary as a shell script and returned:

`/usr/local/bin/python: cannot execute binary file`

No lineage was imported by that failed invocation. The import was then rerun through the already-supported Bash command boundary with `-lc "python ..."`.

The corrected import returned:

- `status=PASS`
- archive SHA-256 `d8df5c0cab365cf55c66fc173d439f34c3d40a409ab45cada1c21cca5a02a182`
- lineage revision `cb23ac483d0ec5bcc9107f895ada0a6a9c8c34ba`
- `private_identity_data=false`
- `private_identity_gradient=false`
- `model_training_performed=false`
- `weights_created=false`

Required corpus, tokenizer, external transfer manifest, and external transfer receipt artifacts all exist after import.

## Corpus verification

`verify-corpus` passed against the imported canonical lineage:

- accepted characters: `412144896`
- accepted rows: `5933`
- source count: `5`
- verified shards: `5`
- verified bytes: `422476606`
- corpus receipt SHA-256: `6cf717095e0eea4e790d0c852014a8047fd5ffc9f30550b2445c8a992a0bc293`
- source config SHA-256: `b0d1fb42dd276e04a249f2d2cb9207dd3b841fb95950adbc2e16ae9fc9bc1a5c`
- `private_identity_data=false`

Verified source shards:

1. `common_pile_project_gutenberg_filtered-00000.jsonl`
2. `common_pile_pre_1929_books_filtered-00000.jsonl`
3. `common_pile_regulations_filtered-00000.jsonl`
4. `common_pile_usgpo_filtered-00000.jsonl`
5. `common_pile_python_peps_filtered-00000.jsonl`

## Compute discipline from this point

This closes the corpus acquisition/import boundary. Do not continue using Magnolia as a general-purpose preprocessing or network troubleshooting environment.

Magnolia's role now is narrow: use the proven 2×P100 route for model training segments that materially advance the N0/EIPM build. Kaggle remains the network-capable fallback for acquisition work only when needed. Local/GitHub work remains the default for code, evaluation logic, manifests, and model-building changes that do not require accelerator execution.

The prepared 200-step 2×P100 run is not another infrastructure qualification. It is the first durable real public N0 MLM training segment. Its checkpoints live in the canonical N0 checkpoint lineage and are resumable. No compute from that segment is throwaway if training is healthy.

After that first segment, use the observed loss/throughput/checkpoint evidence to choose the next real training duration. Do not repeat runtime-smoke, DDP-mechanics, corpus-network, Kaggle-transport, or import gates unless a new concrete failure invalidates their assumptions.

No private E0/E-INF/A-SYN gradient is authorized by this import.
