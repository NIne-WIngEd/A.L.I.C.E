# MFM learned-run execution handoff

**State, 2026-09-29:** owner-authorized training corpus frozen; no optimizer
step or learned checkpoint has been observed. The current corpus is synthetic
and text/structured. The raw-media route exists but needs actual image, audio
and video source histories and a checkpoint/processor run before a multimodal
capability claim. Independent development and separately custodied FINAL also
remain to be built. These are qualification and coverage tasks, not a hold on
starting the authorized weight run.

## Exact training input

Place these three files together in a private training directory, preserving
their names. The JSONL files stay outside the public code branch. The source
recipe and license attribution are in
[`MFM_CORPUS_SOURCE_RECEIPT_2026-09-29.md`](MFM_CORPUS_SOURCE_RECEIPT_2026-09-29.md).

| File | Cases | SHA-256 |
| --- | ---: | --- |
| `multisource_formation_train.jsonl` | 43,819 | `2595e5209c0cf9cc8077e330b9e288c08cf5752f6db46acde6a455ff781519a1` |
| `longitudinal_formation_train.jsonl` | 6,000 | `2c92302390ebcf46ea222df8cf1c79b75710fdfb7a47a684c4897b9e0f6ae830` |
| `mfm_training_mixture_v1.json` | 49,819 | `60fa44a5be62c8d2f0daf508e543dc1a41c34562dc0c4d399e26c9cabb772e21` |

The 71,659/77,659-row draft is superseded: it paired 28,800 identical
inputs with conflicting targets. The corrected source curriculum gives each
plan/report/tracker state one combined six-domain target. A streaming trainer
invariant now rejects repeated exact model inputs before GPU work. The
structured tier uses narrow field-level spans, bounded day/week/month targets
with explicit inclusive `day` granularity in contract v1.5.0,
and separate unverified physical outcomes. The 6,000 fictional cases retain
whole-utterance citations and belong to training only.

## Pinned backbone and routes

The full sensory route uses `google/gemma-4-12B-it` at exact upstream commit
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`. Its raw image, audio and
video inputs go through `AutoProcessor` and `AutoModelForMultimodalLM`; no
caption or base64 substitute counts as seeing the media. The text/structured
route uses a native chat template and can fit the current frozen mixture, but
its weight result alone cannot establish sensory formation. The two scripts
use different Transformers dependency environments. Both produce local,
content-addressed, self-contained weights, and their outputs remain proposals
for the separate authority gate.

## Work before renting GPUs

Run from the exact MFM commit with the three frozen inputs in
`../mfm-candidates`. Prepare a transferable environment with
`scripts/mfm/requirements-formation-multimodal.txt`, `ffmpeg` and `ffprobe`.
Keep the installed versions recorded; the processor receipt binds the exact
Transformers version and the trainer/prompt source bytes. Run the static checks,
install dependencies and stage the pinned model snapshot on a non-billed
machine. The snapshot and its staging receipt must be copied to storage that
the training machine can read before the paid allocation starts. A processor
download by itself does **not** stage the weight files. The receipt records the
original absolute path. A relocated copy can use `--snapshot-dir` for offline
verification and `--staged-model-dir` for the trainer. Both paths verify its
contents against the original receipt.

### Magnolia transfer route observed 2026-09-29

The `node` CPU partition was available, but `node016.cluster` could not resolve
`huggingface.co`. An earlier N0 attempt also failed on Magnolia DNS. Do not
download the model or install from the Hub in a Magnolia job. Stage and seal
the exact snapshot on an internet-connected, non-billed machine, then transfer
the complete snapshot, its receipt and the three frozen corpus files through
the `hpcwoods.olemiss.edu` gateway. The gateway and Magnolia share home
storage. Keep MFM in its own owner-only directory under `rayan-compute`; do
not upgrade the N0 container or use its name for an MFM environment. The
observed 3.1 TB free is shared-filesystem space, not a reserved allocation.

Copying from a Windows staging machine leaves a Windows origin path in the
receipt. On Magnolia, use the committed staging verifier with
`--snapshot-dir /homes/01/mxrayan/rayan-compute/rayan-mfm/gemma4-12b-it` on
the `node` partition. The override is rehashed against every receipt entry;
the original path need not be valid on Linux. Include that same path with
`--staged-model-dir` on the processor preflight and any later training command.
Run the verification and the full processor pass in a CPU Slurm job with a
separate MFM-compatible Python runtime. Do not hash the 24 GB snapshot or
process the 49,819 cases on the login or transfer gateway.

Stage the exact pinned Hub snapshot and write a per-file receipt on the
non-billed preparation machine. Standard Hugging Face authentication must
already be available to that machine; do not put credentials in the repo:

```bash
PYTHONPATH=src:. python -m scripts.mfm.stage_gemma4_model stage \
  --snapshot-dir /persistent/gemma4-12b-it \
  --receipt /persistent/gemma4-stage.json --download
PYTHONPATH=src:. python -m scripts.mfm.stage_gemma4_model verify \
  --receipt /persistent/gemma4-stage.json
PYTHONPATH=src:. python - <<'PY'
from pathlib import Path
from scripts.mfm.multimodal_paid_run import require_complete_weight_export
print(require_complete_weight_export(Path('/persistent/gemma4-12b-it')))
PY
```

The latter two commands check local bytes and model weight structure without
loading tensors or needing network. Repeat the offline staging verification on the
final runtime-visible disk **before** allocating the paid GPU instance. If
that storage is visible only after allocation, the same verification becomes
the first paid action; account for those minutes. A partially copied snapshot
must never be treated as ready.

For a transferred snapshot at a different path, repeat the offline verification
with `--snapshot-dir /new/path/gemma4-12b-it`. Include
`--staged-model-dir /new/path/gemma4-12b-it` on the processor, probe and full
fit commands. The original receipt travels with the snapshot.

On the non-billed machine, validate the actual processor and longest
prompt/answer on the *entire* mixture without loading weight tensors. Keep the
receipt alongside the exact frozen inputs; do not edit its bound code or change
the Transformers version after this pass:

```bash
sha256sum ../mfm-candidates/*formation_train.jsonl ../mfm-candidates/mfm_training_mixture_v1.json
PYTHONPATH=src:. python -m scripts.mfm.train_multimodal_formation \
  --curriculum-manifest ../mfm-candidates/mfm_training_mixture_v1.json \
  --input-sha256 60fa44a5be62c8d2f0daf508e543dc1a41c34562dc0c4d399e26c9cabb772e21 \
  --owner-authorization-ref owner_authorized_service_teacher \
  --output-dir ../mfm-runs/cpu-preflight \
  --preflight-receipt ../mfm-runs/processor-preflight-v1.json \
  --staged-model-receipt /persistent/gemma4-stage.json \
  --staged-model-dir /persistent/gemma4-12b-it \
  --max-sequence-tokens 32768 --preflight-only
```

With the staging receipt and relocated snapshot directory, this processor pass
reads verified local files and needs no compute-node network. Run it inside a
CPU Slurm job on Magnolia's observed `node` partition, not on the login node.

The 32,768-token value is an execution request, not a model or product
ceiling. The preflight reports complete prompt/answer lengths and rejects
overflow. Increase the requested context or change the hardware route if a
complete case does not fit. Do not crop evidence or target JSON. This frozen
corpus contains text and structured inputs, so this pass cannot establish
sensory formation ability.

Do **not** purchase MFM GPU time until all of these are complete and retained
outside the paid instance:

1. The exact three corpus digests above, CPU processor receipt and trainer
   source fingerprint match the committed code and intended run options.
2. The exact `google/gemma-4-12B-it` revision is fully staged, its receipt
   verifies from the runtime-visible disk with network disabled, and the
   processor and model read the same staged snapshot. A model download on
   the paid machine is not the planned route.
3. The intended Python, PyTorch, Transformers 5.17.0, Accelerate and
   DeepSpeed environment is built and import-checked before allocation.
   CUDA kernels and distributed loading still need the paid hardware probe.
4. The run has a durable output volume with at least 750 GiB free for multiple
   full ZeRO optimizer checkpoints, the final model and rank receipts, and a way to export them before
   releasing ephemeral storage. GPU allocation, startup, mounting, model
load, the backward pass and export all consume billed time.

## First paid run: one complete optimizer step

Use a fresh, empty probe output directory. Request one node with four actual
A100 80 GB GPUs. The trainer checks each visible device for `A100`, at least
75 GiB of VRAM, BF16 support and exactly four torchrun ranks. Its selected
64 training cases include the four longest processed inputs and cover one
full 16-microbatch accumulation on each rank. The probe saves a full optimizer
checkpoint, an exported model and per-rank peak-memory/timing receipts. It is
a capacity and code-path qualification, not the full training run.

```bash
PYTHONPATH=src:. torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  -m scripts.mfm.train_multimodal_formation \
  --curriculum-manifest ../mfm-candidates/mfm_training_mixture_v1.json \
  --input-sha256 60fa44a5be62c8d2f0daf508e543dc1a41c34562dc0c4d399e26c9cabb772e21 \
  --owner-authorization-ref owner_authorized_service_teacher \
  --preflight-receipt ../mfm-runs/processor-preflight-v1.json \
  --staged-model-receipt /persistent/gemma4-stage.json \
  --staged-model-dir /persistent/gemma4-12b-it \
  --deepspeed-config scripts/mfm/deepspeed_zero3_a100_4gpu.json \
  --output-dir ../mfm-runs/gemma4-12b-probe \
  --max-sequence-tokens 32768 --probe-only
```

The probe must finish backward, optimizer, sharded checkpoint, gathered model
export and receipt writing on all ranks. Check the measured peak VRAM and
throughput and checkpoint/export duration before deciding whether the full run
fits, its cost, and a `--save-steps` interval that balances checkpoint time
against work lost on interruption. Bind that interval before starting the
full run; do not change it during resume. A
32,768-token request and four 80 GB GPUs are not a proven fit until this passes.

## Full fit and resume

After a green probe, use a new empty output directory for the exact full run:

```bash
PYTHONPATH=src:. torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  -m scripts.mfm.train_multimodal_formation \
  --curriculum-manifest ../mfm-candidates/mfm_training_mixture_v1.json \
  --input-sha256 60fa44a5be62c8d2f0daf508e543dc1a41c34562dc0c4d399e26c9cabb772e21 \
  --owner-authorization-ref owner_authorized_service_teacher \
  --preflight-receipt ../mfm-runs/processor-preflight-v1.json \
  --staged-model-receipt /persistent/gemma4-stage.json \
  --staged-model-dir /persistent/gemma4-12b-it \
  --deepspeed-config scripts/mfm/deepspeed_zero3_a100_4gpu.json \
  --output-dir ../mfm-runs/gemma4-12b-formation \
  --max-sequence-tokens 32768
```

To resume after interruption, run the **same command with the same code,
source, environment, world size, configuration and options**, adding
`--resume-from-checkpoint ../mfm-runs/gemma4-12b-formation/checkpoints/checkpoint-N`
for a complete local checkpoint from this run. Keep the entire checkpoint,
including optimizer and scheduler states; the exported model alone cannot
resume a ZeRO-3 training run. The run manifest rejects a changed configuration
or a new run over an occupied output directory. Retain at least two complete
step checkpoints on durable storage.

Record the exact MFM commit, source mixture digest, tokenizer/processor
revision, actual GPU model and VRAM, framework versions, full training and
checkpoint receipts, peak memory, elapsed time and the final artifact SHA.
The paid route updates full weights with ZeRO-3; LoRA and CPU offload require
separate qualification. If the actual allocation does not fit, replan the
hardware or explicitly qualify offload. Do not shrink context, source coverage
or the backbone to make a receipt appear green.

## Available hardware and remaining proof

The last owner-supplied Magnolia inventory listed two 12 GB P100s on `gpu001`
and four K80s on `gpu002`. That inventory does not establish a feasible
full-weight or native BF16 training allocation for the pinned 12B multimodal
backbone. A later, stronger device allocation has not been observed here.
The prior Kaggle L4 request actually ran on T4s; requested hardware is not
proof of delivered hardware. No target four-A100 optimizer step has been
observed. Do not describe the proposed rental shape as a proved fit.

After training, independently reviewed histories from distinct generators,
languages, media, corrections, deletion/revocation and long-lived people must
qualify the exact model against strong baselines and the downstream governed
memory loop. A synthetic training loss or a passed preflight alone proves
neither that generalization nor full A.L.I.C.E. memory formation capability.
