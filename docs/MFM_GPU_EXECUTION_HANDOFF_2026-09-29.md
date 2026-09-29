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

Run from the exact MFM commit with the three frozen inputs in
`../mfm-candidates`. Install
`scripts/mfm/requirements-formation-multimodal.txt` plus `ffmpeg/ffprobe`
in the Gemma 4 environment. First validate the actual processor and longest
prompt/answer on the *entire* mixture without loading weight tensors:

```bash
sha256sum ../mfm-candidates/*formation_train.jsonl ../mfm-candidates/mfm_training_mixture_v1.json
PYTHONPATH=src:. python -m scripts.mfm.train_multimodal_formation \
  --curriculum-manifest ../mfm-candidates/mfm_training_mixture_v1.json \
  --input-sha256 60fa44a5be62c8d2f0daf508e543dc1a41c34562dc0c4d399e26c9cabb772e21 \
  --owner-authorization-ref owner_authorized_service_teacher \
  --output-dir ../mfm-runs/gemma4-12b-formation \
  --max-sequence-tokens 32768 --preflight-only
```

The 32,768-token value is an execution request, not a model or product
ceiling. The preflight must report actual complete prompt/answer lengths and
refuse overflow; increase the requested context and/or change the hardware
route if a complete case does not fit. Do not crop evidence or target JSON.
On a compatible allocation, execute the same frozen input with an output
directory reserved for this exact run:

```bash
PYTHONPATH=src:. python -m scripts.mfm.train_multimodal_formation \
  --curriculum-manifest ../mfm-candidates/mfm_training_mixture_v1.json \
  --input-sha256 60fa44a5be62c8d2f0daf508e543dc1a41c34562dc0c4d399e26c9cabb772e21 \
  --owner-authorization-ref owner_authorized_service_teacher \
  --output-dir ../mfm-runs/gemma4-12b-formation \
  --max-sequence-tokens 32768
```

Record the exact MFM commit, source mixture digest, tokenizer/processor
revision, actual GPU model and VRAM, framework versions, full training and
checkpoint receipts, peak memory, elapsed time and the final artifact SHA.
The full-weight path is the default; an optional learned adapter must be
merged into a self-contained artifact. If the real allocation cannot fit the
full objective, introduce and verify an explicit sharding/offload route rather
than silently shrinking context, source coverage or the backbone.

## Available hardware and remaining proof

The last owner-supplied Magnolia inventory listed two 12 GB P100s on `gpu001`
and four K80s on `gpu002`. That inventory does not establish a feasible
full-weight or native BF16 training allocation for the pinned 12B multimodal
backbone. A later, stronger device allocation has not been observed here.
The prior Kaggle L4 request actually ran on T4s; requested hardware is not
proof of delivered hardware. Obtain the current device inventory and a
source-bound processor/forward/backward memory measurement before scheduling
the complete fit. No GPU or PyTorch is available in this workspace.

After training, independently reviewed histories from distinct generators,
languages, media, corrections, deletion/revocation and long-lived people must
qualify the exact model against strong baselines and the downstream governed
memory loop. A synthetic training loss or a passed preflight alone proves
neither that generalization nor full A.L.I.C.E. memory formation capability.
