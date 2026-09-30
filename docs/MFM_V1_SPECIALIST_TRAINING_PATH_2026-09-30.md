# MFM V1 specialist training path

**Status, 2026-09-30:** Implemented training code and synthetic CPU component
tests. No actual 23.9 GB source clone has been transformed for MFM. No
processor preflight, GPU probe, full training, independent FINAL evaluation,
or product qualification has occurred on the pinned prepared artifact.

## Boundary

The source ancestry is the exact non-instruction-tuned
`google/gemma-4-12B@023679ed352de9bb66cc873c9009ce3482585c08` weight
file, SHA-256 `fe054ae05ff7f44318fd8ae90d58992531455c7ed31356704088f0f2d8c8009a`.
`alice_foundation.gemma4_v1.verify_derivative` must fully rehash a separate
role=`mfm` transformed snapshot and reject unchanged publisher weights. A
receipt that merely records a copy cannot pass. This check proves custody and
changed bytes. It does not prove suppression or behavior improvement.
The trainer pins the verifier file from foundation commit
`1f76064263f035de41f1c7d71a4d90e87643d181`; the file hash is included
in the processor and training binding.

The prepared base supplies multimodal hidden states. Its pretrained language
head is not used to answer or train. `FormationSpecialist` has new projection,
decoder, token embedding, and output weights initialized from the run seed.
The target is the existing canonical formation contract with exact source
anchors, subject, epistemic status, time interval, and scoped disposition.
Original evidence remains the authority for the separate Claim gate.

The training loader uses the existing exact source processor for text, code,
structured input, image, audio, and video. The non-IT source tokenizer has no
native chat template, so the trainer supplies a first-party template whose
hash is frozen in the CPU preflight receipt. Text is rendered as an escaped
JSON string to keep literal `<|image|>` and similar content from becoming
processor media placeholders. The original bytes and original byte offsets
are preserved in source custody and output grounding.

## Run sequence

Set `PYTHONPATH=src:../alice-gemma4-base/src` from `alice-mfm`. All commands
that open a corpus require a **loopback-only Linux network namespace** before
the first source is read. The process must also run under owner-controlled
storage and without outbound proxy sockets. The runtime check is a local
precondition, not a guarantee against host compromise.

1. Admit an exact SHA-256-pinned owner-authorized curriculum or an independent
   admitted corpus with FINAL sealed:

   ```bash
   python -m scripts.mfm.train_v1_formation_specialist data-preflight \
     --curriculum-manifest /secure/corpus/mixture.json \
     --input-sha256 "$CORPUS_SHA256" \
     --owner-authorization-ref "$OWNER_AUTH_REF"
   ```

2. After materializing and verifying a **changed-weight** MFM base, run the
   complete source and target processor pass on a CPU node. Use the same corpus
   arguments and set `--prepared-base-dir`, `--prepared-base-receipt`, and a
   new `--preflight-receipt`, then select `processor-preflight`. Media are
   decoded and tensorized; no source or target is silently truncated. The
   resulting receipt includes the longest tokenized inputs, modality set, and
   source/target/joint cross-attention stress cases.

3. On an actual BF16 GPU allocation, run `train --probe-only` with the same
   arguments and preflight receipt, plus a new `--output-dir` and an explicit
   `--max-cross-attention-pairs` no lower than the preflight's estimated corpus
   maximum. This is an explicit admission cap, not evidence of GPU fit.
   The one optimizer step includes the stress cases and reloads its saved
   component and optimizer for a continuation check. This is a **hardware
   probe**, not a trained MFM or a free assertion that the full corpus fits.

4. Run `train` without `--probe-only` only after the exact prepared base,
   processor, stress-case backward pass, checkpoint replay, storage, and
   training duration fit the selected allocation. Checkpoints are content
   hashed and `--resume-checkpoint` resumes at the next case after a complete
   optimizer step. No inference or development evaluation has yet been run on
   the resulting specialist component.

`--max-source-tokens` and `--max-target-tokens` are configurable resource
admission bounds. They do not silently shorten a history or define an MFM
capability ceiling. Every case must fit its selected source/target context;
histories that do not fit require a measured long-context or source selection
plan and a new preflight. The specialist computes target loss in recomputed
logit chunks to avoid a full target-by-262K-vocabulary loss tensor. Dense
cross attention still needs an actual full backward measurement. This script
currently loads the prepared base on one GPU. No 48 GB or multi-GPU fit claim
has been established.

The preflight receipt, checkpoint and component receipt all say
`qualified_for_product: false`. A finite changed-weight check, successful
training loss, or successful hardware probe cannot qualify suppression of
inherited behavior or Alice memory formation. The paired untouched-base,
prepared-base, assembled, and specialist-ablated diagnostics and independent
FINAL remain separate required steps.
