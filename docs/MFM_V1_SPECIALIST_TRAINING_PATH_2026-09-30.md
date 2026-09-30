# MFM V1 specialist training path

**Status, 2026-09-30:** Implemented training and evaluation code with synthetic
CPU contract tests. The eight source files were SHA-256 verified on Magnolia.
Owner-provided Slurm stdout for job `576486` reports a completed CPU run on
`node005.cluster` (exit `0:0`), a new source receipt digest
`ad081e28519961d72181815b7a74c65b42238e2c2ab0a65065acbc8e2be606df`,
and a separate MFM role-clone receipt digest
`54484dd523c396648fdf7069abe219c3f8f7c1fdcfec026cda1038810d4d7ac7`.
The clone digest was printed twice. The owner then ran the foundation
`read_receipt` metadata check on Magnolia. It validated both receipt digests,
the clone's parent-source digest, `role=mfm`, `qualification=unqualified`, and
identical eight-file rows against the pinned file manifest. The submitted batch
script and complete JSON were not transferred for independent inspection here;
the exact final verifier invocation remains unconfirmed. This is an
owner-reported custody result, not a processor, inference or training run. No complete 1.6
corpus processor preflight, GPU probe, full training, independent FINAL
evaluation or product qualification has occurred.

## Boundary

The source ancestry is the exact non-instruction-tuned
`google/gemma-4-12B@023679ed352de9bb66cc873c9009ce3482585c08` weight
file, SHA-256 `fe054ae05ff7f44318fd8ae90d58992531455c7ed31356704088f0f2d8c8009a`.
`alice_foundation.gemma4_v1.verify_role_base` fully rehashes a separate
role=`mfm` clone of those exact bytes, or a changed derivative when an
evidence-backed edit is warranted. A bare publisher snapshot and mismatched
role or receipt cannot pass. Cloning establishes custody and a separate
local artifact. It does not confer ownership of Google-origin weights or prove
suppression or behavior improvement. An edit is not a V1 training prerequisite.
The trainer pins the verifier file from foundation commit
`3e1328410dac52ba18be243cb9f7144e7a3964d1`; the file hash is included
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

2. Clone the verified publisher source into a separate MFM role snapshot and
   seal its clone receipt with the pinned foundation tool. Verify the role
   clone once before admission; the trainer independently rehashes all eight
   files at each admission. An optional edited derivative uses its own receipt.
   Then run the complete source and target processor pass on a CPU node. Use
   the same corpus arguments and set `--prepared-base-dir` to that role clone,
   `--prepared-base-receipt` to its receipt, and a new `--preflight-receipt`,
   then select `processor-preflight`. Media are
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
   optimizer step. A new training run saves its exact seeded, untrained
   specialist and a sealed digest before the first update, as a matched
   negative control. A resumed run rehashes that control.

5. On the same local, isolated network boundary, run the specialist evaluator
   separately for `trained` and `seeded-untrained` using the same exact input
   JSONL, preflight and role clone. Each input row has `case_id`, canonical
   `context`, `context_digest`, and `opened_sources` as ordered `{ref_id,
   payload_base64}` entries. The public diagnostic `emit` command produces
   seven such rows without its gold answers. Example:

   ```bash
   python -m scripts.mfm.run_v1_formation_specialist \
     --component-dir /secure/run --prepared-base-dir /secure/mfm-role-clone \
     --prepared-base-receipt /secure/mfm-role-clone.json \
     --preflight-receipt /secure/processor-preflight.json \
     --input-jsonl /secure/diagnostic-inputs.jsonl \
     --output-jsonl /secure/trained-output.jsonl --control trained \
     --inference-run-id evaluation-trained --max-new-tokens 8192
   ```

   Change `--control` to `seeded-untrained`, the output path and run ID for the
   negative control. The runner rehashes the role base, both specialist files
   and sealed component/run/preflight/control receipts, then uses the exact
   training source encoding. It generates BOS to EOS greedily through the
   first-party decoder only. A malformed, ungrounded or truncated JSON
   completion remains in the output with its raw tokens and invalid status;
   there is no repair, authority write or hidden fallback to Gemma's LM head.
   Output publication is atomic, local and mode 0600. The loopback-only Linux
   process boundary is mandatory before private input is opened.

`--max-source-tokens` and `--max-target-tokens` are configurable resource
admission bounds. They do not silently shorten a history or define an MFM
capability ceiling. Every case must fit its selected source/target context;
histories that do not fit require a measured long-context or source selection
plan and a new preflight. The specialist computes target loss in recomputed
logit chunks to avoid a full target-by-262K-vocabulary loss tensor. Dense
cross attention still needs an actual full backward measurement. This script
currently loads the prepared base on one GPU. No 48 GB or multi-GPU fit claim
has been established.

Autoregressive decoding currently recomputes the decoder prefix without a
cache, so long 8K outputs may be slow and require measured runtime capacity.
The public source-only Gemma diagnostic uses a different 1024-token cap and
language head, so its output is a descriptive reference. The role-boundary
qualifier compares the real trained and seeded-untrained specialist outputs
under matched settings and rehashes their common run artifacts. The seeded
control has a distinct weight hash and is **not** a fictional "disabled"
component. Even a passing seven-case diagnostic cannot qualify MFM.

The preflight receipt, checkpoint and component receipt all say
`qualified_for_product: false`. A verified clone, successful training loss or
hardware probe cannot qualify suppression of inherited behavior or Alice
memory formation. Measure an assembled specialist against its matched
seeded-untrained control and independent FINAL. The public source-only
diagnostic is optional for studying
inherited defaults; no separate weight-cleaning program blocks training.

## Magnolia public processor diagnostic after job 576486

`scripts/mfm/magnolia_v16_public_cpu_preflight.sbatch` uses the existing
`rayan-n0-base` CPU container because Magnolia's host Python/glibc is not the
validated Transformers 5.17 runtime. Supply absolute `MFM_REPO_ROOT` and
`MFM_FOUNDATION_ROOT` checkouts under `$HOME/rayan-compute` and submit from the
login node with Slurm logs outside the repository. The job checks the exact
21-case authoring seed, imports processor dependencies before rehashing the
role clone, and writes a new job-specific receipt under
`$HOME/rayan-compute/mfm/receipts`. It opens only public fiction and loads no
Gemma weight tensors. This is a **diagnostic** processor pass for 15 synthetic
train and six same-generator development cases, not the complete signed 1.6
corpus preflight required before a full fit. The complete pass must later bind
the separately admitted full-role corpus, external trust roster and exact
clone on the same isolated runtime path.
