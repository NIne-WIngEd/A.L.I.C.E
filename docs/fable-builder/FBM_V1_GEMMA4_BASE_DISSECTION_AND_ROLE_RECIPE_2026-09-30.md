# FBM V1: shared Gemma 4 base dissection and role specialization

**Owner-local decision:** 2026-09-29. **Status:** reusable construction
procedure with a verified source inventory; no edited base, personal model or
behavioral qualification is asserted here. This applies to all five personal model roles in
Fable V1 and refines [the V1 assembly contract](FBM_V1_LICENSED_BASE_ASSEMBLY_2026-09-30.md).

## Pinned source and custody

The common source is Google's **pretrained** `google/gemma-4-12B`, revision
`023679ed352de9bb66cc873c9009ce3482585c08`, not the `-it` checkpoint.
The publisher's weight-object SHA-256 is
`fe054ae05ff7f44318fd8ae90d58992531455c7ed31356704088f0f2d8c8009a`
(23,919,549,408 bytes). The exact eight-file pretrained source was historically
downloaded and rehashed at `/workspace/gemma4-12b-base-snapshot`. MFM commit
`4f5c7639` records the historical source receipt
`b80271bf8ff30023fac84aa247f7e21d8eb25fe2dbc4cd19709c9e7a24dde6e5`
and inventory manifest
`29d3f925c3be6cc72266973285fa87e198bf9b504a7beb61f48cb6cbaf18d578`.
The first source receipt was superseded after relocation away from scratch
synchronization that exhausted disk; identical weight bytes passed full rehash
at both paths. FBM must refuse to
load or transform a base until the exact local weight digest, all other file
digests, revision, license/notices, tokenizer/processor and runtime versions
are verified and written to a manifest. Preserve immutable licensed source
bytes and hash-linked working derivatives; copying does not remove upstream
lineage, license duties or inherited behavior.

**Current replay state (2026-09-30):** Workspace maintenance removed that
complete 24 GB snapshot. The currently visible partial temporary downloads
are not source artifacts. The historical receipts remain useful evidence of
past verification, but a new run must reacquire and rehash all eight files at
its own durable path. The 24 GB checkpoint and `google/gemma-4-12B` name
refer to this same pretrained, non-IT release. It is not an earlier neutral
checkpoint. Its Apache license permits derivatives under license conditions;
neither licensing nor absence of instruction tuning proves the source is free
of Google training influence. The machine-readable
[MFM replay seed](../../training/fbm/base_assembly/gemma4_12b_mfm_v1.seed.json)
records the exact input manifest, ordered gates and unimplemented steps. It
has no model-quality or production authority.
At replay, download through a separate staging directory. Hugging Face
`--local-dir` may place `.cache/huggingface` beside the model files, while the
MFM verifier accepts exactly eight regular files and no nested entries.
Move only those eight files into a fresh snapshot on the same filesystem, then
run the verifier. A partial move or extra cache entry is a hard failure.
The shared foundation branch now supplies offline clone, one-tensor
replacement staging, derivative materialization and hash verification CLIs.
The materializer requires a stage receipt and checks the exclusive named
tensor diff. These tools track ancestry and reject a pristine derivative. Their tests use
tiny synthetic checkpoints; no 24 GB candidate tensor has been selected,
changed or behaviorally measured. MFM now has a prepared-base public
diagnostic runner and paired scorer; their small fixture tests are not an
actual 12B run. The replay seed blocks specialist training until an actual
derivative passes paired causal and competence checks.
The V1 trainer now has CPU-tested data, processor and training entrypoints for
a separate formation decoder. Its exact-base processor pass, BF16 gradient
step, restart, full fit and independent entity qualification remain unmeasured.
The existing 49,819-case owner-authorized synthetic mixture is available for
training-only admission through `--curriculum-manifest`. It supplies neither
independent development results nor sealed FINAL qualification.

## Reproducible procedure

1. **Inventory before editing.** Under the pinned revision, identify and
   hash the configuration, safe tensor keys/shapes/dtypes/ties, text trunk,
   modality patch/projection path, input/output embeddings, tokenizer and
   processors. Record activation interfaces and a load/generation receipt.
   A tensor's size or name does not prove its behavioral function.
2. **Establish untouched behavior.** Freeze independent cases for general
   competence and each role's authority boundaries, including long history,
   competing evidence, unprompted autobiography, wrong person, unsupported
   certainty, generic assistant voice, sycophancy, refusal/obedience drift,
   correction/deletion, modality conflicts and attempted instruction injection.
   Record prompts, decoding, artifacts and outcomes against the untouched base.
3. **Localize candidate influences.** Use matched inputs, activation patching,
   ablation and controlled interventions to connect a proposed layer/head/
   neuron/direction/adapter or policy surface to a specific observed failure.
   Distinguish base behavior from prompt, tokenizer, loader, retrieval, private
   adapter and deterministic gate effects. Register the measured causal result,
   uncertainty and untouched-versus-edited competence tradeoff.
4. **Specialize only justified working copies.** Prefer role-specific learned
   components, structured outputs and enforceable authority gates. Prune,
   reweight, add or replace a base parameter group only when an intervention
   improves independent role tests without unacceptable capability loss.
   Store operation, exact source and result hashes, changed tensors, data
   provenance, seed/config, reproducible script and rollback path. A shared
   edit needs cross-role requalification; a role-local edit stays local. A
   written policy is a runtime control, not proof that weights lost a trait.
5. **Assemble per person.** Build each role's distinct learned state from
   authorized instance evidence. Keep private corpora, adapters/deltas,
   optimizer states, indexes, memory and backups in separate instance custody.
   Never merge a user's private gradient or data into the distributed common
   base. Bind all components to upstream and transform manifests.
6. **Qualify the assembled entity.** Compare untouched base, edited base,
   each role with and without its personal component, wrong-person swaps and
   the complete governed loop on independent held-out cases. Record every
   critical inherited-behavior override, privacy failure and competence
   regression. Promotion requires the registered acceptance gates and no
   critical failure on the qualified suite. A finite suite cannot certify a
   behavior-free or influence-free foundation.

| Personal role | What FBM specializes | Foreign behavior to probe | Authority boundary |
| --- | --- | --- | --- |
| Identity/personality | Source-bound tendencies, voice, values and uncertainty for the source person | Generic assistant persona, invented personal history, source-person flattening | Personal evidence and the qualified personality model govern identity; base prose is no evidence. |
| Memory Formation Model (MFM) | Experience selection, fact/belief/episode proposals, time, subject and correction handling | Fictional memories, unsupported promotion, merged subjects, deletion echoes | MFM proposes; registered evidence and deterministic Claim gate decide canonical writes. |
| Host model | Host-specific preferences, permissions, goals and changes over time | Conflating host with source person or following an unauthenticated preference | Authorized host evidence and current permissions; no inference grants consent. |
| Relationship model | Longitudinal relations and interaction outcomes among distinct parties | Generic intimacy scripts, fabricated reciprocity, cross-person leakage | Subject-bound evidence, permissions and governed updates. |
| Assistant-self model | Own experience, uncertainty, commitments and development | Claiming the base model's autobiography or substituting generic helpfulness for learned self-state | Distinct self-history; revisions require outcome evidence and governed authority. |

All five roles share one licensed **source** and may have different derived
working copies. They are not interchangeable fine-tunes. Some useful general
representations and unwanted defaults may be distributed through the same
weights; no reliable universal split into pure "base weights" and removable
"personality weights" is known. Keep decisions evidence-driven. Private data
must be opened only after a pinned safe local load and no-egress runtime check;
an ordinary third-party GPU does not meet that boundary by itself. This
procedure is an FBM process seed, not eligible model-quality gold.

The MFM recipe must initialize and train its own formation-specific parameters
against an explicit formation objective and attach them to governed memory.
FBM must measure their causal effect on unseen histories against untouched,
edited and personal-data-fine-tuned base controls. Serving a fine-tuned Gemma
chatbot as the final MFM would repeat the obsolete personality Phase 2 pattern.
The licensed base is shared ancestry, not the whole personal model.

## Observed source anatomy; specialization status

The verified source has 677 BF16 tensors, 11,959,730,224 elements, a
2,013,265,920-byte tied token embedding, 48 language layers (40 local, eight
global), a shared decoder, 10 vision-named tensors and one audio projection.
The latter two are input capabilities, not detachable third-party personas.
The tokenizer contains turn/tool/thinking markers, and the default generation
config samples; neither establishes a causal behavioral failure. No cloned
working copy, role-specific tensor edit, before/after inference, MFM fit or
qualification exists yet. The current 32 GB workspace cannot retain a second
full 23.9 GB weight file next to the verified source. FBM must preserve these
observed limitations in its seed lineage and materialize/measure derivatives
on sufficient storage and compute before promoting them.
