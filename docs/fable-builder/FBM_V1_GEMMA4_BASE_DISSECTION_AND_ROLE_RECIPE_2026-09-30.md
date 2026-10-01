# FBM V1: shared Gemma 4 base dissection and role specialization

**Owner-local decision:** 2026-09-29, revised 2026-09-30. **Status:** reusable construction
procedure with a verified source inventory and owner-reported MFM role clone; no personal model or
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
complete 24 GB snapshot. The owner subsequently downloaded the exact eight
files on Magnolia at `$HOME/rayan-compute/mfm/gemma4-12b-023679ed352de9bb66cc873c9009ce3482585c08/source`.
Their pasted `sha256sum -c` output reports all eight pinned checks passing on
a CPU node. The owner's pasted Magnolia job `576486` stdout reports a
completed CPU source receipt (`ad081e28519961d72181815b7a74c65b42238e2c2ab0a65065acbc8e2be606df`)
and MFM role-clone receipt (`54484dd523c396648fdf7069abe219c3f8f7c1fdcfec026cda1038810d4d7ac7`)
at separate paths, with `COMPLETED 0:0` on `node005.cluster` and empty stderr.
The clone digest appears twice. The owner then ran the foundation
`read_receipt` check on Magnolia: self-digests, parent source digest,
`role=mfm`, `qualification=unqualified`, and the eight file rows matched the
pinned manifest. Full receipt JSON and the submitted batch script were not
transferred for independent inspection in this checkout. Rehash the clone at
point of use. The 24 GB checkpoint and `google/gemma-4-12B` name
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
The shared foundation branch supplies offline source verification, role clone,
role-base verification, and optional one-tensor replacement tools. Its
`verify-role-base` admits the exact clone as the V1 MFM starting base; it
does not claim that the clone is a trained or qualified MFM. A tensor edit is
**not** required before formation-specialist training. Use the intervention
path only after a specific inherited-behavior failure is observed in the
assembled MFM, causally diagnosed, and a candidate repair is justified. The
intervention tools have tiny synthetic tests, not a 24 GB behavior result.
The MFM public diagnostic runner and paired scorer remain optional tools,
not an edit-before-training gate.
The historical v1.5 specialist CLI has CPU-tested mechanics for its narrower
target schema. The V1 seed keeps its CPU and bounded `--probe-only` route as
optional feasibility work; a corpus manifest swap does not upgrade it. The
MFM branch's [1.6 admission contract](https://github.com/NIne-WIngEd/A.L.I.C.E/blob/a5b04504c227347c5b56e6555d7aa7cc4ae86af4/docs/MFM_FORMATION_CONTRACT_V1_6_ADMISSION_2026-09-30.md)
and v1.6 codec, signed full-fit trainer, runner and separate custodian assessor
for one-step or full-fit artifacts are now published with CPU/static tests.
Its narrow P2 candidate bridge stages one owner host profile claim and rejects
richer 1.6 semantics; the full persistent receipt and promotion-time source
recheck remain open. There is still no admitted full-role corpus or real
exact-base 1.6 processor pass, BF16 GPU step, restart, full fit or independent
entity qualification.
The existing 49,819-case owner-authorized synthetic mixture is available for
training-only admission through `--curriculum-manifest`. It supplies neither
independent development results nor sealed FINAL qualification.

## Full-role readiness before full fit

The [hash-bound training coverage audit](../../training/fbm/base_assembly/mfm_v1_training_role_coverage_20260930.json)
has SHA-256 `0f198706d5158f830c79dd3a6a891449025f453c0976f9fa21377c8cd5085cec`
and binds mixture manifest
`60fa44a5be62c8d2f0daf508e543dc1a41c34562dc0c4d399e26c9cabb772e21`.
Its 43,819 multisource rows yield 420,367 proposals, all in the host domain
and only four proposal kinds: goal, host observation, behavior pattern and
uncertainty. The other 6,000 rows contain ten repeated text scenario families.
Across all 426,367 proposals, confidence and uncertainty references are null,
contradiction links are empty, and sensitivity hints are null. Evidence is
text or structured data only. No episode or decision-rationale proposal kind,
source-person domain, or media beyond text/structured is present. The small
templated tier does include 600 deletion requests and 600 relationship norms;
those counts do not fill correction, revocation, relationship and mission
coverage for the full role.

The MFM branch at public commit
`a5b04504c227347c5b56e6555d7aa7cc4ae86af4` has a **diagnostic-only**
amended 1.6 authoring seed:
15 fictional train and six diagnostic development cases. The seed SHA-256 is
`8e967ee6494f2c5ec47ab602bad213df8c07c53b1abe9a9aff950f19c0e78e22`;
its aggregate audit SHA-256 is
`3df1ba7d61db6e4a09106ac80c7fc95353e8b31579fbe14d7ee3226ba695d99d`.
It explicitly marks ten formation dimensions as present, negative or unknown
and exercises the 1.6 codec. Two source-only assistant QA passes prompted
target corrections, but both used the same provider. The pack remains one
author's text-only fiction with shared generator lineage, no authenticated
rights, two real independent target reviews, independent DEV or sealed FINAL.
It cannot admit a paid full fit or support a capability claim. The active
FBM seed pins that exact MFM commit and the two diagnostic file hashes.

The current curriculum is **blocked as the sole source for full paid MFM
training**. FBM must first version the target schema and show coverage of the
full formation role, then obtain additional rights-cleared histories with
independently adjudicated development and sealed FINAL targets. The held-out
families must cover sensitive, cross-person, longitudinal, multimodal,
uncertainty, contradiction, correction and revocation cases with person,
source and generator separation. The full-role trainer, inference runner and
evaluator must all consume the same versioned target semantics. Source
verification, role cloning and optional 1.5 CPU/`--probe-only` feasibility work
may proceed now. The full 1.6 CPU processor receipt and bounded backward pass
require the admitted corpus and matching 1.6 code path. These checks establish
custody and hardware fit, not full-role learning or product qualification.

MFM commit `412ae32a` adds a source-only 1.6 review packet builder for the
pinned Multi-Source inventory. Its in-memory smoke covered 480 simulated hosts
and 1,920 cumulative day windows. Packets exclude simulator truth and prior
QA/targets, preserve source-file hashes and host/generator lineage, and are
marked `candidate_unreviewed`. They are **not** rights-cleared or adjudicated
gold; all hosts share one generator family. The same commit publishes a
Magnolia CPU Slurm script to run the 21-case public 1.6 processor diagnostic
against the verified role clone. Owner-reported job `576488` ran on
`node005.cluster` and `FAILED 1:0` after `00:03:35` in udocker P2. The runtime
printed Torch `2.7.1+cu118` and Transformers `5.17.0`, then
`AutoProcessor.from_pretrained` failed because `Gemma4UnifiedProcessor`
required PIL, which was unavailable in that container. The printed
job-specific receipt path is an intended output; no successful processor
receipt was observed. The [failure trace](traces/FBM_TRACE_20260930_MFM_V16_PUBLIC_CPU_576488_FAILURE.jsonl)
preserves this dependency lesson without changing the verified job `576486`
source/clone metadata. Published MFM commit `079c010c` (tree
`11aa22b85c75dcc8f097609800780ea2fddb8860`) now pins a CPython 3.11
Pillow 11.3.0 wheel with SHA-256
`106064daa23a745510dabce1d84f29137a37224831d88eb4ce94bb187b1d7e5f`.
The launcher verifies those bytes, installs the wheel offline into a fresh
MFM-only job directory, and actually loads the local Gemma 4 `AutoProcessor`
inside udocker before the 23.9 GB clone rehash. This
[published repair procedure](traces/FBM_TRACE_20260930_MFM_V16_PILLOW_RUNTIME_FIX_PUBLISHED.jsonl)
was unrun when published. Owner-reported job `576508` then used `079c010c`:
the exact wheel SHA-256 checked OK and udocker installed Pillow 11.3.0, but
the job `FAILED 1:0` on `node005.cluster` in `00:00:42` when early
`AutoProcessor.from_pretrained` imported `Gemma4UnifiedProcessor`. Its image
module imports `torchvision.transforms.v2` at module load, and the target
container lacked `torchvision`. This import path is reached even with the
text-only public diagnostic. The
[job 576508 trace](traces/FBM_TRACE_20260930_MFM_V16_PUBLIC_CPU_576508_TORCHVISION_FAILURE.jsonl)
records the intended receipt path but no successful processor result.

MFM commit `e93e098a` (tree `2bf4630e71c97cb925da3c082ee055864f182d5f`)
publishes the next isolated dependency repair. The
[pinned Transformers source](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/gemma4_unified/image_processing_gemma4_unified.py)
and [official PyTorch CUDA 11.8 pairing](https://pytorch.org/get-started/previous-versions/#v271)
show why container Torch `2.7.1+cu118` needs matching TorchVision
`0.22.1+cu118`. The launcher verifies the official wheel's SHA-256
`6dd3d825fb4a75eae887665d1da812a360d69273118bfa17616c836bfb466627`,
installs it with Pillow without replacing Torch, asserts both imports inside
udocker, and loads the local processor before clone rehash. The
[matching-runtime procedure trace](traces/FBM_TRACE_20260930_MFM_V16_TORCHVISION_RUNTIME_FIX_PUBLISHED.jsonl)
was unrun at publication. The owner then reported Magnolia job `576509`
`COMPLETED 0:0` on MFM `e93e098a`: the target runtime loaded
`Gemma4UnifiedProcessor` and processed the 21-case public synthetic text
fixture, with stdout reporting a preflight digest. See the
[bounded success trace](traces/FBM_TRACE_20260930_MFM_V16_PUBLIC_CPU_576509_TEXT_SUCCESS.jsonl).
The actual receipt JSON has not been independently inspected here. The fixture
has 38 text references and no media; actual full-role media needs separate
decoder and ffmpeg/ffprobe tests. Jobs `576488` and `576508` remain failures
and do not change source/clone custody from `576486`. Job `576509` supplies no
model forward, training or admitted full-role processor qualification.
Do not claim capability from the present training-only mixture.
The published signed-review verifier tests record format and signatures; an
independent steward must authenticate the reviewers, rights issuer, blind
review and original source rights. The shared distributable MFM gate remains
separate from private per-user adaptation consent.

## Reproducible procedure

1. **Verify and clone the source.** Rehash every pinned publisher file,
   preserve attribution, seal a role-local clone, and rehash it at trainer use.
   An inventory can record tensor keys/shapes/dtypes/ties and processor paths;
   structure alone does not confer MFM authority or prove behavioral causes.
2. **Admit full-role data and code.** Version 1.6 source and target records,
   authenticate rights and two independent reviews, and freeze disjoint DEV
   plus sealed FINAL. Build one matching codec, trainer, inference and evaluator
   path. The 1.5 CPU and bounded hardware checks remain optional feasibility.
3. **Measure the exact 1.6 route.** Complete the corpus-wide CPU processor
   pass on Magnolia or equivalent, then a bounded backward/restart probe on
   the intended GPU topology. Neither receipt proves role quality.
4. **Train the MFM role after readiness passes.** Initialize separate
   first-party formation parameters from sufficiently broad authorized data.
   Bind structured proposals to registered evidence and the deterministic
   Claim gate. The clone supplies inherited representations; new learned
   parameters must causally perform formation on independent histories.
5. **Assemble per person.** Build each role's distinct learned state from
   authorized instance evidence. Keep private corpora, adapters/deltas,
   optimizer states, indexes, memory and backups in separate instance custody.
   Never merge a user's private gradient or data into the distributed common
   base. Bind all components to upstream and transform manifests.
6. **Qualify and diagnose.** Compare the MFM role clone with trained versus
   seeded-untrained specialist weights under matched inputs and decoding,
   each role with and without its personal component, wrong-person swaps and
   the complete governed loop on independent held-out cases. Source-only
   Gemma output is an optional descriptive reference, not a measured formation
   control. Include long histories, competing evidence, unsupported certainty, correction and
   deletion. If a specific Gemma influence causes failure, distinguish it
   from prompt, tokenizer, loader, retrieval, learned-component and gate
   effects. Record the cases, localization evidence and uncertainty.
7. **Repair only an observed failure.** Prefer role-specific training,
   structured outputs and enforceable authority gates. Change base parameters
   only when a controlled intervention improves independent role tests
   without unacceptable capability loss. Store parent/result hashes, changed
   tensors, data provenance, script and rollback. A shared edit requires
   cross-role requalification. A policy is a runtime control, not evidence
   that weights lost a trait.
8. **Requalify any repaired assembly.** Include an edited-base control only
   when an edit was attempted. Record every critical inherited-behavior
   override, privacy failure and competence regression. Promotion requires
   the registered acceptance gates and no critical failure on the qualified
   suite. A finite suite cannot certify an influence-free foundation.

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
FBM must measure their causal effect on unseen histories against a matched
seeded-untrained specialist. A source-only or fine-tuned-base diagnostic may
help characterize inherited behavior, and an edited-base control is relevant
only when one exists. Serving a fine-tuned Gemma chatbot as the final MFM
would repeat the obsolete personality Phase 2 pattern.
The licensed base is shared ancestry, not the whole personal model.

## Observed source anatomy; specialization status

The verified source has 677 BF16 tensors, 11,959,730,224 elements, a
2,013,265,920-byte tied token embedding, 48 language layers (40 local, eight
global), a shared decoder, 10 vision-named tensors and one audio projection.
The latter two are input capabilities, not detachable third-party personas.
The tokenizer contains turn/tool/thinking markers, and the default generation
config samples; neither establishes a causal behavioral failure. The owner's
Magnolia log reports an exact MFM role-clone custody run and the owner-checked
receipt metadata matches the pinned source. No role-specific tensor edit, assembled MFM inference,
MFM fit or qualification exists yet. FBM must preserve these observed limits
in its seed lineage and measure the assembled MFM before claiming role quality.
