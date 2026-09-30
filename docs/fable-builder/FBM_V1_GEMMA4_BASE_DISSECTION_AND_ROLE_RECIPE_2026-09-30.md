# FBM V1: shared Gemma 4 base dissection and role specialization

**Owner-local decision:** 2026-09-29, revised 2026-09-30. **Status:** reusable construction
procedure with a verified source inventory; no role clone, personal model or
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
a CPU node. The foundation verifier has not yet sealed that path or produced
an MFM role clone. The 24 GB checkpoint and `google/gemma-4-12B` name
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
The current v1.5 specialist CLI has CPU-tested data, processor and training
mechanics for its narrower target schema. The V1 seed permits its exact-base
CPU processor pass and bounded `--probe-only` backward/restart check. It
blocks full fit until a versioned full-role target and matching trainer,
inference runner and evaluator exist. A corpus manifest swap does not upgrade
the v1.5 CLI. The MFM branch's [1.6 admission contract](https://github.com/NIne-WIngEd/A.L.I.C.E/blob/df3da321feb56cc33762e38e19f43bbb078a6e36/docs/MFM_FORMATION_CONTRACT_V1_6_ADMISSION_2026-09-30.md)
now has CPU-tested additive semantic checks for source sensitivity, episodes,
relationship counterparts and mission links. It has not migrated the 1.5
target serializer, trainer or inference runner. No exact-base processor pass,
BF16 gradient step, restart, full fit or independent entity qualification has
been measured.
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

The current curriculum is **blocked as the sole source for full paid MFM
training**. FBM must first version the target schema and show coverage of the
full formation role, then obtain additional rights-cleared histories with
independently adjudicated development and sealed FINAL targets. The held-out
families must cover sensitive, cross-person, longitudinal, multimodal,
uncertainty, contradiction, correction and revocation cases with person,
source and generator separation. The full-role trainer, inference runner and
evaluator must all consume the same versioned target semantics. Source
verification, role cloning, CPU
processor admission and a bounded backward/restart hardware probe may proceed
now. Those checks establish custody and fit, not full-role learning or product
qualification. Do not claim capability from the present training-only mixture.

## Reproducible procedure

1. **Verify and clone the source.** Rehash every pinned publisher file,
   preserve attribution, seal a role-local clone, and rehash it at trainer use.
   An inventory can record tensor keys/shapes/dtypes/ties and processor paths;
   structure alone does not confer MFM authority or prove behavioral causes.
2. **Check the data and hardware.** Run a corpus-wide CPU processor pass and
   a bounded backward/restart hardware probe on the verified role clone.
   Version and audit full-role formation targets and independently adjudicated
   development/FINAL histories before approving a full paid fit.
3. **Train the MFM role after readiness passes.** Initialize separate
   first-party formation parameters from sufficiently broad authorized data.
   Bind structured proposals to registered evidence and the deterministic
   Claim gate. The clone supplies inherited representations; new learned
   parameters must causally perform formation on independent histories.
4. **Assemble per person.** Build each role's distinct learned state from
   authorized instance evidence. Keep private corpora, adapters/deltas,
   optimizer states, indexes, memory and backups in separate instance custody.
   Never merge a user's private gradient or data into the distributed common
   base. Bind all components to upstream and transform manifests.
5. **Qualify and diagnose.** Compare the MFM role clone with trained versus
   seeded-untrained specialist weights under matched inputs and decoding,
   each role with and without its personal component, wrong-person swaps and
   the complete governed loop on independent held-out cases. Source-only
   Gemma output is an optional descriptive reference, not a measured formation
   control. Include long histories, competing evidence, unsupported certainty, correction and
   deletion. If a specific Gemma influence causes failure, distinguish it
   from prompt, tokenizer, loader, retrieval, learned-component and gate
   effects. Record the cases, localization evidence and uncertainty.
6. **Repair only an observed failure.** Prefer role-specific training,
   structured outputs and enforceable authority gates. Change base parameters
   only when a controlled intervention improves independent role tests
   without unacceptable capability loss. Store parent/result hashes, changed
   tensors, data provenance, script and rollback. A shared edit requires
   cross-role requalification. A policy is a runtime control, not evidence
   that weights lost a trait.
7. **Requalify any repaired assembly.** Include an edited-base control only
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
config samples; neither establishes a causal behavioral failure. No verified
MFM role clone, role-specific tensor edit, assembled MFM inference, MFM fit or
qualification exists yet. Magnolia has space for a role clone but its CPU
source receipt and clone have not been returned. FBM must preserve these
observed limits in its seed lineage and measure the assembled MFM before
claiming role quality.
