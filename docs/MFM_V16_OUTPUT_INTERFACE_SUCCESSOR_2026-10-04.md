# MFM output-interface successor: design and implementation ledger

**Current decision: the byte-interface proposal below is ON HOLD and is not
implemented.** The owner's test-design challenge triggered an independent
producer/objective/evaluator audit. It found missing prior-memory input,
unobservable local-ID supervision and exact agreement misreported as falsity.
Tokenization cannot correct those defects. Keep the selected original-vocabulary
decoder and correct the complete input/learning/evaluation contract first.

## Current correction and execution plan

- Seed 06 requires `calendar-meeting-claim`; seed 11 requires
  `current-study-preference` / `quiet-study-preference`; seed 18 requires
  `archive-interview-date`. None is present in those context/source bytes.
  Named references in seed 05/12/14 and the longitudinal generator do occur in
  their sources. Do not generalize the missing-input finding to every row.
- The caller case ID is absent from source encoding, but some targets require
  its arbitrary local-ID spelling. Renaming seed 02's case/proposal/value IDs
  preserves the exact model-input digest and semantic score while changing CE.
- A faithful seed 02 paraphrase or expanded valid anchor passes grounding but
  yields exact FP=1/FN=1; the qualifier previously called either `false_memory`.
  That is not evidence of factual falsity. It did not cause the original cached
  no-EOS rejections. [HaluMem's operation-level evaluation](https://arxiv.org/html/2511.03506v2)
  supports separately evaluating extraction, updating and later use; it supplies
  no truth oracle for our teacher targets.
- `formation_decision_v16.py` is a distinct learned-decision wire format. All
  semantic content, source anchors, time/subject/uncertainty status/confidence,
  external correction/deletion/contradiction refs, episode, relationship,
  mission/workspace and dispositions remain learned. A strict compiler allocates
  only new artifact-local proposal/value/local uncertainty IDs. Uncertainty
  identities stay literal unless explicitly declared newly local by the
  fictional authoring recipe. Shared local uncertainty groups and local
  inter-proposal/value references use typed indices. Compilation repairs no
  content or permission and grants no write. An initial presence-only draft
  lost existing/shared uncertainty identities; independent review reproduced
  this defect and the draft was corrected before execution.
- `v16_prior_memory_inputs.py` explicitly authors fictional prior-memory input
  for the three incomplete teaching cases. Multiple candidate stored records
  provide resolvable IDs without expected actions or output labels. Stored
  claims can be wrong/overgeneralized. The target-independent structured snapshot
  is scoped context, not new experience, recovered real history, independent
  truth or a production prior-memory service. Current policies and any eventual
  memory write still belong to the independent gate.
- `prepare_v16_contract_successor.py` re-admits the exact original corpus,
  preflight, failed component and complete bank; processes all 441 full sources
  and decision targets; checks unchanged input IDs/masks against original
  features; reuses 438 unchanged states and exports only three changed sources
  through the fully reverified frozen Gemma role clone. Its new complete source
  view binds all input/target hashes, code, ancestry, lengths, stress cases and
  remaining bounded-literal reference gaps including existing uncertainty.
  Torch/Transformers versions must equal the original export; actual CPU/BF16
  placement and the reverified prepared-base receipt are sealed in the new
  patch and complete view. It is not a relabeled old preflight/bank.
  Remaining gaps require diagnosis before fitting; literal availability does
  not establish entailment. Targets never enter source states.
- The qualifier now labels exact unmatched proposals
  `unreviewed_target_mismatch`, retaining counts and unresolved failing cases,
  and names the metric `exact_target_fields_not_semantic_entailment`. No old
  model failure becomes a pass. No automatic paraphrase/cosine oracle is added.
- Numerical calibration is still supported by the original-seed intervention,
  not sufficient by itself. The next model construction keeps the 262,144-token
  vocabulary, width 768, six layers/twelve heads and full role; a separately
  bound calibrated decoder learns the corrected decision objective on the
  complete admitted source view. Measure actual stress/restart, generation,
  grounding, semantic decisions and counterfactual effects; never interpret
  CE, schema fixtures or broad teacher-prefix swaps as learned memory ability.

Ledger: public root counterexamples independently confirmed; strict complete
decision/compiler and fictional input extension built; 44 relevant local tests
passed initially, including full-field roundtrip across all 21 public cases, local-ID
invariance, malformed-content rejection, source scope, explicit uncertainty,
target-independent index bytes and evaluator mismatch retention. Review added
counterexamples for existing/shared uncertainty, partial-ID matches and runtime
ancestry; all reproduced a failing draft before correction. Runtime source
view and subsequent fitted/generated capability remain pending. Original core,
bank, training/code and failed weights stay reproducible. Full longitudinal,
native modality, calibrated uncertainty, governed consumer/outcome learning and
independent second-host qualification remain required by the full build plan.

## Historical byte-interface proposal (not adopted)

Owner direction: continue the full Alice/Fable MFM build, resolve the design
failure rather than repeat the failed fit, and stop when only external work
remains. This is an unqualified construction candidate, not a release or a
replacement of the full-capability plan.

## Evidence and decision

The complete matched diagnostic from 576702 is sealed at
`a17b610d62887fb8dd371b9a72bee7f75287b8a78cf8f803cf45b81eb625c9c6`:
all nine trained and all nine seeded completions are invalid. Corrected scalar
analysis 576703 is sealed at
`5e42932974ab7c8c083fed21c68b0606e7cbebd44b271f9ea7c2e9df64b5b6a7`.
Its six sampled matched cases have scalar-value token accuracy 24.78–29.74%,
despite the gold prefix; this is neither claim-level accuracy nor independent
evaluation. All 441 targets round-trip correctly. Initial previous-input
copying is saturated, and the original-seed embedding intervention removes it.
The teacher fit partly compensates for a defective numerical interface.

The deeper construction mismatch is giving a fresh decoder the source model's
262,144-token vocabulary without that model's learned lexical weights or
language competence. Only 728 target IDs occur in 199,104 teacher target tokens.
This explains an inefficient learning interface, not a proof that the old
architecture could never learn. Insufficient data, exposure/learning budget,
source-specific grounding and exact copying remain open beyond calibration.

Options considered: calibrate the old BPE interface alone (smallest numerical
repair, but leaves sparse scratch lexical learning); import a pretrained
language decoder (changes the selected independent-formation-weight boundary);
or preserve the selected cross-attention decoder and use a universal byte
output alphabet. Select the third as one justified construction candidate,
with numerical calibration already supported by the causal intervention.
Do not run an all-pairs architecture or hyperparameter tournament.

The [ByT5 paper](https://aclanthology.org/2022.tacl-1.17/) and
[versioned byte tokenizer implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/byt5/tokenization_byt5.py)
establish a 256-byte alphabet as a practical open-text Transformer interface,
and explicitly expose longer sequences and inference cost. This transfers
only the encoding mechanism: no ByT5 weights, task results, denoising objective
or architecture are imported. Unlike the reference tokenizer, invalid UTF-8
must remain a recorded failure; it must never silently discard bytes.
[BART initialization](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/bart/configuration_bart.py)
and the actual-runtime intervention support a 0.02 fresh lexical/position scale
for this tied-head interface. This is not a claim of learned success.

## Construction contract

Keep frozen non-IT Gemma source states, complete native source processing,
768-wide six-layer twelve-head independent formation decoder, all v1.6
proposal/disposition semantics, subjects, time, uncertainty and exact source
anchors. Replace only the fresh output lexical interface with 256 byte IDs
plus PAD/BOS/EOS, initialized at standard deviation 0.02. Original decoder and
export/training modules remain byte-identical so the historical feature bank
can be independently reverified. A new successor preflight binds the exact
bank, original processor preflight, new target lengths/stress cases, codec,
decoder and transitive execution code. It does not relabel an old preflight.

Every target is canonical full JSON encoded as strict UTF-8, with exact
BOS/body/EOS shift. No vocabulary whitelist, grammar mask, repeated-token
penalty, source/target truncation, Gemma language-head fallback or authority
write. Budget overflow rejects the run and requires an explicit larger budget.
Training uses only the admitted training split; development remains diagnostic
teacher material, FINAL remains closed, and product qualification remains false.

Before a full run, measure the complete byte-length corpus, full source/target
stress backward/restart, seed entropy/copying and time. A deterministic KV
inference path must agree with full-prefix logits within measured FP32 tolerance
and produce the same sampled greedy decisions; controls use the same path.
Measure complete generated output, grounding and full semantic teacher agreement,
including dispositions, rather than interpreting CE/JSON validity as MFM.
Source swaps are sensitivity observations with confounded context; do not call
them clean semantic causal evidence. Source-native counterfactual families,
expanded training, independent hosts/modalities and downstream consumer/outcome
learning remain required by the full-capability roadmap.

## Implementation plan and ledger

Execute inline under existing authorization. Preserve original failed artifacts.

- [ ] Add strict reversible byte codec and calibrated independent decoder;
  first reproduce absent interface behavior in tests, then verify exact Unicode,
  literal special-marker text, loss shifts, no truncation and trainability.
- [ ] Add incremental inference with reference parity and cache isolation tests.
- [ ] Add separately sealed successor admission/training/generated diagnostic,
  reusing the existing verified bank and checkpoint mechanics. Reject code,
  feature, corpus, split, recipe or component mismatches. Preserve invalid bytes.
- [ ] Verify targeted regression tests and independent code review; package a
  clean immutable checkout and execute protected Magnolia construction under
  mxrayan, with restart state and an externally pinned recipe.
- [ ] Record actual results and pending jobs, update FBM methods and Graphify
  source navigation, publish branch checkpoints, and stop if only jobs remain.

Full training may proceed after the actual stress measurement; a hardware or
teacher diagnostic is not an independent qualification gate or capability claim.
