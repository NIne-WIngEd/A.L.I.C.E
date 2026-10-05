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
  features; reuses unchanged states and exports changed admitted sources
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
view and corrected teacher fit now completed; all generated controls remain invalid (see completed ledger below). Original core,
bank, training/code and failed weights stay reproducible. Full longitudinal,
native modality, calibrated uncertainty, governed consumer/outcome learning and
independent second-host qualification remain required by the full build plan.

Actual issued-corpus correction: job 576713 passed all 49 runtime tests then
stopped at a roster assertion before loading Gemma or exporting new features.
The initial preparation wrongly assumed every public fixture was admitted.
Read-only re-admission of the exact pinned corpus establishes 432 train / 9
development, with seed06/11 admitted and seed18 absent. The 25 base cases
include a separate source-person revocation case and nine deterministic
development cases; 416 longitudinal cases complete the roster. All 441 cases
were audited: only seed06/11 source inputs change, and the corrected complete
inputs have zero unresolved existing-reference gaps. A strict intersection of
the explicit augmentation mapping and admitted roster therefore reuses **439**
states and exports **two**, preserving all 441 rows. No missing fixture is
fabricated, no source is dropped, and no failure becomes a model pass. The
initial fixture-only tests missed this producer-boundary assumption; a subset
regression and the actual-corpus audit now cover it. Source job 576713 is retained
as failed preparation; neither new features nor fitted weights came from it.

## Actual decision-model construction

The new `FormationDecisionSpecialist` inherits the original complete model and
scales only its freshly initialized token/position embeddings by the causal
0.02 intervention. A parameter comparison confirms every other initialization
is unchanged. No source lexical weights or language head are imported.

`formation_source_view.py` re-admits the original corpus, verifies the external
new-view producer pin, every input/target/source feature/code/runtime binding,
new patches, complete lengths, derived stress maxima and zero unresolved input
identities. Training rejects a different or incomplete view before gradients.
`train_v16_decision_specialist.py` uses all 432 training cases with original
FP32 geometry, seed, learning rate, 16-case accumulation and two epochs. Every
development case remains gradient-free. This is a matched construction
comparison, not a claim that 441 single-author examples suffice for full MFM.

A separate actual-runtime 16-case stress probe includes all admitted augmented
inputs and new longest source/target/joint cases, takes three optimizer steps,
reloads the step-two checkpoint and compares the next weights and Adam moments.
Its externally published receipt is required by the fit, and the initial fit
seed must equal the probe's seed. Full epoch checkpoints retain sampler/RNG and
optimizer state; a long session can stop at an optimizer boundary for resume.

`run_v16_decision_diagnostic.py` decodes every diagnostic development case for
trained and seeded controls without target tokens, using the same full budget.
It reuses the existing incremental decoder under explicit FP32/disabled
autocast and checks the first 32 actual greedy prefix logits and argmaxes per
case/control against reference prefix computation. This bound is recorded,
not presented as exhaustive whole-sequence cache qualification. The receipt
names the actual device, dtype and runtime. Invalid raw completions remain
invalid. Valid outputs compile strictly and pass the original source grounding
check; exact teacher-decision agreement is a separate field, not truth.

All 58 relevant local tests pass, including real numerical cache comparison,
next-step weights/moments after restart, admission-before-read, resealed target
mismatch rejection and faithful-paraphrase disagreement retention. Independent
read-only review found no further construction blocker; a missing diagnostic
device/dtype receipt field was corrected. Actual source export, new stress/
restart and the complete teacher fit now have verified runtime receipts below;
all generated controls remain invalid and semantic capability is unproved.
Full-role longitudinal and native-media coverage, independent judgments,
meaning-preserving evaluation, learned memory-to-judgment/outcome behavior and
both consumers remain required. This teacher candidate cannot activate MFM.

### Completed corrected construction and remaining language failure

Source job **576714 completed**: all 441 cases retained, 439 original states
reused, two corrected inputs exported, and zero remaining bounded reference
gaps. Source-view seal: `0c10f71e4af006c159e7995d79d80ac6a8a7ae76fea98315faa719b04ca8fd60`.
The public receipt is an aggregate projection; the protected original remains
the complete row-level authority. All 50 source-runtime tests passed.

Decision job **576715 completed, exit 0**, including 58 actual-runtime tests,
the full 16-case stress/restart proof (zero next-weight and Adam-moment
differences), all 432 train cases for two epochs / 54 updates, and all nine
development completions in both matched controls. The fit took 4,387.6 seconds;
final training-group loss was 3.1232 and development gold-prefix loss 4.2142
versus seeded 12.5750. The original sealed probe, fit/run and diagnostic receipts
are retained in `docs/receipts`; original failed weights and raw outputs remain
in protected custody.

**All nine trained and all nine seeded outputs are invalid without EOS at
2,048 tokens.** The corrected model starts JSON then breaks ordinary actions
and references before repetition. Initial malformed prefixes are inside each
actual 32-step reference/cache comparison (argmax equal, maximum differences
about 2.9e-6–3.8e-6). The cache therefore does not explain that initial failure.
Completed construction and improved teacher loss do not establish useful MFM.

Read-only job **576724** (`4b37c6b09e88a69dd50013133c98a37a26c270bf`) checks
the whole admitted corpus, measures positive target corpus frequency and
original input-token membership, and stratifies the same full gold-prefix
forward passes by field and frequency (0 / 1–9 / 10+). It retains the original
validity separately from exact teacher divergence. Unseen positive labels
are not tokenizer OOVs and still receive negative softmax gradients. Source
membership is only a proxy. Matched, all-BOS-prefix and zero-source observations
are bounded sensitivity tests, not clean semantic counterfactuals. Twelve
targeted tests pass locally and in the actual isolated runtime. No optimizer,
new weights, new architecture or repaired completion is produced by this job.

The next construction choice depends on whether ordinary trained fields are
also poorly learned, whether errors concentrate in unseen/rare identities,
and whether correct teacher prefixes conceal generation collapse. Do not
blindly add epochs, pointer generation, EOS weighting, grammar repair or the
byte interface. The byte proposal remains **ON HOLD**.

Primary research checked before this decision: [pointer generation and its
official implementation](https://github.com/abisee/pointer-generator/blob/master/model.py)
mix learned vocabulary generation with source-position attention mass; its
[paper](https://aclanthology.org/P17-1099/) treats repetition coverage separately.
This is an option for source identity, not proof of our cause. The
[pretrained seq2seq study](https://aclanthology.org/2020.tacl-1.18.pdf) compares
initialization routes; even its pretrained-encoder/random-decoder baseline
shares pretrained embeddings. Its task/data assumptions cannot be applied
directly to our independently random lexical head. [MemReader](https://arxiv.org/html/2604.07877v2)
explicitly builds its extraction/action models on Qwen3-0.6B/4B (§4.1–4.2),
so its memory teaching assumes existing language competence. It keeps extraction
and contextual memory actions in scope; neither its reported
results nor [copy/recall steering of instruction-tuned models](https://arxiv.org/abs/2601.12075)
provides scratch language competence or MFM authority. Frontier-watch exact
head `3bea3318904ec2881d3f981d46a5b8d5bca91ba7` reinforces governed memory
programming and verifier-owned outcome claims, without prescribing this neural
repair. Full native modality, longitudinal decisions, independent semantic
evidence and real consumer/outcome effects remain required.

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

### Language diagnosis completed: 576724

Completed in 4m52s, exit0, original seal
`4c841f1afbd630db067290b496198a81fe8c0dab14858ae1a205b533fde149c9`.
All441 corrected targets have 175,357 training tokens over582 positive token
types; development has365 types,146 never-positive in training. Of1353 dev
reference tokens,306 are unseen positive labels and270 ofthose occur in the
original encoded source. In all measured unseen/rare frequency strata,
top1 accuracy is zero. This is an open identity/lexical-alignment gap, not OOV.

It is also insufficient to explain the broad failure: all9 retained completions
first differ from the teacher at token7, whose expected common action token
has495 corpus occurrences. With correct teacher prefixes, common keys are only
44.6–52.5% accurate on3 training and3 development cases; EOS is never top1,
with probability about0.004–0.007. Matched loss3.79–4.20 increases to6.33–6.60
without source states and10.25–10.85 with all-BOS teacher input. Source states
affect the model; those artificial controls do not prove correct decisions.
The earliest exact divergence alone still does not locate first malformed text.

The supported design diagnosis is broad incomplete sequence learning plus an
additional sparse/no-positive lexical identity channel. A pointer, grammar or
EOS-only patch cannot establish the missing common-field generation. The prior
two-epoch fit was a matched numerical/contract construction comparison, not a
defensible full training budget. Any continuation must bind its changed recipe,
parent checkpoint/optimizer/RNG and full corpus; record train-only learning
progress, finite budget and stop/failure interpretation; retain target-free
generation and the unresolved open-identity/full-consumer requirements. Do not
declare this analysis a semantic MFM pass or repeatedly extend a flat run.

### Bounded full-corpus learning continuation: implementation plan

Goal: establish or falsify further basic sequence learning in the selected
full-role model before introducing a different lexical/copy architecture.
The owner authorized autonomous continuation; execute inline. This is a
learning-budget baseline supported by measured train/common-field underlearning,
not a selected open-identity repair or full-capability claim.

Keep all432 training cases, frozen source/view, full targets, FP32 geometry,
learning rate1e-4, accumulation16 and original global seed/shuffle unchanged.
Restore completed parent checkpoint54: epoch1,next_case432,step54, exact weights,
Adam moments and RNG. Begin globalepoch2 and globalstep55. Create a distinct
child run binding original component/run/checkpoint, all code, extra budget and
stopping rule; never modify/relabel the completed parent manifest.

Add at most six complete epochs /162 updates. Evaluate all432 training cases
at baseline and after every two additional epochs. The stopping metric is
equal-case mean of each case's token-mean CE, matching the optimized objective;
report token-weighted CE separately. Two consecutive intervals improving by
less than1% stop as stalled. The maximum stops as budget exhausted. Neither is
convergence, learned memory or proof against the architecture. Record full
train field/frequency/EOS measurements on the three already selected augmented/
stress examples, with no development-based selection. Keep development target
loss and all9 trained/seeded target-free completions as end diagnostics.

- [x] Add `scripts/mfm/continue_v16_decision_learning.py` and focused tests for
  admission-before-read, exact completed parent/cursor/external checkpoint pin,
  restored next-step weights/moments/RNG, equal-case versus token-weighted
  reporting, plateau versus finite-cap status, and declared resume boundaries.
- [x] Add a separately bound continuation diagnostic using the original FP32
  reference-checked generation and strict grounding; retain invalid output and
  existing no-write boundaries. A deliberate finite stop is not a completed
  eight-epoch fit and must never be mislabeled as one.
- [x] Verify targeted tests and independent review, then package immutable
  code and submit one protected Magnolia CPU12/64GiB job as mxrayan. Preserve
  all original artifacts and checkpoint interruption state.
- [x] Record actual results or pending job in existing MFM/FBM/Graphify records.
  Do not automatically schedule another budget increase. If familiar sequence
  learning progresses but unseen identities remain unsupported, use that
  measured state to decide the source-identity/lexical route. If it stalls,
  reassess the learning interface/data instead of repeating this job.

### Bounded construction submitted: 576741

Magnolia job576741 uses immutable `a4fd0c5fcf34755299fe33907793e7262e137d9d`
under mxrayan, protected CPU12/64GiB,12h allocation and8h train-session limit.
It restores the externally pinned completed parent54 weights/Adam/RNG, creates
a distinct child manifest and starts globalepoch2/update55. Maximum162 new
updates over six complete432-case epochs; complete equal-case trainCE determines
the declared insufficient-progress stop. Token-weightedCE is separate.

Twenty-one targeted tests pass, including actual next-step weights/moments/RNG
identity and interrupted-child/final-weight identity. Independent read-only
review found no remaining launch blocker; session seconds are sealed and the
actual incremental-decoder hash is bound and checked. Tests establish machinery,
not learned competence. The child uses distinct run/component schemas, so a
deliberate plateau is never reported as completing the maximum eight epochs.
Its strict diagnostic byte-reuses the original pinned parent9 controls and
generates all9child/9original-seed controls at the same2048-token FP32 settings.
Original malformed outputs and the146 never-positive development token types
remain unresolved. No automatic second budget increase or architecture choice
follows from this run. Await actual full-train progress, terminal receipt and
raw generated behavior before selecting the next learning/lexical route.

Runtime snapshot:576741 RUNNING after all21 actual-runtime tests (10.892s).
Initial Slurm stdout includes tiny test-fixture progress; those one-case rows
are not432-case MFM training. Real progress is in the separate protected
`receipts/rayan-mfm-budget-fit-576741.stdout` and run progress receipts.
No actual full-corpus child update or behavioral result is asserted yet.
