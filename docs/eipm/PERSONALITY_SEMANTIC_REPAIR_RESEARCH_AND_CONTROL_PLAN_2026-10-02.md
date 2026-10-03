# Research before semantic-readout repair: measured failures and bounded candidates

The completed candidate-only experiment diagnoses the supervision prior. The
next comparison tests one researched semantic repair candidate without adopting
it as the personality architecture, modifying Gemma or approving N0. Full ALICE
personality and full Fable v1 FBM remain the purpose; public relation labels
cannot become Alice identity targets.

## Actual observations and competing explanations

Original learning 576600 and independently reproduced diagnostic 576638 give
learned TRAIN 9/56, DEV 1/16 and sentence-unseen DEV 1/14, versus fixed 15/56,
4/16 and 3/14. Destructive source substitutions change most learned decisions;
the scorer is source-dependent on these examples, but correct relational use is
weak. All DEV targets lack TRAIN-positive exposure. Learned choices favor
TRAIN-positive descriptions on 15/16 DEV examples, versus fixed 11/16 and
pool-uniform expectation 0.78125. Candidate counts and small family counts
confound that association. The 49 global depth weights remain nearly uniform.

The independently pinned plan has 1,528 TRAIN candidate slots: 56 positive
slots and 1,472 distractor slots. Each of the 56 TRAIN descriptions is positive
once and appears as a distractor 21 to 33 times. All eight DEV target
descriptions have ZERO TRAIN candidate exposure, positive or negative. Thus the
current evidence does not show that DEV descriptions were taught as negatives.
These are static example counts, not optimizer-weighted counts from 256 draws.
The sealed derived audit is preserved as
`evaluation/eipm/gemma_n0/receipts/public_candidate_static_exposure_576638.json`:
file SHA-256 `c11f1d2cc7fbb7fba790f232bd7cb100f6cc0fc78b6dee19d207f8b04c0729bc`,
canonical receipt `a8c9a7d508b105387975e05b491266feac6d99d4c99e3fe13ddb60d4de40f579`.
Root independently recounted all descriptions from the pinned plan and verified
both seals. Its 1,965 recorded source/candidate links agree with original cache
and closed-package inventories; this metadata cross-check is not a fresh local
tensor-byte rehash. Actual 576638 separately reverified original features.

These observations admit several explanations: an exposure-induced description
prior, incompatible source/description geometry, weak optimization, insufficient
independent contexts, or a poor shared interaction. They do not isolate inherited
Gemma persona, overfitting, missing depth information or a required network size.
Lower target cross entropy and weaker argmax can coexist. Calibration cannot be
treated as semantic repair. The current DEV results are already observed and
must stay exploratory/model-selection evidence.

## Primary research and applicability

Three parallel, read-only research workstreams used nine Exa searches with six
requested results each: 54 requested slots, 53 delivered URL items. Separately
they verified 7, 9 and 9 unique primary studies within their workstreams, with
cross-workstream overlaps. Those counts are not 25 independent studies. Original
methods, evaluation designs and relevant limitations were read; search snippets,
blogs and third-party implementations do not establish the conclusions below.

| Primary evidence | Useful lesson for this failure | Constraint on adoption |
| --- | --- | --- |
| [FewRel](https://aclanthology.org/D18-1514.pdf) | Separate support-shot count from the much larger meta-training corpus. | Our description-only disjoint-family evaluation is zero-shot relation matching, not a canonical support-based FewRel one-shot reproduction. One source per TRAIN family supplies no within-family independent query test. |
| [RE-Matching](https://aclanthology.org/2023.acl-long.369.pdf), [AlignRE](https://aclanthology.org/2024.findings-acl.174.pdf) | Ordered roles, context and description geometry should be matched explicitly and tested with ranking controls. | These methods optimize BERT and often evaluate unseen-only candidate sets. Their scores, hyperparameters, types and aliases do not transfer to fixed Gemma banks or our mixed pools. Existing appended role states already have the complete sentence to their left. |
| [Generation-Augmented Retrieval](https://aclanthology.org/2025.findings-emnlp.974.pdf), [CLIP-Adapter](https://arxiv.org/abs/2110.04544) | A shared residual adjustment above a frozen source can preserve an existing semantic path. | Different embeddings, generated triplets or image/text alignment matter. This is a later comparison hypothesis, not a selected architecture, ratio or suppression proof. |
| [Generalized zero-shot study](https://arxiv.org/abs/1605.04253) | Familiar classes can dominate mixed candidate spaces. | Calibrated stacking requires a seen-membership flag; that flag is forbidden as a model input or ranking penalty here. Familiarity may be diagnostic metadata only. |
| [Partial-input baselines](https://aclanthology.org/S18-2023.pdf), [misleading failures](https://aclanthology.org/P19-1554.pdf) | Fit a partial-input baseline on TRAIN; evaluation-time source removal is a different question. | Baseline success reveals a learnable prior, without proving the full scorer uses only that prior. Failure cannot certify shortcut-free semantics. |
| [Context editing](https://aclanthology.org/2022.naacl-main.350.pdf), [contrast sets](https://aclanthology.org/2020.findings-emnlp.117.pdf) | Hold candidates fixed and independently validate both sources and changed gold. Measure both members correct and justified probability movement. | Our constant/cyclic substitutions retain old labels and are not these semantic pairs. Merely changing the winner is insufficient. Entity reversal cannot invent inverse-relation gold. |
| [Counterfactually augmented data](https://arxiv.org/abs/1909.12434) | Compare grounding/exposure designs with matched sample counts and original-domain retention. | Human coherent edits and independent labels are required. Repeating 56 existing rows is not new semantic coverage; public fabricated facts are not identity evidence. |
| [Calibration](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf), [uncertainty under shift](https://proceedings.neurips.cc/paper_files/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf) | Track NLL/Brier and retained-count risk alongside ranking, correctness and shifted support. | Positive scalar temperature preserves argmax. Smooth confidence, low ECE or entropy alone cannot prove correct judgment or epistemic abstention. |

The strongest immediate lesson is to test training exposure and conditional
matching before capacity, depth selection or longer optimization. No paper
identifies the causal failure of our checkpoint or guarantees a borrowed recipe.

## Precommitted next experiment: freshly fitted candidate-only control

Use only the closed PUBLIC package from 576635 and externally pinned original
576600 plan, export and midpoint. Keep original complete banks immutable; no
publisher model, tokenizer, forward, source snapshot, dependency installation,
private input or FINAL payload is needed. Run under `mxrayan` on Magnolia's
existing CPU Torch 2.7.1 runtime, with four threads.

Use the same `SemanticReadout` geometry, initialization seed and optimizer as
the original run: 49 states, hidden width 3840, learned width 64; AdamW learning
rate 0.0003, weight decay 0.01; seed 20261002, sampler seed 20261003; 256
single-example TRAIN updates. Candidate banks, candidate order, targets and
TRAIN example order remain identical. No family, ID, position or familiarity
flag enters the scorer.

Replace every source with ONE explicitly artificial, detached, all-zero BF16
two-token bank of the declared geometry. Its constant masks contain separate
head and tail positions. It is not a claimed Gemma observation. No per-example
source length, masks, text, entity metadata or ID is retained. Descriptions are
the sole varying semantic input; a global learned query/prior is permitted.
Nominal parameter counts match, but effective degrees of freedom can differ.
Record blockwise gradients and updates; zero gradients in degenerate constant
paths are not automatically an implementation failure.

The sampler is a separate CPU `torch.Generator`, independent of model random
initialization. Reconstruct its sequence in the actual original runtime and
compare its state after 128 draws with the externally pinned original midpoint:
`fcaec4867816532b7b8d139f1b33460606b2bd71d75f621da6aacc961e48f4f4`.
Verify the midpoint's original binding and step. Save all 256 drawn TRAIN indices
and initial/midpoint/final sampler-state digests. This verifies matched exposure
without retraining the original full scorer or calling a different runtime's
sample replay the original run.

Reproduce original full-scorer and fixed metrics before fitting the control.
Preserve the original export unchanged. Record control initialization, loss
history, final per-row logits/ranks/margins, exact reload, candidate-permutation
equivariance and strict independence from source metadata. Recheck complete
bank bytes and package closure after use. Bind receipts to actual source,
code, runtime, plan, sampler and exported control bytes; use fresh outputs.

Report TRAIN, DEV and sentence-unseen DEV separately, plus family macro results,
candidate count, target slot and TRAIN-positive/distractor exposure strata.
Multiclass Brier is the sum over candidates of squared probability errors
against this public single-winner target. NLL, Brier and pool-uniform baselines
do not replace top-1 accuracy. DEV remains exploratory; do not select architecture,
width, updates, threshold or loss by this result. No acceptance gate is invented.

The comparison addresses whether the original supervision supports a familiar
description prior without varying source evidence. It cannot establish that
the full model ignores context, determine a winning repair, prove correct
semantic counterfactuals or qualify any Alice role.

## Decisions deferred until the control is observed

Actual positive/distractor exposure is counted before interpreting the result.
Optimizer-weighted exposure requires the actual runtime's verified sampler;
static candidate counts alone are not actual step counts. If a shared residual
matcher, normalization, ranking objective or a depth comparison is warranted,
precommit matched TRAIN-family holdouts and candidate pools before fitting it.
The old full-trained scorer cannot be a valid holdout comparator when it already
saw those families. A candidate-only baseline must be newly fitted within each
fold. Holdout descriptions must not enter positive/negative optimization merely
because they exist in the cache; evaluate mixed familiar/unfamiliar pools without
providing their membership to the scorer.

Independent same-relation contexts and legitimate fixed-pool changed-gold pairs
may be required to distinguish coverage from readout formulation. Preserve
ambiguity, co-validity, UNKNOWN and support spans; never manufacture inverses,
types or identity labels from benchmark descriptions. Keep all 49 states while
evaluating compression, and separate calibration from ordering.

N0 supplies general semantic representations. N1 must learn conditional identity
and governed evidence, N2 owns contextual judgment and expression intent, and
N3 calibrates authorized confidence and fidelity. Public relation controls cover
only part of N0's meaning/argument-role requirement. Full voice, pragmatics,
uncertainty/plurality, source/host/self separation, inherited interference and
complete grounding/scale remain separately evaluated. Practical residual Gemma
influence is accepted for v1; upstream weight modification remains retired.

## Completed control and predeclared TRAIN-family repair comparison

Actual 576648 completed: candidate-only TRAIN/DEV/sentence-unseen DEV were
4/56, 0/16 and 0/14, unchanged in top-1 from its initialization. It selected
familiar descriptions on 16/16 DEV rows. Actual matched exposure was 256
positives and 7,160 negatives; DEV descriptions had no positive or negative
TRAIN exposure. Root replayed all 216 scorer records and 204 aggregate/strata
groups. See `PERSONALITY_CANDIDATE_ONLY_RESULT_576648.md` for original evidence.
This weak partial-input fit leaves conditional geometry, optimization and
limited source coverage unresolved. No persona interference cause is asserted.

The stronger fixed path is a useful measured starting point, while its absolute
accuracy is still weak. [Generation-Augmented Retrieval, section 3.4](https://aclanthology.org/2025.findings-emnlp.974.pdf)
and the [authors' implementation](https://github.com/lizehan1999/RE-GAR-AD/blob/main/retrieval_adjuster.py)
use a shared residual MLP above frozen sentence/relation embeddings. We re-read
that method and its source before implementing this candidate. It operates on
different embeddings and generated triplets; its objectives, reported results,
types, hyperparameters and data are not imported. Our candidate instead adds
the existing shared readout's correction to the deterministic fixed **scores**.
Zero initialization of its final layer preserves fixed scores exactly before
updates. This is an adaptation hypothesis, not a replication, a post-training
retention guarantee or a persona-suppression result.

Before any production fold fit, choose the lowest hash-ranked quarter of the
original 56 TRAIN families using fixed seed 20261003: 14 held out, 42 admitted.
Each training row uses all and only the 42 admitted positive descriptions as
its candidate pool. Held-out descriptions enter neither positive nor negative
optimization. Score all 56 original TRAIN sources against the same pool of
all 56 original TRAIN descriptions; report the 42 training-fit and 14
pseudo-unseen families separately. DEV contributes neither fit nor scoring.
IDs, family assignment, exposure, target and split are metadata, never model
features or familiarity penalties. Complete 49-state token banks remain intact.

Compare three freshly seeded models: original readout, fixed-score-plus-residual
full model, and the same anchored model with one artificial zero source.
Keep width 64, 256 updates, AdamW rate 0.0003/decay 0.01 and initialization seed
20261002. All three consume the same precomputed TRAIN index sequence, drawn
with seed 20261003. A sealed plan binds the fold and sampled exposure before
updates. This pool/sampler differs from 576600; the old full-family-trained
export is never a held-out comparator. Cached fixed training scores are purely
deterministic; the learned correction and exported consumer use the real
forward. Full reload scores, candidate permutation, final optimizer update
replay, unchanged feature bytes and closing package/code checks are recorded.
Both midpoint and last-step checkpoints are retained; only the last update is
replayed, so no full midpoint-continuation claim is made. Receipt/export scoring
uses inference without gradients, following the [PyTorch 2.7 API](https://docs.pytorch.org/docs/2.7/generated/torch.no_grad.html).

Nine local fixture tests cover the exact initial fixed scores, downstream-only
updates, invalid bank refusal, candidate permutation, fold exclusion, pointer
consistency, executing-helper hash bindings, changed artificial-source refusal
and all three complete consumer/export/checkpoint paths. Fixture
mechanics do not establish model quality; actual runtime compatibility precedes
the production fit. This is one already-observed-source developmental fold with
one source per family, not an untouched confirmation set. No numeric acceptance
threshold or automatic architecture approval is invented. The role's full
identity, pragmatics, judgment, uncertainty, spoken fidelity and inherited
interference requirements remain separate from this public relation comparison.
The expanded candidate pool uses the benchmark's single reference class; it
does not prove that every other description is false in natural language or
authorize single-winner identity targets.

## Completed comparison 576651 and owner-directed reset

Job 576651 completed `0:0` in 2h19m29s under `mxrayan`, at immutable
`a0c5dbc68d8fa575c99d27a0d9960635888c90e7`. All three variants completed 256
updates, with exact final-update optimizer replay, exported full-forward score
reload and 56-row candidate permutations recorded by the actual consumer.
The unchanged feature package and artificial source passed closing checks.
Slurm batch MaxRSS was 5,188,192 KiB. N0 approval, repair selection, personality
qualification, private data, publisher forward and neutrality flags remain false.

| Scorer | Training-fit correct / 42 | Pseudo-unseen correct / 14 |
| --- | --- | --- |
| Fresh original | 18 | 0 |
| Anchored full | 2 | 0 |
| Anchored candidate-only | 1 | 0 |
| Fixed semantics | 3 | 4 |

Anchored full matched fixed scores before learning but failed to retain their
pseudo-unseen correctness. All learned winners were from training families.
Original training-fit NLL was 2.112327 and pseudo-unseen NLL 5.371212; anchored
full was 3.936766 and 4.019917, versus fixed 4.023935 and 4.023305. Lower NLL with
zero correct decisions is not a retained-semantic success. These observations
do not identify a single mechanism, prove inherited persona interference or
justify a new regularizer, projection, capacity or update-count change.

Original fit bytes: 924,043; SHA-256
`2aa493f50129d430020d4879f0d27e05b246deb7b0a9654bfff2c97c2965b7af`;
canonical receipt `b84b5f9b91c5ad8ec9354ecc8d0851409787ba66030bbeca517d3bbc3e8ab590`.
Original plan bytes: 14,470; SHA-256
`ae19f902b4aec5cb0112cc42c2d38339ec6f8cf2e2088371299d80762360779e`;
canonical receipt `d72a7531455f36640e35443dfba9ed807e262f44ed525149e76ea9c8cb72f184`.
Byte-identical originals are in
`evaluation/eipm/gemma_n0/receipts/public_semantic_fold_{fit,plan}_576651.json`.
Root independently verified seals, seven exact Git implementation hashes, fold
assignment, all 168 logits/winners/ranks/margins/NLL records, six final aggregate
groups and 256 positive/10,496 negative sampled slots. Actual family sampling
ranged from 2 to 12 positive updates. Source rehash, original-runtime sampler,
optimizer and export tensor operations were not repeated locally; their actual
consumer evidence remains separately attributed.

The owner explicitly called for recalibration rather than another trial-and-error
cycle. This auxiliary experiment sequence is stopped, with no successor job or
selected repair. The role/data/objective/consumer reassessment is recorded in
`FULL_PERSONALITY_BUILD_STATE_2026-10-02.md`. Public relation classification covers
only a small part of N0 and does not teach N1-N3/EIPM's governing identity. This
negative result must not become a reason to expand a proxy optimization program
while conditional identity teaching and integrated expression remain missing.
