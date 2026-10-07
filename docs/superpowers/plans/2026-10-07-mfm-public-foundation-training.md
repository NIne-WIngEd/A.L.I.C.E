# Full-roster public MFM foundation training implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Root reviewed this plan against the selected full-build contract. Implementation is authorized; model training requires the completed bank and a measured execution recipe.

**Goal:** Train the unchanged fresh MFM producer on every prepared public conditional training pair, with explicit task binding, equal semantic-MR mass, complete source-bank custody and exact next-step resume, then transfer its learned weights into full formation construction.

**Architecture:** A public-only reader joins sealed encoded pairs to the complete frozen Gemma source bank. A parameter-free task view binds `conditional_meaning_to_text` to the existing calibrated fresh producer; no new neural parameters, pretrained head or old reconstruction task is introduced. A dedicated public training entrypoint owns its objective/run/checkpoint ancestry while reusing applicable serializer and tree-comparison helpers.

**Tech stack:** Current PyTorch/safetensors runtime; existing `FormationDecisionSpecialist`, `SpecialistConfig` and source-only public exporter APIs. Runtime placement, optimizer values and full learning allocation are fixed only after actual resource measurements; no device fallback is implicit.

**Spec:** [producer contract](../../MFM_PUBLIC_FOUNDATION_PRODUCER_CONTRACT_2026-10-07.md), plus [full-role successor](../../MFM_FULL_ROLE_SEQUENCE_SUCCESSOR_2026-10-07.md).

## Global constraints

- Full intended material: 40,868 pairs, 36,597 training, 4,271 language development; 4,862 exact processed sources, 4,356 training semantic groups and 506 development groups.
- Fresh width768/layers6/heads12/full262144 tied vocabulary, source width3840; existing full formation positional capacity. Public data maxima71 source/84 target tokens are observed material properties, not role ceilings.
- `fresh-token-position-scale-0.02-v1`; seeded untrained weights retained before gradients. No failed-parent import, Gemma lexical/head reuse, public target in feature extraction, FINAL access or personal corpus read.
- No reuse of `source_reconstruction` for realization, no forced empty formation labels, no truncation/vocabulary reduction and no fixed50/50 mixture.
- New public run/curriculum/seed/component schemas must identify public conditional foundation learning honestly; they are not v1.6 teacher-example schemas or product qualification.
- Complete bank and its external bank pin are pending. No training can accept an ordered partial prefix or merely valid progress seal.

## Actual artifact/API evidence

Local recovered census directory: `work/recovered-576995` relative to the outer workspace.

| Artifact | Actual bytes | SHA-256 |
|---|---:|---|
| `receipt.json` | 38,201 | `e61ba06467a4812c4c30697aaa7c123862c113e82599e391bf5c67191732ca95` |
| `encoded-pairs.jsonl` | 39,131,578 | `490a84406ad4d1988d42e929782206c4437fce9aea447e3f2098086e283e11c0` |
| `source-inputs.jsonl` | 6,793,856 | `721567a70c7f9c287c6a29569fbd8defd03dde7e94e099ec86b7c2fb5843f172` |

Census schema: `mfm-public-conditional-processor-census-v1`; externally pinned self-seal `0885f9a52b8ea967127a5a6b7708331163b214d3dea81ed598d24c57de6df27d`.

Actual pair keys are exactly `case_id, decoder_inputs, labels, semantic_input_sha256, semantic_reference_count, source_ids, source_payload_sha256, split`. Pairs do not themselves contain task or source_key; derive source_key only as `canonical_sha256({'task': census.task, 'source_ids': pair.source_ids})`.

Source-input keys are exactly `schema, task, source_key, source_ids, attention_mask, split, case_ids, semantic_input_sha256s, source_payload_sha256s`. The local audit checked all40,868 source joins, source case/group/raw-hash membership and target shifts; all passed. No semantic group crosses splits. Training reference counts range1–46. Summed group weights equal1 with maximum floating group-mass discrepancy2.44e-19.

The actual completed bank has not been opened or verified by this worker. The exporter contract is inspectable: `export.json` binds clone/weight/census/template/runtime/code and source-only flags; `sources/<source_key>/source.json` seals exact source input and tensor SHA; tensor keys are `states,input_ids,attention_mask`; `bank.json` is published only for the complete roster and binds final progress/quarantine/source receipts. Use `read_census(directory, expected_sha256)`, `read_source(root,row,export_digest,width)` and `verify_bank(root,census,sources,export)`. Check the returned bank seal against the independently supplied expected bank SHA as well.

The old `formation_sequence_curriculum.balanced_case_weights()` is deliberately tied to432 cases/29 strata and cannot be repurposed. `train_v16_cached_specialist._payload()` also expects formation examples and re-tokenizes their formation targets; do not pass E2E rows to it. The old `FormationSequenceSpecialist.TASKS` contains only `formation` and `source_reconstruction`; its task vector and failed-parent recipe are not this fresh public task.

## Exact objective and optimizer coefficients

For training semantic group g with n_g retained pairs, let T_i be a pair's complete label length including EOS, excluding BOS and any padding. Current per-pair forward returns `L_i = sum_t CE_i,t / T_i`. Define:

`J_train = sum_i [L_i / (G_train * n_g(i))]`, with `G_train = 4356`.

Each MR receives mass1/4356 and every original retained realization divides that mass. This is sequence-mean then reference-mean then group-mean, not global token pooling. Include JSON wrapper tokens and EOS; no teacher label is ignored except explicit future padding. Reconcile n_g against the entire pair roster before gradients, never trust the row field alone.

Shuffle all36,597 cases once per epoch from immutable encoded-file order with `random.Random(seed + epoch)`. For a minibatch of actual size b, multiply each per-pair mean by `N_train / (b * G_train * n_g)`. This is the existing importance-weighted minibatch estimator pattern. Do not divide by the batch's weight sum, which erases singleton weighting. The final partial minibatch uses its actual b. Report the exact global weighted objective separately; a sum of minibatch reports is not automatically J_train.

Development uses its506 groups and their own reference counts under no-grad. It never enters sampler/gradients, stopping or checkpoint selection in this construction recipe. E2E content-selection/addition limitations remain visible; development likelihood is not independent formation qualification.

## Review focus

1. A correctly sealed but incomplete bank must never start learning.
2. Processing collisions or raw slot-order variants must retain their exact source/case ancestry rather than pool by meaning alone.
3. Last partial minibatches, alternative-reference counts and different target lengths must preserve the declared global weighting.
4. Explicit task binding must preserve every tensor key and exact logits while rejecting the old reconstruction task.
5. Restart after an epoch boundary or scheduler interruption must produce the same next batch, loss, model and Adam state without reusing partial checkpoint files.

## Task 1 — Public reader and objective contract

**Create:** `scripts/mfm/public_conditional_training.py`.

**Test:** `tests/governance/test_mfm_public_conditional_training.py`.

**Interfaces:**
- `load_public_training(census_dir: Path, census_sha256: str, feature_bank: Path, feature_bank_sha256: str) -> PublicConditionalTraining`: complete validated census/source/pair/bank mappings and train/development tuples. It loads no base model or teacher/private corpus.
- `payload(prepared, pair, *, device) -> dict`: detached frozen states/mask plus decoder inputs/labels; joins through exact processed key and verifies source receipt/tensors using the exporter API.
- `case_weights(pairs, *, split: str) -> dict[str,float]`: complete unique case weights; checks grouped split ancestry and exact reference counts.
- `batch_coefficients(batch, weights, *, total_train_cases: int) -> dict[str,float]`: unique train-only importance coefficients.
- `epoch_order(prepared, *, seed: int, epoch: int) -> tuple[Pair,...]`: complete deterministic training permutation.

- [ ] Write red tests for valid independent source/target join; swapped source IDs, wrong case membership, raw hash/group membership and mixed split rejection; duplicate JSON keys/case IDs; invalid shifts/BOS/EOS/vocabulary; incomplete source coverage and incorrect n_g; partial/missing bank and false source-only flags; changed externally pinned seals and resealed internal joins.
- [ ] Implement strict pair whitelist and census file hash checks, recompute complete statistics via the census helper, reconcile all source/pair/case/group/raw-hash rosters and verify complete bank plus external seal. Do not change exporter guards or treat flags alone as proof.
- [ ] Test objective on unequal-length reference sequences with groups of two and one references: weights1/4,1/4,1/2. Verify full objective, singleton and partial-batch coefficients and gradients against a direct scalar loss. Reader fixtures prove contracts only, not actual model learning.
- [ ] Run only the new reader tests plus relevant existing public preflight/material/features tests.

## Task 2 — Explicit parameter-free public task binding

**Create:** `src/cognitive_kernel/formation_conditional_task.py`.

**Test:** `tests/governance/test_mfm_conditional_task.py`.

**Interfaces:** `bind_conditional_task(specialist, *, task: str) -> ConditionalTaskView`, accepting `conditional_meaning_to_text`. The ephemeral view delegates existing forward/next-token/cache-visible configuration and embeddings; it registers no tensors and modifies no positions. The source input already contains the task tag under the pinned processor, and the reader verifies that binding. A later full-formation stage has its own explicit caller/source/target contract; this current task name is not a role-capability ceiling.

- [ ] Write red tests for public binding with unchanged parameter/state-dict keys and exact prefix logits; unknown and `source_reconstruction` rejection; cache/reference generation through the view; no added task vector or pretrained tensor access.
- [ ] Implement delegation only. Preserve the existing `FormationDecisionSpecialist` and old sequence class unchanged.
- [ ] Verify that a checkpoint saved from the underlying specialist reloads directly into the same specialist class; the task view must not insert a checkpoint prefix or prevent later formation weight reuse.

## Task 3 — Real public foundation trainer and resumable lineage

**Create:** `scripts/mfm/train_public_conditional_foundation.py`.

**Test:** `tests/governance/test_mfm_public_conditional_foundation.py`.

**Interfaces:** `optimizer_step(model, optimizer, batch, *, prepared) -> metrics`, `run(args) -> sealed_result`, and `arguments(argv=None)`. CLI input binding: `--census-dir`, `--census-sha256`, `--feature-bank`, `--feature-bank-sha256`, `--output-dir`; explicit reviewed device/config/seed/optimizer/allocation arguments; `--resume-checkpoint` and a scheduling-time checkpoint stop. No separate smaller-roster/model mode.

- [ ] Write red tests that only train pairs enter gradients, each epoch exhausts the exact roster once, task binding remains explicit, development has no gradient/selection effect, and objective coefficients survive target-length/reference-count imbalance.
- [ ] Create fresh calibrated full-geometry producer and new AdamW; save a public-specific seed-control receipt before updates. Persist public run/curriculum schemas with exact input pins, clone/tokenizer/code/config/init, label counts, objective/normalization/order version, Python/Torch/runtime precision, optimizer fields, full epochs/roster and checkpoint interval. A scheduling-time allowance is an execution receipt, not a changed learning objective.
- [ ] Reuse applicable `train_v16_formation_specialist._checkpoint`, `_verify_checkpoint`, `_resume` serialization mechanics explicitly as their existing parameter/RNG container format, and `train_v16_cached_specialist._cpu_tree/compare_tree`. Do not reuse formation admission, probe prerequisites, public-v1.6 seed meaning, generation qualification or the failed-parent trainer. If these generic serializers require changes to preserve the public contract, use a public-specific serializer rather than mislabeling data.
- [ ] Pin deterministic epoch order and next-case/step cursors at complete optimizer boundaries. Verify latest complete checkpoint/run/seed bindings and reject changed pair/bank/objective/weight/config/task/optimizer/sampler inputs, cursor corruption, foreign directories and partial checkpoints. Torch/CUDA RNG and Adam state are restored; Python global RNG is not used for sampling. The base tuple and Python shuffle implementation/version are bound.
- [ ] Write a real next-step restart test using a different next minibatch, unequal weights and a final partial minibatch: checkpoint, continue, reload, regenerate the same next IDs, then compare loss/model/Adam/RNG and cursor. Include the completed-epoch cursor case. No monkeypatched checkpoint or optimizer substitute may establish restart behavior.
- [ ] Preserve abrupt-exit scratch evidence safely within the owned run, reject unexpected paths/links and never overwrite a completed checkpoint. A process killed mid-step resumes the last complete boundary. Reuse the learned source-export transaction lesson; do not mutate the source bank or clean arbitrary files.
- [ ] Before selecting fit allocation, measure actual full geometry on the actual longest-source/target/joint material, backward/Adam peaks, throughput, checkpoint IO and next-step replay in the same construction workflow. There is no standalone qualification campaign. Public cached training needs no12B Gemma load; source-only ancestry stays frozen.
- [ ] Export final fresh learned weights and public component/run/curriculum/checkpoint lineage; reload actual generation with the bound prepared tokenizer and report free sequence/source-change behavior separately from teacher likelihood. Same-seed untrained generation is retained. No unsupported formation or owner-independent qualification claim follows.

## Transfer boundary

The public artifact's unchanged learned tensor names/config permit direct parent-weight loading into a separately versioned full formation stage. That stage binds current caller input, expanded full-role teaching, typed candidate/span supervision, byte alignment and its new objective/optimizer decision. Do not quietly preserve an old task vector or Adam moments across an objective change. Public sequence transfer does not fill currently untrained long positions, open semantic transitions, native modalities, authority policy or personal memory; those remain actual subsequent construction requirements.

## Self-review and handoff

The reader covers complete-bank custody, exact joins, split ancestry, labels and global weighting. The task view preserves model architecture and checkpoint compatibility. The trainer covers full-roster learning, restart and honest learned-artifact lineage. Typed selection/copy/full-role authoring are intentionally subsequent work under the full-role contract, not fabricated as already implemented. Root reviewed the concrete plan; independent task reviews and actual measurements remain part of construction.


## Root implementation rulings

- The source export is the full4,862-source build. Magnolia577007 continues the
  same exact bank after successful577006 with the parent terminal/progress pin.
  Scheduling blocks preserve the full roster and never become a smaller model.
- Sequence-mean/reference-mean/MR-mean weighting is selected explicitly for this
  conditional realization objective. Complete label count includes EOS. Data
  reference counts must be recomputed from the full roster before gradients.
- The public task view is parameter-free and uses the existing fresh decision
  specialist directly. No old task vector, reconstruction-name alias, pretrained
  language head or failed formation checkpoint enters initialization.
- Public checkpoint lineage must identify the public objective. Reuse existing
  serialization mechanics only where they preserve that truth; otherwise write
  a public-specific atomic container. Do not modify old accepted formats.
- Existing Windows runtimes are available. Use them without installation; actual
  Linux helper-file pins can differ from Windows checkout line endings. Preserve
  exact code custody and perform real artifact consumption in its bound runtime.
- Small tensor fixtures verify joins, weights, gradients and restart only. The
  intended model retains its full geometry; resource and learning claims require
  actual complete material and the selected measured runtime.
