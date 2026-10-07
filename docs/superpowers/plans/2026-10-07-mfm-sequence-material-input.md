# MFM sequence material and caller input implementation plan

> For agentic workers: use superpowers:executing-plans or subagent-driven-development. Complete the listed test cycles and retain original failures.

**Goal:** Supply real diverse conditional teaching material and target-independent
caller inventories for the next full-semantic MFM producer.

**Architecture:** Two independent preparation paths. A pinned public-material
worker retains human source-to-text pairs; a caller-input boundary validates
source bytes and explicit scoped prior state without teacher access. Neither
changes the current trained producer. The exact Gemma processor performs the
complete public-material census on the existing Magnolia CPU runtime.

**Tech stack:** Python standard library, existing v1.6 contracts, prepared Gemma
processor, actual Torch 2.7.1/Transformers 5.17 runtime. No environment mutation.

**Spec:** [full-role successor](../../MFM_FULL_ROLE_SEQUENCE_SUCCESSOR_2026-10-07.md).

## Constraints and review focus

Keep original 441 inputs, weights and generation failures unchanged. Do not read
FINAL or invent registration from teacher labels. Preserve source/target bytes,
license and attribution, input ancestry, grouped splits and complete token limits.
Reject foreign scope, stale binding, mismatched source bytes, metadata-only
citations and source-subject promotion to registered entity targets.

## Task 1: Pinned conditional material

- [x] Create `tests/governance/test_mfm_public_conditional_material.py`; observe
  missing preparation API fail before implementing.
- [x] Create `scripts/mfm/prepare_public_conditional_material.py` and
  `configs/mfm/e2e_conditional_material_v1.json`. Consume only pinned training
  CSV, original grant/notice and exact manifest hash. Emit grouped/deduplicated
  conditional records plus complete custody/split receipt.
- [x] Verify corrupt pins fail before publication, alternate references remain
  together, source/target changes are bound, reordering preserves assignment,
  original UTF-8 survives and no formation/FINAL/write capability is asserted.
- [x] Stage all real upstream training bytes outside Git and run the worker.

Local result: all 42,061 original rows retain ancestry; 40,868 exact unique
pairs form 4,862 semantic MR groups. Grouped language splits contain 36,597 train
and 4,271 development pairs. Nine material tests and the independent full-row
custody audit pass. This is conditional teaching eligibility, not formation
admission or learned capability.

## Task 2: Caller-input boundary

- [x] Create `tests/governance/test_mfm_caller_input.py`; first verify the expected
  missing interface, then implement `scripts/mfm/formation_caller_input.py`.
- [x] Consume only `FormationContextV16`, exact ordered opened bytes and explicit
  read-only registry entries. Emit a canonical independently hashed input
  manifest and input-only selection roster with path eligibility.
- [x] Verify exact-byte/host/namespace/generation bindings, no metadata-only
  citations, no subject-to-entity promotion, raw text mentions do not register
  identities, and scopes remain semantic output labels. Round-trip unchanged
  input records. No teacher reader or authority operation is allowed.

## Task 3: Complete actual processor census

- [x] Add `scripts/mfm/preflight_public_conditional_material.py` and focused
  tests for full records/frequencies/BOS/EOS, hash mismatch and length failures.
- [x] Implement exact prepared-processor, task-wrapper, worker, material and
  receipt bindings, complete source/target tokenization, maxima and training-only
  positive-frequency counters. Validate round trips, interior BOS/EOS, original
  notices, grouped splits and the independent target-free source export contract.
  Local material/preflight tests pass; this does not establish actual Gemma counts.
- [x] Deploy immutable code and actual material to an owner-only new Magnolia
  CPU job. Reuse the proven runtime, owner account and namespace wrapper.
- [x] Recover and verify the complete actual processor census and its external
  receipt-file pin, source-input roster, encoded-pair hashes, length maxima and
  training-only frequencies. Preserve incomplete results and stop when that
  external result is the remaining dependency.

## Task 3b: Target-free frozen source-feature export preparation

- [x] Create `scripts/mfm/export_public_conditional_features.py` and
  `tests/governance/test_mfm_public_conditional_features.py` through a test-first
  cycle. Consume only the completed census's independent `source-inputs.jsonl`
  roster and receipt; never open teaching labels, encoded pairs or teacher targets.
  Bind the exact task, source IDs/masks, original payload hashes, split and opaque
  case associations. Preserve semantic MR grouping separately from feature keys.
- [x] Require the census's externally published terminal record pin and source-input
  file hash, the same exact prepared Gemma clone receipt bound by that census,
  processor and weight pins, and immutable exporter/helper code bindings.
  Feature reuse requires exactly
  identical processed IDs and masks under those bindings. Preserve every raw
  payload hash when normalized processing collides, and keep slot-order variants
  distinct whenever their processed inputs differ. No base-head answers, optimizer,
  private evidence or authority writes enter this export.
- [x] Implement a bounded, resumable CPU export using the existing protected
  owner route, frozen evaluation mode and complete source lengths. Write shards
  to staging, verify their source roster, finite tensors, dtype, complete shape,
  IDs/masks and byte hashes, then atomically commit their manifest entries.
  Resume only from fully verified committed shards with unchanged ancestry.
  Quarantine interrupted or uncommitted artifacts; never consume them or silently
  skip an unfinished source. Preserve the interrupted-run evidence.
- [ ] Measure longest-source and representative CPU exports before choosing the
  full export budget. Record wall time, throughput, peak memory and disk use;
  project the complete census roster's cost from actual processed-entry lengths.
  Verify committed tensors again on reload and declare the completed roster and
  external export-receipt pin. This measured feature and training budget must
  precede a model-training allocation or fit; no such fit is authorized by this
  preparation task, and incomplete export does not become a qualified bank.

## Task 4: Evidence and continuation

- [ ] Review actual changes, run new and affected existing contract tests and
  verify complete staged rosters. Preserve red/green evidence with the scope
  of what it proves.
- [ ] Update MFM, existing FBM seed/append-only method trace, Graphify navigation
  and private handoff. Publish exact heads. Do not claim learned improvement
  from data preparation or independent qualification from same-author material.

Actual census submitted as Magnolia **576995**, immutable code `480f8bff7334d8691f84d5896de09de0deaeb815`, under `mxrayan`. The completed0:0 census and external pin `0885f9a52b8ea967127a5a6b7708331163b214d3dea81ed598d24c57de6df27d` are verified. All40,868 original pairs and4,862 exact processed source inputs reconcile. The prepared material transferred with exact original hashes.

The first actual frozen-feature job is a **30-minute pilot** with a two-hour scheduler safety limit. It reuses the complete pinned source roster and preserves its verified ordered prefix for later same-code resume. Throughput, setup, peak memory and disk are measured before selecting the longer continuation. The longest source has71 tokens; if absent from the pilot prefix, measure it before fixing the full export/training budget.

Pilot576999 failed before Gemma because the adversarial test retained a mapped
file during same-path replacement on NFS. The unchanged-runtime577003 controls
confirm that exact cause. The [ordered diagnosis](../../MFM_SOURCE_FEATURE_FIXTURE_DIAGNOSIS_2026-10-07.md)
preserves original failures, every custody predicate and valid controls. The
test-only correction expands ID/mask/geometry/dtype/nonfinite and valid-bank
coverage;31 focused independent and91 affected root local tests pass. The v2
recipe gates real source export on those91 checks in the actual protected runtime.
Measured export and its budget remain outstanding.
