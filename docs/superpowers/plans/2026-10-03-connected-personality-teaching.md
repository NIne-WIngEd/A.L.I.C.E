# Connected personality teaching implementation plan

> For agentic workers: use executing-plans to implement inline; one whole-change review at completion.

**Goal:** Connect separately reviewed complete teaching cases to conditional N1 representations, contextual N2 judgment and separate N3 calibration, without another isolated relation-readout experiment.

**Architecture:** Preserve the current full-layer frozen-provider readout and rich authority-separated concept streams. Add explicitly supervised conditional concept applicability/uncertainty and independent concept-source support/disconfirmation; embed these learned signals into the concept streams consumed by N2. A strict named-pointer case codec and deterministic materializer supply sparse targets to the existing admitted trainer for every phase.

**Tech stack:** Existing Python/Torch; no dependency or compute-environment change.

**Spec:** `docs/eipm/PERSONALITY_GOVERNING_TARGET_MATERIALIZATION_REVIEW_2026-10-02.md`, original EIPM role/voice contracts at `021c5021a98104b35f9c8e94b19e48d21f25f132`, owner recalibration and build-before-capability-evaluation instruction of 3 October.

## Constraints and decisions

- Full ALICE personality and full Fable v1 FBM; FloRA excluded. Fixed publisher weights; private lineage/N0 acceptance remains unresolved. Code construction proceeds independently.
- Complete target-free situation/source/concept/candidate/label documents; explicit annotation/review/source/protocol bindings outside provider text. IDs resolve pointers and never encode meanings.
- Missing ratings, rivals, source support or acoustic observations cannot become negatives or invented gold. Independently reviewed probabilities may preserve plurality; owner identity is not population voting.
- Source authority stays deterministic. Conditional support/disconfirmation cannot grant history, identity or lived-experience permission. No hard semantic threshold, provider weight suppression or ontology ceiling is invented.
- Existing representation, graph and N2/N3 mechanisms are retained. Explicit conditional outputs are an ALICE-contract-derived extension, not a copied concept bottleneck or PAL prototype architecture.
- CB-LLMs author implementation supplies the distinction between explicit concept supervision and task labels, but its class-label concepts, pseudo-label correction and generator LoRA are unsuitable identity teaching. PAL supplies contextual preference conditioning, but population prototypes do not define ALICE. Pairwise Calibrated Rewards uses observed preference fractions and assumes tie-free rewards; its ensemble and majority supervision are not adopted.
- Distribution annotations declare closed versus partial comparisons. Partial singleton distributions are accepted as sparse facts, never reported as discrimination-bearing training.
- Build correctness checks now; full behavior/owner fidelity evaluation after the connected trained build. No successor public semantic experiment is authorized by this plan.

## Task 1: Reviewed case codec and materializer

Create `teaching/cases.py` for strict serializable cases, named sparse facts and annotation/review lineage. Create `teaching/materialize.py` for deterministic target layouts and exact document/frame/closed-session/target bindings. Add an optional materialization binding to `TeachingBatch`; its existing complete batch fingerprint/admission binds it. Export the interfaces from teaching.

Consumes: `CognitiveFrameDocument`, `ProducedFrame`, closed producer receipt, exact externally reviewed case digest/artifact pins and explicit family split allocation.
Produces: `ReviewedTeachingCase`, `NamedTarget`, `TargetFact`, `materialize_case(...) -> MaterializedCase` with a real `TeachingBatch` for N1, N2 or calibration; never a gradient or acceptance grant.

- [x] Write and observe failing mechanics tests for pointer-order resolution, partial/closed comparisons, explicit unknowns, source authority, family splits, target separation and changed/unfinished production.
- [x] Implement codec/materializer and connect all currently supported output families. Reuse existing trainer admission; do not produce private authorizations.
- [x] Pass affected tests; retain zero-discrimination partial annotations in explicit coverage counts.

## Task 2: Conditional N1 connected to N2

Modify `identity/model.py`, `identity/contracts.py`, `identity/codec.py` and `teaching/objectives.py`. Extend packet/output and teaching schema versions; old scientific checkpoints remain immutable and explicitly incompatible with changed code.

Produces: concept applicability/uncertainty and independent source support/disconfirmation probabilities, sparse reviewed objectives and learned conditional signal integration into each authorized N1 concept stream consumed by N2. Keeps high-dimensional explicit/residual representations.

- [x] Write and observe failing tests proving reviewed conditional targets update intended N1 parameters, unknowns carry no gradient, provider/N2/N3 remain fixed during N1 and N2 consumes conditional signals.
- [x] Implement shared context-conditioned readouts and sparse objectives, deterministic authority masks and complete output serialization.
- [x] Exercise materialized cases through actual admitted N1, N2 and separate N3 optimizer steps and resumable state. These are public mechanics, not personality fidelity results.
- [ ] Review the whole change, run affected and required repository checks, capture the concrete method/gaps in existing FBM/build state and PR100.

## Review focus

Unknown/colliding named pointers; candidate-order changes; unavailable sources/concepts; partial distribution singleton zero loss; unknown pair directions; wrong historical/lived-memory authority; target/review text entering features; unfinished or resealed mismatched feature sessions; source-family calibration crossover; preserved source/private training authority.

## Research sources

- https://github.com/Trustworthy-ML-Lab/CB-LLMs (author code/README fetched 3 October)
- https://pal-alignment.github.io/ (author method fetched 3 October)
- https://proceedings.neurips.cc/paper_files/paper/2025/file/53dbd7e34fab703a639964e2d3ee9e84-Paper-Conference.pdf (original paper fetched 3 October)

No method or semantic success is inferred from a search hit. The above sources inform distinctions and limits; the ALICE role and existing interfaces determine this implementation.

## Inline construction ledger

Baseline `a19d538df3c06e1502781ab88f185cc13c13df51`; existing feature checkout reused.
Shared seam: task 1 deterministic N1 layouts match task 2 conditional objectives;
all N2/calibration output families match existing packet masks and objectives.
Task 1 import-failure red observed; initial 8 case tests then exposed missing N1
objectives/output. Task 2 red observed before conditional implementation.
Whole-change reviewer found source averaging and admission cross-binding defects;
both reproductions and missing-annotation-role negative were observed failing,
then fixed. All 92 affected tests passed in 19.269s, including exact phase-specific
optimizer/checkpoint reload. Foundation constitutional validation passed.

Ruling: conditional consumer projections belong to N2, because N2 judgment
losses must train them; placing unsupervised relay projections in frozen N1 would
repeat the original incomplete-learning error. Full source-weighted messages
replace the lossy probability-only average. Source/annotation pin completeness
is enforced both at case parsing and admitted materialization.

The full pre-final-fix local check was 628 pass / 354 subtests / one unchanged
Windows symlink privilege failure. Final Linux CI will run the complete check.
This ledger records code construction, not learned identity capability. Source
lineage, actual reviewed curriculum and training remain substantive next work.
