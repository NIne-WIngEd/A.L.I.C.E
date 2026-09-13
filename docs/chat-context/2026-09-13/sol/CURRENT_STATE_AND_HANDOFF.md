# A.L.I.C.E. Current State and Continuation Handoff — 2026-09-13

**Branch purpose:** durable chat continuity only. This is not canonical `main`, not a training authorization, and not a claim that A.L.I.C.E. weights have been created.

## Source continuity

This handoff records the state recovered from the complete 131-page continuation chat export supplied on 2026-09-13 and reconciled against `main`, `alice-context`, `alice-eipm-v1-build`, `alice-mc10b-live`, `alice-mc10c-live`, the orphan `alice-telemetry` branch, and the uploaded Curated Frontier v2 archive.

Chat export source:

- filename: `ChatGPT-Update Alice Context-20260913-1610.pdf`
- size: `1,967,248` bytes
- SHA-256: `f4dd62fd26f2d3a186d9c73b6a3f5882c3ac97b8025e1a5ad58477d6d268dfb3`
- pages: `131`

The PDF itself remains a conversation attachment. This branch records its cryptographic identity and the decisions needed for continuity.

Private curated source package:

- filename supplied to the chat: `ALICE_EIPM_CURATED_FRONTIER_v2(1).zip`
- size: `1,881,844` bytes
- SHA-256: `3867ff04d1e326086b9086b2f106b9156b3a3ec8d637d3161e7bf01616183ee9`
- this hash is the same artifact identity recorded by `docs/eipm/EIPM_CURATED_FRONTIER_V2_RECEIPT.md` on `alice-eipm-v1-build`.
- the raw archive is **not** committed here because it contains private EIPM source/training material and the repository's custody rules prohibit public-Git placement of private companion payloads.

## Authoritative current direction

The old MC10D validator/judge tournament is historical evidence, not the active A.L.I.C.E. mainstream. Its infrastructure lessons remain valuable; its multi-model validation loop is not to be resumed by default.

The active EIPM path is:

```text
canonical Elaina source evidence (E0)
        +
curated reconstruction inference (E-INF)
        +
curated synthetic behavioral completion (A-SYN)
        +
provenance / clone-awareness / identity rules
        ↓
model-agnostic canonical identity substrate I*
        ↓
private learning structures (ACFP / graph / concepts / preferences / IDP targets)
        ↓
N0 native semantic-language bootstrap design
        ↓
N1 identity representation learning
        ↓
N2 identity judgment / preference learning
        ↓
N3 calibration and owner-directed refinement
        ↓
first A.L.I.C.E.-native EIPM candidate
```

The present location is **N0 research and architecture selection before target-scale training**.

## A.L.I.C.E.-native ownership rule

Permanent EIPM weights must be A.L.I.C.E.-owned and trained from A.L.I.C.E.-controlled initialization/training lineage. Third-party models may be disposable teachers, research tools, generators, mechanics references, or external reasoning engines where licensing and custody permit. Their weights must not be a structural dependency of the permanent EIPM.

This supersedes the older working idea that the permanent personality model would simply be `Qwen + LoRA`.

## EIPM scope

A.L.I.C.E. is not one model. The EIPM is the first learned identity component and should answer, in effect, **“What would A.L.I.C.E. make of this?”**

The EIPM should learn Elaina-derived personality-relevant semantics, interpretation, values, boundaries, characteristic judgment, emotional/social reading, disagreement, affection, restraint, humor, relationship posture, communication tendencies, uncertainty, evidence use, and resistance to generic-assistant drift.

It must not become the sole store for biography, Rayan host state, mission state, current world facts, tools, calendar state, canonical memory, or A.L.I.C.E.'s post-activation continuity. Those remain separate governed components and inputs.

The current architecture hypothesis is **structured-first, not structure-only**: ACFP supplies governed structured state and provenance-bound raw spans; the EIPM owns enough native semantic understanding to detect nuance that schemas can erase. The EIPM produces a portable Identity Decision Packet rather than needing to be a giant general-purpose generator.

## Provenance layers that must remain separate

- `E0` — immutable historical/source evidence authority.
- `E-INF` — revisable reconstruction inference with uncertainty; never historical fact merely because it is useful.
- `A-SYN` — revisable synthetic behavioral completion; may shape A.L.I.C.E. behavior but is not Elaina lived history.
- `A-EXP` — later A.L.I.C.E. experience.
- `A-SELF` — A.L.I.C.E.'s evolving current self/continuity state.

Hard continuity rule: **UNKNOWN historical behavior is not the same thing as a runtime behavioral blank.** When E0 does not establish what Elaina historically did, the historical answer may remain UNKNOWN while A-SYN supplies an explicitly synthetic/provisional A.L.I.C.E. policy for that situation.

The retained 13,719 alternate behavioral branches are **unordered competitors/alternatives**. They must not silently become rejected examples or automatic negatives.

## Curated frontier state

The broad-generation phase is closed as a saturation candidate. Do not restart broad generation merely because more rows could be produced.

Current curated sources:

- canonical E0 semantic units/router: `290`
- curated E-INF: `205`
- curated direct A-SYN: `1,009`
- factorized A-SYN base policies: `822`
- targeted A-SYN additions: `119`
- regenerated contextual A-SYN variants: `15`
- total curated A-SYN across those active sets: `1,965`
- historical UNKNOWN competitors: `3,496`
- alternative-policy competitors retained unordered: `13,719`
- raw transform instances retained as blueprints/non-independent evidence: `12,924`
- legacy reserve E-INF: `720`
- legacy reserve A-SYN: `288`
- targeted gap queue: `0` remaining
- direct core-E0 anchoring after targeted curation: `167/187`

The Curated Frontier v2 archive was independently rechecked on 2026-09-13: ZIP CRC passed, all manifest/root/nested SHA-256 checks passed, every JSON/JSONL file parsed, canonical E0 router IDs matched canonical E0 units, E0 text hashes matched, curated references resolved, no active identity-support/excluded-context overlap was found, no exact normalized active-behavior duplicate group was found, and the retained targeted proposal IDs/source payloads matched their generated inputs.

Current v2 router counts observed in the supplied archive are authoritative over older historical counts in earlier chat notes:

- `context_only_conditioning`: 226
- `conditional_identity_supervision`: 145
- `direct_identity_supervision`: 123
- `episodic_behavior_supervision`: 71
- `exclude_from_identity_loss`: 64
- `style_voice_supervision`: 23

## Private learning structures already prepared

The continuation chat records the prepared private model-learning substrate as:

- `2,460` canonical private training records = 290 E0 + 205 E-INF + 1,965 A-SYN
- `3,496` historical UNKNOWN targets
- `3,930` runtime preference pairs
- `1,965` historical-provenance pairs
- `205` E-INF uncertainty sets
- `13,719` alternative branches retained unordered
- `2,460` ACFP frames
- `2,460` symbolic IDP targets
- `2,830` identity-concept-bank entries
- persona graph: `5,290` nodes / `19,930` edges

These private structures are not public-Git payloads.

## Parameter-count rule — owner clarification 2026-09-13

**No numerical parameter target, research envelope, or ceiling is ratified.**

The prior `~100M–400M` working envelope and shorthand `~400M` references were research guesses based on the then-proposed specialist/discriminative architecture. They are superseded as sizing constraints.

The permanent model may be 100M, 400M, 1B, 10B, 100B, or another size if evidence shows that scale is required to reach the EIPM objective. Conversely, no larger model is preferred merely because it is larger.

Size must be derived from:

1. the exact functional competency contract;
2. the architecture selected after frontier research;
3. semantic/pragmatic capacity actually required;
4. scaling and ablation curves;
5. public-data quantity/quality and legally permitted teacher signals;
6. private-data overfitting/generalization risk;
7. inference/training cost only as an engineering tradeoff, not an arbitrary capability ceiling;
8. owner-evaluated identity fidelity and calibration.

Do not optimize the architecture to hit a preconceived parameter count.

## Current research questions before N0 freeze

Do not assume the previous `ModernBERT-style encoder + Qwen teacher` idea is optimal. Before freezing N0, research and decide:

- whether the permanent EIPM should remain a discriminative identity/judgment model, become partly generative, or use a hybrid topology;
- what semantic encoder/fusion architecture is strongest for this objective;
- which capabilities must be learned internally versus supplied by ACFP/graph/memory/reasoning fabric;
- which teacher families, if any, are best for each competency and whether teacher ensembles beat a single teacher;
- exactly what the teachers teach: lexical/compositional semantics, inference, contradiction, temporal/causal relations, pragmatics, sarcasm, implicature, politeness/register, emotion/appraisal, social roles, theory of mind, uncertainty, provenance, candidate comparison, dialogue/discourse, structured-frame semantics, and other personality-relevant foundations;
- which objectives are appropriate for each competency rather than distilling a teacher indiscriminately;
- tokenizer design and vocabulary size based on measured fertility/coverage, not a tutorial default;
- public corpus mixture, contamination controls, licensing, provenance, and commercial/startup compatibility;
- graph/structured-state integration and whether explicit + learned residual identity concepts remain the best representation;
- scaling strategy and model-size selection from measured capability curves;
- public N0 competency/evaluation suite that must be defined before architecture selection;
- mechanics pilots that are disposable versus the first permanent A.L.I.C.E. checkpoint.

## Training boundary

No target-scale A.L.I.C.E. EIPM weights are authorized by this handoff. No private E0/E-INF/A-SYN gradient step should occur merely because the curated data is ready.

The immediate task is research/design. Freeze the corpus, tokenizer, teacher strategy, architecture, objectives, scaling plan, lineage, and training configuration only after the research case is coherent. Then perform the planned final pre-weight owner/Astra review for material mistakes before first private identity-gradient work.

Throwaway public-data mechanics tests may be used later to prove software/architecture viability without converting them into promoted A.L.I.C.E. identity weights.

## Historical infrastructure lessons that remain active

Do not discard lessons merely because MC10D is no longer the active path:

- bind every material run to exact hashes, manifests, model/runtime identity, code, and hardware;
- distinguish technical-invalid from semantic failure;
- verify packaged source lineage rather than assuming a local file is the packaged file;
- prefer deterministic rebuilds and byte-identical manifests where practical;
- do not spend scarce GPU budget before controller/package/self-tests pass;
- keep provider/runtime mechanics separate from semantic authority;
- preserve durable checkpoints with round-trip/hash verification for expensive work;
- stop repeated hotfixing when the root cause is unclear; do a root-cause/Astra-class audit before consuming more compute;
- preserve source, inference, synthetic completion, host data, A.L.I.C.E. continuity, and evaluation artifacts as different authority classes;
- generated row count is not evidence count; mechanical transformations do not become independent personality evidence;
- route/context-only/excluded units must not silently become direct identity evidence;
- plausible alternative personality branches are not automatically negatives;
- no validator, generator, teacher, or training script grants itself source-person truth or promotion authority.

## Next action

Perform the requested thorough 2026 frontier research on N0/EIPM architecture, teacher strategy, public-corpus strategy, tokenizer, semantic/pragmatic competency coverage, training objectives, scaling, and evaluation. Only then recommend what should be frozen.
