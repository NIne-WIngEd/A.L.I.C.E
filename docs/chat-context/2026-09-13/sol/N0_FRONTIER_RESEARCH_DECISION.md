# A.L.I.C.E. N0 Frontier Research Decision — 2026-09-13

**Status:** continuity handoff. No permanent EIPM weights created.

The requested post-curation N0 frontier research pass has now been completed far enough to proceed to **qualification tooling**, not target-scale training.

The full implementation-facing research decision was committed to `alice-eipm-v1-build`:

- commit: `ed9ad923b4b19e6fb913ee4bf216fb28bccd4d2d`
- file: `docs/eipm/EIPM_N0_FRONTIER_RESEARCH_AND_BOOTSTRAP_PLAN_2026-09-13.md`

## Ratified owner correction carried forward

There is **no fixed parameter target or ceiling** for the permanent EIPM.

The previous `~100M–400M` / `~400M` language was a working research hypothesis for one specialist architecture. It is superseded as a sizing constraint. A.L.I.C.E. may use a much smaller or much larger EIPM if measured competency and owner-evaluated identity fidelity justify it.

Scale must be selected by controlled capability/data/architecture scaling evidence, not by a round-number target.

## Current strongest architecture direction

The EIPM remains a specialist A.L.I.C.E.-native identity/judgment policy rather than a full general-purpose generator.

Primary baseline topology:

- scalable from-scratch bidirectional semantic encoder;
- typed ACFP structured-state encoder;
- relation-aware persona/evidence graph encoder;
- explicit E0-grounded identity concept bank;
- bounded learned residual concept bank;
- cross-context fusion;
- adaptive multi-view latent pooling;
- multi-head/distributional identity outputs including ranking, stance, value priority, relationship posture, emotional/appraisal posture, communication posture, provenance/literal-grounding risk, uncertainty, drift risk, and evidence pointers;
- portable Identity Decision Packet output.

This preserves the existing architecture rule: **structured-first, not structure-only**. Raw provenance-bound source spans remain available where a schema would lose nuance.

A modern bidirectional Transformer is the primary baseline family, but it is not yet frozen to an exact ModernBERT clone. At least one materially different efficient encoder challenger should be qualified if engineering maturity permits.

## N0 must teach competencies, not copy a teacher

N0 competency classes include:

- lexical/compositional semantics;
- paraphrase/STS;
- NLI/contradiction/negation/modality;
- reference/coreference/deixis;
- discourse and long-context consistency;
- temporal and causal relations;
- implicature/presupposition/indirect meaning;
- sarcasm/irony/teasing;
- politeness/register/social stance;
- intent/belief/theory-of-mind distinctions;
- emotion/appraisal/relationship-sensitive interpretation;
- literal grounding versus pragmatic over-inference;
- known/inferred/unknown and evidence sufficiency;
- calibrated uncertainty/OOD detection;
- pairwise/listwise candidate comparison where real order exists;
- tie/plural-policy behavior where no unique order exists;
- text <-> ACFP and text <-> graph alignment;
- frame/raw-evidence conflict detection.

## Teacher strategy

No single teacher is privileged.

Candidate open/self-hosted teacher families should be qualified **per competency**. Current candidates include Qwen3 embedding/reranking models, Qwen3.8-27B, OLMo 3.1 32B, and gpt-oss 20B/120B, plus challengers after exact license/terms review.

Teacher outputs may shape general semantic/pragmatic representations but do not receive authority over E0 historical truth, E-INF truth, A-SYN promotion, owner state, or A.L.I.C.E. lived memory.

Astra remains advisory for the one planned final pre-weight review unless a separate terms/contract review explicitly permits some other use.

## Tokenizer decision

No vocabulary size is frozen.

Run an A.L.I.C.E.-owned tokenizer bake-off over the exact legally cleared N0 corpus. Compare BPE/byte-fallback, Unigram-like, morphology-aware, and tokenizer-free challenger approaches where relevant. Measure TokLens-style metrics plus private EIPM fertility/fragmentation without exporting private text.

## Public corpus decision

Use an open/public-domain-first provenance strategy. Common Pile v0.1 is a strong foundation candidate, not an instruction to train blindly on all 8 TB.

Every source receives an explicit `TRAIN_ALLOWED`, `EVAL_ONLY`, `RESEARCH_ONLY`, or `REJECT` status with license/revision/provenance/hash information.

Keep evaluation manifests contamination-separated from training and synthetic curriculum generation.

## Private-data implication

The active curated E0/E-INF/A-SYN natural-language content is only on the order of hundreds of thousands of tokens. Therefore private identity material must specialize a general N0 semantic substrate rather than create broad language competence from scratch.

Later N1/N2 train/eval partitions must be grouped by underlying E0 support / base policy / gap / coverage family so mechanically related rows cannot leak between partitions.

The 13,719 alternate policy branches remain **unordered alternatives**, not automatic negatives.

## Current next action

The next active task is **N0-R1: build the competency/evaluation registry**. It should be machine-readable and define each competency, candidate public datasets, license status, training/eval role, contamination key, metric, and gate.

Then:

1. N0-R2 public data/license manifest builder;
2. N0-R3 tokenizer bake-off harness;
3. N0-R4 teacher qualification harness;
4. N0-R5 public-data architecture mechanics matrix;
5. N0-R6 scaling pilot;
6. N0-R7 target N0 freeze;
7. N0-R8 final owner/Astra pre-weight review;
8. N0-R9 first permanent native semantic checkpoint;
9. N1/N2/N3 private identity learning.

No target-scale/private identity-gradient run is authorized before that sequence reaches its gate.
