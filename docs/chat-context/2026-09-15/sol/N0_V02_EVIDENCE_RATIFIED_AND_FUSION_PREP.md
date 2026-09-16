# N0 v0.2 — Evidence Specialist Ratified and Cross-Context Fusion Preparation

## Ratified evidence stack

Frozen relation-essential job 575618 passed on the untouched 80-example / 40-pair / 10-family challenge.

Selected evidence checkpoint: relation-repair-v0.1/step-00000080.

Selection rule: earliest checkpoint clearing all frozen relation and ordinary replay retention gates. Step40 solved the relation challenge but failed ordinary replay retention; step80 was the first fully acceptable checkpoint.

Ratified public stack before fusion:
- semantic: targeted-repair-v0.1/step-00000080 — 136,594,435 parameters
- structured: structured-state-pilot-v0.1/step-00000080 — 1,656,064 parameters
- evidence adapter: expanded specialist step80 — 16,002,561 parameters
- repaired dual-endpoint graph: relation-repair step80 — 6,664,963 parameters
- descriptive total before fusion: 160,918,023 parameters

Graph hash: ec9942bd71a39c8552804322f1a36b7a6c5f009446e56423c80e59479ad8fa79.

Frozen relation metrics at selected step80:
- pair flip accuracy 1.0
- family minimum pair flip accuracy 1.0
- mean relation flip margin 0.910261157155037
- permutation cosine 1.0
- ordinary replay macro target support 0.9831965605417887
- ordinary replay minimum family target support 0.934688925743103
- ordinary replay summary cosine 0.9651773393154144

No private identity data or gradients were used.

## Capacity doctrine

There is no hard parameter ceiling. Parameter counts are lineage measurements only. Capability and eventual personality fidelity outrank parameter minimization. Efficiency is a secondary selector among capability-sufficient architectures.

`configs/eipm/n0/alice_n0_stack_state_v0.2.json` is authoritative for current stage state and explicitly supersedes stale pre-pilot fields in `alice_n0_semantic_v0.2.json`, including the old 2.5M structured parameter budget.

## Cross-context fusion design

Implemented `src/alice_personality/n0/cross_context_fusion.py`.

Fusion inputs:
1. raw semantic context tokens
2. typed structured-state field tokens
3. relation/evidence graph field tokens

The fusion module:
- preserves all three contextualized views separately;
- uses explicit view identities;
- uses explicit reliability and availability;
- supports missing views;
- uses cross-view Transformer interaction without new positional embeddings;
- exposes contextualized per-view tokens, per-view summaries, view weights, fused state, and cross-view cosine geometry;
- does not replace the later multi-view latent pooling stage;
- has no hard parameter ceiling and no private identity parameters.

Implemented fusion objectives:
- fused semantic alignment
- soft/plural view routing
- source-view preservation
- disagreement-geometry preservation

## Fusion curriculum

Initial v0.1 compiler incorrectly marked deterministic synthetic template text as non-generated. This was caught before any gradient.

Superseding compiler:
`scripts/eipm/n0/build_n0_v02_cross_context_fusion_curriculum_v0_2.py`

Correct provenance:
- data_origin = deterministic_public_synthetic_template
- source_authority = public_synthetic_training_only
- generated_text = true
- identity_authority = false
- private_identity_content = false

Planned curriculum: 320 examples, 240 train / 80 dev, 10 families covering semantic-decisive, structured-decisive, evidence-decisive, multi-view agreement, unresolved conflict, and missing-view cases.

## Current authorization

Fusion gradient remains CLOSED.

Next action:
run CPU-only `scripts/eipm/n0/run_n0_v02_cross_context_fusion_prepare_v0_2.sh` on Magnolia. If mechanics/tests and curriculum provenance pass, inspect once and then authorize one bounded public fusion pilot on frozen ratified parents.

Do not train on private E0/E-INF/A-SYN in this stage.
