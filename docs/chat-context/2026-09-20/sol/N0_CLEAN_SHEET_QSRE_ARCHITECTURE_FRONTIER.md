# N0 Clean-Sheet QSRE Architecture Review — Frontier

Date: 2026-09-20

## Why this frontier exists

The recent N0 query/relation/edge sequence was part of a longer failure family, not four isolated mistakes.

Valid learned-model failures around the same boundary include:
- 575797 relation-semantic grounding
- 575798 semantic-role residual
- 575800 query-relation-role router
- 575825 relation-conditioned multilayer interface
- 575888 query-edge cross-attention bridge
- 575907 competitive edge router
- 575912 dual-view late-interaction specialist

575908 was a pre-gradient trainer-boundary failure and is not model evidence.

Counterfactual inference job 575913 showed:
- specialist activation is not the primary blocker;
- hard support materially improves over diffuse soft support;
- hard support still does not close the task;
- endpoint/argument-role execution remains nonsystematic.

The repeated architectural invariant was:
query-conditioned capability was forced to modify or mix with the same global parent field-decision surface.

## Clean-sheet decision

Research branch:
alice-eipm-v1-n0-clean-sheet-relational-execution-review

Current frontier:
5687469a5a3913b8abce889a1ef397ae95125bb2

Authoritative state:
configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.41.json

Selected family:
QSRE — Query-Conditioned Sparse Relational Executor

QSRE is an N0 relational capability, not the whole N0 model.

Required shape:
1. multi-layer continuous query operator state
2. token-level adaptive sparse structural support
3. query-conditioned node/edge relational execution with explicit source/target argument positions
4. structural relational readout on claimed support
5. separate general-parent fallback/applicability
6. output compatible with existing fusion/latent fabric

Closed pattern:
parent global field score + specialist/query scalar residual is no longer an acceptable relational executor.

## Components preserved

- AliceN0V02 semantic backbone: keep frozen initially; token and multi-layer states remain first-class.
- StructuredStateEncoder: keep.
- EvidenceViewAdapter: keep frozen initially.
- DualEndpointEvidenceGraphEncoder: reuse learned representations/relation knowledge/general fallback; do not preserve field_weights as universal relational authority.
- ratified CrossContextFusion: keep.
- adaptive latent-pool lineage: not modified by this audit.

## Deterministic preimplementation proof

Proof obligations:
configs/eipm/n0/n0_v02_relational_execution_proof_obligations_v0_1.json

Diagnostics:
configs/eipm/n0/n0_v02_qsre_preimplementation_diagnostics_v0_1.json

Fixtures:
evaluation/eipm/n0/n0_v02_qsre_oracle_contract_v0_1.jsonl

Evaluator:
scripts/eipm/n0/evaluate_n0_v02_qsre_oracle_contract_v0_1.py

CI run 35496944177: SUCCESS.
17 fixtures passed D1-D5:
- role reversal and parent-global distractor isolation
- zero/one/many support
- ambiguity/plurality
- ordered two-hop composition
- fallback/defer

This proves interface expressivity only. It does not prove learnability.

## Concrete blueprint

docs/research/eipm-n0-qsre-architecture-blueprint-v0.1.md

Key design:
- operator anchors attend across multi-layer semantic states; no hard-coded universal layer
- relation and argument-role anchors are migratable, with continuous residual state
- token late interaction remains the semantic binding basis
- adaptive exact-zero sparse support preferred over permanent top-k
- support selection and applicability remain separate
- executor uses explicit node + edge states and role-specific endpoint positions
- query operator participates inside edge/node updates
- structural readout operates only over support
- QSRE does not blend its field distribution into the parent global field distribution

## Future causal training order

No training is authorized yet.

If later authorized:
T1 executor/readout with oracle operator + oracle support
T2 learned operator with oracle support + proven executor
T3 learned support with proven operator + executor
T4 end-to-end QSRE
T5 N0 fusion integration only after standalone QSRE capability

This ordering is mandatory to avoid blaming routing for executor failures or executor for semantic parsing failures.

## Current boundary

Authorized:
- static tensor-interface/mechanics decision only

Not authorized:
- QSRE source implementation
- optimizer
- gradient
- GPU training
- TEST/challenge
- semantic/parent retraining
- scaling
- private identity gradient
- promotion

Next artifact:
one exact static tensor-interface/mechanics decision. Only after that may CPU/no-gradient mechanics implementation be considered.
