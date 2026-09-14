# N0 v0.2 Public Teacher Curriculum Contract

**Status:** active authoring contract  
**Private identity data:** forbidden  
**Teacher:** Sol primary, optional specialist support under the same governance

## Purpose

N0 v0.2 uses teacher material to shape generic semantic, pragmatic, epistemic, social, causal, temporal, ranking, and structured-alignment representations. It does **not** teach Elaina-specific identity.

The key change from N0 v0.1 is that a teacher row must teach both **what is preferred** and **why**. The `rationale` field is no longer inert metadata. It becomes a gradient-bearing supervision target through candidate/rationale alignment and semantic contrastive learning.

## Minimum row contract

Every trainable row must contain:

- stable unique `id`;
- `competency`;
- `split`;
- `task=candidate_ranking`;
- `prompt`;
- two or more `candidates`;
- one or more `preferred_indices`;
- a concise `rationale` explaining the governing principle;
- governed `source` / origin metadata.

Rows may optionally include:

- `principle_tag` for a reusable rule such as `preserve_uncertainty`, `respect_information_state`, or `do_not_invent_missing_fields`;
- `difficulty`;
- `hard_negative_type`;
- `invariance_group`;
- `context_requirement`;
- `notes` for authoring/audit only.

## Quality rules

1. Do not create hundreds of lexical clones of one easy pattern.
2. Preferred answer position must vary.
3. Hard negatives should be plausible enough to expose the intended distinction.
4. Rationales should state the governing reason, not merely repeat the preferred answer.
5. Ambiguous evidence must permit ties/plurality when warranted.
6. Training examples must not copy the frozen benchmark wording, entities, or exact scenarios.
7. Dev/test material must be written independently rather than produced by candidate shuffling alone.
8. Social/emotional cases must avoid stereotypes and unsupported mind-reading.
9. Epistemic cases must distinguish direct evidence, inference, uncertainty, hypothetical/counterfactual provenance, and out-of-distribution inputs.
10. Structured-alignment cases must preserve relation direction, evidence linkage, unknown fields, and update/conflict semantics.

## Scale policy

Existing governed v0.1/v0.2/v0.3 public teacher rows remain valid seeds if they satisfy the rationale contract. There are currently 128 such seed rows.

Full N0 v0.2 multi-objective training does not begin until at least **1,000 distinct teacher rows** exist. The initial target before N0 readiness review is **2,500 high-quality rows**.

This is not a quota to fill mechanically. Expansion is organized by competency coverage and observed failure modes. If quality would fall, stop authoring and review instead of padding the count.

Recommended minimum distribution at the 1,000-row gate:

- every one of the 43 competencies has train and dev support;
- at least 15 independent train scenarios per competency where applicable;
- at least 5 independent dev scenarios per competency;
- remaining rows concentrate on harder cross-competency and long-context cases;
- candidate-order and superficial lexical patterns are balanced.

## Objective use

For one teacher row:

- candidate preference loss trains which candidate set is supported;
- candidate/rationale compatibility trains whether a candidate is consistent with the governing reason;
- semantic/rationale contrastive loss pulls the supported semantic representation toward its governing rationale and away from unrelated rationales.

The implementation lives in:

- `src/alice_personality/n0/v02_objectives.py`
- `src/alice_personality/n0/v02_model.py`

## Governance

Teacher supervision must remain public/non-identity in N0. An origin manifest must authorize training and state `private_identity_data=false`.

The N0 teacher curriculum is not a route for introducing Elaina source text, Elaina behavioral completion, Rayan host material, private relationship history, or A.L.I.C.E. continuity state into the weights. Those belong to later governed stages and separate authorization boundaries.
