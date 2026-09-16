# EIPM Expansion Generator Handoff Spec v0.1

**Status:** implementation handoff; subordinate to owner-ratified EIPM hard rules  
**Date:** 2026-09-10  
**Clarification:** 2026-09-16 capability-limit audit  
**Scope:** generation of new E-INF and A-SYN proposals only

## Purpose

Expand the Elaina-derived identity/personality substrate before EIPM-v1 corpus freeze. The generator is a temporary development tool. It has no authority to modify E0, accept candidates, promote candidates, create Alice lived memories, authorize training, or begin weight generation.

## Non-negotiable rules

1. E0 is immutable historical evidence.
2. UNKNOWN is an epistemic label, not a required runtime behavior.
3. When historical behavior is unknown, attempt evidence-constrained E-INF first. When historical truth remains underdetermined but Alice still needs a practical starting behavior, propose A-SYN behavioral priors.
4. A-SYN is never historical Elaina truth and never Alice lived memory.
5. Maximize useful personality-space coverage while minimizing unsupported invention and duplicate paraphrases.
6. Generate competing branches, not one supposedly correct answer.
7. Preserve uncertainty, boundary conditions, counterevidence, and lineage.
8. Existing E0/E-INF/A-SYN inputs are read-only.
9. Generator outputs remain proposals until independently curated.
10. No gradient update, model training, or target-scale EIPM weight generation is authorized.
11. No planning batch size, row target, context packet size, coverage taxonomy, or corpus freeze in this handoff is a permanent personality-capability ceiling.

## Input set

The generator should receive the current compiled identity substrate and coverage map plus the canonical E0 semantic source, owner-ratified EIPM router, existing E-INF candidate material, and existing A-SYN candidate material. A packaged handoff may use the following logical paths:

- `compiled/identity_substrate.jsonl`
- `compiled/coverage_map.json`
- `compiled/judgment_obligations.jsonl`
- `compiled/unknown_competitors.jsonl`
- `compiled/corpus_manifest.json`
- `e0/g2a_gold_candidate_pass2.jsonl`
- `e0/eipm_global_fidelity_router_ratified_entries.jsonl`
- `existing_einf/candidate_review_queue.jsonl`
- `existing_einf/packet_review_dossiers.jsonl`
- `existing_asyn/mc10c_asyn_raw_candidates_v1.jsonl`
- `existing_asyn/mc10c_asyn_selected_packets_v1.jsonl`

## Expansion method

### Phase A — derive coverage cells

Build a behavioral coverage taxonomy from the actual substrate. Coverage cells should represent meaningful combinations of personality dimension, social/relationship context, emotional state, stakes/intensity, public/private setting, temporal state, and uncertainty state. Do not create a giant blind Cartesian product. Include a cell only when the combination could plausibly change behavior.

At minimum consider values, judgment, boundaries, privacy, trust, affection, humor, teasing, criticism, disagreement, conflict, jealousy, embarrassment, shame, vulnerability, ambition, risk, injustice, loyalty, authority, failure, mistakes, disappointment, grief, fear, stress, reconciliation, social status, strangers, friends, close-circle relationships, Rayan-specific historical relationship context, public/private differences, and intensity changes.

Mark each cell as `covered`, `weak`, `missing`, `redundant`, or `unsupported` using only the supplied evidence and existing proposals.

The taxonomy is an audit/search instrument, not a closed ontology. New dimensions or combinations must be added whenever source evidence or measured behavior reveals territory the current map does not represent.

### Phase B — generate proposals for weak/missing cells

For each selected cell, use the **smallest sufficient** E0 evidence packet plus accepted/usable inference context available in the supplied substrate. "Smallest sufficient" means no irrelevant context is sent merely for bulk; it does not authorize truncating evidence that could materially change the judgment. Expand across sources, time, relationship context, or contradictory evidence whenever necessary.

Generate multiple competing branches.

For E-INF proposals, distinguish evidence-derived tendencies from unsupported invention. Include an explicit historical-UNKNOWN competitor when the evidence does not justify a historical claim.

For A-SYN proposals, construct practical Alice starting behaviors that are compatible with known Elaina patterns while remaining explicitly synthetic. A-SYN may bridge a runtime behavioral blank. It may not rewrite history.

Prefer conditional policies over absolute traits. Include boundary conditions and situations in which the behavior should change.

### Phase C — novelty control

Before emitting a proposal, compare it with existing E-INF/A-SYN material and proposals already generated in the current run. Reject near-paraphrases. A candidate should add a new context, boundary condition, trade-off, intensity regime, relationship condition, or genuinely distinct behavioral branch.

### Phase D — saturation

Generation is coverage-driven rather than quota-driven. Continue while new proposals add meaningful non-duplicative behavioral territory. Report cells that remain unsupported instead of fabricating historical certainty.

Use iterative waves so independent curation can influence the next wave. The historical Wave 1 planning target of up to 500 E-INF proposals and up to 1,000 A-SYN proposals was an execution/batching control only. It was never a corpus or personality-space ceiling. Later waves and targeted regeneration remain available whenever accepted/rejected results or later fidelity measurements expose additional coverage needs.

## Required proposal schema

Each proposed row must contain at least:

- `proposal_id`
- `provenance_class` (`E-INF` or `A-SYN`)
- `coverage_cell_id`
- `personality_dimensions`
- `relationship_context`
- `social_context`
- `emotional_context`
- `stakes_or_intensity`
- `scenario`
- `behavioral_proposal`
- `supporting_E0_unit_ids`
- `supporting_EINF_ids`
- `counterevidence_or_tension_ids`
- `historical_Elaina_truth` = false unless the row is merely quoting an E0 anchor, in which case it must not itself be emitted as a new proposal
- `Alice_lived_memory` = false
- `autobiographical_recall_allowed` = false for the generated proposal itself
- `runtime_behavioral_prior_allowed`
- `confidence_or_support_strength`
- `uncertainty_reason`
- `boundary_conditions`
- `disconfirmation_conditions`
- `novelty_claim`
- `candidate_role` (`primary_branch`, `alternative_branch`, `historical_unknown_competitor`, or `counterfactual_competitor`)
- `generator_has_acceptance_authority` = false
- `model_training_authority` = false

Do not present generator confidence as probability that a historical claim is true.

The schema itself is extensible. Additional fields must be added when necessary to preserve source nuance, provenance, temporal dependence, relationship dependence, uncertainty, or another fidelity-relevant distinction. Schema convenience must never force information loss.

## Required output package

Return a ZIP containing:

- `einf_proposals.jsonl`
- `asyn_proposals.jsonl`
- `historical_unknown_competitors.jsonl`
- `counterfactual_competitors.jsonl`
- `coverage_cells.jsonl`
- `coverage_summary.json`
- `unresolved_or_unsupported_cells.jsonl`
- `generation_manifest.json`
- `README.md`
- `SHA256SUMS.txt`

The manifest must record input file hashes, generator/model identity, generation timestamp, prompt/spec version, row counts, coverage counts before and after proposal generation, and an explicit statement that no candidate was accepted or promoted and no training occurred.

## Independent curation boundary

The generation model must not perform the final acceptance decision. Sol independently curates each proposal. Ambiguous Elaina-fidelity decisions may be escalated to the owner.

A plausible branch that is not selected remains a competing or co-valid hypothesis when evidence permits. It may become a hard negative only when it is genuinely contradicted, incompatible, provenance-invalid, or otherwise wrong for the supervised judgment. "Not selected" must never be treated as equivalent to "false."

Only after curation and global consistency analysis can accepted material enter a review-frozen EIPM training corpus. That freeze is a lineage snapshot and may be reopened through governed targeted regeneration when later fidelity evidence exposes a gap.

## Weight gate

This handoff does not authorize training. Before the first private EIPM gradient update or target-scale weight generation, the owner must receive the explicit final pre-weight review request and approve proceeding. That gate validates provenance and lineage; it is not a capacity limit.

## Third-party service terms

Before using any hosted model service to produce material that will be used to develop EIPM, confirm that the applicable service agreement permits that use. If it does not, use a locally/self-hosted generator with licensing compatible with the intended model-development workflow.
