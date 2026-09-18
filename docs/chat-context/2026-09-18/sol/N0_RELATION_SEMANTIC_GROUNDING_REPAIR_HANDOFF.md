# N0 Relation-Semantic Grounding Repair Handoff

**Date:** 2026-09-18  
**Status:** missing-evidence failure localized to relation-semantic endpoint grounding; one bounded repair is CI-ready; no frozen-challenge rerun, threshold change, promotion, scaling, or N0 completion authorized

## Current experiment frontier

`alice-eipm-v1-relation-semantic-grounding @ 09fdd7c1c8cadda4ac61b65f1ff9c32e35ced53d`

Stable build remains:

`alice-eipm-v1-build @ 021c5021a98104b35f9c8e94b19e48d21f25f132`

The semantic-grounding frontier is 51 commits ahead and 0 behind stable. It is experimental and noncanonical.

## Preserved evidence chain

### Downstream arbitration

Magnolia job `575792` showed that the selected endpoint-role repair graph has real downstream value.

Selected graph SHA-256:

`3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f`

Arbitration classification:

`IMPROVEMENT`

Do not discard this graph merely because a later capability remains unresolved.

### Final frozen challenge

Magnolia job `575794` completed with exit `0:0`.

Only failed frozen gate:

`counterfactual_family_min_drop`

Failing family:

`missing_structured_evidence`

Observed family mean target drop:

`-0.009834006428718567`

Required frozen minimum:

`0.005`

All other frozen gate checks passed.

Do not rerun job 575794. Do not change its frozen thresholds.

### Fresh path localization

Magnolia job `575795` completed with exit `0:0`.

Fresh diagnostic result SHA-256:

`1733bf1a5fdc4f979beab6d5198bfb6a77a0e17b4fc96adc134dafd03dd5ebbe`

Localization:

`EVIDENCE_GRAPH_RELATION_SELECTION_REMAINS_DEFECTIVE`

Key signals:

- graph argmax target rate = `0.15625`
- graph field-selection relation-flip pair accuracy = `0.0`
- graph field-selection mean target-minus-foil weight = `-0.6037212647497654`
- mean relation-bias target-minus-foil = `-2.322943687438965`
- graph relation direction changes pooled state = true
- fusion preserves relation difference = true
- evidence causally increases latent relation separation = false
- frozen rows reused = false
- prior diagnostic rows reused = false
- gradient performed = false

The graph reacts to relation direction, but the query-conditioned field selector usually chooses the wrong endpoint. Fusion is therefore not the first defect.

## Why endpoint repair passed but this still failed

Endpoint repair v0.2 trained and tested queries that explicitly describe endpoint roles, such as:

- correcting source;
- object being corrected;
- replacement source;
- superseded target.

That proves the graph can represent and distinguish source and target roles.

The failed `missing_structured_evidence` behavior asks a different semantic question:

> given a correction relation, which value is current/correct?

This requires mapping natural relation meaning to an endpoint role without the query naming `source` or `target`.

Therefore the current defect is:

`query semantics + relation semantics -> correct endpoint role grounding`

It is not evidence that:

- the graph lacks source/target representational capacity;
- structured-state is the bottleneck;
- fusion is the bottleneck;
- latent capacity is insufficient.

## One causal repair

New branch state:

`configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.11.json`

Builder:

`scripts/eipm/n0/build_n0_v02_relation_semantic_grounding_curriculum_v0_1.py`

Trainer:

`scripts/eipm/n0/train_n0_v02_relation_semantic_grounding_v0_1.py`

Runner:

`scripts/eipm/n0/run_n0_v02_relation_semantic_grounding_v0_1.sh`

Magnolia wrapper:

`scripts/eipm/n0/magnolia_n0_v02_relation_semantic_grounding_v0_1.sbatch`

### Trainable scope

Only the already-proven relation-read path is trainable:

- `source_relation_pool_mlp`
- `directed_relation_pool_mlp`
- `pool_relation_embedding`
- `query_projection`
- `pool_query`

Frozen for this causal repair:

- evidence adapter
- graph message passing
- conflict pool
- semantic parent
- structured parent

This freeze is an experiment scope, not a permanent architecture ceiling.

## Fresh semantic curriculum

12 semantic families:

- corrects_current
- corrects_previous
- supersedes_current
- supersedes_previous
- temporal_current
- temporal_previous
- causes_cause
- causes_effect
- supports_supported
- supports_supporter
- derived_item
- derived_basis

32 pairs per family:

- 20 train
- 6 dev
- 6 heldout test

The queries do not explicitly name source/target endpoint roles.

Within each pair:

- fields are identical;
- query is identical;
- metadata is identical;
- only relation direction flips;
- the answer flips according to relation semantics.

No rows from:

- frozen latent challenge;
- job 575795 localization;
- endpoint repair v0.2 heldout test

are used for training.

## Preservation strategy

The repair must preserve both:

1. ordinary graph behavior through the original graph replay cache;
2. explicit endpoint-role behavior through replay against the step-200 parent on endpoint-repair training rows.

The prior endpoint heldout test is not reused.

Checkpoint selection first requires preservation, then semantic dev performance.

## Scale policy

No scaling is authorized now.

The failure is localized to semantic endpoint-role grounding in the existing relation-read path. There is no current evidence of insufficient graph width, graph depth, fusion capacity, latent width/depth/slots, or total parameter count.

Current sizes remain operating points, not ceilings.

If a later valid trace proves a correct signal reaches a module input and that module still cannot preserve/use it, capacity becomes eligible for a bounded scale study with no preset hard ceiling.

## Pre-GPU CI

Workflow:

`N0 Relation Semantic Grounding Contract Check`

Run:

`35383267306`

Result:

`SUCCESS`

Passed:

- Python syntax;
- Bash syntax;
- fresh semantic curriculum isolation;
- no observed diagnostic/heldout wording leakage;
- preservation replay contract;
- exact parent/localization/arbitration provenance binding;
- no training on frozen/localization rows;
- no prior endpoint heldout reuse;
- no Git mutation in runtime runner;
- no permanent parameter ceiling;
- udocker output forwarding.

## Exact next action

Run one Magnolia P100 relation-semantic grounding repair from:

`09fdd7c1c8cadda4ac61b65f1ff9c32e35ced53d`

Do not make another model-changing patch before its result.

If semantic heldout passes while ordinary and endpoint-role behavior are preserved, the next action is one fresh downstream causal check on independent semantic analogues. Do not rerun the frozen challenge.

If it fails, preserve the result and localize the single heldout defect. No automatic hotfix chain.

## Standing doctrine

- N0 is the full production personality-model foundation, not a pilot.
- no current parameter/slot/view/relation count is a permanent product ceiling.
- efficiency must not silently truncate required capability.
- scale only from localized evidence.
- one causal change per failed model gate.
- infrastructure failure is not model evidence.
- frozen challenge and diagnostic rows are never training data.
- private identity gradient remains closed at N0.
- no production promotion while this repair remains unresolved.
