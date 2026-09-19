# N0 relation-conditioned multi-layer training decision v0.1

## Decision

One bounded interface-only training experiment is scientifically justified **after** a parent-only preservation calibration passes.

This is not a return to QRR or the previous semantic-role residual path.

The candidate uses the real relation-conditioned multi-layer interface and its per-edge `edge_relation_query_state`.

## Why the experiment is justified

The current evidence chain is coherent:

1. QRR v0.3 failed dev by collapsing toward one global endpoint preference.
2. The no-gradient semantic representation audit showed token-level late interaction preserved more directional information than pooled views.
3. The layerwise audit showed that the hard relations are depth-local:
   - `causes` becomes informative across a mid-layer band;
   - `supports` appears around layers 12-13 and is lost again at the final layer.
4. The relation-conditioned layer map was compiled directly from that audit.
5. The real interface passed exact-map runtime qualification without changing its parameter state.
6. The fresh causal curriculum now independently varies query role and edge direction.
7. The complete CPU preparation hash-bound the exact semantic, structured, adapter, parent graph, ordinary replay, endpoint replay, map, qualification, curriculum, and preservation artifacts.

That is enough evidence to test the architecture once.

## Candidate architecture

Implementation:

`src/alice_personality/n0/evidence_graph_multilayer_interface.py`

Class:

`RelationConditionedMultiLayerEvidenceGraphEncoder`

The selected dual-endpoint graph stays frozen.

The new language-graph path is:

`frozen semantic intermediate token layers -> RelationConditionedMultiLayerQueryInterface -> edge_relation_query_state -> zero-initialized signed source/target residual -> frozen parent graph pool logits`

The residual readout starts exactly at zero.

Therefore a loaded parent checkpoint must produce exact parent behavior before training.

The trainable parameters are only:

- the relation-conditioned multi-layer query interface;
- the zero-initialized scalar edge readout.

The following remain frozen:

- semantic backbone;
- structured-state encoder;
- evidence-view adapter;
- selected parent graph.

There is no generic SOURCE/TARGET router.

There is no raw mean-pool router.

There is no final-layer-only shortcut.

There is no hardcoded layer 12.

There is no QRR v0.4.

There is no reuse of the previous semantic-role residual.

## Parent-only calibration

Before the optimizer is allowed to run:

`scripts/eipm/n0/run_n0_v02_calibrate_multilayer_preservation_v0_1.sh`

must produce:

`PASS_PARENT_ONLY_PRESERVATION_CALIBRATION`

The calibration receives no candidate.

It evaluates the frozen parent twice on:

- ordinary graph replay;
- endpoint-role replay.

It freezes numerical tolerances using:

`atol = max(1e-6, 2 * max_pairwise_abs_repeat_delta)`

with `rtol = 0`.

Candidate output cannot influence this policy.

The calibration itself is CPU-only.

## One bounded training run

Trainer:

`scripts/eipm/n0/train_n0_v02_relation_conditioned_multilayer_interface_v0_1.py`

Runner:

`scripts/eipm/n0/run_n0_v02_relation_conditioned_multilayer_interface_v0_1.sh`

Magnolia wrapper:

`scripts/eipm/n0/magnolia_p100_n0_v02_relation_conditioned_multilayer_interface_v0_1.sbatch`

The operating training budget is 160 steps with checkpoints every 40 steps.

This is a causal-study budget. It is not a permanent product limit.

Only the fresh causal **train** quads contribute gradient.

The preservation lanes are evaluation-only.

The causal dev split is evaluation-only.

The causal test split remains unopened.

The frozen challenge remains unopened.

## Causal dev readiness

A checkpoint can become eligible for a later heldout decision only if all of these are true:

1. every metric in the parent-only preservation policy passes;
2. no relation family's dev row accuracy regresses relative to the exact parent;
3. `causes` strictly improves relative to the parent;
4. `supports` strictly improves relative to the parent;
5. overall causal-quad accuracy strictly improves;
6. worst-family quad accuracy does not regress;
7. mean target margin strictly improves.

The selected checkpoint is then chosen only among preserved, causally ready candidates.

Test data does not participate in checkpoint selection.

## After the run

### Dev or preservation failure

Stop.

Do not patch the interface automatically.

Do not widen it automatically.

Do not rerun with more steps automatically.

Do not open test.

Reopen the language-graph boundary and determine whether the relation semantics should instead be represented by another interaction mechanism, by semantic objective changes, or by a higher reasoning layer.

### Dev and preservation pass

Do not open test automatically.

Make one separate explicit decision about whether to open the fresh heldout split without changing the selected candidate.

## Capability policy

N0 remains the full-production personality-model foundation.

This bounded experiment tests one architecture hypothesis. Its current layer candidates, interface width, dataset size, and step budget are not permanent A.L.I.C.E. capability ceilings.

Efficiency remains important. It must come from selective layer access, caching, batching, and stronger architecture rather than removing needed representational capacity.
