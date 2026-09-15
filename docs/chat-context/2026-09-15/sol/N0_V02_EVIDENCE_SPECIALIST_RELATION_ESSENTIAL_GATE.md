# N0 v0.2 Evidence Specialist / Relation-Essential Gate

Date: 2026-09-15

## Specialist pilot evidence

Magnolia job 575615 completed successfully. The specialist architecture preserved the ratified structured step80 checkpoint by construction and trained an evidence-specific view plus graph.

Best compact checkpoint observed on the original relation dev set:
- variant: specialized_compact_384x2_graph256x1
- step: 240
- specialist parameters: 5,767,555
- family macro target support mass: 0.9976738224
- family minimum target support mass: 0.9908929269
- supersession historical target mass: 0.9963491460
- mean summary cosine: 0.9704250395
- counterfactual positive rate under the old drop-first-edge metric: 0.65

Best expanded checkpoint observed:
- variant: specialized_expanded_640x3_graph512x2
- step: 240
- specialist parameters: 21,617,923
- family macro target support mass: 0.9981253316
- family minimum target support mass: 0.9909800490
- supersession historical target mass: 0.9985884726
- mean summary cosine: 0.9768679877
- counterfactual positive rate under the old drop-first-edge metric: 0.65

Expanded-vs-compact gain was negligible on evidence mass. Raw capacity is therefore not the current bottleneck.

## Critical interpretation

The old relation counterfactual removes the first decisive edge. The public training examples often also encode relation semantics in text (for example current/previous, authority, verification). A model can therefore remain correct after an edge is removed by using redundant node semantics. Forcing every edge removal to break the answer would reward artificial edge dependence.

Do not use counterfactual_positive_rate from that evaluation as a standalone ratification or scaling gate.

## New gate

A frozen relation-essential paired challenge is now required before graph specialist ratification.

Challenge principle:
- same field texts within each pair
- same query within each pair
- only relation topology/type changes
- correct evidence target flips
- training is forbidden on these frozen pairs

Coverage: supersession, correction, temporal direction, causal direction, support direction, conflict resolution, mixed update/support, temporal chain, causal chain, support aggregation topology.

Selection priority:
1. worst-family pair-flip accuracy
2. overall pair-flip accuracy
3. relation-flip margin
4. target evidence mass
5. semantic summary fidelity
6. efficiency only after capability

If compact and expanded both pass with materially tied capability, prefer compact. If expanded materially wins on the relation-essential test, retain the larger capacity. If neither passes, create independent analogue repair data and keep the frozen challenge untouched.

Build head containing gate: 7bd15482cf11f2bfcce903fde8bb3dea17227e2a
