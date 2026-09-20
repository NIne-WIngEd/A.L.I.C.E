# N0 Query-Edge Routed Specialist Failure Localization v0.1

**Date:** 2026-09-20

## Classification

Magnolia job `575888` is a valid model experiment, not an infrastructure failure.

- scheduler state: `COMPLETED`
- exit code: `0:0`
- source revision: `cf15e2b13a5ca4d22a7070e54c95460959206c33`
- result SHA-256: `99fc21394bdfbb8e3500bf0efa7c3ce836a2e071b8b11a1c733f73aa99b6fea2`
- result status: `FAIL_QUERY_EDGE_CAUSAL_DEV_OR_PRESERVATION_STOP_NO_HELDOUT_NO_AUTOMATIC_HOTFIX`
- eligible checkpoints: none
- selected checkpoint: none
- causal test evaluated: false
- frozen challenge evaluated: false
- parent graph parameters changed: false
- preservation DEV used for gradient: false

The single authorized P100 experiment is consumed.

Do not rerun it. Preserve `training-v0.1` as evidence.

## What improved

The query-edge architecture is materially better than the rejected query-only endpoint-polarity residual.

Initial query-edge DEV:

- row accuracy: `0.0625`
- quad accuracy: `0.0`
- routing accuracy: `0.3402777777777778`
- mean target margin: `-0.6462395703420043`

Best observed overall field behavior was around steps 160–200.

Step 160:

- row accuracy: `0.18055555555555555`
- quad accuracy: `0.05555555555555555`
- routing accuracy: `0.4930555555555556`
- mean target margin: `-0.34061604355358416`
- endpoint pair accuracy: `0.9791666666666666`
- endpoint row accuracy: `0.9791666666666666`
- endpoint mean margin: `0.8355223461985588`

Step 200:

- row accuracy: `0.18055555555555555`
- quad accuracy: `0.05555555555555555`
- routing accuracy: `0.4861111111111111`
- mean target margin: `-0.345081679833432`
- ordinary macro top-1 support accuracy: `1.0`
- ordinary min top-1 support accuracy: `1.0`
- endpoint pair accuracy: `0.9583333333333334`
- endpoint row accuracy: `0.96875`
- endpoint mean margin: `0.8325983931620916`

At step 200 ordinary preservation missed only macro target-support mass:

- observed: `0.9806338856617609`
- required: `0.9810003264417648`

This is evidence that parent distillation is doing useful work. It is not evidence that thresholds should be relaxed.

## What still failed

### Routing never became reliable

Routing loss was near the random three-way cross-entropy baseline `ln(3) ~= 1.0986` for most of training.

Examples:

- step 1: `1.0980912446975708`
- step 40: `1.096488356590271`
- step 80: `1.1115995645523071`
- step 120: `1.0958815813064575`
- step 160: `1.0592268705368042`
- step 200: `1.0671123266220093`

Worst-family routing remained poor:

- step 80: `0.08333333333333333`
- step 120: `0.125`
- step 160: `0.16666666666666666`
- step 200: `0.16666666666666666`

At step 200 temporal-successor routing remained `0.16666666666666666`, below the uniform same-relation chance floor `1/3`.

### The routing loss did not supervise the runtime gate

The qualified bridge computes actual specialist contribution using:

`edge_specialist_gate = tanh(gate_mlp(...))`

and then:

`source_residual = edge_specialist_gate * source_residual_proposal`

`target_residual = edge_specialist_gate * target_residual_proposal`

However, the training script defined routing supervision as a different quantity:

`routing_score(edge) = dot(edge_query_language_state, edge_binding_query) / sqrt(width)`

The routing cross-entropy was applied to this proxy score.

Therefore:

- the supervised routing score did not directly control specialist contribution;
- the actual independent gate could activate an edge even when the routing proxy did not select it;
- improving the proxy did not guarantee that runtime computation moved to the selected edge;
- field-selection loss could train the gate/residual path without solving the proxy routing task.

This is a control-plane mismatch, not a missing cross-attention capability.

### Independent gates did not enforce competition

Each edge has its own `tanh` gate.

There is no normalization or competition across edges.

The irrelevant-edge gate penalty did not solve this. It became large:

- step 40: `0.341525673866272`
- step 80: `0.6995546221733093`
- step 160: `0.7658759355545044`
- step 200: `0.8111457824707031`

Because this is mean squared gate magnitude, the late values imply large specialist activation remained on query-irrelevant edges.

That explains why parent preservation oscillated even with explicit parent distillation.

## Updated localization

The language↔graph representation boundary is no longer the primary blocker.

The CPU qualification proved that:

- edge content changes query-token attention;
- same-relation edges produce distinct query-conditioned states;
- the parent can remain exact at initialization.

The failed P100 run localizes the current bottleneck to:

`query-edge specialist routing / contribution control plane`

Specifically:

`supervised routing proxy != runtime contribution gate`

plus:

`independent per-edge gates lack explicit competition and no-op routing`

## Frontier evidence

Current routing research supports treating route assignment as a first-class control problem.

Microsoft Research's 2026 counterfactual routing analysis reports that trained top-k routers can select suboptimal routes even when better equal-compute routes already exist in the frozen model; router-only correction can recover reasoning performance.

2025–2026 expert-router coupling work likewise argues that downstream task loss does not guarantee alignment between router decisions and expert capability.

2026 work on routing-free and geometric-coupling MoE systems also emphasizes that routing and expert activation should be coupled directly rather than supervised through an unrelated proxy.

The transferable lesson for N0 is not to copy generic MoE load balancing. N0 has a stronger form of supervision: the causal curriculum knows the relevant graph edge.

The supervised quantity should therefore be the same quantity that actually scales specialist contribution.

## Rejected training/control pattern

Close this pattern:

`proxy routing score + independent tanh contribution gates`

Do not repair it by:

- increasing routing-loss weight;
- decreasing learning rate;
- increasing steps;
- relaxing preservation thresholds;
- adding PCGrad;
- increasing bridge width;
- adding another proxy router;
- keeping the same gate and merely changing its penalty.

Those would not repair the control-plane mismatch.

## Next causal architecture hypothesis

Preserve the qualified cross-attention representation path.

Replace only the routing/control plane with a **Competitive Edge Router with No-Op Route**.

For each active directed edge:

1. use the already-qualified edge-specific fused representation;
2. emit one route logit;
3. add an explicit parent/no-op route;
4. normalize route logits competitively across all active edges + no-op;
5. multiply each edge's source/target residual proposal by its actual route probability.

The route distribution used by training must be the exact route distribution used by runtime contribution.

### Exact-parent initialization

Remove the independent `tanh` edge gate.

Zero-initialize source and target residual readout heads instead.

At initialization all specialist residual proposals are exactly zero, so parent output remains exact even if route probabilities are non-zero.

This is not a dead learning path:

- routing cross-entropy directly trains route logits on the first step;
- field-selection CE directly trains the zero-initialized residual readout weights on the first step;
- upstream specialist representation receives field gradients from subsequent steps.

### Routing targets

For fresh query-edge causal TRAIN rows:

- target route = the known `relevant_edge_index`;
- route competition spans all active directed edges plus no-op, not only same-relation edges.

This makes relation selection and edge selection one production-real decision.

For ordinary and endpoint preservation TRAIN anchors:

- route target = explicit parent/no-op;
- frozen-parent KL remains as a second stability signal.

The old irrelevant-edge gate penalty is removed because competitive routing plus no-op supervision directly expresses the intended behavior.

## Why no new causal corpus is required yet

The existing v0.1 query-edge curriculum remains suitable for this one causal change:

- it has three same-relation candidate edges;
- it has a different-relation distractor;
- it labels the relevant edge;
- its causal TEST remains unopened.

DEV has been used for architecture development, which is its intended role.

Do not open TEST.

Changing both the architecture and the corpus now would destroy causal attribution.

## Next authorized work

Authorized:

1. preserve this failed result and close the proxy-routing control pattern;
2. create a new competitive-router branch;
3. implement route-logit + no-op competition using the existing qualified cross-attention representation;
4. zero-initialize residual readouts for exact-parent behavior;
5. update training-objective design so supervision targets the actual runtime route;
6. static CI;
7. CPU/no-gradient exact-parent and route-control qualification.

Not authorized yet:

- optimizer;
- gradient;
- P100;
- causal TEST;
- frozen challenge;
- scale;
- parent graph retraining;
- semantic backbone retraining;
- private identity gradient;
- promotion;
- N0 completion.

A new P100 decision requires a separate qualification receipt.
