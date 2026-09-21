# Production QSRE P2 job 575956 failure localization v1

**Status:** zero-gradient diagnosis authority; no rerun or GPU authorization  
**Failed model source:** `dba3d4b100f081b0c09fe63ae002205244af5b44`  
**Magnolia job:** `575956`  
**Observed result:** `FAIL_QSRE_PRODUCTION_P2_OPERATOR`, exit 21 after 1600 gradient steps

## 1. What is already proven

This is a genuine learned-model failure, not a Magnolia or harness failure.

The P100 runtime, CUDA binding, uDocker, preserved lineage, and P1 checkpoint all executed successfully. The corrected P1 checkpoint at step 50 requalified with 1.0 row success, 1.0 family minimum, 1.0 path-family minimum, and 1.0 open-schema downstream success without changing its weights.

P2 then completed the full governed 1600-step optimization budget. Its final training loss was approximately 0.0042, but no DEV checkpoint was eligible. The final/best checkpoint reached only:

- relation-sequence exact: 0.440476
- open-schema relation exact: 0.093750
- traversal accuracy: 0.789474
- unknown termination accuracy: 0.0
- downstream row success: 0.773810
- three-hop downstream success: 0.0
- ordered-path downstream success: 0.25
- path-role downstream success: 0.375
- path-latest downstream success: 0.5

The failure is therefore not under-training. Training fit became nearly perfect while transfer remained far below the precommitted contract.

## 2. Primary localization

### A. Relation identity/order is still the dominant unresolved capability

The failure pattern is strongly sequence-shaped.

One-step families such as relation filtering, endpoint role, aggregate selection, and several arbitration families can reach high or perfect downstream scores. Multi-step families remain the weakest, especially three-hop and ordered-path.

The v1 operator does not explicitly segment an ordered query into consumed and remaining relation evidence. At every recurrent step it re-scores the entire query against the entire schema. The prior relation influences the recurrent state, but there is no coverage state, no ordinal pointer, no consumed-token mask, and no direct mechanism enforcing that step 2 grounds to the phrase after step 1 rather than the same globally salient relation phrase again.

This is an architecture-level hypothesis supported by the observed family pattern. It is not yet isolated from termination-mass effects, so the checkpoint diagnostic must measure identity-only sequence accuracy separately from active-mask accuracy.

### B. Relation selection and STOP/UNKNOWN were incorrectly coupled in v1

The v1 operator concatenates all relation logits, STOP, and UNKNOWN into one sparsemax. Relation logits are scaled by an exponential relation scale while STOP and UNKNOWN come from independent scalar heads.

That gives semantically different events a shared competitive normalization and lets relation-cardinality / semantic-match scale affect termination.

This is especially damaging for UNKNOWN. Job 575956 reached only 0.3125 unknown-termination accuracy at its best observed point and ended at 0.0.

### C. Cumulative survival can erase otherwise-correct longer programs

In v1, active mass at step t is the product of all prior continuation/relation masses:

`effective_mass_t = survival_t * relation_mass_t`

and `survival` is multiplied by continuation after every step.

The evaluator calls a step active only when effective relation mass is at least 0.5. A three-step path therefore requires every preceding continuation decision to remain sufficiently close to one. The diagnostic must separate:

- relation identity correct but effective mass below 0.5;
- active-mask failure;
- true relation identity/order failure.

This distinction cannot be recovered from the aggregate DEV metrics alone.

### D. Open-schema matching was not sharing one exact semantic metric

P1 trains `schema_encoder.schema_projection`, `schema_norm`, layer embeddings, and pooling under oracle relation IDs. P1's geometry penalty preserves relation-to-relation cosine geometry, not absolute query-to-schema alignment.

P2 v1 then initializes only the query projection and layer embedding from the P1 schema encoder. It does **not** copy the learned schema LayerNorm affine state. It subsequently trains the query-side projection while schema matching continues to use the separately frozen P1 schema path.

The code comment says query and schema late interaction begin in the "exact same learned semantic coordinate system", but that is not literally guaranteed by the implementation.

This is consistent with the open-schema relation-exact result collapsing to 0.09375 despite much higher open-schema downstream success under oracle support.

### E. Factor imbalance is real but secondary

Role, direction, modifiers, control, and termination repeatedly rise far above relation-sequence accuracy, while traversal plateaus around 0.789474.

The v1 trainer uses row-weighted losses. It does not macro-balance the factor classes. This can explain the stable traversal plateau and some head-specific ceilings, but it does not explain the relation-sequence/open-schema failure by itself.

### F. Oracle support makes downstream execution less diagnostic than operator exactness

P2 executes with oracle support edges. A wrong relation argmax can still leave nonzero soft probability on the correct relation and produce a partly correct downstream answer.

That is why final open-schema downstream success can be 0.71875 while open-schema relation-program exactness is only 0.09375.

The operator program metrics are the primary evidence for this stage.

## 3. What the unrun operator-v2 recovery fixes

The current unrun v2 recovery is directionally aligned with the observed failure:

- relation selection is separated from CONTINUE/STOP/UNKNOWN;
- query and raw runtime-schema text use one shared projection/normalization metric;
- all runtime schema candidates are present during training while open-schema query labels remain withheld;
- symmetric query-to-schema and schema-to-query late interaction replaces one-sided max matching;
- factor-specific query slots separate role/traversal/direction/control from the terminal program state;
- macro-balanced factor/event losses address class imbalance;
- schema-only self-calibration is allowed without DEV query labels.

These are causal architecture changes, not LR/step/width tuning.

## 4. What v2 does **not** yet prove

The v2 source still uses a recurrent program state that re-attends the complete query at every relation step. It does not implement an explicit consumed-span / remaining-span state.

Therefore ordered relation decoding remains an unresolved risk until either:

1. the v1 checkpoint diagnostic shows most long-path failure was actually active-mass / event coupling rather than relation ordering, or
2. a future architecture explicitly proves stepwise query-evidence differentiation.

The v2 implementation also retains survival multiplication. Because CONTINUE is now an independent event this may cease to be a practical bottleneck, but it must be measured rather than assumed.

## 5. Required checkpoint diagnosis before interpreting or modifying v2 further

Run exactly one zero-gradient CPU diagnostic over the preserved **best/final v1 P2 checkpoint**. It must report for both query views:

- full relation-program exactness;
- relation-identity exactness ignoring active mask;
- active-mask exactness;
- relation top-1 by target step;
- exactness by target sequence length;
- reversed-order match rate for length >= 2;
- repeated-relation collapse rate;
- effective active mass by target step and target length;
- extra active-tail rate;
- step-to-step query-attention cosine similarity;
- step-to-step relation-distribution cosine similarity;
- family-level relation identity vs full-program exactness;
- core versus open-schema relation identity;
- UNKNOWN versus STOP terminal behavior;
- P1 schema-normalization / projection alignment versus the trained P2 query metric.

No optimizer, backward pass, parameter mutation, TEST opening, frozen-final opening, or private identity data is authorized.

## 6. Decision rule

After the diagnostic:

- **Identity good, active mask bad:** event/survival coupling is primary. v2's decoupled event design is justified; no ordered-parser expansion yet.
- **Identity bad primarily at later steps and step attentions remain nearly identical:** ordered query-evidence consumption is primary. Strengthen the recurrent relation decoder before any v2 GPU run.
- **Core identity good, open-schema identity bad:** shared semantic metric / schema transfer is primary. v2's shared query-schema matcher is justified.
- **UNKNOWN still loses because terminal relation mass remains high:** decoupled rejection is required; v2 is justified.
- **All of the above occur:** keep the factorized v2 corrections but do not treat them as sufficient until ordered-step differentiation is explicitly qualified.

This diagnosis does not authorize a P2 rerun. It exists to prevent another GPU architecture loop.
