# N0 job 575990 — semantic localization resolved; semantic-operator foundation selected

**Date:** 2026-09-21  
**Status:** job 575990 completed cleanly; old frozen semantic-authority path closed; new architecture decision statically qualified; no gradient/GPU authorized yet

## Source diagnostic

Magnolia job `575990` completed `0:0` on CPU with empty stderr.

Source branch:

`alice-eipm-v1-n0-p2a-semantic-localization-v1@8180ed500fcf2181d3a70f5a01481d00ebbbb24a`

The audit reconstructed job 575986 successfully before interpretation.

Observed headline metrics:

| Surface | Aux holdout | Aux seen | Factor macro | Production core |
| --- | ---: | ---: | ---: | ---: |
| Current P2A | 0.2604166567 | 0.2708333433 | 0.3666666731 | 0.4068181813 |
| Teacher-native prompt geometry | 0.2604166567 | 0.2291666716 | 0.4166666731 | 0.3840909004 |
| Teacher-native + best-layer token oracle | 0.3541666567 | 0.3333333433 | 0.4166666731 | 0.4090909064 |

The best-layer result is post-hoc diagnostic only.

## Causal conclusion

The three remaining hypotheses are now separated enough for one architecture decision.

### Prompt/readout mismatch is not primary

Restoring teacher-native prompt/candidate geometry did not materially recover relation semantics.

### Equal-weight P2A fusion is not primary

Even the diagnostic best-layer token replacement plus teacher-native trained-head geometry remained far below the semantic gates.

### Frozen-layer selection is not sufficient

No best-layer diagnostic upper bound approached the required broad open-schema performance.

### Semantic representation objective must reopen

The ratified semantic checkpoint remains historically valid for the capabilities on which it was ratified.

It is no longer sufficient authority for dynamic runtime relation/factor semantics.

Job 575966 remains important: learned P2S-v2 adaptation recovered much stronger semantics than the frozen zero-gradient views, proving useful information exists but is not organized into a sufficiently general open-schema geometry.

## New architecture decision

Branch:

`alice-eipm-v1-n0-semantic-operator-foundation-v1`

Qualified head:

`907dcba89c5c2dbb7ac9119905bbde710f980845`

Static qualification:

GitHub Actions run `35635353225` — SUCCESS

Architecture authority:

- `docs/research/eipm-n0-semantic-operator-foundation-v1.md`
- `configs/eipm/n0/n0_v02_semantic_operator_foundation_v1.json`

The selected family is a **schema-conditioned semantic-operator foundation**.

Key properties:

- shared bidirectional semantic token foundation;
- runtime relation/factor schemas are first-class input;
- query/schema token interaction is trained jointly with the representation;
- no relation-ID or factor-ID semantic authority;
- continuous operator latent;
- shared recurrent relation-step transition;
- structural stop;
- uncertainty/plurality;
- ordered query-evidence coverage;
- continuous relation hypotheses until structural binding;
- Binder v2 retains the first exact sparse structural boundary;
- corrected QSRE structural execution and the structured/evidence/fusion/latent fabric remain preserved.

## Semantic parent policy

The current `alice-n0-semantic-v0.2` step-80 checkpoint SHA

`6c2706984c0e05123c4d88ba456788fbd4c9e7f0fca41d43d9e575ac6e53bf43`

is preserved as:

- an initialization candidate;
- a comparison baseline;
- historical semantic-base evidence.

It is no longer frozen semantic authority for full QSRE.

## No-ceiling policy

This is the full-capability architecture.

Current 136.6M parameter count, 16-layer depth, width, relation count, factor count, hop count, and context-view count are operating points, not product ceilings.

Blind parameter scaling is not authorized. Scale only after the objective/data/mechanics are correct and measured evidence localizes capacity as the remaining bottleneck.

## Closed paths

Do not build:

- P2S-v3;
- P2A-v4;
- another frozen-head ensemble;
- prompt-template search;
- learned fusion weights over the same failed frozen views;
- relation-ID embeddings;
- fixed semantic factor heads;
- fixed-top-k semantic commitment;
- LR/step/width/batch search as a substitute for objective redesign.

## Next authorized work

No GPU run yet.

Next:

1. full-envelope public semantic/operator curriculum;
2. schema-conditioned semantic/operator interfaces;
3. shortcut/leakage audits;
4. static and no-gradient mechanics;
5. one precommitted joint training/evaluation plan.

The next gradient run, when separately authorized, must train the joint semantic-operator foundation rather than another repair head.
