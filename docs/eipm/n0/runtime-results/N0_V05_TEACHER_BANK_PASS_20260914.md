# N0 v0.5 Public Teacher Bank — PASS

**Date:** 2026-09-14  
**Status:** PASS / full public multitask floor open  
**Private identity data:** false  
**Private identity gradient authorized:** false

## Runtime result

The owner executed the deterministic v0.5 public coverage wave on Magnolia and then ran the governed teacher-bank audit against the resulting runtime registry.

Observed final state:

- registered rows: **1,020**;
- unique row IDs: **1,020**;
- registered competencies: **51**;
- core competencies: **43**;
- voice-expression competencies: **8**;
- every competency: **15 train + 5 dev** scenarios;
- generated v0.5 wave: **637 rows** = 490 train + 147 dev;
- distinct reusable principle tags across v0.4+: **82**;
- fixed core readiness suite: 43 base competencies;
- fixed voice readiness suite: 8 base competencies;
- `coverage_gate_ok=true`;
- `full_multitask_gate_open=true`;
- failures: none;
- private identity data: false;
- private identity gradient authorized: false.

Preferred-answer positions also remained acceptably distributed across the v0.4+ bank rather than collapsing to one slot:

- position 0: 349 / 39.84%;
- position 1: 294 / 33.56%;
- position 2: 233 / 26.60%.

## Interpretation

The public N0 v0.2/v0.5 teacher bank has now crossed the **real coverage floor**, not merely the numerical 1,000-row threshold. The owner-approved requirement that every registered competency have at least 15 independent train and 5 dev scenarios is satisfied.

The remaining gap to the historical 2,500-row readiness target is **1,480 rows**, but that target is not a prerequisite for beginning the first bounded native training tranche. Additional teacher material should now be driven primarily by observed model failures and difficult cross-competency cases rather than blind quota filling.

## Voice-first significance

Voice is no longer a late add-on. All eight generic spoken-expression competencies are at the same 15/5 coverage floor as the core semantic/judgment competencies. The private Elaina-derived voice overlay remains separate and is reserved for N1/N2 under the existing private-gradient authorization boundary.

## Next build action

Proceed to the first bounded **alice-n0-semantic-v0.2** native multi-objective weight update using:

- the passed native 136,594,435-parameter v0.2 model;
- the governed 48k v0.2.1 tokenizer;
- the balanced public v0.2.1 train corpus already materialized on Magnolia for the first bounded tranche;
- the 1,020-row public teacher bank;
- objective mix: span MLM + candidate preference + principle/rationale alignment + multi-positive semantic contrastive learning;
- pre/post fixed semantic and voice-readiness evaluation;
- no private Elaina gradient.

The v0.1 352M step-1000 checkpoint remains a pathfinder baseline and must not initialize v0.2.
