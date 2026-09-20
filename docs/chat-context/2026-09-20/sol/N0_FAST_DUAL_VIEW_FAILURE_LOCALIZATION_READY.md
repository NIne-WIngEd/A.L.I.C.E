# N0 Fast Dual-View Specialist Failure Localization — Ready

Date: 2026-09-20

## Why v0.1 was slow

The first CPU localization attempt was interrupted by the user after ~20 minutes.

It had not produced a result and therefore is not model evidence.

Root cause:
- it rebuilt ordinary and endpoint preservation query hidden states;
- that required running the semantic backbone on CPU before causal counterfactual evaluation;
- this work was unnecessary because the source training receipts already freeze preservation/no-op metrics and the causal cache already contains the full causal query hidden-state stack.

The interrupted directory must remain preserved:
query-edge-dual-view-late-interaction-v0.1/failure-localization-v0.1

## Fast v0.2 audit

Authoritative state:
configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.37.json

Branch:
alice-eipm-v1-dual-view-specialist-failure-localization

Exact frontier:
0a01d6b30e9d97c2780c84daf40c411e10d50e14

CI:
35493567399 — SUCCESS

Script:
scripts/eipm/n0/audit_n0_v02_dual_view_specialist_failure_localization_v0_2.py

Runner:
scripts/eipm/n0/run_n0_v02_dual_view_specialist_failure_localization_v0_2.sh

v0.2 reuses:
- causal query hidden states from the existing causal cache;
- causal field token states from the existing training field-token cache;
- preservation metrics from immutable source checkpoint receipts.

It does not instantiate or execute AliceN0V02Model.
It does not call attach_query_hidden_states.
It does not re-encode ordinary or endpoint preservation data.

Scientific question unchanged:
1. actual activation + soft binding
2. forced specialist + soft binding
3. actual activation + predicted hard-top1 binding
4. forced specialist + predicted hard-top1 binding
5. relevant-edge source/target residual proposal role accuracy

No optimizer, gradient, GPU, TEST, challenge, retraining or tuning is authorized.
