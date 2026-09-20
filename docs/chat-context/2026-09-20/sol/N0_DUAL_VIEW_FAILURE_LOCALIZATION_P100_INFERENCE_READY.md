# N0 Dual-View Specialist Failure Localization — P100 Inference Placement

Date: 2026-09-20

The causal-only v0.2 localization audit remained too slow on Magnolia login-node CPU. It reached checkpoint 40 counterfactual inference and ran for ~25 minutes without completing.

This is not model evidence.

Root cause: even after semantic re-encoding was removed, the trained dual-view bridge still performs heavy 17-layer token projections and counterfactual forwards. That workload is appropriate for GPU inference, not the login-node CPU.

No scientific question changed.

Preserve:
- failure-localization-v0.1 (interrupted; no result)
- failure-localization-v0.2 (interrupted; no result)

Authoritative state:
configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.38.json

Branch:
alice-eipm-v1-dual-view-specialist-failure-localization

Exact frontier:
d5d2a4f265d5d10a0e9cb6c12c60c864d257c495

CI:
35494643335 — SUCCESS

v0.3:
- uses one P100 for inference only;
- no optimizer;
- no backward;
- no gradient;
- no training;
- no TEST/challenge;
- semantic backbone remains unexecuted;
- preservation metrics are reused from source checkpoint receipts;
- same four causal counterfactuals and endpoint-role diagnostic.

Wrapper:
scripts/eipm/n0/magnolia_p100_n0_v02_dual_view_specialist_failure_localization_v0_3.sbatch

Runner:
scripts/eipm/n0/run_n0_v02_dual_view_specialist_failure_localization_v0_3.sh

Audit:
scripts/eipm/n0/audit_n0_v02_dual_view_specialist_failure_localization_v0_3.py

Output:
query-edge-dual-view-late-interaction-v0.1/failure-localization-v0.3/audit.json

One GPU job only. Training remains closed after the result until the counterfactual localization is interpreted.
