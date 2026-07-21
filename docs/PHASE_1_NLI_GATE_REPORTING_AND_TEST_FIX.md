# P1.11 NLI Gate Reporting and Stale-Test Fix

Two non-behavioral issues were found after enabling the NLI claim-support gate.

1. The MS MARCO reranker is intentionally disabled because the seven-case
   support audit regressed from 0.421053 to 0.083333, but an older test still
   asserted that the reranker must be enabled.
2. The NLI gate actually ran, but the gate-result dictionary replaced the
   initial `enabled=True` marker and omitted the `enabled` key. The user-visible
   response summary therefore incorrectly displayed `support_gate_enabled=false`
   even when claims had been evaluated and dropped.

This patch updates the stale test and ensures the gate result reports:

- enabled
- entailment_threshold
- contradiction_threshold
- input_claim_count
- kept_claim_count
- dropped_claim_count

It does not change NLI thresholds or claim filtering behavior.
