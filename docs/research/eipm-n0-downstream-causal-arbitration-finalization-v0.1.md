# N0 downstream causal arbitration finalization v0.1

## Purpose

`finalize_downstream_causal_arbitration_v0_1.py` closes the frozen arbitration evidence chain without adding another model, proxy, tuning pass, or promotion path. It consumes the already-frozen preflight receipt and the completed canonical-vs-repaired downstream arbitration result, independently revalidates the evidence, and then materializes the exact input for `post_arbitration_decision_gate_v0_1.py`.

The lifecycle is therefore:

1. prepare and preflight the full production graph -> parent-cache -> adaptive multi-view latent stack;
2. execute that frozen arbitration exactly once;
3. finalize the immutable result and run the deterministic post-arbitration decision gate.

No step in finalization trains, tunes, ratifies the endpoint repair, promotes the candidate graph, scales the model, or marks N0 complete.

## Independent evidence checks

Before the decision gate is allowed to run, finalization requires all frozen fields shared by the preflight and completed result to match exactly. This includes source revision, manifest hash, evaluator argv hash, common inputs, common fingerprint, graph-arm bindings, metric policy, and invariants.

The finalizer also re-hashes both raw arm outputs and independently recomputes every declared metric comparison from those raw outputs using the frozen comparison policy. The stored classification must equal the recomputed classification. A hand-edited comparison, classification, raw output, manifest, or common-input binding therefore fails closed.

At normal CLI execution the checked-out Git HEAD must equal the frozen source revision. The finalizer, arbitration protocol, and post-arbitration gate source files must also be clean relative to that commit. Their SHA-256 values are written into the finalization receipt.

## Decision authority

The finalizer does not make a new policy decision. It passes the verified immutable evidence to the existing decision gate. The gate remains authoritative:

- `HARM` -> retain canonical;
- `EQUIVALENT` -> retain canonical;
- `IMPROVEMENT` with no frozen-metric degradation -> authorize exactly one final frozen challenge.

Even the positive path keeps the canonical graph as the current baseline and leaves repair ratification, candidate promotion, training, scaling, `n0_ready`, and `n0_complete` false.

## Outputs

The finalizer refuses to overwrite any output and materializes three artifacts:

- the exact gate config derived from the completed arbitration receipt;
- the decision-gate receipt;
- a finalization receipt binding the preflight receipt, arbitration result, arbitration manifest, both raw arm outputs, gate config, gate receipt, and committed source files by hash.

This receipt is the deterministic handoff to whichever branch of the N0 workflow the gate authorizes. It is not itself N0 readiness evidence.
