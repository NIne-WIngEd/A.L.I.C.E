# N0 downstream causal arbitration v0.1

## Purpose

This protocol answers one narrow causal question: after the relation endpoint-role repair changed graph state, does that change survive, disappear, or become harmful when both graph states are passed through the **same already-frozen downstream latent frontier**?

It is a diagnostic. It is not a repair stage, training stage, promotion stage, scale gate, or ratification mechanism. The endpoint-role repair remains unratified unless a later explicit decision process says otherwise.

## Frozen two-arm design

The run has exactly two semantic roles, not two separately tuned systems:

- `canonical`: the canonical pre-repair graph state.
- `candidate`: the selected relation endpoint-role repair graph state. The current frontier identifies the selected step-200 artifact, but its path and hash must be taken from the authoritative run receipt before launch.

Both arms must use one evaluator command template and one set of content-addressed common inputs. Those common inputs include the evaluator implementation, latent checkpoint, evaluator configuration, and evaluation/challenge set. The manifest also binds the source revision. All bound files are SHA-256 verified before either arm runs.

The driver runs the canonical arm first and the candidate arm second through the same command template. The only arm-specific substitutions are the graph path and raw output path. There is no optimizer, gradient path, challenge-row training, retraining, tuning, scaling, promotion, or automatic mutation path in this protocol.

## Classification

Metrics are predeclared in the manifest. Each metric declares whether higher or lower is better plus absolute and relative numerical tolerances. For every metric, the candidate is classified against canonical as `IMPROVEMENT`, `EQUIVALENT`, or `HARM`.

The run-level classification is deliberately conservative:

- any metric outside tolerance in the harmful direction -> `HARM`;
- otherwise, at least one metric outside tolerance in the beneficial direction -> `IMPROVEMENT`;
- otherwise -> `EQUIVALENT`.

These labels describe the frozen downstream comparison only. Even `IMPROVEMENT` does **not** ratify the failed endpoint repair and does not authorize promotion, scaling, or training. The result always records `ratifies_repair=false`, `scale_authorized=false`, and `promotion_authorized=false`.

## Fail-closed preflight

`downstream_causal_arbitration_v0_1.py` refuses to run when any frozen input is absent or has the wrong SHA-256, when the two graph hashes are identical, when protocol invariants are weakened, when the evaluator command does not carry graph/output substitutions, or when known training/mutation flags appear in the evaluator command.

The training-argument deny-list is only a secondary safety check. It does not prove that an arbitrary program is evaluation-only. The evaluator itself therefore must be a known evaluation-only implementation and must be pinned in `common_inputs` by SHA-256.

A preflight-only invocation validates and fingerprints the complete frozen manifest without executing either arm:

```text
python scripts/eipm/n0/downstream_causal_arbitration_v0_1.py \
  --manifest <pinned-manifest.json> \
  --output <preflight-receipt.json> \
  --preflight-only
```

The example manifest is intentionally non-runnable. Replace every placeholder from authoritative execution receipts and compute hashes from the frozen artifacts. Do not infer historical paths or hashes from stale summaries.

## Capacity and scaling interpretation

This diagnostic introduces no model-size, memory-size, metric-count, graph-size, or compute-capacity ceiling. The two-arm design is a causal-control structure for this experiment only. Resource limits required by a specific runtime are operational launch settings and are not architectural limits.
