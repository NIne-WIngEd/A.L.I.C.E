# N0 QSRE T2 CPU/GPU Reality-Gap Audit Active

**Date:** 2026-09-20
**Status:** all T2 GPU/gradient work frozen pending one coherent reality/data audit

## Trigger

Job 575934 failed before entering the container:

- job: 575934
- state: FAILED 1:0
- elapsed: 00:00:01
- stderr showed Bash expanding a malformed doubled-dollar shell parameter expression into a process-id prefix before the literal brace expression.

Direct source inspection confirmed both recovery scripts contained malformed doubled-dollar parameter expansions.

More importantly, the recovery CI encoded the same malformed spelling as the expected source form. The validator therefore shared the implementation defect.

This is classified as:

`SHELL_EXPANSION_AND_VALIDATOR_CORRELATION_FAILURE_NOT_MODEL_EVIDENCE`

## Governing state

- authoritative state: `configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.55.json`
- audit: `docs/research/eipm-n0-cpu-gpu-reality-gap-and-data-audit-v0.1.md`
- old v0.54 recovery authorization is revoked.
- recovery runner and sbatch are fail-closed with exit 90.

## Main architectural finding

The repeated pattern is not simply CPU success versus GPU failure.

Three classes must stay separate:

1. deterministic/oracle mechanics success;
2. genuine learned-model failure/success;
3. infrastructure/harness failure.

Historical examples:

- v0.7 CPU import harness mismatch — no model evidence
- v0.22 / job 575825 — genuine learned-model failure
- v0.34 CPU bookkeeping bug — no model evidence
- v0.36 / job 575912 — genuine learned-model failure
- v0.47 — pretraining validity audit found T1 shortcuts before GPU and retracted authorization
- v0.51 — uDocker PYTHONPATH boundary, infrastructure only
- job 575914 — genuine QSRE T1 P100 PASS
- job 575933 — evaluator totality failure, no model-gate result
- job 575934 — shell/runtime + validator-correlation failure, no model-gate result

The repeated systemic weakness is that CPU/static tests often validate contracts authored from the same assumptions as the implementation. They may prove representability/plumbing while not exercising the real learned-output domain or exact runtime semantics.

## Direct data read

T1 v0.2 is intentionally structural:
- every field text is `neutral evidence record <hash>`;
- oracle operator/support;
- valid for T1 executor causality, not natural-language realism.

T2 v0.1 is also synthetic:
- 720 TRAIN / 288 DEV;
- small relation/role/operation/control phrase banks;
- DEV wording banks are disjoint but generated from the same family-specific grammar;
- highly available cues include “all/aggregate”, “most reliable”, “latest”, “follow ... then”, “non-relational/fallback”, and “defer”.

Therefore T2 v0.1 is currently a unit test of generated operator-language learnability, not yet a strong realistic public-N0 operator benchmark.

The frozen semantic backbone itself was trained on richer public N0 language including pragmatic implicature, evidence conflicts, counterfactual provenance, uncertainty, causal paraphrases, social/relationship context, and domain-sensitive semantics.

## Early RAG analogy recovered

Earlier A.L.I.C.E. Phase 1 added MS MARCO cross-encoder passage reranking. Direct downstream support audit showed support fell from 0.421053 to 0.083333, so that reranker was disabled for the passage-selection role.

Transferable lesson:
- do not trust sophistication or a local stage metric;
- inspect the actual data and downstream evidence;
- a test/benchmark can be internally clean while measuring the wrong thing.

## Current QSRE conclusion

QSRE is **not rejected**.

Supported:
- T1 executor/readout (real P100 PASS)
- explicit relation/role/operator factorization
- full semantic hidden-state access
- support-local execution
- separate fallback

Unresolved:
- whether current T2 operator/data recover these factors robustly from realistic public language.

T2 has no valid model PASS/FAIL yet.

## Next work — one coherent batch, no micro-gate chain

Before any new gradient/GPU work, complete together:

1. runtime-parity audit that executes shell/container semantics rather than grepping source;
2. representative public-only relation/evidence set from real N0-style natural language;
3. shortcut/cue baselines for relation/role/operation/control;
4. frozen-semantic layer sufficiency probes on the representative set;
5. evaluator totality over deliberately invalid/provisional operator outputs;
6. one combined architecture decision: keep T2, change operator interface, or reopen semantic representation.

Still closed:
- optimizer
- gradient
- GPU training
- T1 rerun
- T3 / learned support
- TEST/challenge
- private identity gradient
- promotion
