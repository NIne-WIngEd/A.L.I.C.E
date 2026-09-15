# N0 v0.2 step80 ratified; structured-state work started

Date: 2026-09-15

## Semantic checkpoint decision

Selected semantic checkpoint: `targeted-repair-v0.1/checkpoints/step-00000080`.

Evidence:
- repair-dev top1: 0.80 versus parent 0.60
- teacher-dev top1: 0.8736842105, unchanged from parent
- teacher-dev full invariance: 0.8745098039, unchanged from parent
- novel cross-competency top1: 0.875 versus parent 0.8125
- novel full invariance: 0.875 versus parent 0.7916666667
- remaining novel failures: ALIGNTEMP-02, SEMRANK-02, VOICESOC-01
- step120 tied step80 on novel and repair-dev correctness but had worse failed-example confidence and a small teacher-dev regression

Warm P100 three-candidate latency:
- end-to-end p50 32.4606 ms
- end-to-end p95 37.8038 ms
- model-only p95 16.1453 ms
- 512-token model-only p95 67.3630 ms

Checkpoint hashes:
- full model: `6c2706984c0e05123c4d88ba456788fbd4c9e7f0fca41d43d9e575ac6e53bf43`
- ranker: `b9f98bf2827580a9e266276d08e221c568885df2c04c4437b77d6e6f9638cbf7`

This is the N0 v0.2 semantic base. It is not the completion of the full EIPM.

## Structured-state capability

Added on `alice-eipm-v1-build`:
- `src/alice_personality/n0/structured_state.py`
- `configs/eipm/n0/n0_v02_structured_state_v0.1.json`
- `tests/eipm/test_n0_structured_state.py`
- `configs/eipm/n0/n0_v02_semantic_base_ratification_v0.1.json`

Mechanics:
- 640-d semantic field vectors
- 256-d internal structured state
- 2 set-style transformer layers
- 4 attention heads
- FFN 512
- no positional embeddings
- explicit type, provenance, relation, temporal, confidence, and missingness channels
- 1,656,064 parameters
- semantic core unchanged

Next work: build the public identity-neutral structured-state curriculum and semantic-state alignment objective. Do not create another broad validation program before that capability work.
