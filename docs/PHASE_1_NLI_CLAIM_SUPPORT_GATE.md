# P1.11 NLI Claim-Support Gate

The generic MS MARCO reranker reduced the mismatch-only support rate from
0.421053 to 0.083333, so it is disabled.

P1.11 now adds an independent local NLI gate after generation and before final
verification. Each generated claim is checked only against the passages from
the citations attached to that claim.

Model:
cross-encoder/nli-deberta-v3-base

Labels:
contradiction, entailment, neutral

Default behavior:
- entailment >= 0.70: keep claim
- contradiction >= 0.80: drop claim
- otherwise neutral/uncertain: drop claim
- if every claim is dropped: return insufficient_evidence

The visible answer is rebuilt only from claims that survive the gate.

This improves faithfulness without forcing A.L.I.C.E. to cite benchmark labels.
The model is public and downloaded locally. Private evidence is never uploaded.
