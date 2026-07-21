# P1.11 — Private Pre-Gate Claim Diagnostics

The NLI gate correctly removed an AFM claim as neutral, but the original
pre-gate claim was not retained in the private response package. This prevented
diagnosing whether the low entailment score came from fragmented evidence,
owner-attribution phrasing, or an actually unsupported claim.

This patch stores the normalized model output immediately before the NLI gate
under:

`pre_gate_model_output`

This field exists only in the private grounded-response package. The public
summary exposes only `pre_gate_claim_count`; rejected claim text is not copied
to the exports summary.

The diagnostic script compares, without printing private text:

1. each cited passage individually;
2. all cited passages combined;
3. the combined passages with explicit trusted owner-self-record attribution.

No NLI thresholds or filtering behavior are changed.
