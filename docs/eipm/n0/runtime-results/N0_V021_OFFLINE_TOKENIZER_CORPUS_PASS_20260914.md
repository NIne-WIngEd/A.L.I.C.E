# N0 v0.2.1 Offline Tokenizer Corpus — PASS

Date: 2026-09-14

## Result

The governed offline derivation from the previously materialized public corpus passed without network or GPU use.

- status: PASS
- source count: 21
- target characters: 100,000,000
- accepted characters: 99,999,292
- accepted rows: 27,798
- parent corpus receipt SHA256: `2da91b9f0441b63b3b12d23d596e177f79038a6257aed92a37a43022cf5988a3`
- derived corpus receipt SHA256: `fa0cf5fe2a42d740d423409699fd1d46c77e3b6de66cae2807281ea799de8270`
- network_access_required: false
- private_identity_data: false
- private_identity_gradient: false
- shell exit: 0

Every retained source filled at least 99.9976% of its assigned budget. The largest prior problem, OER Commons, had already been removed from the v0.2.1 activated mixture rather than widening the license gate. Long-document sources were bounded through deterministic chunking instead of allowing a few whole books/reports to dominate their source share.

## Decision

The tokenizer-corpus construction gate is closed. Do not reacquire this tokenizer corpus or retry Magnolia networking for this stage.

The next stage is a CPU-only governed 48K byte-complete BPE tokenizer fit. The tokenizer must:

- train only on rows marked `split=train` so dev/test documents remain outside tokenizer fitting;
- bind its receipt to the exact derived corpus receipt and v0.2.1 activated source manifest;
- preserve special token IDs PAD=0, UNK=1, CLS=2, SEP=3, MASK=4;
- pass byte-completeness / Unicode round-trip probes with zero `[UNK]` use;
- pass a post-fit audit on held-out corpus rows before any model-weight training is considered.

No GPU training and no private identity gradient are authorized by this result.
