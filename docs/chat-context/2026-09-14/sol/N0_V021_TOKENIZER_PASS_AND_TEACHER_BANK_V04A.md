# N0 v0.2.1 tokenizer PASS and teacher-bank wave 0.4a

Date: 2026-09-14

## Closed gate: tokenizer

The governed N0 v0.2.1 tokenizer passed on Magnolia.

- tokenizer SHA-256: `cd4cb8025918891c84b908c2a8aceba1d49f08b23f99b7913a0a279f8a43c9aa`
- tokenizer receipt SHA-256: `52bc2cb70251d8b3ba315eaa1dcf8f78a3cc4bfe6b78306d802d55afba3372ad`
- corpus receipt SHA-256: `fa0cf5fe2a42d740d423409699fd1d46c77e3b6de66cae2807281ea799de8270`
- 48,000 byte-fallback BPE
- PAD/UNK/CLS/SEP/MASK IDs = 0/1/2/3/4
- train-only fitting: 27,730 rows, 99,767,869 chars
- dev rows skipped from fitting: 30
- test rows skipped from fitting: 38
- unknown token count: 0
- round-trip failures: 0
- GPU: false
- model training: false
- private identity gradient: false

Do not rebuild the tokenizer unless a concrete downstream failure demonstrates a tokenizer bottleneck.

## Teaching resumed

Wave 0.4a adds 215 public Sol-authored principle/rationale rows across all 43 registered N0 competencies. Each competency receives four new train scenarios and one independently authored dev scenario.

New shards:

- semantic/causal/temporal: 55
- pragmatic/social/emotion: 70
- epistemic/ranking/alignment: 90

The existing 128 governed seed/coverage/repair rows remain unchanged. Registered total is now 343. The 1,000-row full-multitask minimum remains closed; 657 additional high-quality rows are still required before that threshold can open.

New v0.4a rows require reusable `principle_tag`s and gradient-bearing rationales. The next action is one CPU-only teacher-bank audit, then continue authoring toward the 1,000-row gate if the audit passes. Do not spend P100 time on multitask training before the teacher bank is ready.

Private Elaina identity material remains excluded from N0 and no private gradient is authorized.
