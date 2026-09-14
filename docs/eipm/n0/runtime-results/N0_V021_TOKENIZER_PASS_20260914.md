# N0 v0.2.1 governed tokenizer — PASS

Date: 2026-09-14

The governed N0 v0.2.1 tokenizer build completed successfully on Magnolia without GPU use, model-weight training, private identity data, or private identity gradient.

## Frozen lineage

- corpus: `$HOME/rayan-compute/rayan-n0/n0-v02/tokenizer-corpus-v0.2.1-offline`
- corpus receipt SHA-256: `fa0cf5fe2a42d740d423409699fd1d46c77e3b6de66cae2807281ea799de8270`
- tokenizer: `$HOME/rayan-compute/rayan-n0/n0-v02/tokenizer-v0.2.1`
- tokenizer SHA-256: `cd4cb8025918891c84b908c2a8aceba1d49f08b23f99b7913a0a279f8a43c9aa`
- tokenizer receipt SHA-256: `52bc2cb70251d8b3ba315eaa1dcf8f78a3cc4bfe6b78306d802d55afba3372ad`
- vocabulary: 48,000 byte-fallback BPE
- special IDs: PAD=0, UNK=1, CLS=2, SEP=3, MASK=4

## Fit boundary

Tokenizer fitting used only rows whose deterministic document split was `train`:

- train rows used: 27,730
- train characters used: 99,767,869
- dev rows skipped during fit: 30
- test rows skipped during fit: 38

The dev/test documents therefore did not contribute to merge learning.

## Audit

- build status: PASS
- audit status: PASS
- unknown-token count: 0
- round-trip failures: 0
- private identity data: false
- private identity gradient: false
- model training performed: false

Observed characters/token in the audit sample:

- train: 4.1201
- dev: 4.2102
- test: 3.5119

The small held-out sample sizes make split-level compression differences diagnostic only. They are not a reason to retune the tokenizer. All 21 governed source families tokenized without unknown tokens, and source-level compression remained usable across prose, dialogue, code, science, law, books, and reference material.

## Decision

Tokenizer acquisition, fitting, and byte-completeness gates are closed for N0 v0.2.1. Do not rebuild the tokenizer unless a concrete downstream failure demonstrates that this tokenizer is a bottleneck.

The next substantive N0 work is teacher-bank expansion and preparation of the first durable public pretraining tranche. The 48K tokenizer above is the tokenizer lineage for those tasks.
