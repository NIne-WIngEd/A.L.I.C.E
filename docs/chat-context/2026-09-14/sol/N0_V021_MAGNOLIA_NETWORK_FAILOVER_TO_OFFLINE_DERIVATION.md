# N0 v0.2.1 Magnolia network failure -> offline tokenizer corpus derivation

Date: 2026-09-14

The corrected 120M-character tokenizer-corpus materialization on Magnolia did not expose a new model/data-design defect. It failed during Hugging Face access on the first source with a read timeout followed by repeated `Temporary failure in name resolution` errors. Do not enter a network retry/patch loop and do not change N0 architecture or source semantics because of this infrastructure failure.

The prior failed 120M-character v0.1 materialization remains useful governed evidence. Although its final status was FAIL because OER Commons filled only 32.12% of its assigned budget, the retained 21 v0.2.1 sources each already contain enough accepted, exact-revision, row-license-gated text to derive a balanced 100M-character tokenizer-only corpus without network access.

Production decision:
- keep `public_corpus_v0.2.1.activated.json` unchanged;
- do not widen licenses;
- do not retry Magnolia/HF acquisition for the tokenizer corpus;
- derive a fresh 100M-character corpus offline from the already acquired parent materialization;
- re-chunk long documents to <=16K characters while preserving parent document split lineage;
- reapply exact chunk dedup and current allowlists;
- use the derived corpus only after its receipt passes;
- no GPU and no private identity gradient.

Implementation:
- `scripts/eipm/n0/derive_tokenizer_corpus_v021_from_v01.py`
- `scripts/eipm/n0/magnolia_offline_n0_v02_tokenizer_corpus.sh`
- `tests/eipm/test_n0_v021_offline_derivation.py`

This is an infrastructure failover, not an architectural reset. The objective remains: build the native N0 v0.2 representation model efficiently and spend GPU only after corpus/tokenizer/teacher/eval inputs are ready.
