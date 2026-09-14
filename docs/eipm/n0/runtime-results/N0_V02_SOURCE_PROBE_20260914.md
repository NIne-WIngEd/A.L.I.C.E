# N0 v0.2 Source Probe — 2026-09-14

Status: **PASS — source activation completed for bounded public N0 v0.2 corpus construction.**

## Probe result

The Magnolia login-node schema probe inspected 64 streamed rows from each of the 22 planned Common-Pile-derived sources.

- source_count: 22
- schema_pass_count: 22
- schema_failure_count: 0
- all_schema_passed: true
- exit code: 0
- GPU requested: false
- training authorized by probe: false
- private identity gradient: false

For every sampled source, the probe found the required `text`, `id`, `metadata.license`, and `metadata.provenance` fields on all sampled rows. Exact Hugging Face revision SHAs were resolved and recorded.

## Activation decision

The exact revisions and the observed, manually reviewed open/public license strings are now frozen in:

`configs/eipm/n0/public_corpus_v0.2.activated.json`

Activation is deliberately fail-closed:

- only the exact pinned revision is allowed;
- every accepted row must carry a license string;
- only source-specific allowlisted strings are accepted;
- an unseen license value is rejected rather than inferred safe;
- provenance remains required;
- private identity data remains forbidden;
- the source probe itself did not authorize training.

The allowlist contains only the license families observed in the successful probe: public domain / CC0, CC BY, CC BY-SA, Open Parliament Licence, Apache-2.0, BSD, MIT, Unlicense, and explicit combinations of these in the code sources. Attribution/share-alike and notice obligations remain preserved in row provenance/license metadata. The project must not discard those records merely because model training begins.

## Data-construction correction

The v0.2 corpus builder must preserve target mixture shares rather than giving every source the same character cap. A new deterministic tranche builder therefore uses the activated source `target_share` values to allocate per-source character budgets.

Implementation:

- `scripts/eipm/n0/materialize_public_corpus_v02.py`
- `scripts/eipm/n0/magnolia_login_n0_v02_tokenizer_corpus.sh`

The materializer also:

- refuses non-empty output directories;
- verifies the activated manifest and exact 40-hex revisions;
- requires source-specific row-license allowlists;
- exact-deduplicates across all 22 sources;
- supports deterministic hash partitions for future disjoint tranches;
- records source balance, fill ratio, hashes, provenance, and exact revisions in a v0.2 receipt;
- fails if a source cannot fill at least 95% of its assigned share.

## Immediate next boundary

Build one bounded, 120M-character, source-balanced tokenizer corpus on the login node. This is a data-construction action only. It uses no GPU and contains no private identity data.

Do not train the v0.2 tokenizer until that tranche returns PASS and its source-balance/fill receipt has been inspected.
