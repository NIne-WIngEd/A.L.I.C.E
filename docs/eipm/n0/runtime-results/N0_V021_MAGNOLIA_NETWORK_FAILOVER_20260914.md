# N0 v0.2.1 tokenizer corpus: Magnolia network failover

Date: 2026-09-14

## Observed failure

The corrected v0.2.1 tokenizer-corpus materialization started on the Magnolia login node with the intended 21-source manifest and 120M-character target. During the first source, Hugging Face access produced a read timeout and then repeated `Temporary failure in name resolution` errors while fetching a pinned Wikimedia shard.

This is classified as an acquisition-network failure. It does not invalidate:
- the 136,594,435-parameter N0 v0.2 architecture;
- the 21-source v0.2.1 manifest;
- exact source revisions;
- row-level license allowlists;
- the OER Commons removal/rebalance;
- the long-document chunking correction.

## No patch loop

Do not change model architecture, source semantics, license policy, scheduler, or GPU plan in response to this DNS failure. Do not repeatedly rotate Magnolia nodes or increase network retry counts.

## Reuse decision

The prior materialization at `tokenizer-corpus-v0.1` completed all 22 sources before correctly returning FAIL on the OER Commons fill gate. Its retained 21 production sources already contain enough accepted characters to support a 100M-character tokenizer-only corpus under the v0.2.1 shares.

Therefore the tokenizer path now derives a fresh corpus offline from those already acquired shards:
- parent shard bytes and SHA-256 are verified against the parent receipt;
- active source revision and license allowlists are rechecked;
- OER Commons is excluded;
- long rows are deterministically chunked to <=16K characters;
- all chunks retain the parent document split, preventing document-level split leakage;
- exact chunk dedup is reapplied globally;
- each retained source is capped at the v0.2.1 target share;
- target total is 100M characters;
- no network access, GPU, private identity data, or model training is required.

## Implementation

- `scripts/eipm/n0/derive_tokenizer_corpus_v021_from_v01.py`
- `scripts/eipm/n0/magnolia_offline_n0_v02_tokenizer_corpus.sh`
- `tests/eipm/test_n0_v021_offline_derivation.py`

Kaggle/public-network acquisition remains available later for genuinely new pretraining tranches. It is not necessary merely to finish the tokenizer corpus when governed material already exists locally.
