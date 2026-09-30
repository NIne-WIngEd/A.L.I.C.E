# Native MFM Stage A public corpus: acquisition to review candidates

**Status:** Implemented CPU acquisition, per-upstream-shard candidate staging,
and cross-shard exact-lineage reconciliation. No public shard has been fetched,
no rights have been authenticated, no corpus is admitted, and no optimizer has
run. These commands are a foundation data path, not a full-capability claim.

The frozen 21-source N0 v0.2.1 evidence is copied unchanged to
`configs/mfm/n0_public_corpus_v0.2.1.activated.pinned.json` with SHA-256
`8bfbc11a2974af8f9d03aff558c413dafc699d45743d674a3db0180d17be7c31`.
The MFM candidate inventory binds that hash. N0's per-row license strings are
**only a conservative metadata filter proposal**. An N0 right to train N0 does
not grant commercial MFM training or distribution automatically.

## Sequence for each exact upstream file

1. On a CPU environment with Hugging Face access and persistent working space,
   freeze the MFM inventory hash, N0 evidence hash, dataset commit, the full
   list of repository shard paths selected for that source, and an exclusion
   manifest. Do not cap rows or characters to meet a budget. `part_bytes` only
   chooses the size of transport files; a long source row is never truncated.
   Freeze **all** source/file choices before the global reconciliation that
   assigns split candidates.
2. Run `native_hf_acquire.py` for each shard. It resolves the exact dataset
   commit, obtains the exact file's LFS SHA-256 or regular Git blob ID from that
   commit, downloads with the same revision, checks all bytes against the
   object ID, and writes `acquisition_receipt.json`. Missing/verifiably wrong
   upstream object IDs fail closed. This authenticates *repository byte
   lineage*, not the legal grant or underlying authorship.
3. Run `native_public_corpus.py stage` with the acquisition receipt, its SHA,
   the downloaded raw path and SHA, the source ID, and file path within the
   repository. It streams all JSONL, gzipped JSONL or Parquet rows. Its exact
   N0 source license match, nonempty provenance, syntactically valid HTTP(S) source URL,
   source record ID and nonempty text are necessary to create a **review
   candidate**. A held row is recorded by upstream ordinal, reason, and row
   hash in `held_rows.jsonl`; the receipt counts every row. No row is marked
   PII-safe, rights-cleared or benchmark-clean. Original URLs, source IDs,
   provenance, revision, license expression and content hashes survive.
4. Hash and transfer the acquisition receipt, original upstream shard,
   staged candidate parts, hold ledger and stage receipt by owner-controlled
   storage. On Magnolia, verify the raw SHA and stage receipt with the `verify`
   command. The compute nodes in the observed Magnolia check could not resolve
   `huggingface.co`; download on a CPU environment with access and transfer
   bytes, then verify on Magnolia. Stage output can be retried per upstream
   file: an identical completed output verifies and returns its receipt;
   incomplete output is rejected rather than silently appended.
5. Freeze a JSON receipt list and the separate exclusion JSON, including
   reserved independent evaluations, exact content hashes, source URLs,
   source record references and excluded families known by that point. Run
   `reconcile` over **all** accepted shard receipts together. It verifies each
   part and receipt hash, uses a disk-backed connected-component index across
   source URLs, upstream record IDs and exact content hashes, propagates
   exclusions to related rows, holds exact duplicate copies, and assigns a
   stable train/development candidate split by whole component. Its FINAL
   split is deliberately absent: independent FINAL must be sealed separately.
   Adding shards or changing exclusions requires a new global reconciliation.
   Output is written to a sibling temporary directory and renamed after
   success, so a failed check leaves no purported completed destination.

Use `python -m scripts.mfm.native_hf_acquire` and
`python -m scripts.mfm.native_public_corpus` from the MFM repository root with
`PYTHONPATH=src:.`. For example, this is one file's **shape**, with paths and
hashes taken from real frozen manifests and acquired bytes before execution:

```bash
export PYTHONPATH=src:.
python -m scripts.mfm.native_hf_acquire \
  --candidate-inventory configs/mfm/native_foundation_source_candidates_v1.json \
  --candidate-sha256 "$CANDIDATE_SHA" \
  --n0-manifest configs/mfm/n0_public_corpus_v0.2.1.activated.pinned.json \
  --n0-sha256 8bfbc11a2974af8f9d03aff558c413dafc699d45743d674a3db0180d17be7c31 \
  --candidate-id common_pile_wikimedia_filtered \
  --upstream-repo-path "$EXACT_REPO_FILE" \
  --output "$ACQUIRED_DIR"
python -m scripts.mfm.native_public_corpus stage \
  --candidate-inventory configs/mfm/native_foundation_source_candidates_v1.json \
  --candidate-sha256 "$CANDIDATE_SHA" \
  --n0-manifest configs/mfm/n0_public_corpus_v0.2.1.activated.pinned.json \
  --n0-sha256 8bfbc11a2974af8f9d03aff558c413dafc699d45743d674a3db0180d17be7c31 \
  --candidate-id common_pile_wikimedia_filtered \
  --acquisition-receipt "$ACQUIRED_DIR/acquisition_receipt.json" \
  --acquisition-sha256 "$ACQUISITION_SHA" \
  --raw-file "$ACQUIRED_DIR/$EXACT_REPO_FILE" \
  --raw-sha256 "$RAW_SHA" \
  --upstream-repo-path "$EXACT_REPO_FILE" \
  --format parquet --output "$STAGE_DIR"
python -m scripts.mfm.native_public_corpus verify \
  --receipt "$STAGE_DIR/shard_receipt.json" \
  --raw-file "$ACQUIRED_DIR/$EXACT_REPO_FILE"
```

The `reconcile` input has shape:

```json
{"schema":"mfm-native-public-receipt-list-v1","receipts":[{"path":"shards/one/shard_receipt.json","sha256":"<actual 64-hex receipt SHA>"}]}
```

The exclusion file has shape:

```json
{"schema":"mfm-native-public-exclusions-v1","reserved_evaluations":["LongMemEval-V2","LoCoMo","CareCall"],"candidate_ids":[],"content_sha256s":[],"source_urls":[],"upstream_item_refs":[]}
```

`reconcile` takes `--receipt-list`, `--receipt-list-sha256`, `--exclusions`,
`--exclusions-sha256` and `--output`. The receipt list's paths are relative
to its parent directory; transfer that directory with all staged shards and
verify every SHA on the destination. The SQLite file and candidate index are
reproducible from that frozen bundle; their presence alone is no admission.

## Human handoff before any Stage A gradient

The global index stores shard receipt, candidate part, byte offset and length,
content hash, source URL, source ID, exact duplicate group and proposed split.
Independently audit each candidate's upstream work and current source terms,
license, commercial training/model distribution permissions, attribution,
withdrawals, PII, consent, and independent FINAL or benchmark overlap. Audit
related works and semantic near duplicates, which this exact hash pass cannot
identify. For accepted items, a future reviewed packer must extract payloads
from these exact byte ranges, attach recorded terms and review evidence, and
create the *separate* `mfm-native-foundation-admission-v1` manifest checked by
`scripts/mfm/native_source_admission.py`. That verifier currently loads its
whole manifest in memory; foundation-scale pack admission still needs sharded
global verification before a long optimizer run. Teacher labels must live in
separate source-bound ledgers with teacher/prompt/output digests and cannot be
silently appended to this public raw corpus.

The code expects `huggingface_hub` for acquisition and `pyarrow` for Parquet
staging on the CPU staging machine. The package versions and Kaggle image must
be frozen and validated there before a production-scale job. JSONL fixture
tests require neither package nor network.
