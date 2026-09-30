# Native MFM Stage A public corpus: acquisition to review candidates

**Status:** Implemented exact-commit complete repository inventory and explicit
file selection, CPU acquisition, per-upstream-shard candidate staging, and
cross-shard exact-lineage reconciliation. No public shard has been fetched,
no rights have been authenticated, no corpus is admitted, and no optimizer has
run. These commands are a foundation data path, not a full-capability claim.

The frozen 21-source N0 v0.2.1 evidence is copied unchanged to
`configs/mfm/n0_public_corpus_v0.2.1.activated.pinned.json` with SHA-256
`8bfbc11a2974af8f9d03aff558c413dafc699d45743d674a3db0180d17be7c31`.
The MFM candidate inventory binds that hash. N0's per-row license strings are
**only a conservative metadata filter proposal**. An N0 right to train N0 does
not grant commercial MFM training or distribution automatically.

## Sequence for each pinned repository and its upstream files

1. On a CPU environment with Hugging Face access and persistent working space,
   hash the MFM candidate source manifest and pinned N0 source evidence. For
   each of the 21 pinned N0 Common Pile repositories, run `native_hf_acquire.py
   inventory`. It verifies the exact 40-hex commit, exhausts the paginated
   recursive `HfApi.list_repo_tree` iterator, and freezes **every** repository
   file path, byte size, LFS SHA-256 or regular Git blob ID, and folder tree
   IDs. It writes a separate file-selection template. This fetches repository
   *metadata*, not dataset contents. No maximum shard count is applied.
2. Review the template: assign `include` to each data shard intended for the
   foundation and `exclude` with an explicit reason to every other file,
   including `README.md`, `.gitattributes` and other metadata. Supported
   staged formats are Parquet, JSONL and gzipped JSONL. Unrecognized files are
   `review_required`, so freezing fails until each one has a decision. An
   intentionally omitted data shard needs a recorded reason; it cannot simply
   disappear from the tree. `review_record` must name the actual source/file
   review evidence; the generated template has `null` there and cannot be
   frozen as-is. Hash the reviewed selection and run
   `native_hf_acquire.py freeze`; the resulting manifest binds the full tree
   and all decisions. Freeze **all** source/file choices before the global
   reconciliation that assigns split candidates. No row or character caps are
   allowed; `part_bytes` only sizes transport files, never truncates a row.
3. Run `native_hf_acquire.py fetch` for **each included file**. The downloader
   requires its frozen inventory, rechecks exact-commit Hub metadata against
   the selected path/size/object ID, downloads from that revision, checks all
   bytes, and writes `acquisition_receipt.json` tied to the tree and selection
   hashes. An excluded, unknown or changed path fails before download. This
   authenticates *repository byte lineage*, not the legal grant or underlying
   authorship. Freeze a list of acquisition receipt paths and hashes; run
   `native_hf_acquire.py coverage`. It rechecks raw bytes and fails if any
   selected file lacks exactly one receipt. Match the later stage receipts to
   the same complete set before global reconciliation; acquisition coverage
   alone cannot prove every selected shard was staged.
4. Run `native_public_corpus.py stage` with the acquisition receipt, its SHA,
   the downloaded raw path and SHA, the source ID, and file path within the
   repository. It streams all JSONL, gzipped JSONL or Parquet rows. Its exact
   N0 source license match, nonempty provenance, syntactically valid HTTP(S) source URL,
   source record ID and nonempty text are necessary to create a **review
   candidate**. A held row is recorded by upstream ordinal, reason, and row
   hash in `held_rows.jsonl`; the receipt counts every row. No row is marked
   PII-safe, rights-cleared or benchmark-clean. Original URLs, source IDs,
   provenance, revision, license expression and content hashes survive.
5. Hash and transfer the frozen repository inventory, acquisition receipt,
   original upstream shard,
   staged candidate parts, hold ledger and stage receipt by owner-controlled
   storage. On Magnolia, verify the raw SHA and stage receipt with the `verify`
   command. The compute nodes in the observed Magnolia check could not resolve
   `huggingface.co`; download on a CPU environment with access and transfer
   bytes, then verify on Magnolia. Stage output can be retried per upstream
   file: an identical completed output verifies and returns its receipt;
   incomplete output is rejected rather than silently appended.
6. Freeze a JSON receipt list and the separate exclusion JSON, including
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
`PYTHONPATH=src:.`. The commands below are the **shape** for one repository;
compute each SHA from the actual completed file, edit the template's unresolved
decisions, then repeat `fetch` and `stage` for **every included shard**:

```bash
export PYTHONPATH=src:.
python -m scripts.mfm.native_hf_acquire inventory \
  --candidate-inventory configs/mfm/native_foundation_source_candidates_v1.json \
  --candidate-sha256 "$CANDIDATE_SHA" \
  --n0-manifest configs/mfm/n0_public_corpus_v0.2.1.activated.pinned.json \
  --n0-sha256 8bfbc11a2974af8f9d03aff558c413dafc699d45743d674a3db0180d17be7c31 \
  --candidate-id common_pile_wikimedia_filtered \
  --output "$TREE_FILE" --selection-template-output "$SELECTION_TEMPLATE"
# Review every path in a copy of the template, then hash that reviewed file.
python -m scripts.mfm.native_hf_acquire freeze \
  --repo-tree "$TREE_FILE" --repo-tree-sha256 "$TREE_SHA" \
  --selection "$REVIEWED_SELECTION" --selection-sha256 "$SELECTION_SHA" \
  --output "$FROZEN_FILE"
python -m scripts.mfm.native_hf_acquire fetch \
  --candidate-inventory configs/mfm/native_foundation_source_candidates_v1.json \
  --candidate-sha256 "$CANDIDATE_SHA" \
  --n0-manifest configs/mfm/n0_public_corpus_v0.2.1.activated.pinned.json \
  --n0-sha256 8bfbc11a2974af8f9d03aff558c413dafc699d45743d674a3db0180d17be7c31 \
  --candidate-id common_pile_wikimedia_filtered \
  --frozen-inventory "$FROZEN_FILE" --frozen-inventory-sha256 "$FROZEN_SHA" \
  --upstream-repo-path "$EXACT_REPO_FILE" \
  --output "$ACQUIRED_DIR"
# Repeat fetch for every included file, freeze a receipt list, then:
python -m scripts.mfm.native_hf_acquire coverage \
  --frozen-inventory "$FROZEN_FILE" --frozen-inventory-sha256 "$FROZEN_SHA" \
  --receipt-list "$ACQUISITION_LIST" --receipt-list-sha256 "$ACQUISITION_LIST_SHA" \
  --output "$COVERAGE_FILE"
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

The acquisition receipt list has shape (with one entry for each included
shard, relative to the list's parent directory):

```json
{"schema":"mfm-native-hf-acquisition-receipt-list-v1","frozen_repo_inventory_sha256":"<actual 64-hex SHA>","receipts":[{"path":"shard-0/acquisition_receipt.json","sha256":"<actual 64-hex SHA>"}]}
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
