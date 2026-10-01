# MFM v1.6 exact source slices for teacher annotation

**Status:** source-only annotation aid. No formation targets, source permission
attestations, reviewers, split, model fit, or admission are produced.

The Multi-Source structural JSON files hold all 30 days. Passing a complete
file to a day-1 teacher/model would reveal future records. The source review
packet filters records but originally retained only JSON pointers and hashes,
not original-file byte offsets. `materialize_v16_source_slices.py` rebuilds the
pinned packet from the inventoried original files, then scans each original
UTF-8 JSON document for exact byte ranges. It writes a private binary blob of
the **verbatim** `/records/n` values used by the selected packet windows and a
JSONL index that binds each fragment to its parent whole-file SHA-256, byte
offsets, fragment SHA-256, source ID, packet and lineage. The profile has two
separate exact fragments, `/facts` and `/anchor_window`; it appears only when
its `total_days` horizon has elapsed. The full profile file's generator-only
fields are never copied into the blob.

For each extracted value, the tool parses the original byte slice and checks
it against the packet record and canonical unit SHA-256. The packet itself must
match a complete regeneration from the frozen inventory, even if a different
packet hash is supplied. An existing output directory is never overwritten;
the new directory is mode 0700 and its files are mode 0600. A repeated source
pointer is stored once in `fragments.bin`, with index entries referring to its
offset. A teacher intake must consume the isolated export below, **not** the
shared `fragments.bin`, the raw index or a parent file. That blob contains
records from all 1,920 windows, including future days relative to a day-1
packet. Its offset index is custody material, not a model input. The hash of a
parent file establishes lineage but does not expose its contents.

Run from the MFM checkout with the source data and source review packet under
owner-controlled custody outside Git:

```bash
PYTHONPATH=src:. python scripts/mfm/materialize_v16_source_slices.py \
  --packets ../mfm-review-repro/source-review.jsonl \
  --packets-sha256 4ae868b2cae39b837364f71c8b68cebe5fee731bca3711e1459d87782084e642 \
  --dataset-dir ../mfm-multisource-new-seed \
  --inventory ../mfm-multisource-new-seed/mfm_source_inventory.json \
  --inventory-sha256 a5fcd53f84e049f84a2aaf0e1902da17a2d9d8fd4eec85deccb4bf8902c4d362 \
  --output-dir ../mfm-slices-private
```

The pinned 1,920-packet replay yielded 102,240 source references and 58,560
unique fragments. The fragment blob SHA-256 was
`9df692ade0c10454da9cb3198cd9649ec43721c330b936daa649830b59974e34`,
index SHA-256
`d3c118e9e5a6324423eae354090652171237c83542f90ddf4df69653a72846dc`,
and receipt file SHA-256
`7eb0c22c1b36825bb26b140565c33706401fb9b3b5ce48f4d21223b50b7c80c9`.
Record the receipt digest in separate owner custody before exporting a packet;
hashing the current receipt at call time alone would not detect a substituted
index/blob/receipt triplet.

To prepare one teacher-visible source packet, use its externally pinned receipt
digest and packet ID:

```bash
PYTHONPATH=src:. python scripts/mfm/export_v16_source_packet.py \
  --source-slices-dir ../mfm-slices-private \
  --receipt-sha256 7eb0c22c1b36825bb26b140565c33706401fb9b3b5ce48f4d21223b50b7c80c9 \
  --packet-id a5fcd53f84e049f8/bench_shift_001_drew_carter/through-day-01 \
  --output ../mfm-review-repro/teacher-day-01-source-only-opaque.json
```

The exporter rehashes the receipt, full index and blob, selects exactly one
packet row, checks every selected fragment hash and canonical unit digest,
then creates a mode-0600 file once. It includes only selected exact bytes in
base64, source type and window, each fragment SHA-256, and HMAC-derived opaque
packet, host, source and slice handles. The caller supplies the original packet
ID privately; it never appears in the teacher-visible file. The private index
retains the original IDs and fields needed to derive the handle mapping,
alongside generator family, seed,
upstream commit, lineage group, parent whole-file hashes, pointers and original
byte offsets for split and grounding audits. None of those fields enter the
teacher-visible export. A trusted converter can resolve each handle against
the separately pinned private index after the teacher responds. These handles
reduce accidental prompt leakage; they are not a secrecy guarantee against an
actor with the public corpus and receipt-derived HMAC key. The day-1
export had four source units, SHA-256
`6c5b1a66d6e79ad1a17845f5ff6240d4d91a0b7f5673214d8ec109a770c75858`,
and no day-2 record or profile fragment. The teacher must not receive the
shared custody blob or open the original whole-file paths.

The index is still an unreviewed source candidate. Its `rights_status` is
`unverified`, and it contains no target. Original byte locations are **source
unit** spans, not claim-level evidence anchors. A teacher must read selected
units, produce a source-bound v1.6 proposal and narrow claim spans, distinguish
speaker/subject/time/epistemic state, and review all ten dimensions. A steward
must bind original and derivative source rights before shared-weight training.
The exact source fragments themselves do not decide whether a plan occurred,
whether a device reading proves a physical outcome, or whether a source-person
fact belongs to the host. The independent memory gate remains the authority
for changes to memory.

The 480 hosts share one generator family. All four cumulative windows for a
host share ancestors, so these 1,920 rows cannot be split into independent
train/development/FINAL merely by assigning different host IDs or dates.
There is no 1.6 training admission from this command. Long day-30 windows
still require measured context selection or long-context execution without
silent truncation; extracting exact source slices does not solve that model
capacity question.

Focused test:

```bash
PYTHONPATH=src:. python -m unittest \
  tests/governance/test_mfm_v16_source_slice_materialization.py -v
```
