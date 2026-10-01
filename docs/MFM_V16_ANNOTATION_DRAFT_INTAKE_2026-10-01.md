# MFM 1.6 source packet to annotation draft intake

**State:** incomplete annotation aid. This command creates no formation target,
training row, reviewer decision, rights attestation, split, sealed FINAL, model
weight, or qualification result.

`scripts/mfm/prepare_v16_annotation_drafts.py` checks the exact SHA-256 of a
source-only review packet, regenerates its entire selected host/day matrix from
the pinned inventory and original structural source files, and compares every
byte. This rejects modified or duplicated rows and source-file drift, even if
someone supplies a new packet hash. The packet builder also excludes simulator
truth, published QA, generation knobs and previous target text. The intake
does not establish permission or independent annotation.

From the MFM worktree, after generating the source packets described in
`MFM_V16_SOURCE_REVIEW_PACKET_INTAKE_2026-09-30.md`, create a directory under
owner-controlled custody outside Git and run:

```bash
mkdir -m 700 -p ../mfm-review-private
PACKET_SHA256="$(sha256sum ../mfm-review-private/source-review.jsonl | cut -d' ' -f1)"
PYTHONPATH=src:. python scripts/mfm/prepare_v16_annotation_drafts.py \
  --packets ../mfm-review-private/source-review.jsonl \
  --packets-sha256 "$PACKET_SHA256" \
  --dataset-dir ../mfm-multisource-new-seed \
  --inventory ../mfm-multisource-new-seed/mfm_source_inventory.json \
  --inventory-sha256 a5fcd53f84e049f84a2aaf0e1902da17a2d9d8fd4eec85deccb4bf8902c4d362 \
  --output ../mfm-review-private/annotation-drafts.jsonl
```

Pin the packet hash in custody before review; computing it immediately before
intake protects accidental drift but does not prove who approved the packet.
The output is created once with mode `0600` and an existing output is never
overwritten. It omits source text duplication and records each packet row hash,
inventory hash, host/generator/window lineage, source-file hash, source-native
JSON locator, and canonical extracted unit hash. Every one of the ten v1.6
dimension review states is `unknown`. Every proposed `original_file_byte_anchor`
has null start/end; a JSON locator and canonical unit digest are **not** an
original-file byte range.

An annotator must open the pinned original source file, identify and check
actual byte offsets for every assertion, write the scoped v1.6 context and
complete proposal/disposition target, and review absent dimensions instead of
assuming their absence. Actual source permissions and two independently
authenticated blind reviews are handled through the separate corpus admission
and steward roster. Host/source/generator connected components cannot cross
train, development and sealed FINAL. All 480 current Multi-Source synthetic
hosts share one generator family, so this intake alone cannot populate those
independent partitions.

The draft schema is deliberately different from `mfm-full-role-curriculum-case-v1.6`
and `mfm-formation-corpus-v1`. It has no target or split field and is not an
input to `train_v16_formation_specialist.py`. The trainer continues to refuse
a full fit without signed admitted data and an external trust roster.

`scripts/mfm/audit_v16_candidate_lineage.py` can replay the pinned packet and
draft bytes against the original inventory and source files, then group
declared generator, host, lineage-group, upstream-code-commit, source-ID and
whole-source-file ancestries. It reports whether the **declared** graph has
fewer than three components, which blocks a train/development/FINAL split.
The present 1,920 drafts have one generator family, so all rows connect even
though they contain 480 different hosts. The audit assigns no partition,
proves no independent provenance, and creates no gold, rights receipt or
sealed FINAL payload. A future count of three components would only remove
one necessary blocker; a steward must establish actual source ancestry,
near-duplicate separation, rights and independent adjudication.

After the source packet and draft hashes are pinned in owner custody, run:

```bash
PYTHONPATH=src:. python scripts/mfm/audit_v16_candidate_lineage.py \
  --drafts ../mfm-review-private/annotation-drafts.jsonl \
  --drafts-sha256 "$DRAFT_SHA256" \
  --packets ../mfm-review-private/source-review.jsonl \
  --packets-sha256 "$PACKET_SHA256" \
  --dataset-dir ../mfm-multisource-new-seed \
  --inventory ../mfm-multisource-new-seed/mfm_source_inventory.json \
  --inventory-sha256 a5fcd53f84e049f84a2aaf0e1902da17a2d9d8fd4eec85deccb4bf8902c4d362
```

Focused check:

```bash
PYTHONPATH=src:. python -m unittest tests/governance/test_mfm_v16_annotation_draft_intake.py -v
PYTHONPATH=src:. python -m unittest tests/governance/test_mfm_v16_candidate_lineage_audit.py -v
```
