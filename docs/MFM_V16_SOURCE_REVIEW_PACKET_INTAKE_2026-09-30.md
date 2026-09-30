# MFM 1.6 source-only review packet intake

**State:** deterministic annotation candidate builder. It creates no formation
target, reviewer decision, adjudication, source rights receipt, split admission,
trained weight, or model qualification.

`scripts/mfm/build_v16_source_review_packets.py` reads the frozen
`mfm_source_inventory.json` and the five inventoried structural source files
for each selected synthetic host. It checks the supplied inventory SHA-256 and
every exact source-file SHA-256 before writing JSONL. Each row is a cumulative
history through a specified day. The default windows are days 1, 7, 15 and 30.
The output is sorted by host and day, then by day and stream within a window.
It is repeatable byte for byte given the same inventory, sources and options.

From the MFM worktree, make a small review packet outside the public repo:

```bash
python scripts/mfm/build_v16_source_review_packets.py \
  --dataset-dir ../mfm-multisource-new-seed \
  --inventory ../mfm-multisource-new-seed/mfm_source_inventory.json \
  --expected-inventory-sha256 a5fcd53f84e049f84a2aaf0e1902da17a2d9d8fd4eec85deccb4bf8902c4d362 \
  --host-family bench_shift_001_drew_carter \
  --output ../mfm-review-private/bench_shift_001_source_review.jsonl
```

Omit `--host-family` to package all hosts, or repeat it to select a set. Use
`--days 1 7 15 30` explicitly to specify the windows. The builder rejects an
existing output if its bytes differ. It reports the output SHA-256 and row count.

Each row carries `status=candidate_unreviewed`, `training_admitted=false`, the
exact inventory SHA-256, host/history/generator lineage, a shared host lineage
group and the cumulative interval. Every evidence unit carries its stable source
ID, relative source path, exact whole-file SHA-256, source-native JSON pointer
(and a second anchor pointer for the profile),
and a SHA-256 of the *canonicalized extracted JSON unit*. The latter is **not**
an original-file byte span; an annotator must open the pinned original file to
bind a narrow byte anchor. Date fields are calendar-day markers, not precise
observation or ingestion timestamps.

Daily planner, self-report, objective and device records are available in the
window for their day and later days. `profile_ltm` has a whole-run horizon and
staleness fields, so it appears only after its declared `total_days` have
elapsed (day 30 in this pinned run). Its generator-only `difficulty_type` field
is omitted. The builder reads no `event_table.json`, `ground_truth.json`,
`generation_metadata.json`, published QA, or v1.5 authored target. It never
copies the inventory's simulator-ancestor digest into a reviewer row.

All 480 hosts share one generator family. Cumulative windows repeat earlier
source records and remain in the same host lineage group; do not assign them
independently across train, development or FINAL. The packets are simulated
source views for review. They do not establish rights or independent gold and
cannot serve as a full-role training handoff. Separate source authority,
blind human review, complete v1.6 target authoring, connected-component split
custody and sealed FINAL admission are described in
[`MFM_V16_CORPUS_AUTHORING_AND_ADMISSION_2026-09-30.md`](MFM_V16_CORPUS_AUTHORING_AND_ADMISSION_2026-09-30.md).

The focused fixture test is:

```bash
PYTHONPATH=src python -m unittest tests/governance/test_mfm_v16_source_review_packets.py -v
```
