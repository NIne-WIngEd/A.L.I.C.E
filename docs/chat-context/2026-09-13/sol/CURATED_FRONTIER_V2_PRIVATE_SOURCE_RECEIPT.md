# Curated Frontier v2 — Private Source Receipt — 2026-09-13

This file records a **sanitized receipt only** for the private A.L.I.C.E. EIPM curated package supplied to the 2026-09-13 continuation chat.

The raw archive is intentionally not committed to this public repository branch.

## Artifact identity

- supplied filename: `ALICE_EIPM_CURATED_FRONTIER_v2(1).zip`
- size: `1,881,844` bytes
- SHA-256: `3867ff04d1e326086b9086b2f106b9156b3a3ec8d637d3161e7bf01616183ee9`
- ZIP integrity: PASS
- package manifest entries: 34
- root SHA256SUMS entries: 34
- nested targeted SHA256SUMS entries: 8
- manifest/hash mismatches: 0
- malformed JSON/JSONL rows: 0

The SHA-256 matches the artifact recorded in `docs/eipm/EIPM_CURATED_FRONTIER_V2_RECEIPT.md` on `alice-eipm-v1-build`.

## Sanitized file/count inventory

- `alternative_policy_competitors_v1.jsonl`: 13,719
- `raw_transform_instance_index_v1.jsonl`: 12,924
- relationship transforms: 21
- state transforms: 56
- `curated_asyn_base_policies_v2`: 822
- `curated_asyn_context_variants_v2`: 15
- `curated_asyn_direct_v2`: 1,009
- `curated_asyn_targeted_v2`: 119
- `curated_einf_v2`: 205
- `historical_unknown_bank_v2`: 3,496
- targeted gap dispositions: 67
- targeted queue remaining: 0
- repair ledger: 211
- targeted curation ledger: 134
- reserve raw A-SYN: 288
- reserve raw E-INF: 720
- reserve legacy UNKNOWN: 60
- canonical router: 290
- canonical E0 semantic units: 290
- G2A gold candidate pass2 rows: 171
- targeted raw A-SYN proposals: 119
- regenerated targeted context variants: 15
- targeted E-INF proposals: 0

## Rechecked invariants

- canonical E0 router unit IDs equal the canonical E0 unit IDs: PASS
- canonical E0 text SHA-256 checks: PASS
- active curated support references resolve: PASS
- active identity support overlapping `exclude_from_identity_loss`: none found
- exact normalized behavior-text duplicate groups across active curated E-INF/direct/base/targeted/context sets: none found
- targeted 119 proposal IDs/source payloads preserved into curated targeted output: PASS
- targeted 15 context-variant IDs/source payloads preserved: PASS
- 67 gap dispositions link to 119 unique generated A-SYN proposals: PASS
- generated targeted E-INF: 0

## Current router counts in this artifact

These are the observed v2 values and supersede older historical count snapshots when discussing the supplied Curated Frontier v2 artifact:

- `context_only_conditioning`: 226
- `conditional_identity_supervision`: 145
- `direct_identity_supervision`: 123
- `episodic_behavior_supervision`: 71
- `exclude_from_identity_loss`: 64
- `style_voice_supervision`: 23

## Authority boundary

This receipt does not promote proposals to E0, does not declare historical Elaina truth, does not authorize model training, and does not grant generator/curator authority. The raw private rows remain under owner-controlled custody.
