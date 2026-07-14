# Phase 1 — Deterministic Chunking and Provenance Catalog

**Subphase:** P1.7  
**Input:** verified P1.6 extraction outputs  
**Cloud use:** none  
**Embeddings:** deferred

P1.7 converts verified extracted text into deterministic overlapping chunks while preserving complete path-level provenance.

The machine-readable policy is `policies/chunking_policy.json`. The default uses a 2,400-character maximum, 1,800-character target, 500-character minimum, and 240-character overlap. Paragraph, sentence, line, and word boundaries are preferred over hard cuts.

Stable chunk IDs include the source SHA-256, policy digest, chunk index, normalized offsets, and chunk-text SHA-256. The chunk-set ID includes the pilot-manifest hash, parser-registry digest, and chunking-policy digest.

Private output:

```text
C:\ALICE_Vault\derived\pilot-v1\chunks\<CHUNK-SET-ID>\
├── chunk-manifest.json
├── chunk-records.jsonl
├── source-map.json
├── chunks.sqlite3
└── text\<CHUNK-ID>.txt
```

Each chunk records extraction truncation, parser identity, all original file paths, duplicate controls, source buckets, year hints, and contradiction labels. Cross-document chunk deduplication is deliberately disabled so independent provenance is not collapsed.

Build:

```powershell
py scripts\build_pilot_chunks.py `
  --vault "C:\ALICE_Vault" `
  --pilot-name "pilot-v1"
```

Verify:

```powershell
py scripts\verify_pilot_chunks.py `
  --vault "C:\ALICE_Vault" `
  --pilot-name "pilot-v1"
```

P1.7 passes only when `deterministic_rebuild_match` and `ready_for_indexing` are true and `error_count` is zero.
