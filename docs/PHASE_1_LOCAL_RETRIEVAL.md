# Phase 1 — Local FTS5 Retrieval and Evaluation

**Subphase:** P1.8  
**Input:** verified P1.7 chunk catalog  
**Cloud use:** none  
**Embeddings:** deferred

## Purpose

P1.8 builds a local lexical retrieval baseline over the private chunk catalog.
It measures retrieval before any embedding model or answer-generation model is
introduced.

## Index design

The index uses SQLite FTS5 with:

- Porter stemming over the Unicode tokenizer;
- Latin diacritic normalization;
- 2-, 3-, and 4-character prefix indexes;
- weighted BM25 ranking;
- filename and source-path fields;
- snippet generation;
- ordinary relational tables for filters and provenance.

A contentful FTS5 table is used instead of an external-content table. The
retrieval database is derived and can be rebuilt from the verified P1.7
catalog.

## Query behavior

Natural-language queries are reduced to searchable content words. Search first
requires all retained terms. If that returns no matches, it falls back to an OR
query.

Results are ranked with BM25 and then collapsed so overlapping chunks from one
source do not dominate the result list.

Available filters:

- family;
- year;
- source bucket;
- contradiction label;
- truncated-source inclusion.

## Private outputs

```text
C:\ALICE_Vault\derived\pilot-v1\retrieval\<INDEX-ID>\
├── retrieval.sqlite3
└── retrieval-manifest.json
```

## Build and verify

```powershell
py scripts\build_retrieval_index.py `
  --vault "C:\ALICE_Vault"

py scripts\verify_retrieval_index.py `
  --vault "C:\ALICE_Vault"
```

Verification includes:

- SQLite integrity check;
- FTS5 integrity-check command;
- database digest;
- chunk, FTS, and provenance counts;
- chunk-catalog and policy digests;
- sampled chunk-body hashes.

## Search

```powershell
py scripts\search_pilot.py `
  --vault "C:\ALICE_Vault" `
  --query "AFM segmentation research"
```

Search results contain private snippets and provenance. Do not paste or upload
them unless deliberately reviewed.

Example filters:

```powershell
py scripts\search_pilot.py `
  --vault "C:\ALICE_Vault" `
  --query "transfer plan" `
  --family education `
  --year 2026 `
  --exclude-truncated
```

## Lexical smoke benchmark

The automatic benchmark selects rare indexed terms and associates them with
their known source documents. This validates index plumbing and lexical
ranking; it is not a semantic personal-question benchmark.

```powershell
py scripts\create_lexical_benchmark.py `
  --vault "C:\ALICE_Vault"

py scripts\evaluate_retrieval.py `
  --vault "C:\ALICE_Vault"
```

Aggregate metrics:

- hit rate at 1, 3, 5, and 10;
- mean reciprocal rank at 10;
- missed-case count.

Private benchmark questions and evaluation details remain in the vault.

## Deferred

- embeddings and vector search;
- hybrid lexical/vector fusion;
- semantic reranking;
- curated natural-language personal QA benchmark;
- answer generation;
- cloud training.
