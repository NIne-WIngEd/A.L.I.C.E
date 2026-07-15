# Phase 1 — Local Semantic and Hybrid Retrieval

**Subphase:** P1.9  
**Input:** verified P1.7 chunk catalog and P1.8 FTS5 index  
**Private text uploaded:** no  
**Cloud training:** deferred

## Purpose

P1.9 adds local dense embeddings and combines semantic retrieval with the P1.8
lexical baseline.

The selected model is pinned to:

```text
intfloat/e5-small-v2
revision ffb93f3bd4047442299a41ebb6fa998a38507c52
```

The model is English-only, produces 384-dimensional embeddings, and accepts at
most 512 tokens. The build records how many chunks exceed that token limit.

## Security boundary

Model preparation downloads only public model files. It does not read pilot
chunks.

After preparation:

- the model is loaded from the private vault;
- Hugging Face and Transformers offline modes are enabled;
- telemetry and implicit-token use are disabled;
- `trust_remote_code` remains false;
- only safetensors-based prepared models are accepted;
- private chunk and query text stay local.

## Private model location

```text
C:\ALICE_Vault\models\embeddings\
└── intfloat__e5-small-v2__ffb93f3bd404\
```

## Semantic index

```text
C:\ALICE_Vault\derived\pilot-v1\semantic\<INDEX-ID>\
├── semantic-manifest.json
├── chunk-map.jsonl
└── embeddings.f32
```

The embedding file contains normalized little-endian float32 vectors. At the
current pilot size, exact brute-force cosine search is small enough that an
approximate vector database is unnecessary.

## Prepare the model

```powershell
py scripts\prepare_embedding_model.py `
  --vault "C:\ALICE_Vault"
```

This is the only P1.9 step that requires internet access.

## Build and verify

```powershell
py scripts\build_semantic_index.py `
  --vault "C:\ALICE_Vault"

py scripts\verify_semantic_index.py `
  --vault "C:\ALICE_Vault"
```

## Semantic search

```powershell
py scripts\search_semantic.py `
  --vault "C:\ALICE_Vault" `
  --query "research involving nanoscale surface images"
```

## Hybrid search

```powershell
py scripts\search_hybrid.py `
  --vault "C:\ALICE_Vault" `
  --query "research involving nanoscale surface images"
```

Hybrid retrieval uses reciprocal-rank fusion over:

- P1.8 FTS5/BM25 results;
- P1.9 dense cosine-similarity results.

## Human-curated benchmark

Create a private UTF-8 text file with one natural-language question per line.
Questions should use wording that may differ from the source documents.

Example categories:

- research projects;
- education and transfers;
- work and internships;
- publications and awards;
- technical skills;
- future academic goals;
- recurring workflows.

Generate a private benchmark draft:

```powershell
py scripts\create_semantic_benchmark_draft.py `
  --vault "C:\ALICE_Vault" `
  --questions "C:\ALICE_Vault\private\semantic-questions.txt"
```

The draft contains suggested source hashes and filenames. For each question,
set:

```json
{
  "status": "approved",
  "expected_source_sha256": ["<CORRECT-SOURCE-HASH>"]
}
```

or set `status` to `excluded`.

Validate and evaluate:

```powershell
py scripts\validate_semantic_benchmark.py `
  --vault "C:\ALICE_Vault" `
  --benchmark "<PRIVATE-BENCHMARK-PATH>"

py scripts\evaluate_semantic_benchmark.py `
  --vault "C:\ALICE_Vault" `
  --benchmark "<PRIVATE-BENCHMARK-PATH>"
```

The final aggregate report compares lexical, semantic, and hybrid Hit@K and
MRR. Questions, expected sources, snippets, and detailed results remain private.

## Deferred

- embedding fine-tuning;
- cross-encoder reranking;
- cloud training;
- answer generation;
- long-term memory writes;
- UI integration.
