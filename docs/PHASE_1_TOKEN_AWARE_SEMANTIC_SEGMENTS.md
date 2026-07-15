# Phase 1 — Token-Aware Semantic Segments

**Subphase:** P1.9a  
**Reason:** The first semantic index embedded P1.7 character chunks directly. Most exceeded the 512-token E5 limit and were silently truncated by the model runtime.

## Correction

P1.7 chunks remain unchanged because they provide stable provenance for lexical retrieval and later citations.

Before dense embedding, each P1.7 chunk is now divided into deterministic model-aware segments:

- maximum total model input: 480 tokens, including prompt and special tokens;
- overlap: 64 content tokens;
- minimum content window: 48 tokens;
- every segment retains its parent P1.7 chunk ID, source hash, offsets, family, truncation state, and all provenance paths;
- stable semantic-segment IDs are derived from parent chunk ID, policy digest, segment index, character offsets, and segment-text hash;
- every segment is counted with the actual tokenizer before embedding;
- the build aborts rather than accepting a segment that exceeds the configured token limit.

The 32-token margin below E5's 512-token maximum protects the prompt and tokenizer boundary behavior.

## Output schema v2

The rebuilt private semantic index contains:

```text
semantic-manifest.json
segment-map.jsonl
embeddings.f32
```

Important manifest fields:

```text
source_chunk_count
embedding_count
chunks_split_for_embedding
maximum_segments_per_chunk
maximum_segment_tokens
token_truncated_segment_count
```

A valid build requires:

```json
{
  "token_truncated_segment_count": 0,
  "error_count": 0,
  "ready_for_semantic_search": true
}
```

The old schema-v1 semantic index is left in the vault for auditability. The changed policy digest creates a different semantic index ID, so `--replace` is not required.
