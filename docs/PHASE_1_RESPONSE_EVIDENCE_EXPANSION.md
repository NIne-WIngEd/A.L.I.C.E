# P1.11 Evidence Passage Expansion

The v3 benchmark produced 11/11 structurally verified responses but 9/11 were
`insufficient_evidence`.

This exposed a source-level versus passage-level mismatch:

- P1.10 correctly retrieved the expected source document.
- The context package carried only one representative snippet for that source.
- For broad personal questions, that one snippet often did not contain the
  evidence needed to answer, even though another chunk or semantic segment in
  the same source did.

This patch preserves the P1.10 source ranking and expands each selected source
with up to three query-relevant passages from the existing local semantic
segment index.

It does not:

- consult benchmark expected-source labels;
- introduce sources outside the P1.10 top source set;
- upload private text;
- enable memory writes, tools, web access, or external actions.

Passage selection combines local semantic cosine similarity with a small
lexical-overlap bonus and removes heavily overlapping segments.

The response benchmark checkpoint generation advances to v4 so v3 results are
not resumed after the context-construction contract changes.
