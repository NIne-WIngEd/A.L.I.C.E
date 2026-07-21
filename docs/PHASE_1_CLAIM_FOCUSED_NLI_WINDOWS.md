# P1.11 — Claim-Focused NLI Evidence Windows

The AFM diagnostic produced:

- best individual entailment: 0.026417
- combined evidence entailment: 0.009089
- explicit owner-attribution entailment: 0.038998

This rules out simple evidence concatenation and owner-attribution wording as
the primary fix.

The NLI model is trained for sentence-pair classification, while the current
premises are long multi-topic resume/project passages. Sentence Transformers
also truncates CrossEncoder inputs that exceed the model's maximum sequence
length.

This patch keeps the strict thresholds unchanged and changes only NLI premise
construction:

1. use only evidence from the claim's cited sources;
2. split cited evidence into overlapping 3-sentence windows;
3. cap each window at 900 characters;
4. rank windows by deterministic lexical overlap with the claim;
5. score at most the top 12 compact windows;
6. keep a claim only if one compact cited-evidence window reaches the existing
   entailment threshold.

No private text is uploaded. No benchmark labels are used. The answer remains
fail-closed when no cited window supports the claim.
