# P1.11 — Evidence-Constrained Atomic Claim Generation

The latest diagnostic showed:

- the generator produced one claim;
- atomic decomposition received one claim;
- atomic decomposition returned the exact same seven-word claim;
- the FEVER-aware NLI verifier still classified it as neutral with
  entailment probability 0.130780.

Therefore the claim was already atomic. Post-generation decomposition is not
the bottleneck.

This patch moves claim creation upstream.

## Pipeline

```text
question + retrieved cited evidence
        ↓
local evidence-constrained claim generator
        ↓
atomic factual claims with exact citations
        ↓
FEVER-aware NLI support gate
        ↓
keep entailed claims only
        ↓
render visible answer
```

The original free-form answer generator remains temporarily for compatibility
and diagnostics, but its claims are replaced before the support gate whenever
evidence-constrained claim generation is enabled.

The evidence claim generator:

- sees the user question and retrieved evidence;
- generates only atomic factual claims directly supported by cited evidence;
- uses exact package citation IDs;
- may attribute owner-self-record facts to `the user`;
- returns an empty claim list when evidence is insufficient;
- cannot browse, call tools, write memory, or perform external actions.

Post-generation atomic decomposition is bypassed when evidence-constrained claim
generation is active because the new stage already requires atomic claims.

The independent FEVER/ANLI-aware NLI gate remains the final authority.

Grounded-response evaluation advances to checkpoint/schema generation v10.
