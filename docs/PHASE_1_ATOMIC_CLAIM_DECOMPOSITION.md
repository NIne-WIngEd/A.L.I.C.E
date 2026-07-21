# P1.11 — Atomic Claim Decomposition Before NLI Verification

The FEVER-aware verifier improved the AFM claim's best entailment probability
from roughly 0.019 to 0.131, but the claim still remained strongly neutral and
well below the strict 0.70 keep threshold.

Verifier-side experiments have now covered:

- long cited passages;
- combined cited passages;
- explicit owner attribution;
- compact claim-focused windows;
- generic NLI;
- FEVER/ANLI-aware NLI.

The next change moves upstream.

## New pipeline

```text
Qwen grounded response
        ↓
normalized structured claims
        ↓
private atomic decomposition
        ↓
atomic claims inherit parent citations
        ↓
FEVER-aware NLI gate
        ↓
keep entailed atomic claims only
        ↓
rebuild visible answer from surviving claims
```

The decomposer uses the existing local `qwen3:8b` model and is not allowed to
browse, call tools, write memories, or perform external actions.

It receives only the generated parent claims, not raw vault evidence.

Rules require it to:

- preserve only information already asserted by the parent claim;
- split compound claims into independently checkable propositions;
- avoid adding details;
- decontextualize user-facing pronouns into self-contained wording such as
  `The user ...`.

Every atomic subclaim inherits exactly the citations of its parent claim. The
independent NLI gate remains the final authority. Unsupported decomposed claims
are dropped.

## Privacy

The original generator claims, decomposed pre-gate claims, and decomposition
provenance remain in the private grounded-response package.

The exported summary contains only aggregate counts.

## Evaluation

Grounded-response evaluation advances to schema/checkpoint generation v9 so
older pre-decomposition responses are not resumed.
