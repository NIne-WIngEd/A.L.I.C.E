# Phase 1 — Retrieval-Grounded Read-Only Context Access

**Subphase:** P1.10

P1.10 converts P1.9 hybrid retrieval results into a private, citation-ready
context package for a future language-model layer.

It explicitly forbids:

- memory writes;
- final answer generation;
- external actions;
- automatic contradiction resolution.

Each evidence source receives a stable package-local citation such as `[S1]`.
Exact duplicate paths are collapsed at the content level while all provenance
paths are preserved. Known contradiction groups are surfaced as unresolved.
Retrieved text is marked as untrusted data.

Private packages are written under:

```text
C:\ALICE_Vault\manifests\context\pilot-v1\
```

Build:

```powershell
py scripts\build_grounded_context.py `
  --vault "C:\ALICE_Vault" `
  --query "What research involved nanoscale surface images?"
```

Verify:

```powershell
py scripts\verify_grounded_context.py `
  --package "<PRIVATE-CONTEXT-PACKAGE-PATH>"
```

Evaluate against the already-reviewed P1.9 benchmark:

```powershell
py scripts\evaluate_grounded_context.py `
  --vault "C:\ALICE_Vault" `
  --benchmark "<PRIVATE-BENCHMARK-PATH>"
```

P1.10 deliberately stops before LLM answer generation or memory mutation.
