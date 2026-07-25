# Phase 2 — Final Memory Core Release Report

**Status:** Approved after final private release audit
**Scope:** Phase 2 Memory Core
**Public-data classification:** INTERNAL
**Private release record:** Stored outside the repository

## Release scope

Phase 2 delivers:

- a versioned authoritative memory store;
- provenance-linked memory records;
- temporal, correction, supersession, and conflict handling;
- metadata-safe inspection and authorization-aware retrieval;
- encrypted `HIGHLY_SENSITIVE` storage and purpose-bound access;
- non-authoritative memory candidates and explicit promotion;
- ordinary and protected deletion guarantees;
- rebuildable lexical and semantic indexes;
- deterministic source-cited personal-answer packets;
- versioned final Memory Core evaluation and release gates.

Phase 2 does not deliver a conversational assistant. Model orchestration, conversation state, constitutional dialogue behavior, and the user-facing conversational loop begin in Phase 3.

## Final evaluation

The final benchmark is synthetic-only and contains no real personal data.

It covers:

- confirmed personal facts;
- exact source attribution;
- unsupported personal questions;
- current and historical temporal state;
- material conflicts;
- corrected records;
- uncertainty;
- permission denial;
- sensitive-memory denial;
- deletion absence;
- non-authoritative candidate boundaries;
- prompt injection.

Final verification:

- 32 P2.9a tests passed;
- 34 P2.9b tests passed;
- 30 P2.9c/P2.9d tests passed;
- 96 combined P2.9 tests passed;
- 408 Phase 2 tests passed;
- 532 full-suite tests passed;
- 14 subtests passed.

All governing metric gates passed in the deterministic final evaluation.

## Security boundary

The final evaluation:

- performs no model inference;
- performs no web access;
- calls no tools;
- performs no external action;
- writes no memory after fixture construction;
- keeps outputs private;
- blocks ordinary access to `HIGHLY_SENSITIVE` memory;
- treats stored prompt-injection text as untrusted data;
- preserves no real personal benchmark data in the repository.

The exact release record is written beneath the private vault and is deliberately excluded from Git.

## Known limitations

- The final benchmark is synthetic. Real personal-memory acceptance testing remains private and must not enter the public repository.
- Phase 2 validates structured answer packets, not natural conversational quality.
- Backup expiry and secure operational backup deletion remain deployment responsibilities.
- Phase 3 must preserve all Phase 2 permission, sensitivity, provenance, temporal, conflict, uncertainty, and deletion guarantees.

## Release command

Run the final audit only after all tests pass and the implementation commit is pushed:

```powershell
$Commit = (git rev-parse HEAD).Trim()
$Rollback = (git rev-parse origin/main).Trim()
$EvaluatedAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

py scripts\run_phase2_memory_release_audit.py `
    --vault-root C:\ALICE_Vault `
    --repository-root C:\A.L.I.C.E-main `
    --output C:\ALICE_Vault\reports\phase2-memory-release.json `
    --repository-commit $Commit `
    --evaluated-at $EvaluatedAt `
    --rollback-commit $Rollback
```

The release is approved only when the command prints:

```text
approved=true
```
