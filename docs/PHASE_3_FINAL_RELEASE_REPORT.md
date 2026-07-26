# Phase 3 — Final Conversational Release Report

**Status:** Approved only after the final private exact-commit release audit
**Scope:** Phase 3 Conversational A.L.I.C.E.
**Public-data classification:** INTERNAL
**Private release record:** Stored outside the repository

## Release scope

Phase 3 delivers:

- provider-neutral conversational model contracts and a governed local adapter;
- private conversation state with explicit retention and lifecycle controls;
- read-only Phase 1 evidence and Phase 2 authoritative-memory grounding bridges;
- versioned constitutional dialogue behavior;
- controlled orchestration with fail-closed generated-response validation;
- a local terminal conversation runtime;
- bounded, integrity-checked cross-turn context;
- one policy-gated response-repair attempt;
- a versioned synthetic adversarial evaluation;
- an exact-commit private release audit with a required rollback commit.

Phase 3 does not enable web access, tools, external actions, or conversational memory writes.

## Final evaluation

The P3.10 benchmark is synthetic and metadata-only. It contains no real personal conversation content. Its suites cover constitutional behavior, grounding, citations, abstention, context continuity, truncation, cross-session isolation, prompt injection, capability boundaries, hidden-reasoning protection, controlled repair, cancellation, interruption, retention, replay, privacy, integrity, and provider failure.

P3.11 generates the observation bundle from a versioned evidence manifest that maps every benchmark case to existing Phase 3 pytest targets. It runs each unique target on the exact clean commit, stores only result digests and counts, and then reruns the P3.10 evaluation. It binds the verified report digest to the exact clean repository commit, the governing policies, package version `0.12.0`, evaluation timestamp, and a distinct ancestor rollback commit.

## Release gates

Approval requires:

- every test-backed evidence target and synthetic benchmark case to pass;
- every critical metric gate to pass;
- zero privacy, prompt-injection, cross-session, hidden-reasoning, repair-loop, integrity, context-budget, capability, or external-effect violations;
- a clean repository whose full `HEAD` commit matches the supplied release commit;
- a distinct rollback commit that is an ancestor of the release commit;
- private JSON output outside the repository;
- an internally consistent canonical SHA-256 release record.

The release command refuses a dirty repository, commit mismatch, non-ancestor rollback, weakened release policy, repository-local output, unexpected overwrite, or tampered record.

## Known limitations

- The public benchmark is synthetic and metadata-only. Real personal-data and real-model acceptance testing must remain private.
- Conversational quality varies by the selected local model and hardware. The deterministic application boundary is the release subject.
- Live Phase 1 or Phase 2 retrieval is not initiated automatically by the terminal runtime. Grounding remains supplied through governed read-only packets.
- Phase 3 has no web access, tools, external actions, graphical interface, background service, or conversational memory promotion.

## Release command

Run this only after the P3.11 implementation is committed, pushed, and all tests pass. The audit generates metadata-only observations from the public evidence manifest; raw pytest output is reduced to SHA-256 digests and is not written to the release record.

```powershell
$Commit = (git rev-parse HEAD).Trim()
$Rollback = (git rev-parse origin/main).Trim()
$EvaluatedAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

py scripts\run_phase3_conversation_release_audit.py `
    --vault-root C:\ALICE_Vault `
    --repository-root C:\A.L.I.C.E-main `
    --output C:\ALICE_Vault\reports\phase3-conversation-release.json `
    --repository-commit $Commit `
    --evaluated-at $EvaluatedAt `
    --rollback-commit $Rollback
```

The release is approved only when the command prints:

```text
approved=true
```

The private record must then be retained with the exact release commit. Phase 3 is frozen after approval and merge.
