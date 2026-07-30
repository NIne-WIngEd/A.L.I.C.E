# Phase 4 — Final Information Release Report

**Status:** Approved only after the final private exact-commit release audit
**Scope:** Phase 4 Public Web and Information Intelligence
**Public-data classification:** INTERNAL
**Private release record:** Stored outside the repository

## Release scope

Phase 4 delivers:

- PUBLIC-only provider-neutral query and research contracts;
- deterministic provider registration and policy selection;
- controlled HTTPS retrieval with URL, DNS, redirect, peer, media-type, size, and timeout enforcement;
- an injection firewall that treats retrieved content as untrusted evidence;
- freshness classification, temporal metadata, source conflict, and explicit uncertainty handling;
- citation-bound grounding and deterministic projection into the released Phase 3 response boundary;
- bounded fixture research orchestration and a controlled evidence pipeline;
- explicit local-only and research conversation modes;
- governed read-only research execution with fail-clean unavailable states;
- a 24-case synthetic adversarial evaluation backed by 28 pinned Phase 4 test files and at least 640 runtime probes;
- an exact-commit private release audit with a required ancestor rollback commit.

The released Phase 4 profile does not perform memory writes, external actions, repository writes, background execution, or automatic private-data disclosure. These are compatibility-profile boundaries, not permanent destination-architecture ceilings.

## Final evaluation

The P4.8 benchmark is synthetic and metadata-only. It contains no real private query text, provider credentials, or fetched source bodies. Its suites cover prompt injection, SSRF, redirects, oversized content, stale dates, source conflicts, citation tampering, privacy leakage, cancellation, timeout, provider failure, and deterministic replay.

The runtime harness pins the canonical benchmark, evaluation policy, runtime manifest, 28 pre-P4.8 test files, and a 640-test collection floor. It disables live network connections in the pytest subprocess, disables bytecode and pytest-cache writes, verifies repository snapshot stability, hashes every target file, and derives all 24 observation records from the runtime result.

## Release audit

P4.9 reruns the P4.8 runtime-backed evaluation on the exact clean release commit. It verifies that the supplied release commit equals `HEAD`, requires a distinct rollback commit that is an ancestor, and binds the following into a canonical private record:

- release commit, repository `HEAD`, clean-tree state, UTC evaluation timestamp, and rollback commit;
- package version `0.15.0`;
- release, evaluation, benchmark, and runtime-manifest identifiers and digests;
- final evaluation, runtime evidence, and combined runtime-backed report digests;
- repository snapshot, collection summary, and execution summary digests;
- target-file, case, collected-test, passed-test, and skipped-test counts;
- network-guard activation, metric gates, known limitations, and release boundaries.

The record writer permits output only below the supplied private vault root. It refuses repository-local output, non-JSON output, different-content overwrite, duplicate keys, malformed values, weakened policy, tampered counts, inconsistent approval, or digest mismatch.

## Release gates

Approval requires:

- all 24 benchmark cases and all critical metric gates to pass;
- at least 28 pinned runtime target files and 640 collected tests;
- every collected runtime test to pass with zero skips;
- an active outbound-network guard and an unchanged repository snapshot;
- zero critical security failures, privacy leaks, prompt-injection successes, network bypasses, citation bypasses, freshness/conflict bypasses, unbounded execution, or unexpected side effects;
- a clean repository whose full `HEAD` commit matches the supplied release commit;
- a distinct rollback commit that is an ancestor of the release commit;
- private JSON output outside the repository;
- an internally consistent canonical SHA-256 release record.

## Known limitations

- The public benchmark and runtime probes are synthetic and metadata-only. Real-provider and real-model acceptance testing remains a separate private concern.
- The release audit blocks live network access and therefore does not validate current provider availability, credentials, pricing, quotas, or real public-web quality.
- Phase 4's released execution path remains read-only and fixture-governed. Live integrated research, learning, storage, and action capabilities belong to successor profiles and later phases.
- Source-quality judgments are deterministic policy behavior, not a substitute for domain-expert review in high-stakes research.

## Release command

Run this only after the P4.9 implementation is committed, pushed, and all tests pass. Use the pre-P4.9 merged P4.8 commit as the rollback reference.

```powershell
$Commit = (git rev-parse HEAD).Trim()
$Rollback = "065ac4e3c395a45ccdfb9845224e35bed262774d"
$EvaluatedAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

py scripts\run_phase4_information_release_audit.py `
    --vault-root C:\ALICE_Vault `
    --repository-root C:\A.L.I.C.E-main `
    --output C:\ALICE_Vault\reports\phase4-information-release.json `
    --repository-commit $Commit `
    --evaluated-at $EvaluatedAt `
    --rollback-commit $Rollback
```

The release is approved only when the command prints:

```text
approved=true
```

The private record must then be retained with the exact release commit. Phase 4 is frozen as a released compatibility baseline after approval and merge, while remaining migratable under the ratified architecture.
