# A.L.I.C.E. Phase-Boundary Audit Standard

**Version:** 1.0.0
**Status:** Ratified development standard
**Owner:** MK Rayan

## 1. Purpose

Every top-level phase ends with an adversarial audit of the entire Git-tracked repository, not only the files changed during the final milestone.

The audit asks:

- Does the system still preserve the ratified destination architecture?
- Did a local milestone restriction become a hidden permanent ceiling?
- Do documentation, policy, runtime, tests, workflows, and release claims agree?
- Does the phase work in the environment implied by its name and purpose?
- Did private data, credentials, or generated artifacts enter Git?
- Are rollback, restoration, and next-phase prerequisites real?

A phase is not cleared merely because its unit tests pass.

## 2. Required inventory

The audit must enumerate every path returned by:

```text
git ls-files
```

For every tracked file, record at minimum:

- path;
- byte size;
- SHA-256 digest;
- file category;
- parse or syntax result where applicable;
- critical and advisory findings.

The complete machine report remains private or ephemeral unless its contents are approved for Git.

## 3. Required checks

### Repository integrity

- every tracked path exists;
- no merge-conflict markers;
- JSON, YAML, TOML, and Python syntax parse;
- no committed Python caches, test caches, editor state, or OS metadata;
- no unexpected executable or binary payload;
- no malformed links between controlling documents and policies.

### Privacy and credentials

- no private vault artifact;
- no live database or private index;
- no private release record;
- no credential, token, private key, cookie, or authenticated-session material;
- no raw private prompt, response, browsing history, or training payload;
- no host Identity Capsule or host-specific adapter.

### Architecture and capability continuity

- Constitution, roadmap, authority model, memory policy, storage policy, evaluation charter, capability catalog, Friday parity, shared-kernel separation, and Research Frontiers remain mutually consistent;
- every restrictive released component is phase-local, compatibility-scoped, or paired with a successor path;
- no test, validator, schema, or workflow silently establishes a permanent capability ceiling;
- no capability is marked available without working evidence;
- no planned capability is silently deleted.

### Operational acceptance

A phase whose purpose depends on an external environment must include acceptance evidence against that environment.

Examples:

- web phases require real-provider and real-web acceptance;
- model-training phases require real training and held-out evaluation;
- tool phases require real connector execution;
- storage phases require restoration and deletion drills;
- operating-environment phases require real device and recovery tests.

Synthetic tests remain mandatory but are not a substitute for operational acceptance.

### Release evidence

- exact clean commit;
- rollback ancestor;
- private release record outside Git;
- complete required tests;
- environment and dependency identity;
- known limitations;
- post-deployment monitoring plan;
- updated README, roadmap, capability catalog, and handoff.

## 4. Severity

### Critical

Blocks phase closure and the next phase:

- private data or credential exposure;
- false release claim;
- missing operational acceptance for the phase's defining capability;
- hidden permanent capability ceiling;
- unverified irreversible behavior;
- broken rollback or restoration;
- cross-host or cross-product private-state leakage;
- critical test or syntax failure.

### Major

Requires correction or an explicitly approved migration plan:

- inconsistent status across controlling documents;
- missing evaluator or benchmark for a material capability;
- stale policy binding;
- unowned compatibility debt;
- dependency or workflow gap that can invalidate CI evidence.

### Advisory

May be scheduled without blocking when it does not affect truthful release status, privacy, authority, or recoverability.

## 5. Required outputs

Each phase produces:

- public post-phase audit report;
- private full-file inventory and findings;
- exact commit and rollback reference;
- remediation PRs;
- updated continuation handoff;
- a clear `approved`, `conditionally_approved`, or `blocked` result.

## 6. No safety theater

The standard does not require keeping a system weak.

It requires truthful claims, empirical evidence, recoverability, and explicit authority so capability can expand without self-deception.

A restriction that exists only because an early phase was narrow must be migrated rather than treated as sacred.
