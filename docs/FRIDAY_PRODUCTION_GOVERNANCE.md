# Friday Production Governance

**Version:** 1.0.0
**Status:** Owner-ratified production-promotion standard

## 1. Independent development

A qualified Friday team may research, design, prototype, implement, test, open pull requests, build internal artifacts, and propose product or kernel changes without prior production approval. Candidate work remains in development, laboratory, sandbox, simulation, or explicitly non-production channels.

## 2. Dual production approval

Every new or changed production behavior requires:

1. an A.L.I.C.E. audit attestation eligible for owner review; and
2. MK Rayan's explicit production approval.

Both records bind the exact source commit, kernel version, dependency lock, build artifact hashes, model packs, policy and schema versions, migrations, evaluation bundle, deployment manifest, release channel, and rollback manifest. Either A.L.I.C.E. or Rayan may veto or return a candidate. Only Rayan may amend this rule.

## 3. Production scope

The rule covers user-facing capability, authority/autonomy, data and learning, architecture, schema, identity, encryption, storage, distribution, models, feature flags, remote activation, analytics, accounts, and commercial behavior.

## 4. Emergency exception

Without prior dual approval, the team may stop a rollout, disable or quarantine a vulnerable capability, revoke a compromised key, isolate an affected service, or roll back to the last approved release. It may not introduce new behavior, broaden permission, start new collection, weaken privacy, or deploy an unapproved replacement.

## 5. Enforcement

Production signing fails when either approval is missing, mismatched, expired, revoked, or refers to different artifacts. The Friday team does not control A.L.I.C.E.'s attestation identity, Rayan's approval identity, every signing key, and audit-history deletion.
