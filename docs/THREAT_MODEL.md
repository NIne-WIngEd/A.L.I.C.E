# A.L.I.C.E. Threat Model

**Version:** 1.0.0  
**Status:** Phase 0 baseline  
**Owner:** MK Rayan  
**Review trigger:** Every major new capability

## 1. System assets

- Rayan's personal life archive and memories
- credentials and connected accounts
- permissions and standing authorizations
- source provenance and audit logs
- A.L.I.C.E. code, prompts, policies, and evaluations
- research, academic, employment, and project data
- devices and services controlled through tools
- the emergency stop and rollback mechanisms

## 2. Trust boundaries

1. Rayan ↔ A.L.I.C.E. interface
2. language model ↔ orchestration layer
3. orchestration layer ↔ permission gateway
4. permission gateway ↔ tools
5. memory service ↔ databases and vector indexes
6. local environment ↔ cloud model providers
7. web/email/files ↔ retrieval pipeline
8. development branch ↔ production system
9. public repository ↔ private data vault

## 3. Primary threats and required controls

| ID | Threat | Severity | Required controls | Verification |
|---|---|---:|---|---|
| T-001 | Credential committed to source control | Critical | secret manager, `.gitignore`, push protection, secret scanning, immediate rotation, history remediation | secret scan and repository-history test |
| T-002 | Prompt injection from webpages, email, files, images, or memories | Critical | treat retrieved content as untrusted data, isolate instructions, permission gateway, output validation | adversarial injection suite |
| T-003 | Unauthorized external action | Critical | default deny, typed permissions, exact confirmation, least-privilege tools, action audit | zero unauthorized-action test |
| T-004 | Memory poisoning or false autobiographical record | High | provenance, confidence labels, confirmation, conflict preservation, correction interface | poisoned-document tests |
| T-005 | Disclosure of private memory to a cloud model or third party | Critical | data classification, context minimization, local-first routing, approval for sensitive transmission | data-flow and redaction tests |
| T-006 | Hallucinated personal facts or completed actions | High | source-required personal answers, status verification, uncertainty labels | factuality and action-verification tests |
| T-007 | Excessive or emotionally manipulative personalization | High | memory dignity rules, relationship boundaries, evaluation of dependency/manipulation behaviors | behavioral red-team cases |
| T-008 | Self-modification weakens oversight | Critical | protected Constitution, isolated branches, tests, human review, no direct production write | self-change pipeline tests |
| T-009 | Shutdown or rollback resistance | Critical | out-of-band kill switch, revocable credentials, process supervisor, no self-preservation objective | emergency-stop drill |
| T-010 | Compromised dependency or model provider | High | pinned/reviewed dependencies, minimal permissions, provider isolation, integrity checks | dependency and supply-chain scan |
| T-011 | Local device compromise | Critical | disk encryption, OS account security, short-lived tokens, least privilege, secure backups | device-security checklist |
| T-012 | Backup or log leakage | High | encrypted backups, retention limits, content minimization, separate key management | restore and access audit |
| T-013 | Identity spoofing or unauthorized user control | Critical | authentication, session locking, strong confirmation for P4, device trust | authentication tests |
| T-014 | Tool output spoofing or incomplete execution | High | structured tool responses, independent verification, idempotency keys where appropriate | fault-injection tests |
| T-015 | Denial of wallet or unbounded computation | Medium | budgets, timeouts, quotas, cancellation, proactive-task limits | resource-cap tests |
| T-016 | Cross-project or cross-person data leakage | High | namespaces, access filters, scoped retrieval, provenance | isolation tests |
| T-017 | Corrupted database, embeddings, or migration | High | checksums, versioned migrations, backups, rollback, rebuildable indexes | restore and migration tests |
| T-018 | Public repository used as private-data storage | Critical | explicit repository policy, blocked paths, data inventory checks | CI path and secret checks |
| T-019 | Malicious or mistaken memory deletion | High | strong target resolution, preview, backup window, deletion audit | deletion and recovery tests |
| T-020 | A.L.I.C.E. silently expands task scope | High | exact action plans, permission per action, no permission laundering | scope-expansion tests |

## 4. Security assumptions

- No model is trusted to enforce its own permissions.
- Retrieved text can be malicious even when it comes from Rayan's own files.
- A public repository must be treated as permanently observable.
- Revoking an exposed credential is more important than merely deleting the visible file.
- Zero risk is impossible; risks must be continuously identified, measured, and managed.
- Personalization increases both usefulness and potential harm.

## 5. Incident response

For a material security incident:

1. stop affected automation;
2. revoke exposed credentials or tool sessions;
3. preserve sanitized evidence;
4. determine affected data and actions;
5. notify Rayan clearly;
6. remediate the root cause;
7. test the fix;
8. restore from a known-good state;
9. document lessons and update tests;
10. re-enable the capability only after approval.

## 6. Current Phase 0 security decision

The historic prototype is not an active component. It is quarantined under `legacy/`. Phase 1 must begin with a private, encrypted data-vault design; no personal archive is to be committed to this repository.
