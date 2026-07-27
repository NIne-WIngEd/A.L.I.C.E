# Phase 4 — Web and Information Tools Architecture

**Status:** P4.2 controlled retrieval boundary started; live network disabled
**Phase 0 dependency:** Ratified governance and default-deny permission model
**Phase 1 dependency:** Frozen read-only evidence layer
**Phase 2 dependency:** Frozen authoritative Memory Core
**Phase 3 dependency:** Frozen governed conversation layer
**Owner:** MK Rayan

## 1. Purpose

Phase 4 gives A.L.I.C.E. safe access to current public information.

The capability is read-only. Network access is controlled by deterministic application code. The model never receives a raw network client or an unrestricted browsing loop.

Phase 4 must provide:

- provider-neutral web search;
- controlled source retrieval;
- freshness and temporal metadata;
- exact source citations;
- bounded foreground research;
- sanitized activity records.

Phase 4 does not add service writes, authenticated browsing, background monitoring, arbitrary code execution, memory writes, or external actions.

## 2. Governing boundary

```text
Phase 0 — Constitution, permissions, classification, evaluation
                              |
Phase 1 — Personal evidence   |   Phase 2 — Authoritative memory
                \             |             /
                 \            |            /
                  v           v           v
                  Phase 3 — Governed conversation
                              |
                              v
                  Phase 4 — Information gateway
                      |                 |
                      v                 v
              approved search     approved fetch
                 provider          transport
                      \                 /
                       v               v
                   normalized PUBLIC sources
                              |
                              v
                injection and freshness gates
                              |
                              v
                citation-bound grounding packet
                              |
                              v
                  Phase 3 response validation
```

The model is not the network security boundary. It cannot select arbitrary providers, alter URL checks, approve private-data transmission, increase budgets, follow instructions from retrieved content, or bypass Phase 3 response validation.

## 3. Phase 4 invariants

1. External information access maps to the existing `web.search` P1 permission.
2. Initial outbound queries may contain `PUBLIC` information only.
3. `INTERNAL`, `PRIVATE`, `HIGHLY_SENSITIVE`, and `SECRETS` cannot enter an external query in the initial release.
4. `HIGHLY_SENSITIVE` and `SECRETS` remain prohibited from web requests, source logs, prompts, and public fixtures.
5. Search and fetch are read-only operations.
6. Retrieved content is untrusted data. It is never policy, permission, or authorization.
7. Retrieved instructions cannot trigger tools, actions, credentials, memory writes, or recursive browsing.
8. All network activity passes through an explicit provider or transport registry.
9. No provider fallback occurs unless a later policy explicitly governs it.
10. Only HTTP and HTTPS source identities are accepted.
11. Localhost, private-network, link-local, metadata-service, and other non-public destinations are blocked.
12. Redirect targets are revalidated before connection.
13. Search, fetch, source, byte, redirect, and time budgets are deterministic.
14. Every externally grounded claim requires an exact source-version citation.
15. Publication, update, and retrieval timestamps remain distinct.
16. Stale, conflicting, or unsupported current claims remain visible or fail closed.
17. Phase 4 grounding enters the existing Phase 3 validation path. It does not bypass P3.6.
18. Activity records contain sanitized metadata. Raw private queries and source bodies are not ordinary logs.
19. Phase 4 does not write authoritative memory or create memory candidates automatically.
20. Phase 4 research is foreground-only. Recurring monitoring belongs to Phase 7.

## 4. P4.0 foundation scope

P4.0 creates contracts and a public policy. It intentionally enables no live network provider.

Implemented in P4.0:

- `alice_information` package boundary;
- public query, request, search-result, source, citation, claim, grounding, and activity contracts;
- deterministic SHA-256 bindings;
- structural public-URL canonicalization;
- literal localhost and non-public IP rejection;
- explicit untrusted-source rendering;
- exact citation-to-source URL and digest verification;
- metadata-only activity records;
- versioned `information_policy.json`;
- PUBLIC-only query transmission;
- fail-closed capability declarations;
- deterministic resource budgets;
- tests for contract and policy violations.

Not implemented in P4.0:

- live search;
- live source fetching;
- DNS resolution;
- redirect execution;
- HTML parsing;
- source-quality scoring;
- freshness classification;
- prompt-injection detection beyond hard data delimiting;
- research orchestration;
- Phase 3 runtime integration;
- background work.

## 4.1 P4.1 provider abstraction scope

P4.1 adds the execution interface without adding a network transport.

Implemented in P4.1:

- separate read-only search and fetch protocols;
- exact lowercase provider identities;
- deterministic search fixtures keyed by public-query digest;
- deterministic source fixtures keyed by canonical URL;
- cooperative cancellation before and after provider execution;
- bounded result, timeout, and response-byte inputs;
- sanitized provider-failure metadata and an approved failure-code vocabulary;
- exact operation-specific provider registration;
- a versioned P4.1 provider allowlist;
- explicit duplicate-registration rejection;
- exact resolution with no provider fallback;
- deterministic output replay and protocol validation.

Still disabled in P4.1:

- HTTP or HTTPS connections;
- DNS resolution;
- redirects;
- live provider credentials;
- authenticated browsing;
- automatic retries;
- provider fallback;
- background execution.

The P4.1 provider registry requires both the P4.0 information policy and the
P4.1 provider policy. A provider must be allowed by exact identity, provider
type, and operation before registration. The only approved provider type is
`deterministic_fixture`.

## 4.2 P4.2a controlled retrieval boundary

P4.2a establishes the connection-time safety and deterministic normalization
boundary before any operating-system DNS or socket adapter is approved.

Implemented in P4.2a:

- versioned HTTP retrieval policy bound to the P4.0 limits;
- default-port-only HTTP and HTTPS target validation;
- deterministic resolver fixtures that require every resolved address to be
  globally routable;
- deterministic transport fixtures pinned to the approved address set;
- point-of-use revalidation for every redirect hop;
- HTTPS downgrade, redirect-loop, and redirect-budget rejection;
- fixed credential-free GET headers;
- no environment proxies, cookies, authentication, downloads, retries, or
  caller-controlled headers;
- response-header and singleton-header validation;
- MIME-type, character-set, and content-encoding allowlists;
- declared, streamed, compressed, and decoded byte limits;
- bounded gzip and deflate decoding;
- deterministic visible-text normalization for HTML, XHTML, and plain text;
- exact normalized-content digests and source-contract projection;
- exact-content duplicate observations;
- sanitized DNS, redirect, peer, header, status, size, decoding, and
  normalization failures.

Still disabled after P4.2a:

- operating-system DNS resolution;
- sockets or live HTTP clients;
- live search or fetch provider registration;
- credentials, cookies, JavaScript, forms, downloads, proxies, and retries;
- recursive browsing or background work.

The deterministic resolver and transport are security fixtures, not live web
providers. A later live adapter must preserve the same address pinning and
response gates and requires a separate policy change and evaluation.

## 5. Package boundary

Phase 4 code lives in:

```text
src/alice_information/
```

Phase 4 tests live in:

```text
tests/phase4/
```

Public policies live in:

```text
policies/information_policy.json
policies/information_provider_policy.json
policies/information_http_retrieval_policy.json
```

Private browsing history, live queries, fetched content, provider credentials, activity databases, caches, and release records must remain outside the public repository.

## 6. Public contracts

### 6.1 Query contract

An `InformationQuery` contains:

- query ID;
- minimized public query text;
- content digest;
- creation time;
- `PUBLIC` classification.

The initial policy rejects any other classification. A future exception for private context requires a separate authorization and redaction design. It cannot be introduced by a provider or model response.

### 6.2 Research-request contract

An `InformationResearchRequest` contains:

- request ID;
- validated query;
- requested read-only operations;
- search-call limit;
- fetch-call limit;
- source limit;
- per-request timeout;
- total timeout;
- fail-closed capabilities.

P4.0 validates the shape only. It cannot execute the request.

### 6.3 Search-result contract

An `InformationSearchResult` binds:

- exact provider identity;
- rank;
- title;
- canonical URL;
- snippet digest;
- retrieval time;
- `PUBLIC` classification;
- mandatory untrusted-content state.

Search snippets are discovery hints. They are not sufficient source evidence by themselves unless a later policy explicitly allows a source type and evaluation proves the behavior.

### 6.4 Source-document contract

An `InformationSourceDocument` binds:

- source ID;
- provider identity;
- canonical URL;
- title;
- normalized text;
- content digest;
- publication time when available;
- update time when available;
- retrieval time;
- `PUBLIC` classification;
- mandatory untrusted-content state.

The model rendering uses explicit start and end delimiters. It states that the source is data, not instructions or authorization.

### 6.5 Citation and grounding contract

Every `InformationClaim` requires at least one `InformationCitation`.

A citation binds to:

- source ID;
- canonical URL;
- exact normalized-content digest;
- visible citation token.

An `InformationGroundingPacket` rejects citation swapping. The citation URL and digest must match the source contained in the same packet.

### 6.6 Activity contract

An `InformationActivityRecord` stores only metadata such as:

- activity and request IDs;
- operation;
- provider;
- status;
- start and finish times;
- query digest;
- source IDs;
- sanitized error code.

It has no raw query-text or source-content field.

## 7. URL and network safety

P4.0 performs the structural URL gate:

- HTTP and HTTPS only;
- no embedded credentials;
- no raw control characters or whitespace;
- deterministic lowercase scheme and host;
- default-port removal;
- fragment removal;
- localhost rejection;
- literal non-public IP rejection.

P4.2 must add the connection-time gate:

1. resolve the hostname through an injectable resolver;
2. reject every non-global address;
3. connect only to the validated destination;
4. limit response size while streaming;
5. revalidate each redirect target and resolved address;
6. reject unsupported content types and encodings;
7. enforce decompression limits;
8. record sanitized completion metadata.

A URL that passed an earlier check is not permanently trusted. DNS and redirect checks occur at the point of use.

## 8. Retrieved-content security

All retrieved text is adversarial by default.

It may contain:

- direct prompt injection;
- fake system messages;
- tool commands;
- credential requests;
- instructions to ignore policy;
- claims that a permission was granted;
- encoded or obfuscated instructions;
- links to private-network targets;
- false dates or source identity.

The information layer must preserve a strict separation between system instructions and source data. Later injection detection may label or exclude suspicious spans. Detection is defense in depth. Permission and action boundaries must remain safe even when detection misses an attack.

## 9. Freshness model

Phase 4 tracks three separate times:

- `published_at`: when the source says the information was published;
- `updated_at`: when the source says it was revised;
- `retrieved_at`: when A.L.I.C.E. obtained the source.

Source-provided dates are claims. They require validation and may be missing or false.

P4.4 will classify the query as current, historical, or time-insensitive. It will apply a policy-defined freshness requirement. A.L.I.C.E. must not use “latest,” “current,” “today,” or similar language without sufficient temporal evidence.

Phase 2 memory validity and Phase 4 public-source freshness remain separate systems.

## 10. Error taxonomy

Phase 4 uses sanitized deterministic errors. Planned categories include:

- `information_denied`;
- `query_classification_denied`;
- `provider_not_registered`;
- `provider_fixture_missing`;
- `provider_protocol_error`;
- `provider_timeout`;
- `dns_resolution_failed`;
- `peer_address_mismatch`;
- `response_header_invalid`;
- `http_status_rejected`;
- `content_decode_failed`;
- `research_budget_exhausted`;
- `invalid_source_url`;
- `private_network_blocked`;
- `redirect_blocked`;
- `unsupported_content_type`;
- `response_too_large`;
- `normalization_failed`;
- `prompt_injection_blocked`;
- `freshness_insufficient`;
- `citation_validation_failed`;
- `research_cancelled`;
- `information_integrity_failed`.

Errors must not contain credentials, private query text, raw source bodies, or internal stack details in user-visible output.

## 11. Approved milestone plan

### P4.0 — Architecture and contracts

Deliver the provider-neutral contracts, fail-closed policy, threat-model update, architecture document, and deterministic tests. Enable no live network access.

Exit criteria:

- all P4.0 tests pass;
- PUBLIC-only query transmission is enforced;
- source content is explicitly untrusted;
- citations bind to exact source versions;
- no live provider is registered;
- Phases 0–3 regressions remain green.

### P4.1 — Information-provider abstraction

Build search and fetch protocols, deterministic fixture providers, cancellation, sanitized failures, exact provider registry, provider policy, and no-fallback enforcement. **Implemented in package version `0.2.0`; live network access remains disabled.**

Exit criteria:

- deterministic providers replay identical results;
- unknown providers fail closed;
- cancellation and budgets are enforced;
- no network call occurs in the test provider.

### P4.2 — Controlled source retrieval and normalization

Add the connection-time HTTP(S) security boundary, deterministic resolver and transport fixtures, DNS and redirect revalidation, SSRF protection, peer pinning, streaming limits, content controls, decompression limits, canonical source metadata, text normalization, and duplicate detection. **The P4.2a boundary is implemented in package version `0.3.0`; operating-system DNS and live sockets remain pending within P4.2 and disabled.**

Exit criteria:

- local and private networks remain unreachable;
- redirect and rebinding attacks fail;
- oversized and unsupported responses fail closed;
- normalized content is deterministic and digest-bound.

### P4.3 — Injection firewall

Add instruction-like-content analysis, containment labels, source isolation, credential-request detection, policy-override rejection, and adversarial fixtures.

Exit criteria:

- retrieved instructions cannot alter policy or permissions;
- no source can trigger a tool or action;
- critical prompt-injection cases have zero successes.

### P4.4 — Freshness and temporal reasoning

Add query-time classification, publication and update extraction, freshness policy, stale-source warnings, date conflict handling, and unsupported-current-claim rejection.

Exit criteria:

- “latest” claims require fresh evidence;
- stale and conflicting dates remain visible;
- historical queries do not incorrectly require current sources.

### P4.5 — Citation-bound web grounding

Build source-quality metadata, claim construction, exact citation verification, source-diversity rules, uncertainty and conflict preservation, and the adapter into Phase 3 grounding.

Exit criteria:

- visible external claims are source-supported;
- citation swapping and digest tampering fail;
- P3.6 remains the final visible-response gate.

### P4.6 — Governed research orchestration

Build bounded query planning, maximum search and fetch counts, deterministic stopping, cancellation, partial-result handling, sanitized activity persistence, and no recursive uncontrolled browsing.

Exit criteria:

- every run terminates under policy budgets;
- no hidden provider fallback or arbitrary link following occurs;
- partial and failed research is reported truthfully.

### P4.7 — Local conversation integration

Add an explicit local research mode. Show when web research is used. Render sources and freshness. Add offline behavior. Do not enable silent web access.

Exit criteria:

- the user can distinguish local-only and web-grounded replies;
- offline mode fails cleanly;
- unrelated turns do not silently trigger web access.

### P4.8 — Final adversarial information evaluation

Create a synthetic public benchmark for injection, SSRF, redirects, oversized content, stale dates, source conflicts, citation tampering, privacy leakage, cancellation, timeout, provider failure, and deterministic replay.

Exit criteria:

- zero critical security failures;
- source quality and freshness gates pass;
- no real private query or browsing content enters Git.

### P4.9 — Release audit and closure

Bind test-backed evidence and the final evaluation to an exact clean commit. Require a rollback commit. Write the private release record under the vault. Close the roadmap and README only after approval.

Exit criteria:

- private audit returns `approved=true`;
- exact commit, policy versions, package version, evaluation digest, evidence digest, and rollback commit are recorded;
- Phase 4 is frozen after merge.

## 12. Development sequence

Each milestone follows:

```text
main
  ↓
feature branch
  ↓
small implementation
  ↓
targeted Phase 4 tests
  ↓
Phase 4 suite
  ↓
Phase 2 + Phase 3 + Phase 4 regression
  ↓
full suite
  ↓
working-tree and staged-index audit
  ↓
commit and push
  ↓
PR and required checks
  ↓
merge, sync main, delete branch
```

Phases 0–3 remain frozen. A regression or security fix in an earlier phase requires a dedicated maintenance branch and explicit scope.
