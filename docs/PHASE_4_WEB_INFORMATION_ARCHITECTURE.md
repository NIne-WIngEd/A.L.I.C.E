> **A.L.I.C.E. CAPABILITY SCOPE:** `historical_or_phase_local`; `capability_ceiling=false`; profile `information.phase4.foundation`. Restrictions below reproduce a released or historical configuration and do not limit successor A.L.I.C.E. capabilities.

# Phase 4 — Web and Information Tools Architecture

**Status:** P4.0–P4.9 fixture-governed compatibility release approved; P4.10 operational live-public-information closure required before Phase 5
**Phase 0 dependency:** Ratified Governance 1.1 and mission/profile-driven authority
**Phase 1 dependency:** Released read-only evidence compatibility baseline
**Phase 2 dependency:** Released authoritative-memory compatibility baseline
**Phase 3 dependency:** Released governed-conversation compatibility baseline
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
providers. The P4.2b adapter preserves this boundary through a separate exact
activation policy rather than weakening the frozen P4.0 or P4.2a policies.

## 4.3 P4.2b controlled live HTTPS transport

P4.2b adds the first operating-system network implementation. It is an
explicitly constructed transport component only. It is not registered in the
P4.1 provider registry and is not reachable from Phase 3 conversation code.

Implemented in P4.2b:

- a separate versioned live-transport activation policy bound to `web.search`;
- HTTPS-only source access on the default port;
- bounded operating-system `getaddrinfo` execution with one in-flight daemon
  worker per resolver backend;
- rejection when any DNS answer is non-public, malformed, duplicated beyond
  policy, or outside the address-count budget;
- direct sockets that ignore environment and system proxy configuration;
- connection to one exact approved address with post-connect peer verification;
- default operating-system trust roots with mandatory certificate and hostname
  validation, plus rejection of environment CA overrides and TLS key logging;
- TLS 1.2 minimum and HTTP/1.1-only ALPN;
- fixed credential-free GET requests with no cookies or caller headers;
- one deadline across connect, request transmission, header parsing, and body
  reading;
- a strict HTTP/1.0 and HTTP/1.1 response parser with bounded status line,
  header count, header bytes, and response bytes;
- rejection of obsolete folded headers, ambiguous content lengths, transfer
  encoding, malformed framing, and early EOF;
- exact production resolver and socket-backend checks at live-retriever
  construction and again before every retrieval;
- sanitized DNS, timeout, connection, TLS, protocol, peer, and framing failures;
- metadata-only component digests that contain no hostname, URL, query, or
  source body.

The P4.0 information policy and P4.2a retrieval policy remain unchanged and
network-free. The P4.2b activation policy is a narrow additive authorization
for the exact live resolver and transport classes. It cannot enable a provider,
query transmission, fallback, retries, credentials, or runtime integration.

Still disabled after P4.2b:

- live search providers and live fetch-provider registration;
- Phase 3 conversation or CLI access to the live transport;
- plaintext HTTP, non-default ports, transfer encoding, connection reuse, and
  multiple address attempts;
- environment proxies, TLS environment overrides, TLS key logging, credentials,
  cookies, client certificates, custom CA bundles, JavaScript, forms, downloads,
  and automatic retries;
- recursive browsing, background execution, and memory writes.

Known compatibility boundary:

- responses using HTTP transfer coding are rejected rather than decoded;
- only the first deterministically ordered public address is attempted;
- if an operating-system DNS call ignores its deadline, the caller fails closed
  and that resolver backend refuses another DNS worker until the original call
  returns.

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
policies/information_live_http_policy.json
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

Add instruction-like-content analysis, containment labels, source isolation, credential-request detection, policy-override rejection, and adversarial fixtures. **Implemented in package version `0.5.0` as a deterministic, model-free, digest-bound firewall. Flagged sources are non-renderable and no raw finding excerpt is retained.**

Implemented controls:

- exact versioned firewall policy bound to `web.search` and the P4.0 untrusted-content boundary;
- NFKC and Unicode-format-character normalization for detection only;
- original source text and content digest preservation;
- deterministic detection of role markers, policy overrides, permission laundering, credential requests, tool execution, memory writes, policy mutation, private-data exfiltration, encoded instruction payloads, and source-boundary collisions;
- two-line detection windows for split instructions;
- metadata-only findings containing approved codes, line numbers, and normalized-line digests;
- exact source ID, URL, source digest, policy version, and detection-view binding;
- fail-closed source, line, and finding budgets;
- `InformationInspectedSource` as the only P4.3 source wrapper eligible for future model-facing web grounding;
- blocked-source rendering refusal;
- no model classifier, content rewriting, raw excerpt logging, tool invocation, action, or memory write.

Exit criteria:

- retrieved instructions cannot alter policy or permissions;
- no source can trigger a tool or action;
- critical prompt-injection cases have zero successes.

### P4.4 — Freshness and temporal reasoning

Add query-time classification, publication and update extraction, freshness policy, stale-source warnings, date conflict handling, and unsupported-current-claim rejection. **P4.4a is implemented in package version `0.6.0` as conservative query classification, explicit query-bound temporal intents, and deterministic digest-bound source assessments. P4.4b is implemented in package version `0.7.0` as deterministic HTML/header temporal-metadata evidence extraction, fail-closed resolution, verified date projection, and explicit-subject cross-source conflict aggregation.**

Implemented controls:

- exact versioned freshness policy bound to `web.search`, P4.0, and the clear-source P4.3 firewall boundary;
- conservative model-free query classification into explicit `current`, `latest`, `recent`, `historical`, and `time_insensitive` intents, with ambiguous signals rejected;
- exact query ID and query-content-digest binding for every temporal intent;
- separate publication, update, retrieval, reference, and assessment times;
- retrieval time explicitly prohibited from acting as freshness evidence;
- updated time preferred over publication time when both are valid;
- future, contradictory, and post-retrieval source timestamps rejected with bounded clock skew;
- deterministic age limits of 24 hours for `current`, 7 days for `latest`, and 30 days for `recent`;
- historical-window matching that does not require a current source;
- undated time-sensitive sources marked `unknown` and made non-renderable;
- stale and historical-mismatch sources preserved as assessments but blocked from model-facing claim support;
- exact source-content, source-metadata, query, intent, policy-version, and reference-time binding;
- re-derivation during validation to reject forged `fresh` assessments;
- `InformationTemporallyQualifiedSource` as the only P4.4 wrapper eligible for future model-facing temporal grounding;
- no model temporal inference, background activity, external action, memory write, or raw temporal-metadata logging;
- exact versioned P4.4b temporal-metadata evidence policy with fixed candidate and aggregation limits;
- deterministic extraction limited to recognized Open Graph article timestamps, schema-style `datePublished` and `dateModified` metadata, matching `<time datetime>` elements, and an HTTP `Last-Modified` update fallback;
- strict RFC 3339 timestamps for HTML metadata and strict IMF-fixdate parsing for HTTP `Last-Modified`;
- visible source prose and model-based date extraction explicitly prohibited;
- original temporal values retained only inside digest-bound evidence candidates, with log-safe records exposing hashes rather than raw values;
- explicit HTML update metadata preferred over the HTTP `Last-Modified` fallback;
- duplicate-body handling includes the temporal-candidate-set digest so conflicting head or header dates are never discarded as body duplicates;
- malformed candidates, within-source date disagreement, and materially pre-publication update times preserved as non-supporting `invalid` or `conflict` resolutions;
- no automatic winner selection for conflicting timestamps;
- raw retrieved resources refuse caller-supplied publication or update times, while `InformationResolvedTemporalResource` is the verified path for projecting dates into source documents;
- cross-source aggregation requires an explicit subject digest and at least two distinct canonical source URLs, and it never infers that sources describe the same temporal fact;
- matching observations produce deterministic consensus, undated observations remain insufficient, and conflicting dates remain unresolved;
- live provider registration, citation grounding, and Phase 3 conversational web access remain disabled.

Exit criteria:

- “latest” claims require fresh evidence;
- stale and conflicting dates remain visible;
- historical queries do not incorrectly require current sources.

### P4.5 — Citation-bound web grounding

Build source-quality metadata, claim construction, exact citation verification, source-diversity rules, uncertainty and conflict preservation, and the adapter into Phase 3 grounding.

#### P4.5a — Deterministic citation-grounding foundation

Package version `0.8.0` adds a model-free grounding boundary before any Phase 3 adapter exists.

The boundary:

- accepts only injection-cleared and freshness-supported source versions;
- derives structural source-quality metadata without publisher-reputation inference;
- requires HTTPS, minimum normalized content, exact source digests, and exact query binding;
- constructs claims only from exact character spans whose text equals the visible claim text;
- binds every citation to the canonical URL and normalized source-content digest;
- requires two distinct canonical domains before an extractive claim may be labeled `verified_fact`;
- preserves `uncertain` and `disputed` states instead of upgrading them silently;
- requires two distinct canonical domains for a conflict packet;
- rejects unused packet sources, citation swapping, support-span tampering, and forged quality assessments;
- renders a digest-bound `VERIFIED WEB GROUNDING` envelope around already governed source renderings;
- stores only span coordinates and digests in metadata-safe support records.

P4.5a does not perform semantic-entailment inference, publisher-reputation scoring, model claim generation, Phase 3 adaptation, external actions, memory writes, or background activity. These are milestone-local limits rather than permanent capability ceilings.

#### P4.5b — Deterministic Phase 3 grounding projection

Package version `0.9.0` adds a read-only adapter from revalidated P4.5a packets into the released Phase 3 grounding and response-validation contracts.

The bridge:

- revalidates the exact P4.5a packet, query digest, source versions, freshness assessments, support bindings, and grounding digest before projection;
- maps `insufficient_sources` to the existing Phase 3 `insufficient_evidence` outcome while preserving `answerable`, `uncertain`, and `conflict`;
- adds `web_source` as an additive Phase 3 citation source kind without giving the Phase 3 model direct network access;
- preserves exact `[WEB:...]` tokens, canonical URLs, source-content digests, knowledge status, confidence, and PUBLIC classification;
- creates deterministic Phase 3 claim and packet identities bound to the P4 grounding and query digests;
- emits a metadata-only projection receipt containing source-version, freshness, citation, P4 packet, P3 packet, and policy bindings;
- emits one metadata-only `grounding_packet` state reference, with no source body or raw support span persisted;
- revalidates the projection before delegating visible-response checks to the unchanged P3.6 validator;
- binds the P3.6 validation report and response digest back to the exact P4.5b projection.

P4.5b does not register the bridge in the local conversation runtime, execute research, generate claims, infer semantic entailment, write memory, perform actions, or persist source bodies. Those successor capabilities remain separately gated.

Exit criteria:

- visible external claims are source-supported;
- citation swapping and digest tampering fail;
- uncertainty and source conflict remain explicit;
- P4 grounding cannot be substituted during Phase 3 projection;
- the exact P3.6 validation report is bound to the projected P4 grounding;
- P3.6 remains the final visible-response gate.

### P4.6 — Governed research orchestration

Build bounded query planning, maximum search and fetch counts, deterministic stopping, cancellation, partial-result handling, sanitized activity persistence, and no recursive uncontrolled browsing.

#### P4.6a — Deterministic research-orchestration foundation

Package version `0.10.0` adds a foreground, fixture-only orchestration layer over the existing exact provider registry. It:

- requires one exact search provider and one exact fetch provider before execution;
- permits one search call and bounds fetch calls, selected sources, response bytes, per-call timeout, and total timeout to the selected Phase 4 foundation policy;
- deterministically orders search results and deduplicates canonical URLs before fetch;
- fetches only URLs returned by the selected search provider;
- checks cancellation and total-time budgets around every provider operation;
- preserves successful sources when later fetches fail or a run is cancelled or times out;
- emits terminal, metadata-safe activity records with approved sanitized error codes;
- records exact operation counters, selected result IDs, source IDs, source digests, outcome, and stopping reason in a digest-bound research-run receipt;
- rejects provider substitution, query substitution, duplicate result identity, unselected URLs, receipt tampering, source reordering, and hidden retries or fallback.

P4.6a does not register live providers, rewrite queries, follow arbitrary links, retry operations, recursively browse, invoke Phase 3, generate claims, persist source bodies, write memory, implement Phase 5 storage, perform actions, or run in the background. Those are milestone-local boundaries rather than permanent capability ceilings. P4.6b may connect verified orchestrated sources through the existing retrieval, injection, temporal, and grounding gates.

Exit criteria:

- every run terminates under policy budgets;
- no hidden provider fallback or arbitrary link following occurs;
- partial and failed research is reported truthfully.

#### P4.6b — Controlled research-evidence pipeline

Package version `0.11.0` composes verified P4.6a research runs through the existing injection, temporal-intent, freshness, and citation-grounding boundaries. It:

- revalidates the exact research run, query, receipt, selected-source sequence, and source-content digests before processing evidence;
- inspects every preserved source through the deterministic P4.3 injection firewall;
- applies the P4.4a temporal classifier and freshness evaluator only to injection-clear sources;
- preserves blocked and freshness-rejected sources as metadata-only dispositions without retaining excerpts in the pipeline receipt;
- accepts only explicit deterministic claim drafts and builds grounding only from qualified source versions;
- prevents partial P4.6a research from being promoted to an `answerable` outcome;
- binds the research receipt, temporal intent, complete source-disposition sequence, qualified source IDs and digests, grounding digest, and exact policy versions into a deterministic metadata-only evidence receipt;
- re-derives inspections and freshness assessments during validation and rejects source substitution, reordered outcomes, policy substitution, receipt tampering, and grounding substitution.

P4.6b remains fixture-only and foreground-only. It does not register live providers or the Phase 3 runtime, persist source bodies, write memory, implement Phase 5 storage, generate model claims, infer semantic entailment, perform actions, retry, recursively browse, or run in the background. Those capabilities remain separately governed successor work.

Exit criteria:

- every source used for grounding passes the exact injection and freshness gates;
- rejected and partial evidence remains explicit and metadata-safe;
- the final grounding packet is cryptographically bound to the exact research run and policy set.

### P4.7 — Local conversation integration

Add an explicit local research mode. Show when web research is used. Render sources and freshness. Add offline behavior. Do not enable silent web access.

#### P4.7a — Explicit local research-mode turn adapter

Package version `0.12.0` adds an explicit adapter over the released Phase 3 turn lifecycle. It:

- requires every turn to select either `local_only` or `research`;
- rejects web grounding, research evidence, and research availability on `local_only` turns;
- returns a deterministic `offline` or `unavailable` result before any conversation-state mutation;
- revalidates the exact P4.6b evidence result before projecting its P4.5a grounding through the P4.5b bridge;
- injects only the exact projected Phase 3 grounding packet into a copied turn command;
- adds an optional Phase 3 pre-commit response-validation hook so the stricter P4.5b citation boundary runs after P3.6 accepts or abstains but before the assistant message is committed;
- exposes metadata-only source summaries containing the exact citation token, canonical URL, source-content digest, and freshness verdict;
- binds mode, availability, research and evidence receipts, grounding, projection, response, validation, source summaries, and selected policy versions into a deterministic metadata-only receipt;
- revalidates completed and replayed results against the exact evidence, projection, response, and conversation-grounding identities.

P4.7a does not execute search or fetch, register live providers, persist source bodies, change the Phase 3 database schema, write memory, implement Phase 5 storage, perform external actions, retry, recursively browse, or run in the background. End-to-end local research execution and live-provider selection remain separately governed successor work.

Exit criteria:

- the user can distinguish local-only and web-grounded replies;
- offline mode fails cleanly before state mutation;
- unrelated turns do not silently trigger web access;
- no research response is committed before both P3.6 and the exact P4.5b validation boundary accept it.

#### P4.7b — Governed fixture research execution

Package version `0.13.0` adds an explicit execution boundary that composes the released P4.6a orchestrator, P4.6b evidence pipeline, and P4.7a research-mode adapter. It:

- requires an explicit `local_only` or `research` execution plan before any provider is touched;
- guarantees local-only, offline, and unavailable turns do not execute search or fetch providers;
- selects the exact approved fixture search and fetch provider identities before execution and rejects provider, request, run, evidence, or mode-result substitution;
- revalidates completed, partial, failed, cancelled, and insufficient-source research-run receipts before deciding whether evidence processing may continue;
- maps failed, cancelled, source-empty, and no-qualified-evidence paths into P4.7a unavailable results before conversation-state mutation;
- preserves partial research without allowing it to become an `answerable` grounding outcome;
- reprocesses the exact evidence plan during validation and binds the research request, query digest, selected providers, run receipt, evidence receipt, mode receipt, result status, unavailable reason, and selected policy versions into one deterministic metadata-only execution receipt;
- keeps P3.6 and the exact P4.5b citation boundary authoritative before any web-grounded response is committed.

P4.7b remains deterministic-fixture-only and foreground-only. It does not register live providers, use provider fallback, persist source bodies, write memory, implement Phase 5 storage, perform external actions, retry, recursively browse, or run in the background. Live provider registration remains separately governed successor work.

Exit criteria:

- local-only and preflight-unavailable turns never execute providers;
- every executed research turn binds the exact request, providers, run, evidence, conversation result, and policies;
- failed, cancelled, partial, and insufficient-evidence outcomes remain explicit and cannot silently become complete answers;
- no conversation-state mutation occurs on any unavailable execution path.

### P4.8 — Final adversarial information evaluation

Package version `0.14.0` adds the synthetic, content-free Phase 4 closure evaluation. It:
- defines 24 deterministic cases with two cases for each required suite: injection, SSRF, redirects, oversized content, stale dates, source conflicts, citation tampering, privacy leakage, cancellation, timeout, provider failure, and deterministic replay;
- requires 100% case, network-security, source-quality/freshness, citation-integrity, privacy-boundary, execution-resilience, and deterministic-replay rates;
- applies zero-tolerance gates to critical security failures, private-content leakage, successful prompt injection, network-boundary bypass, citation-integrity bypass, freshness/conflict bypass, unbounded execution, and unexpected side effects;
- keeps the metadata-only submission parser for contract tests, but prohibits externally supplied submission bundles from serving as release evidence;
- pins a 28-file pre-P4.8 Phase 4 runtime manifest, a 640-test collection floor, and an exact case-to-test-file evidence map;
- executes the pinned suite in an isolated pytest subprocess with outbound socket connections blocked, bytecode and pytest-cache writes disabled, and no recursive P4.8 self-validation;
- derives all 24 case observations from the runtime result rather than from a prebuilt passing fixture;
- binds exact test-file digests, the complete repository snapshot, collection and execution summaries, network-guard activation, and per-case test evidence into the final runtime report;
- rejects a missing network guard, incomplete collection, skipped or failed probes, runtime-manifest substitution, test-file substitution, repository mutation, runtime-evidence tampering, and outer-report tampering;
- rejects duplicate JSON keys, unknown fields, weakened thresholds, missing suites, benchmark substitution, report tampering, and inconsistent metric or release decisions;
- writes evaluation reports only outside the repository and refuses overwrite;
- remains synthetic-only, private-output-only, offline, read-only, and free of raw query text, source bodies, real private queries, persistence, memory writes, actions, repository writes, and background execution.

P4.8 produces the runtime-backed deterministic evidence consumed directly by the P4.9 exact-commit release audit.

Exit criteria:

- zero critical security failures;
- source quality and freshness gates pass;
- no real private query or browsing content enters Git.

### P4.9 — Release audit and closure

Package version `0.15.0` adds the exact-commit private Phase 4 release audit. It:

- reruns the canonical P4.8 runtime-backed evaluation against the exact repository selected for release;
- rejects a supplied commit that differs from `HEAD`, any dirty working tree, a rollback equal to the release commit, or a rollback that is not an ancestor;
- binds the release policy, evaluation policy, benchmark, runtime manifest, package version, final evaluation report, runtime evidence, repository snapshot, collection and execution summaries, test counts, network guard, timestamp, and rollback commit;
- requires at least 24 benchmark cases, 28 pinned runtime test files, 640 collected tests, zero skipped tests, complete test and case passage, zero critical failures, and every metric gate to pass;
- writes a canonical SHA-256 release record only beneath the private vault and refuses repository-local output or overwrite with different content;
- rejects duplicate JSON keys, weakened or substituted release policy, malformed digests, inconsistent approval decisions, tampered counts, boundary changes, and modified records;
- provides metadata-only inspection without case payloads, raw query text, source content, target-file lists, credentials, or other private browsing material;
- closes the README and roadmap while preserving the released Phase 4 behavior as an evolvable compatibility profile rather than a permanent capability ceiling.

The canonical private record path is `C:\ALICE_Vault\reports\phase4-information-release.json`.

Exit criteria:

- private audit returns `approved=true`;
- exact commit, policy versions, package version, evaluation digest, runtime-evidence digest, runtime-backed report digest, repository snapshot, and rollback commit are recorded;
- the record verifies after writing and remains outside the repository;
- Phase 4 release behavior is versioned after merge; its implementation remains migratable under the final architecture.

### P4.10 — Operational live-public-information acceptance and closure

P4.10 is an additive post-release milestone discovered by the Phase 4 post-phase audit. It does not invalidate P4.0–P4.9.

P4.10 must:

- implement at least one exact live PUBLIC search provider;
- register the exact live fetch path through the controlled HTTPS boundary;
- keep provider credentials and configuration outside Git;
- require explicit research mode before network use;
- bind search, fetch, injection inspection, temporal/freshness analysis, grounding, Phase 3 projection, P3.6 validation, and the P4.5b citation gate;
- expose metadata-safe network egress;
- test provider availability, quota, rate limits, timeout, cancellation, outage, and no-silent-fallback behavior;
- keep source-body persistence, Phase 5 storage, private-query transmission, authenticated browsing, external actions, recursion, and background operation outside the initial acceptance profile;
- run private real-provider and real-model acceptance;
- write an exact-commit private acceptance record with rollback evidence.

Exit criteria:

- a live PUBLIC query reaches a real search provider and controlled live fetch path;
- returned evidence survives all Phase 4 trust, freshness, grounding, and response-validation boundaries;
- indirect prompt injection and citation substitution fail in live acceptance;
- provider outage and quota exhaustion fail cleanly;
- private acceptance returns `approved=true`;
- the post-phase repository audit has zero critical findings;
- README, Roadmap, Capability Catalog, architecture, report, and handoff agree that Phase 4 is operationally complete.

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

Phases 0–3 remain released baselines, not immutable architecture. Earlier-phase changes may use dedicated migration branches and must carry regression evidence, but no phase is exempt from redesign when required by the ratified direction.


---

# Roadmap 2.0 Compatibility Amendment

Phase 4 may emit a sanitized, learning-ready activity envelope for future ingestion by Phase 5. This amendment does not create a Phase 4 memory writer, training path, background process, or new authority.

The envelope may contain:

- event and task identifiers;
- query purpose and public classification decision;
- provider, source, canonical URL, retrieval time, and content digest;
- publication/update metadata when available;
- injection, trust, freshness, and contradiction indicators;
- claims and citations actually used;
- budgets, cancellation state, and execution outcome;
- `memory_eligibility: unassessed`.

Raw private context, unrestricted source bodies, secrets, and model hidden reasoning are excluded from ordinary activity records. Phase 5 alone will define retention and candidate-learning ingestion.

Current P4.5 work may continue against the existing release contracts. Completed P4.0–P4.4 components may also be rewritten when necessary for product/host scoping, the Experience Ledger, capability profiles, Friday extraction, or the final learning architecture. Preserve useful behavior through migrations and compatibility profiles rather than treating completion as immunity.

<!-- P4.10 LIVE PUBLIC STATUS START -->
## Additive Phase 4 live-public-information status

- Completed sub-milestone: **P4.10b**
- Package profile: `alice_information 0.17.0`
- P4.10a–P4.10b are complete. P4.10c private live acceptance and exact-commit closure remains active. Phase 5 remains blocked.
- P4.0–P4.9 and the P4.6a/P4.7a/P4.7b fixture profiles remain reproducible and unchanged.
- No source persistence, Phase 5 storage, memory write, external action, recursive browse, retry, fallback, or background execution is activated.
<!-- P4.10 LIVE PUBLIC STATUS END -->
