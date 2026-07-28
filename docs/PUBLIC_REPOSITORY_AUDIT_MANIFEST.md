# Public Repository Capability-Barrier Audit Manifest

**Baseline:** public `main` inspected July 27, 2026
**Purpose:** record where the earlier architecture encoded release-local limits as active project rules.

## Root and governance

- `README.md`
- `SECURITY.md`
- `docs/ALICE_CONSTITUTION.md`
- `docs/PERMISSION_MODEL.md`
- `docs/MEMORY_POLICY.md`
- `docs/DATA_CLASSIFICATION.md`
- `docs/THREAT_MODEL.md`
- `docs/SCOPE_AND_NON_GOALS.md`
- `docs/EVALUATION_CHARTER.md`
- `docs/ROADMAP.md`
- `docs/decisions/ADR-001-system-principles.md`
- `docs/decisions/ADR-002-private-data-architecture.md` when present

## Machine policy and validation

- `policies/permissions.yaml`
- `scripts/validate_phase0.py`
- `.github/workflows/phase0-policy-checks.yml`
- `.github/workflows/phase1-foundation-tests.yml`
- constitutional, permission, and security case suites under `tests/`

## Conversation capability ceilings

Policies:

- `policies/conversation_policy.json`
- `policies/conversation_orchestration_policy.json`
- `policies/conversation_constitutional_policy.json`
- `policies/conversation_cli_policy.json`
- `policies/conversation_context_policy.json`
- `policies/conversation_grounding_policy.json`
- `policies/conversation_model_policy.json`
- `policies/conversation_response_repair_policy.json`
- `policies/conversation_response_validation_policy.json`
- `policies/conversation_state_policy.json`
- related release-audit and evaluation policies

Implementation hotspots:

- `src/alice_conversation/contracts.py`
- `src/alice_conversation/policy.py`
- `src/alice_conversation/orchestration_policy.py`
- `src/alice_conversation/constitutional_policy.py`
- `src/alice_conversation/constitutional_prompt.py`
- `src/alice_conversation/cli_policy.py`
- context, grounding, repair, response-validation, and state policy loaders

The released modules hard-coded or strictly validated combinations including no web, no tools, no external action, no memory write/promotion, no highly sensitive grounding, no live retrieval, no retry, no provider fallback, fixed local provider, session-only retention, exact phase/version bindings, and exact source-document sets.

## Information capability ceilings

Policies:

- `policies/information_policy.json`
- `policies/information_provider_policy.json`
- `policies/information_http_retrieval_policy.json`
- `policies/information_live_http_policy.json`
- `policies/information_injection_firewall_policy.json`
- `policies/information_freshness_policy.json`
- `policies/information_temporal_metadata_policy.json`

Implementation hotspots:

- `src/alice_information/contracts.py`
- `src/alice_information/policy.py`
- `src/alice_information/provider_policy.py`
- `src/alice_information/retrieval_policy.py`
- `src/alice_information/live_policy.py`
- freshness, firewall, transport, registry, provider, and temporal-policy modules

The original foundation encoded search/fetch-only operation, PUBLIC-only transmission, no private networks, no authentication, no JavaScript, no forms, no background operation, no memory output, no arbitrary code, no fallback, closed providers, and fixed budgets.

## Tests

The repository includes extensive Phase 3 and Phase 4 tests. Any test asserting that a capability is false, unavailable, empty, exact-version-bound, or impossible is migrated as a named compatibility-profile test. It remains useful regression evidence but is not a system-wide veto.

## Local-only coverage

This manifest cannot enumerate unpushed Phase 4.5 files. The application script runs `git ls-files -co --exclude-standard` against the actual local worktree, scans every relevant text file, registers reviewed legacy restrictions, annotates text-based compatibility files, and then fails on any remaining active unscoped barrier.

## Product-family and generalization audit additions

The Phase 5+ extraction must also inspect every Phase 1–4 file for:

- hard-coded `Rayan`, A.L.I.C.E.-only, single-user, or fixed local-path assumptions in reusable modules;
- persistent records without `product_id` and `host_instance_id` scope;
- shared caches, embeddings, logs, or training directories that could mix hosts;
- encryption keys or storage roots not bound to one host instance;
- model adapters without training lineage or host ownership;
- mandatory vendor endpoints for local functionality;
- tests that require earlier phases to remain architecturally immutable;
- schemas that cannot be exported through the Identity Capsule;
- interfaces whose names or storage formats depend on the public product brand.

These findings may require direct changes to Phase 1, 2, 3, or 4 files. Completion status does not exempt them.
