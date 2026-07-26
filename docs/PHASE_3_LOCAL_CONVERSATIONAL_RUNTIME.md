# Phase 3.7 — Local Conversational Runtime and CLI

**Status:** Implementation milestone
**Phase 1 dependency:** Frozen read-only evidence layer
**Phase 2 dependency:** Authoritative Memory Core, read-only from ordinary conversation
**Phase 3 dependencies:** P3.0–P3.6

## Purpose

P3.7 provides the first user-facing conversation loop over the governed Phase 3 stack. It runs locally from a terminal and stores private conversation state only under the configured A.L.I.C.E. vault.

The supported entry point is:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
py -m alice_conversation.cli `
    --vault-root C:\ALICE_Vault `
    --provider ollama-local `
    --model qwen3:8b
```

The package also supports:

```powershell
py -m alice_conversation `
    --vault-root C:\ALICE_Vault `
    --provider ollama-local `
    --model qwen3:8b
```

## Controlled runtime path

```text
terminal input
    ↓
private conversation session
    ↓
ratified constitutional system contract
    ↓
optional prebuilt read-only grounding
    ↓
explicit local model adapter
    ↓
controlled turn orchestration
    ↓
deterministic response validation
    ↓
terminal output or sanitized failure
```

P3.7 does not bypass or reimplement the earlier milestones. It composes the P3.2 state service, P3.4 constitutional contract, P3.5 orchestrator, and P3.6 response validator.

P3.7 preserves multiple turns in one private session, but it does not yet assemble prior turns into a cross-turn model prompt. Each generation uses the current P3.5 request contract. Governed cross-turn context assembly remains a separate milestone.

## Local-only provider boundary

The user-facing runtime exposes only the `ollama-local` provider. The model must be named explicitly and must already be permitted by `conversation_model_policy.json`.

The deterministic test provider remains available to tests. It is not exposed through the CLI.

P3.7 does not provide provider fallback. A missing, unavailable, or disallowed model fails closed.

## Private state boundary

The runtime requires an explicit private-vault root. The P3.2 state store rejects repository-local database paths. The default database remains:

```text
<VAULT_ROOT>\conversation\alice-conversation.sqlite3
```

Normal terminal output never displays session IDs, turn IDs, request IDs, generation IDs, message IDs, database paths, or raw database rows.

The runtime supports:

- `session_only`: closing the session purges its conversational content and retains only the sanitized tombstone required by P3.2;
- `retained`: closing the session retains its inspectable private state until explicit deletion.

The default is `session_only`.

## Commands

```text
:help
:new [session_only|retained]
:close
:inspect
:cancel
:resume
:grounding
:grounding off
:grounding reload
:exit
```

`:inspect` is metadata-safe. It displays counts, status values, retention, selected provider/model, validation outcome, failure code, and grounding counts. It does not display message contents, grounding text, hidden reasoning, or raw identifiers.

`:cancel` cancels the active or most recent nonterminal turn. `:resume` resumes exactly one interrupted turn through the existing explicit P3.5 resume contract.

## Optional grounding file

P3.7 can load one prebuilt `ConversationGroundingPacket` JSON file. The file is parsed with an exact schema and validated against the existing P3.0 grounding contracts.

The grounding file:

- must be outside the repository;
- cannot exceed the policy size limit;
- is read-only;
- cannot trigger retrieval;
- cannot modify memory;
- cannot grant permissions;
- remains untrusted model context;
- is revalidated before use.

P3.7 does not perform live Phase 1 or Phase 2 retrieval. A future milestone must add that integration through a separately governed read-only retrieval boundary.

## Citations and validation

Accepted grounded responses render only the exact citation tokens that are present in both the response and the approved grounding packet.

A rejected model response is not printed as an answer and is not stored as an assistant message. The terminal displays only the sanitized validation failure code and issue codes.

Abstentions are labeled explicitly.

## Interruption behavior

A model interruption leaves the turn in the P3.2 `interrupted` state. The user may explicitly run `:resume` or `:cancel`.

A keyboard interruption requests cancellation and records a sanitized cancellation code when a nonterminal turn exists. Hidden reasoning is never captured or displayed.

## Prohibited capabilities

P3.7 does not enable:

- web access;
- tool calling;
- external actions;
- autonomous retrieval;
- memory writes or promotion;
- highly sensitive ordinary grounding;
- response repair;
- model retry;
- provider fallback;
- hidden chain-of-thought persistence or display;
- browser or graphical interfaces.

## Testing

Deterministic tests use fake model adapters and private temporary vaults. They cover:

- policy fail-closed behavior;
- private-vault enforcement;
- exact grounding-file parsing;
- interactive commands;
- retention behavior;
- citation rendering;
- metadata-safe inspection;
- validation rejection;
- interruption, cancellation, and resume;
- absence of hidden reasoning and raw identifiers;
- user-facing provider restrictions.

A separate opt-in Ollama smoke test is skipped unless `ALICE_RUN_CLI_OLLAMA_INTEGRATION=1` is set.

## Scope boundary

P3.7 is the first local terminal surface. It is not the final Phase 3 release evaluation. Adversarial end-to-end evaluation, release audit, and any later retrieval or interface milestones remain separate work.
