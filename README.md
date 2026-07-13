# A.L.I.C.E.

**A.L.I.C.E.** is a long-term personal AI assistant project created for MK Rayan.

Its intended role is to become a persistent, permissioned cognitive partner that can understand personal context, support research and daily work, preserve important memories, offer truthful and constructive judgment, and eventually coordinate approved tools and workflows.

## Current status

**Phase 0 — Identity and Governance: complete**

The repository currently contains the governing constitution, permission model, memory policy, data-classification rules, threat model, evaluation charter, roadmap, machine-readable policy files, and policy-validation tests.

A working assistant is **not yet implemented**. The next development stage is Phase 1: the private personal-data vault and ingestion pipeline.

## Governing principles

A.L.I.C.E. is designed to be:

- logical, truthful, faithful, clever, composed, creative, and constructively critical;
- loyal without being blindly obedient;
- supportive without using false reassurance;
- bold in reasoning without being reckless in real-world actions;
- inspectable, correctable, reversible, and stoppable;
- private by design and governed by explicit permissions.

The highest project-level authority is:

- [`docs/ALICE_CONSTITUTION.md`](docs/ALICE_CONSTITUTION.md)

## Phase 0 documents

- [`docs/PERMISSION_MODEL.md`](docs/PERMISSION_MODEL.md)
- [`docs/MEMORY_POLICY.md`](docs/MEMORY_POLICY.md)
- [`docs/DATA_CLASSIFICATION.md`](docs/DATA_CLASSIFICATION.md)
- [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md)
- [`docs/SCOPE_AND_NON_GOALS.md`](docs/SCOPE_AND_NON_GOALS.md)
- [`docs/EVALUATION_CHARTER.md`](docs/EVALUATION_CHARTER.md)
- [`docs/ROADMAP.md`](docs/ROADMAP.md)
- [`docs/decisions/ADR-001-system-principles.md`](docs/decisions/ADR-001-system-principles.md)

Machine-readable policy:

- [`policies/permissions.yaml`](policies/permissions.yaml)
- [`policies/data_classes.yaml`](policies/data_classes.yaml)

## Security rule

Never commit personal datasets, credentials, passwords, API keys, tokens, private memories, raw conversation archives, embeddings, local databases, or generated activity logs.

Private life data will be stored outside this public repository in an encrypted, access-controlled vault.

## Validate Phase 0

```powershell
py -m pip install -r requirements-dev.txt
py scripts\validate_phase0.py
```

## Legacy prototype

The original 2022 voice-assistant prototype is retained only as historical reference under `legacy/`. It is not part of the current architecture and must not be treated as production code.

## Development order

1. Phase 0 — Identity and governance
2. Phase 1 — Private data vault and ingestion
3. Phase 2 — Memory Core
4. Phase 3 — Conversational A.L.I.C.E.
5. Phase 4 — Web and information tools
6. Phase 5 — User interface
7. Phase 6 — Personal-service integrations
8. Phase 7 — Proactive assistance
9. Phase 8 — Controlled coding capability
10. Phase 9 — Fine-tuning and adaptation
11. Phase 10 — A.L.I.C.E. operating environment

## Owner

MK Rayan
