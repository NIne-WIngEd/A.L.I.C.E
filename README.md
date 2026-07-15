# A.L.I.C.E.

A.L.I.C.E. is a long-term personal AI assistant project created for MK Rayan.

Its intended role is to become a persistent, permissioned cognitive partner that can understand personal context, support research and daily work, preserve important memories, offer truthful and constructive judgment, and eventually coordinate approved tools and workflows.

## Current status

- Phase 0 — Identity and Governance: complete
- Phase 1 — Private Data Vault and Ingestion: in progress
  - encrypted local vault: complete;
  - metadata and SHA-256 inventory: complete;
  - file-signature and inventory analysis: complete;
  - pilot selection, automated review, and immutable `pilot-v1`: complete;
  - safe parser registry and verified extraction: complete;
  - deterministic chunking and provenance catalog: complete;
  - local lexical retrieval and evaluation: complete;
  - local semantic and hybrid retrieval: complete;
  - retrieval-grounded read-only context access: complete;
  - grounded reasoning and response generation: pending.

A working conversational assistant is not yet implemented.
## Governing documents

- [`docs/ALICE_CONSTITUTION.md`](docs/ALICE_CONSTITUTION.md)
- [`docs/PERMISSION_MODEL.md`](docs/PERMISSION_MODEL.md)
- [`docs/MEMORY_POLICY.md`](docs/MEMORY_POLICY.md)
- [`docs/DATA_CLASSIFICATION.md`](docs/DATA_CLASSIFICATION.md)
- [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md)
- [`docs/EVALUATION_CHARTER.md`](docs/EVALUATION_CHARTER.md)
- [`docs/ROADMAP.md`](docs/ROADMAP.md)

## Phase 1 documents

- [`docs/PHASE_1_VAULT_FOUNDATION.md`](docs/PHASE_1_VAULT_FOUNDATION.md)
- [`docs/PHASE_1_INVENTORY_ANALYSIS.md`](docs/PHASE_1_INVENTORY_ANALYSIS.md)
- [`docs/PHASE_1_PILOT_SELECTION.md`](docs/PHASE_1_PILOT_SELECTION.md)
- [`docs/PHASE_1_AUTOMATED_REVIEW.md`](docs/PHASE_1_AUTOMATED_REVIEW.md)
- [`docs/PHASE_1_FINAL_PILOT_POLICY.md`](docs/PHASE_1_FINAL_PILOT_POLICY.md)
- [`docs/PHASE_1_SAFE_PARSER_REGISTRY.md`](docs/PHASE_1_SAFE_PARSER_REGISTRY.md)

## Security boundary

Never commit personal datasets, private memory, vault databases, exported manifests, credentials, tokens, activity logs, extracted text, or other personal content.

The private vault and source archive must remain outside this repository.

## Development checks

```powershell
py -m pip install `
  -r requirements-dev.txt `
  -r requirements-phase1.txt `
  -r requirements-extraction.txt

py scripts\validate_phase0.py
py -m unittest discover -s tests -p "test_*.py" -v
```

## Legacy prototype

The original 2022 voice-assistant prototype is retained under `legacy/` only as historical reference. It is not part of the current architecture and must not be connected to real credentials.
