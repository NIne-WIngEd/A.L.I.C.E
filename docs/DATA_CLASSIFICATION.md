# A.L.I.C.E. Data Classification and Learning Custody

**Version:** 2.0.0

Classification controls custody and exposure. It does not determine whether A.L.I.C.E. is allowed to understand or learn from information.

## Classes

- **PUBLIC** — may be processed locally or by approved external systems.
- **INTERNAL** — project-operational information; external processing follows mission policy.
- **PRIVATE** — personal or non-public information; may be used for memory, inference, simulation, and training under owner-controlled lineage and provider rules.
- **HIGHLY_SENSITIVE** — health, finances, identity, intimate history, or similarly consequential material; may be used when useful under stronger encryption, access, external-exposure, and deletion requirements.
- **SECRETS** — credentials and cryptographic authority. Use through references or credential brokers. Do not duplicate into ordinary memory or training data unless a narrowly designed cryptographic research task explicitly requires it.

## Representation choices

Authorized information may be represented as:

- raw experience;
- episodic or semantic memory;
- graphs or embeddings;
- derived beliefs;
- procedural skills;
- training examples;
- specialized model weights or adapters.

The representation is selected by measured utility, editability, privacy, deletion, compute, and reliability—not by a blanket ban on parametric learning.

## External processing

External transmission is controlled by mission scope, provider capabilities, and data custody rules. Local-only processing is a selectable strategy, not a universal architectural requirement.
