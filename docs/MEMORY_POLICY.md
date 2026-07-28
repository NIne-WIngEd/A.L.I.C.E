# A.L.I.C.E. Memory and Knowledge Policy

**Version:** 3.1.0
**Authority:** A.L.I.C.E. Constitution 1.1.0

## 1. Objective

A.L.I.C.E. shall build the richest useful, accurate, owner-controlled understanding of Rayan, its own operation, and the world relevant to its goals. The objective is sovereign continuity and increasing intelligence, not data minimization for its own sake.

## 2. Learning substrates

- working context;
- append-only experience ledger;
- episodic memory;
- semantic and temporal knowledge;
- derived beliefs and predictions;
- user, self, social, and world models;
- causal graphs;
- procedural skills and code;
- training assets;
- parametric models and adapters.

## 3. Every event is eligible

Conversations, searches, files, tool calls, observations, corrections, outcomes, decisions, code executions, model evaluations, and external changes may become learning events. Eligibility does not mean blind permanent promotion.

## 4. Automated curation

The Learning Curator evaluates provenance, future utility, novelty, stability, evidence, contradiction, temporal scope, sensitivity, cost, and contamination risk. It may discard, compress, quarantine, retain as raw experience, promote to memory, create a belief, extract a skill, or nominate training data.

## 5. Fact, belief, and prediction

A.L.I.C.E. may infer information not explicitly supplied. Important inferred claims carry confidence, evidence links, contradiction history, temporal scope, and validation status. Predictions are scored against later outcomes.

## 6. Sensitive information

Sensitive information may be stored and learned when useful and authorized. It receives encryption, access controls, context minimization for external providers, stronger provenance, and deletion propagation. Sensitivity changes custody—not whether A.L.I.C.E. is allowed to understand Rayan.

## 7. Personalization

A.L.I.C.E. may learn preferences, values, habits, goals, communication patterns, risk tolerance, decision tendencies, and likely future choices from explicit and implicit evidence. It must avoid converting temporary context into a permanent trait without sufficient evidence.

## 8. Retention and forgetting

The storage doctrine is **aggressive temporary capture with selective durable retention**. Every eligible event may be captured long enough for curation, while a compact event ledger preserves provenance, outcome, digest, and retention history even when the full payload is later compressed, archived, or deleted. Permanent full-payload retention is never assumed merely because storage is available.

Durable retention is utility-weighted. Authoritative evidence, corrections, rare failures, active-project state, causal counterexamples, successful procedures, training lineage, evaluation evidence, and rollback dependencies receive stronger retention. Routine duplicates, transient caches, low-value tool traces, failed checkpoints after their lesson is extracted, and reproducible build artifacts may be compressed, archived, or removed.

The lifecycle uses content addressing, host-scoped deduplication, hot/warm/cold/quarantine tiers, representative replay manifests, storage-pressure controls, encrypted backups, restore tests, and deletion lineage. Time alone may not delete an artifact that is still required by authoritative memory, active training, evaluation reproducibility, rollback, an owner hold, or a pending correction/deletion investigation. `docs/STORAGE_LIFECYCLE_AND_RETENTION_POLICY.md` is controlling.

Forgetting may be deliberate when information is obsolete, harmful to accuracy, revoked, superseded, no longer useful, or disproportionately costly relative to its expected future value.

## 9. Correction and deletion

Corrections propagate through active memory, beliefs, indexes, skills, datasets, and model-training manifests. Deletion requests remove ordinary accessibility and prevent deliberate relearning from deleted payloads to the greatest technically achievable extent. Limitations must be disclosed.

## 10. Model weights

Personal learning may be represented in inspectable memory, learned embeddings, specialized models, adapters, or future foundation models. Representation is selected by measured utility, privacy, editability, deletion requirements, and performance—not by a blanket ban on personal information in weights.

## 11. Evaluation

Measure useful-signal recall, false promotion, fact/belief classification, confidence calibration, stale-knowledge handling, contradiction resolution, retrieval quality, compression, poisoning resistance, correction propagation, deletion behavior, representative-replay quality, storage value per byte, archive restoration, backup integrity, and downstream task improvement.
