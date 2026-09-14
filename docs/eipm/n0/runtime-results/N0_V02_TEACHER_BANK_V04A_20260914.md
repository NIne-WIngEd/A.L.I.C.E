# N0 v0.2 teacher bank wave 0.4a

Date: 2026-09-14

## Purpose

After the governed v0.2.1 tokenizer passed, work returned to the actual N0 objective: teaching generic semantic, pragmatic, epistemic, social, causal, temporal, ranking, and structured-alignment principles before any private Elaina identity gradient.

The existing 128 public Sol rows remain immutable seeds. Wave 0.4a adds 215 independently authored public teacher rows: exactly four train and one dev scenario for each of the 43 registered N0 competencies.

## New shards

- `sol_curriculum_principles_semantic_v0.4a.jsonl`: 55 rows across SEM-01..07, CAUS-01..03, TEMP-01.
- `sol_curriculum_principles_pragmatic_social_v0.4a.jsonl`: 70 rows across PRAG-01..07, SOC-01..05, EMO-01..02.
- `sol_curriculum_principles_epistemic_alignment_v0.4a.jsonl`: 90 rows across EPI-01..07, RANK-01..05, ALIGN-01..06.

Each v0.4a row includes a reusable `principle_tag`. Rationales state the governing reason and are intended to supply gradient through principle/rationale alignment and multi-positive semantic contrastive learning. Preferred candidate positions vary. Frozen readiness benchmark wording is not intentionally reused.

## Registered scale

- legacy governed rows: 128
- v0.4a rows: 215
- registered total: 343
- full multitask minimum: 1,000
- remaining to minimum before audit: 657
- initial readiness target: 2,500

The 1,000-row threshold remains a quality gate, not a quota license. No GPU multitask training is authorized merely because rows exist.

## Quality audit

`audit_n0_v02_teacher_bank.py` verifies all registered curriculum manifests and hashes, v0.2 rationale validity, cross-shard ID/prompt uniqueness, no exact copy of frozen evaluation prompts, all 43 competencies with train/dev support, exact v0.4a 4-train/1-dev distribution, reusable principle tags, basic candidate-position balance, and the private/public boundary.

The audit must PASS before the next authoring wave is treated as ratified.

## Boundary

- public semantic teacher data only
- private identity data: false
- private identity gradient: false
- model training performed by this authoring step: false
- tokenizer remains frozen unless a downstream failure demonstrates a real tokenizer bottleneck
