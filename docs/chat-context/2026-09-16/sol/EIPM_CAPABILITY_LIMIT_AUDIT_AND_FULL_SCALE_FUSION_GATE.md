# EIPM Capability-Limit Audit + Full-Scale Fusion Gate — 2026-09-16

## Owner directive

Validation and limitation are different. A.L.I.C.E. may use strict provenance, lineage, held-out, and capability gates. Temporary run controls such as step counts, batches, checkpoint cadence, GPU allocation, or frozen-parent phases are allowed. None of those may become permanent capability limits through inertia.

The personality-model north star remains the comic/docs/clone-aware objective: the highest-fidelity achievable Elaina reconstruction/judgment core, with explicit provenance and uncertainty, not a generic persona overlay. Capability/personality fidelity comes before efficiency. Parameter counts, context windows, dataset sizes, ontology sizes, graph sizes, view counts, and current hardware envelopes are not permanent ceilings.

## Audit scope

Backtraced from Constitution / clone-aware doctrine / comic-level role through E0/E-INF/A-SYN generation, saturation, curation, curated frontier, N0 semantic, structured-state, evidence adapter, graph repair/ratification, and pre-gradient fusion.

Authoritative audit document on build branch:
`docs/eipm/EIPM_CAPABILITY_LIMIT_AUDIT_2026-09-16.md`

## Findings

### Sound lineage / no restart required
- E0 immutable historical authority: sound.
- E-INF revisable evidence-derived inference: sound.
- A-SYN synthetic completion, never historical truth: sound.
- UNKNOWN epistemic, not forced runtime blank: sound.
- saturation-driven expansion and post-curation targeted regeneration: sound.
- curated alternative branches/uncertainty/provenance retained: sound after wording clarification.
- semantic step80, structured step80, evidence adapter step80, repaired graph step80 remain valid; no private gradient occurred under the accidental ceilings found.

### Accidental limitations found and corrected before fusion gradient
1. Structured `max_fields=64` runtime reject removed; value is legacy checkpoint operating-shape hint only.
2. Evidence adapter `max_fields=64` runtime reject removed.
3. Evidence graph `max_fields=64` / `max_edges=256` runtime rejects removed.
4. Relation/type vocabularies are finite checkpoint tensors but explicitly migratable, never closed ontologies.
5. Semantic model now exposes token-level `encode_tokens()` and `encode_views()`; mean pooling is compatibility readout, not bottleneck.
6. Fusion is now checkpoint-view-extensible through `forward_views()`; N0's current three public views are not a permanent ceiling. Future explicit identity-concept and governed residual-concept views may migrate in.
7. Fusion parent token residual scale initializes at zero, preserving ratified input token views exactly before fusion learns.
8. Fusion routing objective rejects target mass on unavailable views instead of silently renormalizing invalid supervision.
9. Older “plausible rejected branches as hard negatives” wording superseded: not selected != false; plausible/co-valid branches remain alternatives unless actually contradicted/incompatible.
10. “smallest relevant evidence packet” clarified to “smallest sufficient”; context must expand whenever necessary for fidelity.
11. Historical 500 E-INF / 1000 A-SYN Wave 1 target explicitly recorded as batching control only.
12. corpus freezes are lineage snapshots, not permanent coverage closure.
13. explicit concept bank is open ontology; residual bank is governance-bounded rather than numerically bounded; latent slot count has no permanent fixed ceiling; IDP cannot collapse to one scalar.
14. serving fast-path defaults are not capability limits.

## No-accidental-ceilings regression gate

Added `tests/eipm/test_n0_no_accidental_capability_ceilings.py`, including:
- 65 structured fields despite 64 legacy hint;
- 65 evidence-adapter fields;
- 65 graph nodes + 257 graph edges despite legacy hints;
- 5-view fusion checkpoint via generic `forward_views()`;
- exact parent token views at fusion initialization;
- token-level semantic API availability.

## Full-scale fusion architecture

Active architecture:
`multi_stream_self_refinement_plus_gated_bidirectional_cross_attention`

Current N0 instantiation: semantic raw-token view + typed structured field view + repaired relation/evidence field view.
Four interaction stages, per-view self-refinement, gated bidirectional cross-attention to all other instantiated views, query/reliability conditioning, preserved per-view outputs, adaptive routing. No reduced-capability pilot and no generic fallback-first model.

Build-side training artifacts:
- `configs/eipm/n0/n0_v02_cross_context_fusion_v0.2.json`
- `configs/eipm/n0/n0_v02_cross_context_fusion_training_v0.1.json`
- `scripts/eipm/n0/train_n0_v02_cross_context_fusion_full_scale.py`
- `scripts/eipm/n0/run_n0_v02_cross_context_fusion_prepare_v0_2.sh`
- `scripts/eipm/n0/run_n0_v02_cross_context_fusion_full_scale.sh`
- `scripts/eipm/n0/magnolia_p100_n0_v02_cross_context_fusion_full_scale.sbatch`

The trainer caches actual semantic token states plus structured/evidence field states; it does not collapse all parents to pooled vectors before fusion.

Parents are frozen in the first fusion optimization run only to measure fusion cleanly. That freeze is explicitly not permanent. Progressive/end-to-end unfreezing or architecture/capacity expansion remains allowed whenever fidelity evidence requires it.

Current build head at handoff: `4b58455a5c82e9ca69b2b3fdd2f50d41d5bcd03b`.

## Next execution

1. Pull exact build head.
2. Run CPU fusion preparation/no-ceiling gate. It emits `cross-context-fusion-curriculum-v0.2/preparation_receipt.json` with exact git revision.
3. Do not modify/pull the repo after that receipt.
4. Submit the full-scale one-P100 fusion optimization job. Runner refuses stale revision receipt.
5. Evaluate 80/160/240 checkpoints capability-first. If the first governed tranche passes, create an untouched fusion challenge before ratification. If it fails, do failure-driven data/architecture work without shrinking the model for efficiency.

Private identity gradient remains CLOSED. N1 requires explicit owner authorization later.
