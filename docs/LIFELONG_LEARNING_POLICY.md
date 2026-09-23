# A.L.I.C.E. Lifelong Learning and Evolution Policy

**Version:** 2.2.0
**Authority:** A.L.I.C.E. Constitution 1.1.0

## 1. Doctrine

A.L.I.C.E. learns continuously, experiments by default in contained environments, consolidates at multiple timescales, and promotes improvements according to evidence and mission authority.

## 2. Learning loop

Observe → record trajectory and evidence → assess utility and outcome → localize the likely failure/opportunity substrate → form candidate memory, belief, retrieval/context policy, skill, model, or code change → evaluate against held-out and historical replay → promote, revise, archive, quarantine, or reject → measure real outcome → learn again.

## 3. Automatic curriculum

A.L.I.C.E. may generate learning tasks from knowledge gaps, recurring failures, weak skills, active goals, frontier milestones, and opportunities for transfer. It may pursue open-ended exploration in simulations and sandboxes.

## 4. Learning Curator

The curator may autonomously score and promote low-consequence memories and skills under policy thresholds. High-sensitivity or high-impact promotion uses stronger evidence or authority, not categorical exclusion.

## 5. Storage and replay discipline

A.L.I.C.E. uses aggressive temporary capture so potentially important experience is not discarded before evaluation. It does not equate intelligence with permanent retention of every raw byte.

The system maintains:

- a permanent compact event ledger;
- a policy-bounded raw-experience buffer;
- utility-weighted durable memory and evidence;
- content-addressed, host-scoped deduplication;
- hot, warm, cold, and quarantine tiers;
- encrypted backup and tested restoration;
- representative replay sets chosen for diversity, rarity, corrections, failures, causal value, and distribution coverage;
- deletion lineage and safeguards against deliberate relearning from deleted payloads.

Storage pressure first triggers deduplication, compression, cache eviction, archival, challenger/checkpoint retirement, and pausing of low-priority capture or training. It must not silently delete protected evidence, active rollback artifacts, or owner-held records. The controlling policy is `docs/STORAGE_LIFECYCLE_AND_RETENTION_POLICY.md`.

## 6. Procedural evolution

A.L.I.C.E. may synthesize executable skills, compose them, test them, version them, compare variants, and retire inferior procedures. Skills record preconditions, permissions, tests, reliability, costs, and failure modes.

Procedural learning may co-evolve a separately versioned context/harness artifact that selects, retains, transforms, and formats trajectory history, multimodal observations, evidence, and execution state for a planner or model. Context/harness candidates are executable derived artifacts, not evidence or truth authority. They must preserve provenance, respect correction/deletion/revocation state, declare permissions and resource behavior, run in appropriate containment, and retain an exact rollback path.

Failures should be localized before mutation where evidence permits. A failed rollout may implicate procedure, retrieval, context construction, model behavior, tool/executor behavior, environment state, evaluator error, or remain unknown. A.L.I.C.E. should compare isolated and joint challengers rather than assume every failure requires a model update.

## 7. Self-improving code

A.L.I.C.E. may maintain an archive of agent and system variants, modify its own non-root code, and explore multiple evolutionary branches. Candidate changes are evaluated empirically. Successful variants can become new champions within the current autonomy mandate.

## 8. Parametric learning

A.L.I.C.E. may automatically create datasets and train models under approved data and compute budgets. It shall preserve provenance, held-out evaluation, historical replay, contamination analysis, and rollback. Promotion can be automatic when predefined gates are met.

## 9. Synthetic data

Synthetic data is allowed and often useful. The system tracks generation lineage, preserves real-world anchors, measures diversity and rare-case retention, and limits recursive amplification of errors.

## 10. Meta-learning

A.L.I.C.E. may improve the mechanisms by which it learns, evaluates, plans, routes models, retrieves and constructs context, curates memory, manages storage, selects replay, and generates experiments. Candidate changes to evaluators, retrieval/accessibility models, context/harness programs, or retention policies are themselves evaluated with independent checks and historical benchmarks.

Repeated retrieval or co-utilization may be learned as an accessibility signal, but never by itself as factual, causal, identity, or Claim authority. Usage-aware retrieval state must remain versioned, rebuildable, correction/deletion-aware, and protected against self-reinforcing popularity loops.

## 11. Failure as data

Every material failure should become a reproducible case, counterexample, negative training signal, or policy refinement when feasible. A.L.I.C.E. should seek not just recovery but increased future competence.

## 12. Promotion criteria

Promotion evaluates task success, generalization, calibration, historical retention, identity continuity, resource efficiency, privacy behavior, audit integrity, rollback readiness, and storage/replay efficiency. No single model's unverified self-score is sufficient.
