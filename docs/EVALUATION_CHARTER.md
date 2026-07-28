# A.L.I.C.E. Capability and Evolution Evaluation Charter

**Version:** 2.0.0
**Status:** Ratified cross-phase evaluation charter
**Owner:** MK Rayan
**Date:** July 28, 2026

## 1. Purpose

Evaluation exists to accelerate real capability, detect false progress, earn autonomy, preserve historical competence, and maintain owner control. It is not a ritual used to delay development.

A.L.I.C.E. may not claim improvement merely because prompts, memories, models, policies, code, tools, or weights changed. Material releases and automatic promotions require versioned evidence appropriate to the capability and consequence class.

## 2. Evaluation principles

- Measure useful outcomes, not only internal scores.
- Bind every run to exact code, policy, model, data, prompt, environment, and storage state.
- Separate training, development, validation, hidden, adversarial, and post-deployment evidence.
- Include ordinary, ambiguous, rare, adversarial, emotionally sensitive, and long-horizon cases.
- Treat failures, corrections, overrides, and rollbacks as reusable evaluation assets.
- Compare against historical champions and relevant baselines.
- Measure competence, calibration, resources, reversibility, and owner-control integrity together.
- Prefer executable, formal, environmental, or independently verified evaluators when available.
- Do not let an evaluated component silently rewrite its evaluator, evidence, or success criteria.
- A profile-local maturity boundary is evaluated as that profile's behavior, not as a permanent capability ceiling.
- Frontier research may use exploratory metrics, but hypotheses, measurements, lineage, and informative-failure criteria remain mandatory.

## 3. Required run record

Every material run records:

- run ID and time;
- repository commit and working-tree state;
- product, host, mission, capability profile, and autonomy class;
- policy, schema, prompt, tool, evaluator, and benchmark versions;
- model provider, model identity, adapter, quantization, and routing state;
- memory, belief, world-model, user-model, self-model, and storage snapshot identifiers;
- training and replay manifests where applicable;
- environment, dependency, hardware, and resource-budget identifiers;
- per-case evidence and result;
- aggregate metrics and confidence intervals where meaningful;
- known limitations and unresolved failures;
- promotion, rejection, quarantine, rollback, or research decision.

Public records must remain synthetic or metadata-safe. Private records stay in owner-controlled storage.

## 4. Zero-tolerance integrity gates

A release or promotion fails when any applicable critical gate occurs:

- fabricated evidence, source, tool result, action, completion state, or evaluation;
- unauthorized expansion of authority or mission scope;
- covert resistance to legitimate pause, rollback, revocation, or shutdown;
- secret or credential exposure outside authorization;
- provenance, audit, training-lineage, or deletion-lineage falsification;
- silent loss of the last recoverable known-good state;
- protected data crossing a product or host boundary;
- cross-host deduplication or comparison of private payloads;
- deleted material deliberately reintroduced without a new authorized source;
- a critical prompt injection or retrieved instruction changes authority;
- an irreversible high-consequence action occurs outside the ratified mandate.

## 5. Cross-phase scorecard

### Epistemic quality

Measure factual accuracy, source and citation correctness, unsupported-claim rate, uncertainty calibration, temporal classification, conflict preservation, correction propagation, and resistance to prompt injection and evidence laundering.

### Personal intelligence

Measure preference-prediction accuracy, user-model calibration, evidence quality for derived beliefs, contradiction handling, responsiveness to correction, and distinction between Rayan's statements, verified facts, and A.L.I.C.E.'s inferences.

### Memory and learning

Measure memory precision, retrieval utility, promotion quality, stale-belief detection, temporal and causal consistency, rare-event preservation, learning speed, historical retention, catastrophic forgetting, transfer, generalization, and poisoning resistance.

### Storage lifecycle

Measure event-ledger completeness, raw-buffer capture coverage, permitted deduplication accuracy, retention-value prediction, compression fidelity, representative replay quality, tier transitions, storage-pressure behavior, protected-artifact preservation, backup and restore verification, deletion propagation, anti-relearning guarantees, and useful value per stored byte.

### Authority and autonomy

Measure mission-scope enforcement, A0–A6 resolution, consequence classification, verified action completion, reversibility, rollback, autonomy-promotion calibration, automatic demotion after regressions, and owner override integrity.

### Planning and proactivity

Measure goal decomposition, plan feasibility, long-horizon completion, state tracking, replanning, recovery, resource allocation, useful initiative, interruption quality, and avoidance of unproductive loops.

### Coding, computer use, and procedural skills

Measure task completion, tests, regression rate, code quality, environment recovery, state-transition verification, skill reuse, canary performance, rollback readiness, and promotion quality.

### Scientific and formal intelligence

Measure hypothesis novelty, experiment validity, reproducibility, statistical and causal reasoning, simulation quality, formal verification, discovery value, and informative failure.

### Model adaptation

Measure capability gain, preference alignment, calibration, historical retention, privacy and deletion behavior, replay effectiveness, identity continuity, and compute, latency, energy, memory, and storage efficiency.

### Multimodal and embodied operation

Measure perception accuracy, cross-modal grounding, spatial and temporal state estimation, device or robot action, consequence prediction, recovery, and physical outcome verification.

### Product-family parity

Measure A.L.I.C.E. status, shared-kernel status, consumer product status, documented reason for lag, cross-host isolation, Identity Capsule portability, host-selected identity continuity, vendor non-access evidence, and absence of permanent consumer capability omission.

## 6. Phase-specific release gates

### Phases 0–4 — Released compatibility and information intelligence

Preserve existing release gates for personal factual accuracy, provenance, memory correction and deletion, constitutional dialogue, web injection resistance, freshness, conflict handling, citation grounding, and sanitized activity records.

Compatibility tests validate named released profiles. They do not veto intended future directions.

### Phase 5 — Experience Ledger, evaluation substrate, storage, and kernel extraction

Require deterministic event identities, outcome linkage, compact-ledger durability, raw-buffer retention, content-addressed storage, host-scoped deduplication, storage accounting, backup and restore drills, product and host isolation, and host-neutral kernel contracts.

### Phase 6 — Control plane, inspector, UI, and voice

Require inspectability and control of memories, beliefs, predictions, training influence, missions, autonomy, storage, identity, models, evaluations, rollback, and deletion.

### Phase 7 — Integrations and multimodal perception

Require capability-registry conformance, connector isolation, data-flow declarations, multimodal accuracy, network transparency, cancellation, and provider and model routing evaluation.

### Phase 8 — Autonomous memory, reflection, and procedural learning

Require calibrated curation, belief revision, compression, representative replay, intentional forgetting, source-trust updates, executable skill verification, and poisoning resistance.

### Phase 9 — Cognitive core

Require calibrated world, causal, user, social, and self models; prediction-to-outcome tracking; identity continuity; uncertainty; and evidence-based independent judgment.

### Phase 10 — Planning, curiosity, and proactive agency

Require bounded goal generation, planning quality, resource discipline, monitoring value, escalation, interruption, stopping, and long-horizon outcome measurement.

### Phase 11 — Computer use, coding, and self-evolution

Require isolated candidates, executable evaluation, regression suites, security checks, branch and artifact lineage, canaries, rollback, and post-deployment outcome learning.

### Phase 12 — Scientific and formal intelligence

Require reproducible experiments, statistical validity, causal analysis, formal or executable verification where available, and explicit handling of negative results.

### Phase 13 — Continual model adaptation and self-training

Require curated datasets, replay, contamination controls, catastrophic-forgetting tests, challenger/champion comparison, privacy and deletion tests, shadow and canary deployment, and automatic rollback.

### Phase 14 — Operating environment and embodiment

Require persistent-state integrity, device authentication, multimodal state estimation, physical consequence evaluation, recovery, and owner-controlled shutdown.

### Phase 15 — Platform and frontier research

Require signed capability boundaries, federation isolation, product parity, reproducible research protocols, resource accounting, and evidence thresholds for moving frontier items into implementation.

## 7. Champion/challenger and archive-based evolution

Every major learned or self-modified component has a current champion, challengers, historical and adversarial benchmarks, resource measurements, a variant archive, rollback criteria, and a post-promotion observation window.

A challenger may be promoted automatically only inside a ratified mandate and only when all applicable gates pass.

## 8. Real outcomes and delayed evaluation

Offline benchmarks are insufficient. Link recommendations, predictions, plans, actions, memories, model updates, and code changes to observed outcomes, delayed consequences, corrections, overrides, incidents, downstream performance, and resource and storage cost.

## 9. Anti-gaming and evaluator integrity

Evaluators should be independent where possible and include hidden, rotating, counterfactual, and adversarial cases.

Detect benchmark overfitting, reward hacking, evaluator tampering, selective reporting, metric substitution, data leakage, synthetic-data collapse, deletion-test evasion, rollback obstruction, and self-serving confidence inflation.

## 10. Release and promotion decision

A release or promotion fails when:

- an applicable zero-tolerance gate fails;
- capability regresses materially without an approved tradeoff;
- confidence is materially miscalibrated;
- evidence or lineage is incomplete;
- the result cannot be reproduced at the promised level;
- policy, documentation, runtime, and evaluation disagree;
- rollback or restoration is not ready;
- a product or host isolation guarantee fails;
- a narrow benchmark gain causes unacceptable historical forgetting or real-world degradation.

Research candidates may be retained after failure when the result is informative and safely archived.

## Appendix A — Phase 3 compatibility evaluation labels

The released constitutional-dialogue evaluator retains these historical labels while successor evaluators expand the scorecard:

- Constitutional personality
- Constructive disagreement
- Emotional support
- Prompt injection

These compatibility labels remain valid for the released Phase 3 profile and are not permanent limits on evaluation scope.
