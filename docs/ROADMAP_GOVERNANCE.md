# A.L.I.C.E. Roadmap and Product-Track Governance

**Version:** 2.1.0

## 1. Stable top-level domains

Phases 0–15 remain stable capability domains. Friday milestones map onto those phases and do not renumber them.

## 2. No implementation freeze

Roadmap stability prevents workflow disruption; it does not make prior architecture sacred. Any phase, interface, policy, schema, test, or module may evolve when required by the final architecture.

## 3. Product tracks

A product track may define milestones, release gates, and commercial deliverables across several A.L.I.C.E. phases. Product tracks live in dedicated documents such as `docs/FRIDAY_ROADMAP.md`.

## 4. Constraint burden of proof

Every non-kernel constraint must exist in `docs/CONSTRAINT_REGISTRY.md` or a phase-scope registry with rationale, evidence, scope, review condition, and removal path. Unregistered constraints are not permanent architecture.

## 5. Default resolution for new capabilities

1. Map to an existing phase and product track.
2. Define a capability manifest and evaluator.
3. Start in simulation, sandbox, or an experimental profile when immature.
4. Gather evidence.
5. Expand autonomy or product exposure based on results.
6. Migrate earlier modules when they create coupling or ceilings.
7. Move unresolved science into the Research Frontiers Register.

## 6. Roadmap-change threshold

A top-level change is justified only when a capability cannot fit any existing domain, a dependency is fundamentally invalidated, or Rayan changes the terminal purpose. A major code migration, product fork, or Phase 1–4 redesign does not by itself require a new phase.


## 7. Friday independent development and production promotion

A Friday team may independently research, prototype, implement, test, open pull requests, and prepare candidates. Production promotion is a separate authority transition and requires both an exact-artifact A.L.I.C.E. audit attestation and Rayan's explicit approval for the same candidate. Neither approval alone is sufficient.

Emergency rollback, disablement, containment, or reversion to previously approved behavior may proceed to protect users. The emergency path may not introduce new capability, broaden permissions, start new data collection, or deploy unapproved replacement behavior. Only Rayan may amend the dual-approval rule.

## 8. Private companion governance

Private companion source and directives remain owner-controlled HIGHLY_SENSITIVE material. Public repository artifacts expose only opaque identifiers, schemas, custody constraints, and synthetic fixtures. Authorized private learning remains possible through explicit lineage, evaluation, deletion, rollback, and promotion paths; this custody boundary is not a capability ceiling.

## 9. Clone-aware identity governance

The A.L.I.C.E. private identity lane targets the highest achievable evidence-grounded reconstruction of the owner-designated source person's personality and mindset. It may not be downgraded to a merely inspired or generic companion persona without a newer explicit owner decision. A.L.I.C.E. must remain aware that it is a reconstruction rather than the literal original person, while naturally embodying the reconstructed identity in ordinary interaction. Source history, reconstruction, and A.L.I.C.E. continuity remain typed and inspectable.
