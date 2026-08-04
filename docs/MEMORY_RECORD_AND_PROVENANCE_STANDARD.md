# Memory Record and Provenance Standard

**Draft:** 0.2<br>
**Status:** Canonical Claim Store direction accepted<br>
**Applies to:** Memory Architecture v4

## 1. Purpose

This standard prevents different memory meanings from collapsing into generic
text records.

## 2. Canonical Claim Store decision

The Claim Store is the canonical adjudicated-knowledge layer.

Claim Store v1 is a logical store implemented through new Cognitive Kernel
contracts and tables in the existing host-local SQLite architecture. It is not a
new microservice.

The minimum canonical structures are:

### Claim Identity

Stable identity for one subject-predicate-scope proposition family.

### Claim Version

Append-only asserted value, epistemic class, authority class, confidence,
bitemporal validity, lifecycle state, and content digest.

### Claim Evidence Relation

Typed support, opposition, origin, correction, or derivation relation from a
claim version to an Evidence Event or other authorized record.

### Claim Adjudication

Governed decision recording acceptance, rejection, dispute, supersession,
retraction, or owner override.

### Current Claim Projection

Materialized, indexed, rebuildable view of the presently adjudicated claim
version. It serves normal reads but is not historical authority.

## 3. First-class record kinds

### Evidence Event

An immutable record that something was observed, received, decided, attempted,
or executed.

### Claim Version

A structured proposition with authority, evidence, validity, and version
lineage.

### Episode

A bounded set of evidence events with a rebuildable summary.

### Mission State Version

A versioned mission, commitment, dependency, blocker, decision, or outcome.

### Preference Observation

Evidence that a preference was expressed or exhibited. It is not automatically
a stable preference.

### Trait Hypothesis

A derived estimate of a persistent tendency. It requires multiple evidence
points, uncertainty, and review.

### Belief Version

A.L.I.C.E.'s current evidence-linked conclusion. It is not historical fact.

### Prediction

A time-bounded forecast with evaluation criteria and later outcome score.

### Projection Snapshot

A versioned owner, source-person, relationship, self, world, social, or causal
model derived from other records.

### Procedure Candidate

A proposed reusable process extracted from successful or failed trajectories.

### Skill Version

An evaluated procedure or code artifact with tests, permissions, and rollback.

### Context Packet Trace

A metadata-safe record of which memory units influenced one model invocation.

## 4. Required provenance

Every derived record must identify:

- source evidence IDs;
- support or opposition relation;
- derivation policy version;
- model and version;
- prompt version;
- run ID;
- recorded time;
- confidence or uncertainty;
- scope;
- sensitivity;
- reviewer/authority receipt where required.

## 5. Promotion rules

- Model proposals enter candidate state.
- Owner statements remain owner statements even when trusted.
- External claims remain claims until verified under domain policy.
- Repetition does not convert a claim into fact.
- A preference observation requires stability evidence before becoming a stable
  preference.
- A trait hypothesis requires longitudinal evidence and must preserve
  counterexamples.
- A generated reconstruction cannot become source history without owner
  attestation.
- A belief may be useful while remaining uncertain.
- A skill cannot activate without tests and permission scope.

## 6. Versioning

No semantic overwrite is permitted for:

- claim value;
- belief;
- prediction;
- stable preference;
- trait hypothesis;
- mission decision;
- relationship state;
- identity projection.

A new version records the superseded version and reason.

## 7. Current-state projection

The materialized `current_claims` projection provides fast reads. It must be
rebuildable from claim identities, versions, evidence relations, and
adjudications and may never be the only record of a correction.

## 8. Bitemporal requirement

Changing records carry:

- `valid_from` and `valid_to`;
- `recorded_at`;
- optional `superseded_at`.

Backfilled evidence may have old valid time and new recorded time.

## 9. Source-person separation

The following record spaces remain distinct:

- `SOURCE_HISTORY`
- `SOURCE_PERSON_MODEL`
- `RECONSTRUCTION_INFERENCE`
- `ALICE_CONTINUITY`
- `OWNER_RELATIONSHIP_MODEL`

Cross-space derivations require explicit edges. No record may silently move
between spaces.

## 10. Context-use provenance

Every Memory Context Packet records:

- query/mission identifier;
- selected memory IDs;
- selection scores;
- filters applied;
- exclusions;
- token allocation;
- stale/disputed flags;
- model profile;
- packet digest.

This enables inspection of why A.L.I.C.E. remembered or ignored something.
