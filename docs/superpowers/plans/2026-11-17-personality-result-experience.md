# Personality result-to-Experience implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline and obtain independent code review.

**Goal:** Persist an explicit link from an existing personality decision event to a later, externally pinned Result Capsule through the existing Experience writer.

**Architecture:** A small sibling of `personality_experience.py` verifies the stored decision, supplied result and scope, then appends a metadata-only reported-result event. The original decision and complete capsule remain immutable; the capsule's status, evidence, outputs and reporting provenance stay in its exact externally stored payload. This is an integration connection, not an outcome judge or N3 trainer.

**Tech Stack:** Existing Python contracts and SQLite Experience ledger; no new dependency, service or model.

**Spec and authorization:** The owner's full personality request and explicit instruction to implement independent work; roots audit's required decision/action/outcome return path, Phase5 Result Capsule and Experience contracts. No new approval dependency. Job582305 uses its pinned remote sources and is untouched.

## Design and research

`record_personality_decision` currently records model/frame/checkpoint lineage but has no later result connection. `ResultCapsule` already owns status, mission/node, output/evidence/source-event references and reporter provenance. The missing connection is therefore an adapter, not a new outcome schema or execution engine.

Chosen API: `record_personality_result(ledger, decision_event_id, expected_decision_event_sha256, capsule, payload, expected_payload_sha256, payload_reference, committed_at)`, using keyword arguments. Require the existing stored development-decision event, its external event digest, the exact serialized capsule metadata and its external byte digest. Require capsule source events to name that decision and product/host/encryption scopes to agree. Schema version evolution alone must not change storage identity. Refuse source-deleted/quarantined inputs and impossible result-before-decision chronology. Existing ledger owns append atomicity and duplicate refusal.

Preserve original parent event, capsule/output/evidence references, policies and combined deletion ancestry. The new event is `personality.reported_result`, derived linkage with unknown confidence and no model producing the result. Original model lineage remains accessible through the exact parent. Do not rewrite the original decision or attach fabricated success, causal credit, teaching labels, owner approval or A-EXP authority. Caller owns result authenticity, payload custody, current lifecycle policy and runtime activation.

[W3C PROV-DM §2.1.2/5.2.1](https://www.w3.org/TR/prov-dm/#term-Derivation) distinguishes linked provenance from demonstrated influence. Accordingly, a source-event reference here records the result owner's claimed association; it is not causal learning credit. [OpenTelemetry links](https://opentelemetry.io/docs/specs/otel/trace/api/#link) provide a comparable explicit link across completed operations; no tracing platform is added or treated as memory authority. Existing ALICE contracts provide the actual implementation, with original immutable decision and later distinct result.

Rejected: infer outcomes from selected candidates, turn capsule status into scalar reward, mutate the decision, introduce a second ledger, or start calibration before actual successor evidence. Those would cross responsibilities or depend on the running experiment.

## Tasks

- [x] Write actual optimized native-decision → existing Experience → Result Capsule → reopened Experience tests. Cover partial/failed/successful reports, exact ancestry, duplicates, wrong scope/decision/payload and original preservation. Confirm missing adapter first.
- [x] Implement `src/cognitive_kernel/personality_results.py` using existing contracts/writer; no scientific-core changes.
- [x] Run new and affected ledger/result/decision tests; independently review. Record failures through the mandatory eight-step diagnosis before corrective edits.
- [x] Capture actual method/evidence in FBMv31, preserving v30 and trace51. Publication and continuation use the existing draftPR100 workflow.

## Verification boundary

Public fixtures use an actual optimized native consumer and real durable ledger, then an explicitly reported Result Capsule. They verify the connection, not private learned judgment, that an action occurred, causal credit, calibration, full lifecycle deletion or automatic runtime activation. No new Magnolia work or polling is required.

## Execution and diagnosis

The first test command failed collection because its invocation omitted the
repository's `src` import root. (1) Evidence stopped before fixture/adapter use;
(2) it proves no feature failure; (3–5) existing ledger and personality tests
import `cognitive_kernel` from `src`, and the prior verified construction
controller explicitly sets `PYTHONPATH` there. (6–7) Reuse that established
controller environment without changing imports, packages or test assertions.
(8) Retain the original collection log and rerun through the versioned
controller to reach the actual missing-connection assertion before implementation.

The corrected invocation reached the real optimized decision and ledger, then
the fixture failed `ProvenanceReference.validate`: `conflict_or_uncertain`
requires a confidence, unlike the connection's derived-link provenance.
(1–2) This is an invalid fictional reporter record, not an adapter failure.
(3–5) The departure is fixture construction before adapter invocation; existing
`contracts.py` explicitly owns that distinction. (6–7) Give only the fictional
reporter an explicit .35 estimate under the unchanged contract; preserve the
link's unknown confidence. This numeric fixture is not private gold or an N3
target. Original PROV research supports preserving provenance without promoting
it to authority. (8) Rerun the same actual native/ledger path, retain the failure,
and later verify reporter confidence survives while the link stays uncalibrated.

The valid fixture reached the explicit missing-adapter assertion after actual
native optimization and durable decision recording. That establishes the absent
return connection described in the design, with no new model failure inferred.
The implementation follows the previously traced producer/consumer boundary and
primary research, retaining both setup failures and this valid RED evidence.

The first implemented connection passed16 native/ledger tests. Exact-source
review then examined the boundary between validated capsule metadata and
canonical Experience values:

1. Two valid resealed capsules with explicit timezone offsets reproduced both
   wrong rejection of a later result and wrong acceptance of an earlier result.
   They pass the existing capsule validator and use an actual stored decision.
2. Temporal ordering must describe actual instants; otherwise the association
   can falsely say that a result followed a decision.
3. `ResultCapsule.validate` parses timestamps and enum values but does not require
   all stored representations to be canonical. Experience normalizes its own
   fields. The adapter had assumed the former implied the latter.
4. The first departure is comparing raw timestamp strings. Review identified
   the analogous raw storage-tier comparison, which can miss an inactive tier
   expressed with valid casing/whitespace; preserve a direct regression too.
5. The adapter must interpret existing contract semantics while retaining the
   original capsule bytes, digest and reporting provenance.
6. Use existing kernel timestamp/identifier normalizers for time ordering and
   lifecycle decisions only. Do not rewrite the pinned payload, change capsule
   validation globally or relax the inactive-input rule.
7. [Python's aware-time comparison rules](https://docs.python.org/3.11/library/datetime.html#datetime.datetime)
   compare differing offsets as UTC instants. The existing kernel
   `normalize_timestamp`/`require_identifier` implement the relevant conversions.
   This fixes the boundary class without a new schema or framework.
8. Re-run both before/after offset cases and inactive representations, followed
   by the complete connection and affected ledger/result/decision tests.

## Verified result

Both inactive representations reproduced the bypass before correction. The
final affected suite passed **44 tests in103.20seconds**. Independent source
review cleared both corrections with zero remaining blockers. The saved
verification receipt binds exact implementation/test/source hashes and all
failure logs. All15 source pins for Magnolia582305 remain unchanged; no remote
poll, new compute, model update or production activation occurred.

FBMv31 preserves v30 and all51 preceding trace rows, appending trace52 with
the actual construction and diagnosis. This completes the reported-result
association component. The runtime result owner still supplies authentic
observations, current lifecycle policy and payload custody; later primary
review and N3 learning remain distinct work.
