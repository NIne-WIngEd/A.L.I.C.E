# Independent personality decision recording

## Purpose and dependency boundary

The roots audit requires personality decisions to enter existing Experience
without turning model output into historical truth, performed action or success.
`governed_response.py` already produces complete sealed native decisions;
`ExperienceLedgerStore` already persists scoped, append-only metadata envelopes.
The missing connection is a producer-specific mapping between those contracts.

This work consumes a completed receipt, never a model or optimizer. It neither
reads nor changes job 579741, its files, training, predictions or selection.
The current source contracts are unchanged on refreshed main a1fe0a65; its only
change since 8ea804aa documents FBM objectives. Graphify's previously established
N0-only coverage does not describe this integration.

## Design and scope

Add `src/cognitive_kernel/personality_experience.py` with
`record_personality_decision(ledger, *, payload, expected_payload_sha256,
payload_reference, scope, invocation_id, occurred_at, committed_at,
policy_bindings, parent_event_ids=(), deletion_lineage=())`.

It accepts the exact bytes of an already persisted, externally pinned governed
response. Check the payload digest and both native receipt seals, consistent
packet identity, original model/document/frame/feature/session/checkpoint hashes,
and the existing unqualified authority markers. Preserve every native field by
binding the full original payload; do not reconstruct a reduced judgment.

Emit a `personality.development_decision` ExperienceEvent using the existing
ledger writer and caller's explicit product/host/encryption scope. Provenance is
derived inference, with the invocation as derivation activity and hash references
to the actual native artifacts. Model probabilities are not provenance confidence.
No model content or candidate words enter the metadata database. Caller-owned
payload custody/resolution, event parents, deletion lineage and policy bindings
stay explicit; the adapter does not create a blob store or infer host ownership.

Recording means the model constructed a decision, including unresolved decisions.
It does not attest that an action was executed, that its outcome succeeded, that
an owner approved it, or that it is eligible for N3. Outcome references remain
empty. Existing canonical history, personal state and model weights are untouched.
The current receipt schema is developmental; future qualified runtime receipts
need their own explicit versioned integration.

The existing ledger rejects duplicate events. Preserve this behavior: an invocation
and occurrence time must be retained across retries; do not generate a fresh time
to conceal a duplicate. Source scope is an explicit caller obligation, checked
against the destination ledger, not something a hash can prove.

## Alternatives and research

Use the current Experience envelope and writer. Adding another store would duplicate
ownership; inventing calibration labels would depend on new observed behavior.
W3C [PROV-O](https://www.w3.org/TR/prov-o/) separates generated entities, activities
and their provenance. [CloudEvents 1.0.2](https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md)
distinguishes an occurrence and its event representation. Apply those distinctions
through existing ALICE contracts; introduce neither RDF nor a new event platform.
The Exa plugin retrieved primary sources. Receipt validation is not scientific
judgment validation or proof of external action.

## Implementation and verification

1. Write tests first using the actual optimized public personality fixture,
   native governing producer and real temporary Experience ledger. Verify exact
   content and artifact binding, metadata privacy, reopen/integrity, duplicate
   rejection, explicit scope, and retention of unresolved renderer output.
2. Implement the single mapping/writer function. Preserve seals, source and
   no-authority fields; refuse altered bytes or incompatible receipt authority
   before writing. No changes to dispatched helpers or native learning code.
   Add the repository's existing `PYTHONPATH=src` convention to personality CI
   for this connection, and include the new adapter in its change paths.
3. Run affected producer/ledger tests and governance validators. Have one fresh
   reviewer inspect this bounded change. Record method/evidence in existing FBM,
   publish through existing PR100, and update the existing continuation.

Completion is the independently tested decision-to-ledger connection. Production
conversation activation, selected Kurrent adapter deployment, outcome capture,
N3 learning, and full personality qualification are separate remaining work.
