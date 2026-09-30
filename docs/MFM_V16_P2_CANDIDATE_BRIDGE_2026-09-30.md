# MFM 1.6 to P2 candidate bridge

**State:** Limited adapter and synthetic gate exercise. No trained 1.6 MFM,
native Alice learning claim, full-role integration, or production admission.

`alice_memory.formation_v16_p2_adapter` accepts a canonical 1.6 output for
**exactly one owner host profile claim**. It checks the complete 1.6 context and
output against exact opened bytes, an authority-owned Alice host/namespace and
artifact pin, and a fresh lookup for each cited source. The lookup must return
the same scoped evidence metadata and SHA-256, an active owner statement with
candidate-write permission, current sensitivity and its immutable P2 locator.
The candidate carries the exact source content/text hashes and its source ID;
the adapter takes the stricter of context floor, current policy and model hint.
HIGHLY_SENSITIVE cannot enter the ordinary P2 candidate table.

The bridge maps the proposed wording to a `model_proposed` P2 `profile`
candidate with `alice_inference` status and `derived_from` provenance. Byte
anchoring establishes the cited span's presence, **not semantic entailment**.
The existing P2 deterministic assessment always requires human review of a
model-origin candidate. Only the existing P2 promotion API, with a separate
confirming actor, can make it authoritative; only then can P2 retrieval serve
it. Adapter authorization only allows *candidate staging*.

The deterministic candidate ID binds the host/namespace, context digest,
whole output bundle digest, proposal digest, artifact/run, proposed time,
classification, evidence metadata and live source registration snapshot. A
metadata-safe receipt returns these digests, candidate ID, content SHA-256 and
exact source SHA-256 values. The current P2 schema stores the candidate's
source IDs/hashes, content hash, artifact digest and inference run. It does
**not** store the complete v1.6 context/output/anchor receipt or the host scope.
An authorized caller must retain that receipt with its original context,
output and source custody and bind the P2 store to the host externally.

The bridge rejects every richer field or action that P2 cannot represent:
episodes and their event/participant/scene links, relationship counterparts,
mission/workspace links, contradictions, correction targets, uncertainty
references, day precision, multiple proposals and non-propose dispositions.
It never reports those dimensions as successfully integrated. P2 also lacks
promotion-time lookup of the 1.6 source registry. An authority must recheck
revocation and sensitivity before promotion; the demonstration does not
authorize unattended promotion of this adapter's candidates. A full-role
Claim Fabric and projection integration needs a versioned persistent receipt,
scoped subject/entity identities, event and relation records, source-policy
resolution within its write transaction, correction/deletion rebuild, and
separate downstream evaluation. This subset does not set a permanent limit on
MFM's proposed semantics.

The focused test uses a hand-authored synthetic proposal. It exercises output
validation, live policy escalation, P2 candidate staging, deterministic
assessment and rejection of model self-confirmation, authorized promotion,
lexical indexing and retrieval. Passing that test proves an interface path,
not learned competence, independent adjudication or native judgment.
