# MFM 1.6 independent gold authentication boundary

**State:** verifier and synthetic tests only. There is no trusted external
roster, independently signed corpus, sealed FINAL assessment, or trained MFM.

The existing corpus admission checks split lineage, byte hashes, declared
rights and two declared blind reviews. Declarations alone cannot authenticate
the people who issued them. `formation_adjudication_v16.py` adds a separate
offline Ed25519 verification step for a shared, distributable MFM corpus. Its
trust roster digest must be pinned by an independent data steward **outside**
the corpus manifest and supplied again at every live optimizer handoff/resume.
The corpus author must not choose the trusted roster for their own admission.

The roster has schema `mfm-v16-external-trust-roster-v1`, `valid_until`,
`revoked_ids`, and maps `authors`, `reviewers`, and `rights_issuers` from IDs to
canonical base64 raw Ed25519 public keys. Reviewer keys must differ from each
other, the author, and the source-rights issuer in the same case. The caller
pins the exact roster SHA-256 outside the corpus and checks current revocation
and expiry immediately before training. Install the pinned dependency in
`scripts/mfm/requirements-formation-review.txt` in that admission environment.

Each train/development rights record retains `mfm-source-rights-v1` fields and
adds `valid_until` and `signature_v16`. Each review retains the existing exact
source/target SHA-256, blind/accept fields and adds `case_id`, `split`,
`author_id`, `valid_until`, `lineage_sha256`, the ordered source-rights
SHA-256 list,
`complete_target_review_v16: true`, `coverage_scope` equal to the four
constants in `formation_adjudication_v16.COMPLETE_SCOPE`, and `signature_v16`.
The lineage digest covers host, source, generator, scenario, duplicate group
and parent-case IDs. The reviewer explicitly attests that the target covers
every relevant proposal and disposition within the specified opened evidence,
including when there is no other relevant memory. The ten target dimension
statuses alone do not establish this exhaustive judgment.

The signature is Ed25519 over a domain prefix followed by
`canonical_json_bytes(record_without_signature_v16)`. Prefixes are
`ALICE-MFM-V16-RIGHTS\n`, `ALICE-MFM-V16-REVIEW\n`, and
`ALICE-MFM-V16-FINAL-RIGHTS\n`. FINAL source/target/rights **payload files are
not opened** by the training verifier. A trusted rights issuer signs an inline
FINAL attestation binding case ID, source ID and SHA-256, rights SHA-256, host
family, evaluation permission and expiry. Two reviewers sign the sealed
source/target digests and lineage. The FINAL custodian still checks those
payloads under a separate isolated evaluation procedure.

The verifier returns only signature provenance. It cannot prove that a key
belongs to a real independent reviewer, that a review was genuinely blind, that
rights are legally sufficient, or that a target is semantically correct. Those
are steward and adjudication responsibilities. Current synthetic authoring
cases contain no such records. They remain construction tests, never full-role
gold or paid-fit authorization.

Per-user private FBM adaptation has a different consent contract and never
requires model-distribution permission or two reviewers for every owner event.
The signed corpus path above qualifies shared, distributable MFM competence.
