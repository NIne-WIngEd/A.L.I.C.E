# Longitudinal MFM corpus admission v1

**Status:** verifier and fictional tests only. No real corpus has been admitted,
independently adjudicated, trained on, or evaluated. Keep raw histories,
labels, consent records and reviewer identity evidence in authorized private
custody. This schema governs a host-neutral shared model; instance-private
adaptation needs a separate permission policy.

`formation_dataset_admission.py` reads a frozen private JSON manifest. It
requires `schema=mfm-formation-corpus-v1`, `corpus_id`, and cases assigned to
`train`, `development`, or `final`. Each case declares `case_id`, `author_id`,
`host_family`, `source_family`, `generator_family`, `scenario_family`,
`duplicate_group`, `parent_case_ids`, an array of sources, one target and at
least two reviews. Source entries carry `source_id`, relative `path`,
`sha256`, `rights_path`, `rights_sha256`, and `parent_source_ids`. Target carries
relative `path` and `sha256`. Each review supplies a distinct `reviewer_id`,
`blind=true`, `decision=accept`, the exact `target_sha256`, and the ordered
`source_sha256s`. Reviewers must differ from the author and each other.

Each train/development source rights file has
`schema=mfm-source-rights-v1`, `issuer_id`, `authority_ref`, `source_id`,
`source_sha256`, `host_family`, `revoked=false`, and separate booleans for
`formation_training`, `formation_evaluation`, and `model_distribution`.
Shared-model training requires training and model-distribution permission;
development requires evaluation permission. All three split families must
exist. Source, target, rights and manifest bytes are hashed. Rights receipts
are checked again at handoff. The external custodian must authenticate the
issuer, active consent, reviewer identities and whether the reviewers really
worked independently; a JSON claim or hash alone cannot do that.

The gate isolates connected host, source, generator, scenario, duplicate,
source digest, source-parent and case-parent lineages across splits. The
corpus steward must identify near duplicates, paraphrases, pseudonym aliases,
shared templates, generator seeds, imported source ancestry and withheld
future events **before** freezing the manifest. Unknown lineage stays out of
admission. This static check catches declared and exact-byte crossings; it
does not discover semantic near duplicates on its own.

FINAL entries provide metadata and declared SHA-256 values only. Admission
never opens their payloads or rights files. The final custodian checks those
independently and keeps them outside model selection. `audit_handoff` accepts
the **actual** gradient and development path lists, rechecks their bytes,
rejects unadmitted or FINAL paths and digests, and returns a versioned receipt
with explicit FINAL exclusions. Only train paths can reach gradients;
development paths are for selection. A boolean claiming FINAL was excluded
cannot replace the entry audit. The existing public eleven-case fixture is
contract diagnostic material and is not an input to this admission path.
