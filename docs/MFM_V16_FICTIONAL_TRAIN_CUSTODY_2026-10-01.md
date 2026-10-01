# Exact fictional v1.6 train candidate custody

Two deterministic materializers make the existing fictional source and label
work concrete for owner review. They do not train a model or claim independent
target correctness. Their source rights are **pending**. Both use the original
`owner-directed-mfm-v16-synthetic` authorization ID already present in the
candidate bytes, not a new authorization claim.

## Original 15 train rows

`scripts/mfm/materialize_v16_authoring_seed_train.py` re-renders the exact
21-row original generator, compares it to the frozen repository JSONL SHA
`8e967ee6494f2c5ec47ab602bad213df8c07c53b1abe9a9aff950f19c0e78e22`
and generator code SHA
`2b6815139e33acd92abf3e215a04ef9bdf57fb2868e6567dde2d734518c057ce`,
then emits **only** its 15 original train cases. The six old development rows
stay out because they share the training generator family.

One original counterfactual pair used the same source ref ID for two different
later-statement byte buffers. The source admission gate correctly rejects
that. The materializer keeps both exact original case files, and makes an
explicitly derived `v16-seed-03` candidate whose later source ID is
`v16-seed-03-temperament-later`. It replaces that ID consistently in its
context evidence, target anchors, dispositions and source list; it does not
alter source text, proposal content or labels. Both old and derived candidate
SHA-256 values and the one-field mapping are recorded. The generator family
remains `assistant-authored-v16-seed-20260930`.

Each case references exact source text files, an original candidate,
admission-shaped derived candidate, canonical v1.6 target JSON, source-only
index, generator input spec, converter provenance, and one source-rights
**draft** per source. The original generator and exact target converter code
bytes are snapshotted. The draft binds source ID, hash and host, while
`issuer_id`, `authority_ref`, `formation_training`, `formation_evaluation`,
`model_distribution` and `revoked` stay null. A draft is not an issued rights
receipt and fails corpus admission.

When the same original source ID and exact bytes occur in two related cases,
both cases point to one frozen source file; the optimizer handoff audits that
physical path once and the parser verifies the bytes again for each case.
The two counterfactual later statements have different bytes and therefore
use different IDs and files.

## Revocation case from another train generator

`scripts/mfm/materialize_v16_role_revocation_train.py` pins the exact six-case
role generator SHA
`2096cb9a148678e9a2a4a2fea9a3c09d44e9f45db448713998667f3f4219d08e`
and emits only `role-source-person-revocation` in its original generator family
`codex-fictional-role-chronicle-20261001`. It has the explicit source-use
revocation and derived-copy deletion requests missing from the original 15.
It contains the registry tombstone and host withdrawal statement, not the
revoked archive payload. Its rights drafts remain unissued. The original
six-case generator had one author and is not independent review.

```bash
PYTHONPATH=src:. python3 -m scripts.mfm.materialize_v16_authoring_seed_train \
  --output-dir "$PRIVATE_BUNDLE/seed-train"
PYTHONPATH=src:. python3 -m scripts.mfm.materialize_v16_role_revocation_train \
  --output-dir "$PRIVATE_BUNDLE/revocation-train"
```

Both commands require new directories outside Git and never overwrite
existing custody. These are 16 potential train candidates from two original
generator families. Their published methods do not authenticate the fictional
owner utterances as observed real data; the owner can review and authenticate
the synthetic MFM teaching permission from the explicit authorization in this
chat. The steward should issue actual source-bound rights receipts only after
that review, using truthful issuer, authority and permission assertions. Do
not edit the draft into a receipt merely by replacing nulls mechanically.

The assembler requires separate rights receipts and an independently sourced
diagnostic development family with the same owner authorization ID. The new
deterministic development generator must be reviewed on its own lineage and
source permissions. A successful structural corpus assembly is still an
unqualified teacher fit, not independently adjudicated FINAL or Alice-level
memory formation capability. Use the Magnolia full-corpus CPU preflight only
after real manifest admission; no paid GPU run is justified by these exports.
