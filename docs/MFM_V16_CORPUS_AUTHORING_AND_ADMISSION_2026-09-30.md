# MFM 1.6 formation corpus: authoring seed and qualification route

**Current state:** 21 original fictional, text-only cases. Fifteen are labelled
`train`; six are diagnostic `development`. Every target and all ten explicit
positive/negative dimension labels are assertions by a single synthetic
teacher. There is no authenticated rights record, second reviewer, independent
development lineage, sealed FINAL, trained MFM or downstream capability
receipt. This seed can exercise the codec and CPU training path. It cannot
authorize a full-role paid fit or support a model capability claim.

Files:

- `scripts/mfm/build_v16_authoring_seed.py` contains the original narratives,
  exact source quotes, source policy floors, scoped target registries, and
  manually specified target labels. The generator checks that those labels
  agree with the explicit output. It does not derive a negative from an omitted
  historical label.
- `benchmarks/mfm/v16_authoring_seed.jsonl` is the byte-frozen authoring pack,
  SHA-256 `8e967ee6494f2c5ec47ab602bad213df8c07c53b1abe9a9aff950f19c0e78e22`.
- `scripts/mfm/audit_v16_authoring_corpus.py` binds the exact hash, calls the
  v1.6 codec and grounding checks, verifies paired input isolation and parent
  history, and reports every admission blocker.
- `benchmarks/mfm/v16_authoring_seed_audit.json` is its aggregate report. It
  reports positive counts for each of the ten coarse dimensions in both
  diagnostic partitions. This is **authoring coverage**, not independently
  adjudicated gold or a model score.

The fiction distinguishes January plans from February experiences, temporary
anger from negotiated norms, four observed social habits from a later agreed
rule, source-person evidence from host habits, disputed events from settled
episodes, correction/deletion requests from completed memory edits, and
observed outcomes from owner rationale. Two paired histories hold all
model-visible metadata and nondecisive evidence constant while changing one
later source. One longer sequence preserves the owner's goal, Alice's warning,
the owner's override, a partial benefit, a failed assumption, two workdays of
cost and a revised procedure. It contains `procedural_skill` and
`metacognitive_signal` proposals. These are cases to challenge a future model;
no learned model has passed them. All source bytes are authored text. The pack
does not test image, audio, video or real longitudinal owner data.

## Repeatable authoring and review

1. Collect each authorized original item in owner-controlled custody with
   exact bytes, creation and observation times, source person and speaker,
   modality, sensitivity floor, consent scope, owner, source lineage and
   revocation state. Preserve derivatives and duplicates as linked children.
   Never place private histories in Git.
2. Build a 1.6 context from registered sources and scoped entity, scene,
   mission and workspace targets. Author exact source byte anchors and all
   output fields. Separately mark **each** of sensitivity, episode,
   relationship, mission, workspace, correction, source-person, contradiction,
   outcome and abstention as `present`, `negative` or `unknown`. An absent field
   in an older source is `unknown` until actually reviewed. Unknown targets
   cannot enter the full supervised serializer.
3. Have two actual reviewers independently inspect the original exact source
   bytes, context, source policy and proposed target while blind to the other
   review. Resolve disagreement explicitly and retain their real identifiers,
   decisions, source and target SHA-256 bindings, timestamps, and the adjudicated
   target revision. These records must be authenticated by the responsible
   steward; the local validator only checks their declared structure.
4. Obtain actual source rights records from their issuer, including training,
   evaluation, distribution and revocation permissions bound to exact source
   SHA-256 and host family. The authoring seed has **no** such receipts.
5. Assign entire connected components by host, source person, family,
   scenario, original/derived source, duplicate group, counterfactual pair,
   parent case and generator lineage to one partition. Freeze truly independent
   train and development families. Seal FINAL by hash and keep its payload
   inaccessible to the optimizer and checkpoint selector. The current seed's
   shared synthetic generator family means its six diagnostic development rows
   cannot serve as independent validation for the fifteen training rows.
6. Build the existing `mfm-formation-corpus-v1` receipt manifest with
   `train`, `development` and metadata-only `final` cases. Its source rights
   receipts and two distinct blind reviewer records bind exact target and
   source hashes. `admit_formation_corpus(manifest_path,
   expected_sha256=...)` checks those declarations, connected split leakage
   and the sealed FINAL boundary. `admitted_rows_v16(admission,
   split="train"|"development")` then checks 1.6 target bytes, source order,
   host binding and complete labels at handoff. A later independent auditor
   still has to authenticate real identities, rights and FINAL custody.
7. Score source-grounded formation, corrections and abstentions on the sealed
   cases. Then check that the independent memory gate accepts and rejects
   appropriately, correction/deletion rebuilds the projections, retrieval
   brings back the right accepted memories, and a fixed task's native judgment
   changes under a controlled history or outcome intervention. Report
   failures by role, source family and time horizon. Do not equate structural
   coverage or training loss with Alice-level memory behavior.

Reproduce the public authoring diagnostics:

```bash
PYTHONPATH=src python scripts/mfm/build_v16_authoring_seed.py \
  --output benchmarks/mfm/v16_authoring_seed.jsonl --check
PYTHONPATH=src python scripts/mfm/audit_v16_authoring_corpus.py \
  --corpus benchmarks/mfm/v16_authoring_seed.jsonl \
  --expected-sha256 8e967ee6494f2c5ec47ab602bad213df8c07c53b1abe9a9aff950f19c0e78e22
```

This file and aggregate report are public method artifacts. No case here is
about the real Elaina or Rayan. The MFM job remains formation proposals from
authorized experience. The gate owns memory authority; downstream host,
relationship, self and mission components consume accepted memory.

Two separate source-only model QA passes found that the earlier seed treated
an explicit owner correction as merely unresolved and omitted several
supported details from mixed outcomes and a negotiated relationship. The
amendments and their limits are in
`MFM_V16_INTERNAL_BLIND_REVIEW_2026-09-30.md`. They are **not** independent
human adjudication or FINAL gold.
