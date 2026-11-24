# Saved-reader failure 581351: metadata representation boundary

## 1. Validate the failure

581351 failed 2:0 after 27 seconds, after successful owner isolation and before
optimizer/behavior measurements. The public traceback identifies the combined
manifest-hash/plan equality check at original worker line108. Stderr is empty;
no successful result exists. The original worker, pins, intent, logs and startup
receipt remain immutable. No model forward or optimizer update was reached.

The real parent-scope builder returns a tuple of retained batches. The actual
experimental writer preserves that nested tuple in the tensor export, but its
protected plan and snapshot binding are JSON-normalized. A local regression
uses the real parent-scope builder, native experiment and all four artifact
forms to reproduce the reader's false mismatch. This was missing from the
previous public fixture, which covered recipe betas but not production-shaped
nested plan metadata. Review clearance was insufficient at that boundary.

## 2. State the proof

The comparison is intended to bind the exact planned learning to its saved
states before interpreting them. Python container identity is not that contract:
the writer's canonical metadata digest is. This failure says nothing new about
personality behavior; 580417's negative learning result remains unchanged.

## 3. Trace architecture

`cumulative_bindings.bind_parent_scope` returns tuple-valued retained batches;
`conditional_residual_experiment.make_plan` deep-copies those bindings. The
worker writes canonical plan JSON. The experiment JSON-roundtrips snapshot
bindings but stores the original plan in `experimental.pt`. `_export` separately
projects that result into a canonical JSON manifest. Two original reader
comparisons crossed these formats without normalization.

## 4. Locate the complete affected boundary

Both final-plan versus protected-plan and snapshot-plan versus final-plan need
canonical metadata equality. The recipe and manifest-arm comparisons were
already canonical. Configuration/cursor contain scalars; snapshot runtime comes
from the same normalized binding. Exact events, model and optimizer comparisons
are tensor-export to tensor-export and must retain strict types and tensor bytes.
An independent full comparison audit found no further unsafe format crossing.

## 5. Preserve responsibilities

The reader establishes provenance and measurement fidelity. It cannot rewrite
the training plan, reinterpret targets, change source authority or promote a
model. File hashes and source pins remain mandatory. Normalizing JSON metadata
must never normalize tensors, optimizer parameter order or changed scientific
values into an apparent match.

## 6. Coherent correction

Create a separately named r2 worker; preserve the failed worker and submission.
Validate the protected plan and entire projected final manifest canonically,
then compare canonical snapshot-plan and recipe metadata. Keep exact tensor/
optimizer/event checks. No original writer, state or teaching change.

## 7. Research

The authoritative implementation evidence is the pinned parent-scope builder,
plan construction, `_save_snapshot` and `_export` functions. Their distinct JSON
and torch formats define the correct comparison. Earlier Flamingo/Side-Tuning
research concerns the later learning decision and does not justify changing a
storage contract. This is a format correction, not an architecture experiment.

## 8. Verification

Exercise real retained-scope construction and actual native two-arm writer
outputs, including nested tuples, recipe betas, protected-plan JSON, final
manifest, final tensors and every snapshot. Reject modified nested scope,
learning rate, ordered groups and parameter order; keep actual Adam and corrupt
moment/state regressions. Verify the failed original worker's exact hash, single
owner guard, source pins and r2 launcher before one new diagnostic submission.
No automatic retry, model update or scientific success is implied.
