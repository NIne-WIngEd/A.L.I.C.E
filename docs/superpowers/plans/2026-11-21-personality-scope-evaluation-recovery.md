# Recover the completed judgment candidates and finish their assessment

**Goal:** finish the original full learning-scope comparison after a Magnolia
node failure, without repeating any training or selecting a candidate early.
**Architecture:** preserve the November17 worker and all its inputs. A separately
versioned recovery worker reconstructs the exact original admission and plan,
restores the externally pinned final group snapshot through the unchanged
scope trainer, then runs the complete original assessment in a new directory.
**Execution:** native implementation in the existing checkout, independent
read-only review before dispatch. Continuing this existing authorized experiment
requires no new model-design decision from the owner.

## Ordered diagnosis

1. **Evidence validity.** Public stderr reports job582305 cancelled on node002
   at 2026-11-20T11:26:15 due to NODE FAILURE. The complete stdout contains80
   acknowledged updates with snapshot manifests, then200/236 intervention
   documents and180/216 objectives. No final summary exists. The accounting
   query failed because the remote database refused its connection; the live
   controller no longer knows the job. Preserve all four observations separately.
   Missing AGENTS.md/admission.py lookup paths were locator errors, not model
   defects; supplied user instructions and actual teaching/trainer.py apply.
2. **Claim and goal.** The full comparison must establish useful natural
   conditional judgment with complete governance/voice retention. Neither an
   interruption nor completed optimizer updates decide that claim. This supports
   N1/N2 construction before actual-outcome N3, connected entity behavior and
   independent personality/N0 qualification. ALICE/Fable v1, not FloRA.
3. **Architecture.** Both arms enabled the conditional route; only intervention
   fixed inherited N2. Same original parent, complete teaching and recipe. Each
   committed group has an fsynced state plus manifest, whose hash is published
   in stdout. Existing run_experiment can restore completed arms and skip updates.
4. **First departure.** The node died during assessment, before final custody
   closure and sealed comparison. No model error or behavioral pass is evidenced.
   Hardware/administrator cause behind NODE FAILURE is unknown, not guessed.
5. **Responsibility.** Recovery must bind exact model/objective/optimizer/RNG,
   runtime, ordered trainable parameters, original admission paths and full
   metadata. JSON tuple/list normalization is for metadata only; tensors remain
   exact. The partial output files lack a committed external seal and are not
   promoted as reusable completed assessment.
6. **Correction.** Reconstruct exact original plan; use original worker hash,
   authorization ID and read-only original allocation/authorization paths in
   that plan. Pin complete group0080 from stdout. Reject any incomplete cursor
   before trainer execution. Require zero new updates and zero training forwards,
   validate all216 batches, export/reload exact recovered state. Recompute all
   evaluation in a new output; preserve the failed output and original files.
   Retraining discards valid work; partial unsealed reuse adds trust assumptions;
   both are rejected. Do not alter optimizer, model, targets, thresholds or scope.
7. **Research.** [Slurm states](https://slurm.schedmd.com/job_state_codes.html)
   distinguish NODE_FAIL from model failure and completion. [sacct](https://slurm.schedmd.com/sacct.html)
   depends on accounting storage; [scontrol](https://slurm.schedmd.com/scontrol.html)
   is an independent controller query. [PyTorch2.7 serialization](https://docs.pytorch.org/docs/2.7/notes/serialization.html)
   supports tensor/state dictionaries and weights_only loading. Existing project
   progress restore is the applicable implementation, not a new checkpointing
   framework. Native .eval()/inference behavior and exact state validation remain.
8. **Verification.** Fresh original local suite; actual completed native restore
   with optimizer/backward forbidden; exact state/outputs; nonfinal snapshot,
   bad external pin, metadata path and altered bindings rejected. Real retained
   tuple metadata must survive. Recovery worker retains guard-first custody,
   unchanged assessment and original scientific sources. Independent review,
   single submission, source-bound startup, FBM and continuation. Remote recovery
   and complete behavior remain pending until actually observed.

## Implementation tasks

- [x] Preserve public interruption, accounting failures and original progress pins.
- [x] Verify original scope/native/metadata suite against dispatched sources.
- [x] Add completed-only restore and exact-source recovery worker with meaningful
      native tests; retain original worker bytes and full evaluation code.
- [x] Bind new worker, original experiment, final manifest and complete source
      pins; stage once under mxrayan and exclude failed node002 plus node016.
- [x] Review and submit evaluation-only recovery if cluster accepts allocations.
- [x] Record outcome, limitations and next action in FBMv32 and current handoff.

## Acceptance and broader continuation

The recovery measures236 natural documents and216 full objectives per arm,
480 corrected probes,160 disabled probes and30 diagnostic documents. Fixed
inherited weights alone never establish retained active behavior. Require
closing custody, exact export reload and all original repairs/harms/voice
reports before a verdict. No automatic candidate adoption, no N0 approval.
After meaningful conditional repair and retention, continue compatible native
adoption and actual-outcome N3 plus connected Memory/Mission/Experience use.
If behavior fails, diagnose that evidence before any new training decision.

## Pre-dispatch recovery wiring diagnosis

1. Initial completed-restore tests passed, but exercised the loader rather than
   the entire worker's ancestor-path wiring. A review found an actual worker
   local-variable collision. A new test executes the worker's actual ordered
   path assignments and failed on the allocation path; retained as path-red.
2. Recovery must preserve the correct source ancestry, not merely load an
   internally valid tensor file. A loader pass could not establish that claim.
3. The new582305 source_output assignment preceded an existing577714 baseline
   source_output assignment; the latter replaced the former in the same run.
4. The first wrong consumer was the original-admission allocation path. The
   authorization and final snapshot used the same overwritten local variable.
5. The old source_output still owns original native/baseline custody. It must
   remain; this correction concerns the new recovery artifact ancestor only.
6. Give the recovered experiment its own recovery_source_output variable and
   update those three consumers, preserving all original baseline connections.
7. [Python3.11 execution model](https://docs.python.org/3.11/reference/executionmodel.html#binding-of-names)
   defines assignment bindings in the function scope; a with block does not
   isolate that local name. This is an integration defect, not a model or
   checkpoint-format defect, and needs no new framework or training method.
8. Re-run the actual path-wiring regression plus completed native restoration,
   real metadata, corruption refusal, guard-first and unchanged assessment checks.
   Renew wrapper hash and independent review. No remote attempt used bad source.

## Verified local result and dispatch

Fresh original suite:68tests/31subtests passed. Final recovery suite:27tests/
19subtests passed (overlap with the original suite). Four governance validators
passed. Independent review verified all19 source hashes, both full test logs,
the source-pin receipt and launch controller, with zero remaining blockers.
The earlier path-wiring failure and all original worker sources are preserved.

Job582473 was submitted once under mxrayan, excluding node002 and node016,
with oneCPU/120GiB/120hours and no requeue. It was observed RUNNING, stderr
empty, summary absent. Initial stdout was empty; isolation entry, protected
snapshot recovery and behavior are not yet verified. The recovery will make
zero training updates and repeats the complete assessment only after exact
saved-state verification. These are planned counters, not observed results.

FBMv32/trace53 retains v31 and all52 prior trace rows. It records node failure,
unavailable accounting, the recovered external plan/snapshot pins, source-path
diagnosis, completed-only recovery implementation and pending assessment.
Procedure evidence only; no private payload, builder training admission, learned
builder weights, N0 approval or personality approval. Current continuation
records the public result paths and exact resume boundary.

## Exact-byte publication diagnosis

1. The publication guard refused before commit: three public receipt files had
   Windows CRLF bytes while the staged copies had LF. All differences were
   independently verified to be solely CRLF-to-LF conversion.
2. Byte-identical publication matters because the already-submitted source and
   dispatch records bind these exact receipt digests; semantic JSON equality is
   insufficient. Compute source and running work were unaffected.
3. Repository `* text=auto` and system core.autocrlf=true caused conversion.
4. The first divergence was git add, detected before commit by index readback.
5. Git normalization serves editable text; these three already-bound immutable
   receipts must retain their original bytes across checkin and checkout.
6. Preserve their bytes and all original digest references. Add exact-path
   -text attributes for only these receipts, with CR-at-EOL whitespace handling.
   Do not change global Git configuration or rewrite historical dispatch hashes.
7. [Git attributes](https://git-scm.com/docs/gitattributes#_end_of_line_conversion)
   explicitly provides -text for data that must not undergo line conversion.
8. Restage the named artifacts and compare every complete index blob with the
   local bytes before commit; verify committed bytes and remote head afterward.
   The failed publication and per-file byte/hash diagnosis remain in local
   receipts. Future JSON writers should request explicit LF before sealing.
