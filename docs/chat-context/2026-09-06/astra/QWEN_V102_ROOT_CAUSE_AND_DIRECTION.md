# V102: evidence-origin failure and the decision it blocks

## Finding

This stop is a defect in Astra's v102 recovery verifier and its test fixture.
The original collector projects files from two locations into one result ZIP.
V102 incorrectly assumes all critical ZIP members physically reside in the old
run directory. Its tests create that same incorrect layout by extracting the
entire result ZIP into a simulated live run. The native Windows pass does not
validate this assumption.

The supplied terminal file has SHA-256
`818e6ce00c2cc9d3993c99ec503c01fc9eba080614b0b7c8ecf2f9f79336728b`.
It records the exact v102 package, 48 Windows tests passed in 114.520 seconds,
and exit 76 at `Source failure evidence differs: package-manifest.json`.

## Exact producer–consumer trace

| Evidence | Actual producer / physical origin | V102 assumption |
|---|---|---|
| `run.json`, submission records, job script, worker start/finish, result and failure | Original live a1 run directory | Correct origin; eight preceding critical hash checks passed on the reported code path |
| ZIP member `package-manifest.json` | Original `remote_agent.evidence_paths()` maps it to `c.BASE / 'PACKAGE_MANIFEST.json'` | Incorrectly reads `old_run / 'package-manifest.json'` |
| Six `authority/*.json` ZIP members | Original package authority directory | Also projected into the ZIP; they are not a physical run-directory inventory |
| `EVIDENCE_MANIFEST.json` and result ZIP | Collector creates these in the live run | Derived export records; they do not materialize package projections in the run |

The submitted job script in the actual a1 result binds the old worker to:

`/homes/01/mxrayan/rayan-compute/packages/alice-qwen38-a1-8e7a384a745496f4/bc7c76c29a4f0fcfbd4567a610737ec5bb2052e7165677562959ac4dca7c959e/ALICE_MC10D_QWEN_PUBLIC_QUALIFICATION_v1.0.0/worker.py`

The package manifest belongs beside that worker. The expected manifest SHA-256
is `b1d94af265eaac40d29314481358aa0f2a37b072bfa8c5542409fd641387175c`.
The original production collector adds its exact bytes to the result ZIP under
a different, lower-case filename. No original production writer found in the
review writes that file into the live run. The original offline evidence fixture
also copies it into a collected-evidence folder; that is appropriate for an
offline verifier, not a live source fixture.

The original result ZIP was already hash-verified. The new reproduction uses
its bytes, the unmodified original collector and the unmodified v102 verifier.
It derives the seven projected members by calling the real collector on an
empty run, reconstructs only run-origin files, then calls the real collector's
manifest and ZIP-writing path. Scheduler status and already-completed telemetry
are mocked because this experiment concerns file origins. The export passes
v102 verification; the live layout produces the exact reported exception.
All eight run-origin critical hashes match and preexisting run files remain
byte-exact. See `tools/qwen-v102-trace/REPRODUCTION.json` and its reproducible script.

This proves the code/fixture defect. It does not replace a stat/hash observation
of Magnolia. The exception merges missing file, symlink and hash mismatch into
one message. The read-only diagnostic distinguishes these states at both origins.

## Where execution stopped

The v102 call order is: verify/stage package, create or verify the a2 run
descriptor, reconcile an existing a2 job, check dependencies, enter source
preflight, then create the job script and submission intent, then call sbatch.
Within source preflight the critical source-file loop precedes source scheduler
reconciliation, CA selection, runtime HTTPS HEAD and model manifest resolution.

The reported exception comes from that file loop. Therefore this invocation
did not reach its new sbatch call, model resolution, throughput probe or any
calibration task. The log contains no a2 job ID. This is distinct from a1 job
575089, which actually ran and failed at TLS preparation before inference.
Current live scheduler absence is still a fact to observe, particularly if
another invocation happened after this transcript.

The a2 `run.json` is created before this failed check and binds its package to
`f196f3b91604f5c35ca594df001c10a6c1cec87aa2962e544fadc3ba99fa9999`.
The local controller also binds that hash. Merely replacing the workload ZIP
under the same a2 ID produces `immutable evidence differs: run.json`, independently
reproduced with the actual contract. It would not be a sound continuation.

Two adjacent boundaries also need explicit treatment in the eventual repair:

- A failing `squeue` query is currently called with `check=False` and its exit
  code is ignored by source reconciliation. Empty failed output cannot itself
  establish that the source job is absent. The diagnostic records query success
  separately from matching rows and uses an explicit accounting start date.
- Source-evidence location and execution identity must be separate contracts.
  A valid exported hash proves a byte identity; the live locator must bind those
  bytes to the exact old package and submitted job. Neither a renamed copy nor
  deleting the check establishes that relationship.

## Why this work exists

ALICE's README and the accepted audit describe one continuing personal cognitive
system. Learned Elaina-derived identity informs judgment; AMFM learns memory
formation; governed memory preserves source, host, relationship and self history;
missions connect actions to goals and outcomes. The executive and specialist
models may change while identity and continuity survive. Authored or synthesized
possibilities must never become fabricated source-person memories.

The current judge calibration is a dependency for evaluating synthetic identity
training candidates. GLM v186 failed the actual semantic gates, including both
fabricated-history anchors. The owner approved Qwen as the fallback family.
Neither Qwen infrastructure attempt has supplied judgments, so there is no Qwen
semantic score or comparative result to act on.

The dependency remains:

1. Establish infrastructure and identity correctness, then execute only the
   already approved bounded public Qwen calibration when that execution is
   concretely reconciled. The existing CPU throughput gate must still reject a
   workload that cannot fit its six-hour budget. Do not silently change provider,
   quantization, seeds, prompt, gates or model to obtain a pass.
2. If Qwen passes, verify all four exact family/profile receipts and create an
   explicit successor binding. Passing reused public tasks remains calibration.
   It does not provide independent validation or make v186 successful.
3. Migrate the breadth prerequisite explicitly before its execution, and retain
   the separate controls for private pointwise screening, acceptance, promotion
   and training. Preserve the 287 candidates, 63 replacements and deferred item.
4. Test the cognitive hypotheses in E01–E07: learned identity, AMFM, memory
   conservation, mission learning, recollection, model replacement and correction
   propagation. Their execution status remains unmeasured in this continuation.
   Stage G's complete selected deployment precedes G+ qualification and the
   downstream Fable/Friday backend under the accepted ordering.

Preparation of independent evaluation protocols and Stage G deployment comparisons
can continue without treating Qwen as a blocker for every project activity.
However, writing another package does not measure any of those hypotheses.
For YC or competitive claims, later evidence must show useful continuing behavior,
measured failure rates, costs and user outcomes. This incident adds no evidence
of product advantage, and this focused trace does not claim a fresh market audit.

## Bounded next action and repair decision

Use `QWEN_READ_ONLY_TRACE_RELEASE.json` and its launcher to return one diagnostic
ZIP. It checks only the source/package origins, a2 lifecycle, scheduler, CA and
two HTTPS HEAD endpoints. The diagnostic needs the SSH identity already on
Rayan's machine; Astra has no direct Magnolia session here. No new qualification
launcher, model request or remote repair is released in this checkpoint.

The returned facts select the next action:

| Observation | Consequence |
|---|---|
| Source package and eight run-origin hashes match; no a2 intent/job/worker/probe/task; source accounting confirms the closed failure | Design an explicit revision from the preserved pre-submission a2 descriptor, binding both old and new execution bytes; implement a single origin mapping used by live reconciliation and export tests |
| a2 has an intent or scheduler job | Reconcile and collect that execution before any mutation; do not infer a fresh attempt |
| Source hash, package identity, or submitted-worker path differs | Preserve the discrepancy and resolve provenance; do not normalize it into acceptance |
| CA, endpoint, or scheduler query fails | Resolve that observed infrastructure condition before submission; do not spend a compute allocation to discover it again |

The exact revision mechanism remains undecided until the a2 state is observed.
No deletion of descriptors, rewriting of a1, copying a ZIP projection into the
source run, or automatic a3 execution is an acceptable substitute for lineage.
The already recorded Qwen activation does not need renewed approval; a different
scientific plan would require a separate explicit decision.

The diagnostic passed six focused tests from its final ZIP. Its literal
PowerShell Python verifier, SSH Python body and bash syntax were checked on Linux.
The original collector reproduction is separate evidence. Native execution of
this new diagnostic and actual Magnolia observations are pending; the earlier
48-test Windows pass belongs to v102, not to this new tool.
