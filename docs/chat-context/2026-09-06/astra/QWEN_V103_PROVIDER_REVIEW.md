# Astra review of Sol's handoff and the actual v103 stop

Parent: `8d67b3d08687aed5dbfd7cd02568255e29e98177`. Sol's master handoff is
retained unchanged. Its uploaded copy differs only by a final newline.
The complete [new terminal](artifacts/S14-terminal-035211.txt) and
[structured review](QWEN_V103_TERMINAL_REVIEW.json) preserve the actual evidence.

## What I got wrong

I did not consolidate and reuse the provider behavior already learned in the
project. I built successive Qwen execution packages around separately implemented
pieces of SSH, TLS, telemetry, evidence origin and scheduling. The known CA and
immutable telemetry requirements were missed. Then the exported evidence layout
was mistaken for the live filesystem layout. These were my execution defects.

V103 repaired the demonstrated evidence-origin defect. It still used a different
scheduler command from the successful read-only host diagnostic. I had real
observation of `squeue -h -u mxrayan -o %A|%j|%T`, but the production source guard
used `squeue -h -j 575089 -o %A|%j|%T`. I did not validate that command against
Magnolia before issuing the release. The test schedulers returned successful
empty output for that query by construction. Sixty passing tests therefore
verified internal behavior while leaving that external assumption unproved.

I also lost the diagnostic evidence. The source guard calls the subprocess
wrapper with `check=False`, then raises a fixed message when its exit is nonzero.
Neither layer journals the complete command result first. The Windows log keeps
the outer SSH response after the inner stdout/stderr has already been discarded.
The earlier read-only trace keeps selected rows and stderr length, not raw streams.
This forced another owner round trip to obtain facts the failing call should
have preserved itself.

The package lineage has also become too coupled. Important old releases belong
in history. Future execution should import a stable provider contract and a
separate scientific workload contract. It should not recursively depend on old
wrappers as live infrastructure. A new version must identify one explicit changed
contract and show its real-provider evidence. Another green fixture suite alone
is insufficient.

## What the new terminal actually establishes

V103's exact release SHA matched. All 60 tests passed on native Windows in
56.284 seconds. The controller printed that it had archived the original local
state, then stopped with `Source queue query failed; absence is unknown` and
exit 76. The handoff reports the remote action as `revise` at
`2026-09-07T03:10:13Z`. That timestamp comes from its quoted controller records;
the raw controller JSONL was not separately attached in this turn.

The released call path supplies useful additional inferences. Before this source
query, `require_unsubmitted` checks a2 lifecycle markers and calls the user-wide
queue query plus a2 accounting. Reaching this particular stop means those checks
returned successfully without finding an a2 job. Source files and their true
package origins were also verified before this assertion. These are inferences
from exact code and the returned stop, not retained raw command receipts.

The failed invocation cannot proceed to its remote revision writes, local active
state update, sbatch or inference. We have not received a post-stop filesystem
inventory. Do not turn that control-flow conclusion into a claim that all current
remote state has been freshly inspected.

## Likely cause versus established cause

The historical-job query is the leading explanation. Slurm's controller can
purge completed job records after its configured retention period; accounting
is the appropriate source for retained historical records.
[Official MinJobAge documentation](https://slurm.schedmd.com/slurm.conf.html#OPT_MinJobAge)
and [official sacct documentation](https://slurm.schedmd.com/sacct.html) support
that distinction. Magnolia's actual retention configuration and the actual
rejection text are not in the returned evidence. A transient or client failure
between the earlier successful call and this call is still possible.

It would be wrong to report `Invalid job id specified` as an observed Magnolia
error. We do not have that stderr. Preserve the nonzero-exit stop. Obtain raw
command results before deciding which provider behavior to change.

## The local revision intent is another real dependency

`begin_local` preserves `controller-state-v102.json` and writes an immutable
`intent.json` before staging or asking the remote agent to revise. The terminal
shows that this stage completed. The expected production v103 intent hash is
`e24f2dd6eb7d725a1099e30abde0c83a26a0bee04928fa47d078d134f67853a8`.
Its actual bytes have not yet been returned.

A replacement package with a different SHA under the same revision ID would
conflict with that local intent before reaching Magnolia. The review-only
[reproduction](reviews/v103-provider/REPRODUCTION.json) invokes the released
functions and demonstrates `immutable evidence differs: intent.json` while
preserving the old state and intent. Its timestamps and scheduler failure are
explicitly synthetic; it does not assert a new actual host failure.

The same reproduction confirms that the released source guard drops the inner
error and never reaches accounting after the nonzero source queue result.
No release source was changed. No network or model call was made.

## Correct next boundary

Follow section 8 of the [master handoff](ASTRA_MASTER_CONTINUATION_HANDOFF.md).
Capture one bounded read-only provider receipt with host/client identity, the
user-wide queue query, exact source/a2 accounting, the failing source-job query
for comparison and raw `{argv, exit_code, stdout, stderr, observed_at}`. Include
the local archived state/intent/acknowledgement and remote descriptor, revision
archives/intent/receipt, submission markers and probe/task markers. These paths
must be inventoried before selecting a repair or supersession transaction.

Do not rerun v103 or produce Qwen v104 to guess around this stop. Do not allocate
compute to obtain scheduler stderr. The owner credentials remain on the Windows
machine, so no live Magnolia result is being claimed in this review.

When that observation identifies the cause, consolidate scheduler invocation,
raw receipts and state interpretation in one reusable provider adapter. A
successful empty active-queue observation and exact terminal accounting must
have explicit, separately tested meanings. A failed query remains unknown.
Release checks should exercise the same read-only provider functions that
production will use. Keep raw diagnostics even when a semantic assertion fails.

## Project direction and preservation

The provider failure chain consumed time without producing a Qwen judgment. The
purpose of this bounded calibration is eligible synthetic candidate evaluation.
ALICE's learned identity, AMFM, host learning, relationship and self-continuity
planes still need direct cognitive evidence. E01–E07 remain unexecuted. The
accepted Stage G → G+ → Fable/Friday sequence and private-state boundaries hold.

Preserve main, MC10B/MC10C evidence, the 287-candidate pool, 63 replacements,
one deferred item, resolved slot64, Gemma/Mistral/Granite receipts and the
complete GLM semantic failure. Qwen remains NOT_EVALUATED. There is no four-family
successor binding, breadth activation, private pointwise result or training.

The current checkpoint files now record this stop. Historical build receipts
and released source remain byte-exact. The master handoff stays the entry point
for Astra-primary / Sol-fallback continuity.
