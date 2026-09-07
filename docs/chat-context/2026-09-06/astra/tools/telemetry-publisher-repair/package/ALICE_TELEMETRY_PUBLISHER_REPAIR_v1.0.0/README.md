# ALICE telemetry publisher repair v1.0.0

The actual Git 1.8.3.1 trace shows cached main/origin-main at ec4d378 and
FETCH_HEAD at 3f42de3. Git fetch origin main did not refresh origin/main before
Git 1.8.4. The old helper then based its commit on stale origin/main. It also
regenerated immutable metadata on retries and suppressed the Git error streams.

The replacement keeps the seven-argument helper interface. It uses an explicit
destination refspec and verifies FETCH_HEAD against the tracking ref. It builds
each append from that fetched parent with a private Git index. No checkout,
local branch reset, force push, or inference is performed. A competing append
is preserved by rebuilding on the next fetched parent. Every Git command gets
raw stdout/stderr, argv, exit status and timestamps in a private remote journal.
The complete spool, including the original metadata timestamp, stays frozen.
An existing three-file connector snapshot can receive its missing wrapper
metadata only if all already-published bytes are identical.

The owner launcher verifies this ZIP and runs local gates. On Magnolia it checks
the old/new helper hash, exact SSH wrapper, expected ledger URL and closed a2
evidence. Seven publisher tests run against the actual host Git using temporary
local repositories BEFORE installation. Changed fixtures stop installation.
The exact original helper is preserved under telemetry/helper-revisions. The
new helper is atomically installed under the existing publisher lock. One closed
a2 terminal snapshot is then published and retried after one second. Both calls
must verify the remote payload and the retry must retain identical spool bytes.

Local Windows and remote a2 controller/result bytes are preserved. The existing
shared checkout HEAD/index/local-main bytes are also checked before/after live
publication. Its local HEAD can remain behind the remote intentionally: future
verification must use the publisher receipt, not a raw push of that cached HEAD.
The helper has a 70-second internal budget, within the existing caller's 90 seconds.

The seven actual-host tests use no model, GPU, scheduler or real GitHub writes.
The subsequent live publisher calls write only the already-reviewed terminal
telemetry to the existing ledger. No new Qwen execution identity, calibration,
private evaluation, training, Stage G acceptance or Phase 2 replacement is granted.
Return RESULT_ZIP even on failure; preserve all helper and a2 state for review.

Primary compatibility source:
https://github.com/git/git/blob/v1.8.4/Documentation/RelNotes/1.8.4.txt
Offline modern-Git fixtures emulate the older fetch mapping only for the legacy
source-only fetch. On actual Git 1.8.3.1 that command runs unmodified.
