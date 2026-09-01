# A.L.I.C.E. Current Working State — 2026-08-31

## Project position

A.L.I.C.E. is still in **Stage G** memory validation. The clarified lifelong-memory architecture (persistent addressable canonical memory plus finite working context and dynamic retrieval) remains the post-G direction. The new retrieval/attention architecture is to be validated/ratified through the planned **G+** work rather than silently replacing the Phase 2 foundation during Stage G.

The current immediate execution frontier is **MC10B full E-INF proposal generation**. This is proposal generation only: E-INF acceptance is disabled, A-SYN generation is disabled, model training is disabled, and neither MC10B nor Stage G is closed merely by generating candidates.


## Context-archive isolation rule

This checkpoint is stored on the dedicated `alice-context` branch. **Do not merge the context archive into `main` during the current frozen MC10B run.** MC10B v1.1.0 pins the canonical repository baseline at `0abaed85873c3f8de04765847eb7700b0e20433f`; keeping archival material on a separate branch prevents a documentation convenience change from invalidating the frozen execution authority.

## MC10B v1.1.0 frozen generation contract

- eligible packets before pilot: 65
- pilot packets already audited: 5
- remaining packets: 60
- generation portfolio: 4 methods × 3 seeds
- raw-candidate target: 12 per packet × 60 = 720
- generator selected by the qualification process: `gpt-oss:20b`
- telemetry block size: 10 packets / 120 candidate obligations
- generation release: `ALICE_MC10B_FULL_EINF_FRONTIER_v1.1.0`
- pinned canonical baseline: main commit `0abaed85873c3f8de04765847eb7700b0e20433f`
- live telemetry/control branch: `alice-mc10b-live`

## Latest durable live state

Run ID: `mc10b-full-20260830T212606Z-ce52f5f7`

Latest GitHub live-state evidence (2026-08-31 about 04:03 UTC):

- status: `PARTIAL_CHECKPOINT_READY`
- checkpoint: `SOFT_STOP`
- controller version: `1.1.0`
- GPUs: 2 × Tesla T4
- model: `gpt-oss:20b`
- primary candidates generated: **519 / 720**
- candidate obligations remaining: **201**
- packet indicator: 45 / 60
- last fully passed telemetry block: block 4
- telemetry gate: `PASS`
- E-INF accepted: 0
- A-SYN generated: 0
- training enabled: false
- failure class/message in worker live state: null

Latest telemetry commit: `ea96dc924b0e3486ffb3356141a918dbeeba6328`, message `FULL_GENERATION_PARTIAL_CHECKPOINT`. Its event records 519 completed candidate obligations and 201 remaining.

## Why the local terminal looked like a failure

The local v1.1.0 controller lost reliable Kaggle kernel discoverability while the GPU kernel was still executing. It exited locally with `ControllerError` after the kernel was not discoverable for more than ten minutes, but intentionally preserved the remote kernel and dataset for recovery. A recovery audit immediately afterward proved the exact remote kernel was still `RUNNING`.

The later GitHub telemetry is stronger evidence of the worker's terminal generation state: the worker reached its configured soft-stop boundary and published a valid partial checkpoint at 519 obligations. So the 519-candidate state must be treated as recoverable computation, not discarded as a failed run.

## Critical resume rule

**Do not blindly rerun v1.1.0 from zero.**

The v1.1.0 controller only resumes when the local Stage-G workroot contains a valid `resume-current.json` that points to an in-workroot `full_result` directory. The old local controller exited before it could retrieve the terminal remote result, so the next step is recovery-only:

1. query the preserved Kaggle kernel;
2. if complete, download its output without deleting anything first;
3. validate the partial result and 519 proposal rows;
4. copy the exact `full_result` into the original local run root;
5. create the canonical local resume pointer only after validation;
6. rerun the **same v1.1.0 package**;
7. require startup to report `local_resume_candidates=519` and remote preflight validation of those 519 rows before generating the remaining 201 obligations.

Expected preserved remote identities:

- kernel: `mkrayanyan/alice-mc10b-full-20260830212734-ce52f5`
- dataset: `mkrayanyan/alice-mc10b-full-20260830212734-ce52f5-data`

Original local run root:

`C:\ALICE_Vault\datasets\memory_stage_g2\alice.stage-g2.g2a.gold-semantic-decomposition.v1\audits\alice-mc10b-full-einf-generation-v1.work\kaggle\mc10b-full-20260830T212606Z-ce52f5f7`

Workroot:

`C:\ALICE_Vault\datasets\memory_stage_g2\alice.stage-g2.g2a.gold-semantic-decomposition.v1\audits\alice-mc10b-full-einf-generation-v1.work`

## After MC10B generation

Generating all 720 raw proposals is not acceptance. Once the frontier is complete, continue the already-defined adjudication/falsification path. Preserve provenance and the existing invariant that external proposal generators have no identity or acceptance authority. Do not begin MC10C or close Stage G until their explicit gates are satisfied.

## Product-context boundary

Fable/Fable Sleight and website/YC planning are related product context, but they do not amend the Stage G technical authority unless a later ratified repository decision explicitly does so.

## 2026-08-31 continuation-tooling correction

A first convenience continuation bundle attempted to place substantial recovery logic in `Recover-and-Resume-ALICE-MC10B-v1.1.0.ps1`. The native Windows PowerShell parser rejected that file before execution. **No recovery or MC10B mutation from that bundle occurred.**

That implementation was discarded rather than hotfixed. It violated a lesson already established earlier in Stage G and explicitly encoded by MC10B v1.1.0 itself: **PowerShell should be minimal; Python should own orchestration.** The successor continuation package therefore has no custom recovery PowerShell script. See `TERMINAL_OPERATING_PATTERN.md`.

As of the fresh GitHub check after that incident, `alice-mc10b-live` still points to the same 519-candidate `FULL_GENERATION_PARTIAL_CHECKPOINT`, and canonical `main` is still the frozen `0abaed85873c3f8de04765847eb7700b0e20433f` baseline. The `alice-context` branch had not yet been created at that point.

## Latest continuation attempt — v1.2.0

The Python-only continuation package passed its compile gate and full offline self-test, including exact MC10B v1.1.0 qualification and the 519-row recovery contract. It then failed **before MC10B recovery began** while staging the context archive in a detached `alice-context` worktree.

The failure was Windows/Git path length. The archive had reproduced the MC10B package inside deeply nested descriptive directories, and `git add -- docs/chat-context` hit `Filename too long` on the future-evaluation handoff JSONL.

No `alice-context` branch was pushed by that attempt. The active `main` tree was not intentionally switched or mutated, and the MC10B live/recovery frontier remains the prior 519/720 soft checkpoint until a successor controller proves a newer state.

Continuation v1.2.1 therefore preserves the Python-only architecture while flattening repository paths and adding explicit archive-member and absolute-staging path budgets before any Git staging.

