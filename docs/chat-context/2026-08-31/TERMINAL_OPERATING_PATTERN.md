# A.L.I.C.E. Terminal / Recovery Operating Pattern

This is a contextual operating note for future A.L.I.C.E. working chats. It records the terminal discipline repeatedly established during Stage G so a future chat does not reintroduce already-known failure modes.

## Windows terminal authority

The owner's active terminal environment is Windows PowerShell from `C:\A.L.I.C.E-main` with the Vault at `C:\ALICE_Vault`.

For released PowerShell artifacts, the historical safe pattern is:

1. pin the exact downloaded artifact by SHA-256;
2. clean-extract when the artifact is a ZIP;
3. locate exactly one expected launcher;
4. let the native Windows PowerShell parser reject the launcher before execution if a `.ps1` launcher is unavoidable;
5. only then execute the validated artifact;
6. preserve diagnostics/checkpoints rather than repeatedly mutating the same failed release.

## PowerShell 5.1 lessons already learned

Stage G previously hit repeated tokenizer/parser failures involving here-strings, multiline expressions, quote-sensitive constructions, and duplicated orchestration logic. Those attempts were made safe by parser gates, but they cost time.

The durable lesson is stronger than "add another parser guard":

- avoid large custom PowerShell orchestration;
- do not embed Python inside PowerShell here-strings;
- do not duplicate state-machine contracts across PowerShell and Python;
- use PowerShell only as a very small launch/verification surface when it is needed at all;
- prefer a versioned Python controller for recovery, validation, Git/Kaggle orchestration, checkpoint handling, and idempotency.

MC10B v1.1.0 independently reached the same conclusion: its deep audit explicitly reduced PowerShell to a minimal parameter wrapper and made Python the sole controller.

## Recovery discipline

A remote/platform error is not permission to restart from zero.

When an A.L.I.C.E. run has produced recoverable evidence:

- preserve exact remote identities;
- preserve local run roots and diagnostics;
- retrieve remote output before cleanup;
- validate checkpoint semantics and hashes before creating a resume pointer;
- reuse the exact frozen release when its own resume contract supports continuation;
- require the official controller to independently validate the resumed rows before spending more GPU time;
- do not overwrite an existing valid checkpoint with an older one;
- do not delete old remote evidence until a successor run has independently accepted the resume state.

## Repository isolation

During the frozen MC10B v1.1.0 run, `main` must remain at:

`0abaed85873c3f8de04765847eb7700b0e20433f`

Chat-history/context archival belongs on the separate `alice-context` branch. Use a detached temporary Git worktree for archive commits so the owner's active `main` working tree is not switched or modified.

## Current continuation tooling rule

The continuation package created after the 2026-08-31 parser incident follows this architecture:

- **no custom recovery `.ps1` file**;
- one Python continuation controller;
- exact embedded MC10B v1.1.0 release bytes;
- exact context archive bytes;
- package-wide SHA manifest;
- clean-extract self-test;
- context upload through a detached Git worktree;
- old 519-candidate checkpoint recovery using the official v1.1.0 resume validators;
- final continuation by invoking the exact v1.1.0 Python controller directly.

Future chats should preserve this design unless new machine evidence proves it insufficient. Do not regress to a PowerShell hotfix chain.

## Windows repository path-budget lesson

The v1.2.0 continuation proved that avoiding PowerShell parser failures is not sufficient by itself. Git for Windows can fail before a commit when a context archive reproduces very deep source-package paths. Context archives must therefore use short repository paths, preserve original names in a manifest, and fail their offline self-test if any archive member exceeds the defined path budget. Do not solve this by globally changing the owner's Git or Windows configuration unless there is a separate reason to do so.
