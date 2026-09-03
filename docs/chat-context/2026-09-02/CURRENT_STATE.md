# A.L.I.C.E. Continuation Checkpoint — 2026-09-02 / v1.3.1

**Purpose:** contextual reconstruction for future A.L.I.C.E. working chats. This archive is not canonical execution authority. Ratified repository files, frozen manifests, validators, and preserved Stage G evidence win if anything here conflicts with chat history.

## Stable repository authority

- Canonical `main` remains frozen at `0abaed85873c3f8de04765847eb7700b0e20433f` while the present Stage G/MC10 evidence chain is open.
- Chat archives belong only on `alice-context`; never move `main` to archive chat material.
- Existing `alice-context` parent for this checkpoint is `6e8c9f236e34a73032155a8f34b1de24ff34acd5`.
- Intentional live/evidence branches remain separate from `main`.

## Recovered terminal workflow — this is the operating pattern

The proven successful continuation pattern is **not** a `.pyz` workflow. It is:

1. Stay in the existing Windows PowerShell terminal at `C:\A.L.I.C.E-main`.
2. Use a **versioned ZIP** with an exact SHA-256.
3. PowerShell verifies the ZIP, removes only the disposable extraction directory, and clean-extracts the package.
4. Resolve the plain Python controller from that extraction.
5. Run `python -m py_compile` before execution.
6. Run the controller's full offline `--selftest`.
7. Only after all gates pass, run the same Python controller live.
8. The Python controller owns Git/Kaggle reconciliation, checkpoint validation, idempotency, and state transitions. PowerShell does not duplicate that state machine.
9. When a tiny released `.ps1` launcher is used for later MC10D packages, invoke it with `powershell.exe -NoProfile -ExecutionPolicy Bypass -File ...`; the launcher performs package/freeze hash checks, clean extraction, Python compilation/selftest, then calls the Python controller and classifies exit codes.
10. After any deterministic or reconciliation stop, preserve the complete terminal output, Vault workroot/checkpoint, and remote identities. Do not blind-rerun, manually clean remote evidence, or hotfix the frozen science.

## PowerShell / Windows failure classes already solved

- Do not embed Python inside PowerShell here-strings or build large quote-sensitive recovery scripts. Those caused tokenizer/parser failures.
- Do not paste explanatory/output lines back into PowerShell; prior sessions produced `MissingExpressionAfterOperator` and command-not-found cascades from copied `+`, `---`, line-number, and prose lines.
- Unsigned script policy is handled with the one-shot `powershell.exe -NoProfile -ExecutionPolicy Bypass -File ...` invocation, not a permanent policy change.
- With Windows PowerShell 5.1, harmless native stderr (for example Git CRLF warnings) can become `NativeCommandError` under `$ErrorActionPreference='Stop'`; substantive controllers judge native success by process exit code and keep machine-readable stdout separate from stderr.
- Context archives must use short/flat repository paths. v1.2.0 failed at `git add` from Windows path length; v1.2.1 fixed the archive layout and added path-budget checks.
- Do not assume ChatGPT attachments exist in `Downloads`; packages that require those bytes must carry them.
- Do not hard-pin a working-tree text-file byte hash across Git line-ending conversion. Pin Git object/ref authority and canonical repository content instead.
- `docs/chat-context/.gitattributes` must mark binary context artifacts such as `*.pdf` as `-text`; otherwise the branch-wide `* text eol=lf` rule can alter binary bytes.

## Proven recovery behavior

The successful MC10B continuation recovered the terminal output first, preserved it byte-for-byte, wrote a resume pointer, independently validated all 519 recovered candidates with the exact v1.1.0 controller, preserved old remote resources during validation, reverified frozen `main`, and only then launched the exact frozen controller. The same principle governs MC10D: recover/reconcile before new remote submission.

The adaptive tournament also established that repeated Kaggle 404/absent observations can represent a bricked or unmaterialized identity rather than permission to reuse/reset state. Completed/running predecessor identities were preserved; a fresh retry identity was used only after the controller classified that state, with safe pause exit codes rather than blind repush.

## MC10B / MC10C

- MC10B full frontier generation completed: 720 raw E-INF proposals, zero acceptance, zero A-SYN, no training. Do not resume from the old 519 checkpoint.
- MC10C raw A-SYN generation completed: 24/24 packets, 288 raw A-SYN candidates, 24 UNKNOWN competitors, zero accepted/promoted, no training.

## MC10D frozen geometry

Frozen pre-execution input: `ALICE_MC10D_PREEXECUTION_FREEZE_BUNDLE_v1.1.zip`

SHA-256: `22B0ADBCCF442B0B3654F964E35AB77A044AF7623891E2A60C72B06A94ECE9A3`

- raw candidates: 288
- deterministic hard-blocked originals: 63
- retained originals: 224
- soft scope-review flags: 20
- repair obligations: 64
- judge families: Gemma, GLM, Mistral, Granite
- MC8 remains sealed during repair/public qualification
- A-SYN acceptance/promotion/training remain closed

## MC10D transport progression — do not repeat old diagnoses

- v1.2: private dataset visibility assumptions failed.
- v1.2.1: delayed discoverability/title-slug collisions exposed duplicate-submission risk.
- v1.3: Kaggle auto-extracted the repair-source ZIP and deterministic failures were retried too broadly.
- v1.4: auxiliary local module imports were unreliable in Kaggle code-file execution.
- v1.5: single-file transport fixed imports but deterministic ZIP-byte reconstruction across environments was too strict.
- v1.6: exact canonical source ZIP bytes were transported as an opaque base64 blob, decoded and SHA-verified remotely, while preserving the exact frozen v1.2 scientific worker and preflight.

## Exact current scientific stop

MC10D v1.6 package SHA-256: `A9CA1EB7DD05F57A4BF5F9F45CD54EFA975513257FE5F28D6D6C744ED839900E`

The v1.6 transport preflight passed and legitimately authorized the repair GPU. The worker checkpointed **62 valid replacements** and then exhausted all three frozen prompted repair attempts on obligation 63:

- original: `ASYN-F694986B9D502671BCFC9D58`
- facet: `repair_initiation`
- target role: `peer`
- third attempted replacement: `ASYN-R1-4466155EC92CBAA563FCCE40`
- blocker: `RAYAN_SUBSTITUTED_FOR_PEER`

The worker stopped before obligation 64. No A-SYN was accepted/promoted and no model training occurred.

## Current successor boundary — MC10D v1.7 Exhaustion-Aware Repair

v1.7 does **not** weaken deterministic hygiene and does **not** create a fourth repair attempt. It must first independently bind the preserved 62-row v1.6 checkpoint and exact slot-63 failure receipt. Proven three-attempt deterministic-hygiene exhaustion is deferred to the packet's existing UNKNOWN/ABSTAIN competitor; the hard-blocked original remains excluded and earns zero saturation credit.

The only expected unattempted repair obligation is:

- original: `ASYN-FC065716EC22BF8D0174E4F2`
- facet: `loyalty_protection`
- frozen blocker: `BETRAYAL_DIRECTION_REVERSED`

Only that slot may call the unchanged frozen v1.2 repair generator, still capped at three attempts. Technical/model/JSON/runtime failure remains failure.

Expected effective pool:

- 287 if slot 64 succeeds;
- 286 if slot 64 also genuinely exhausts.

All 24 UNKNOWN competitors remain. MC8 stays sealed until repair obligations, judge binding, and actual scenario registry are frozen. A successful v1.7 boundary authorizes only the blinded pointwise scope-judge screen with the completion frontier still open. It does not authorize full simulation/falsification, saturation, training, Stage G closure, canary authority, cutover, or Phase 2 retirement.

## Superseded failed continuation artifacts

- `ALICE_CONTINUE_20260902.py`: obsolete; failed because it searched Windows Downloads for ChatGPT-uploaded context files.
- `ALICE_CONTEXT_CHECKPOINT_v1.3.0.pyz`: obsolete; synthetic selftest failed on a working-tree README byte-hash/line-ending assumption. No live `--apply` occurred.

Do not reuse either artifact.

## Future-chat start instruction

Read `docs/chat-context/README.md`, then this checkpoint, then `MC10D_V1_7_EXHAUSTION_AWARE_BOUNDARY.md`, and use `01_terminal.txt` for exact chronology. For terminal execution, preserve the recovered versioned-ZIP/PowerShell-wrapper/Python-controller pattern above.
