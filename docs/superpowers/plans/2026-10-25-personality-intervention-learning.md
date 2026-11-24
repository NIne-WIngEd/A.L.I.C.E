# Native intervention learning implementation plan

> **For agentic workers:** use superpowers:subagent-driven-development for the independent teaching helper and superpowers:executing-plans for the owner runner. Independent code review before dispatch.

**Goal:** train native N2 to use admitted same-context interpretation corrections while preserving natural governing behavior and the full curriculum.

**Architecture:** a typed correction schedule and temporary N1 replay supply the second of two equally weighted native task losses. A distinct grouped runner compares unchanged versus corrected replay using the same native architecture, data, groups and optimizer, with durable experimental progress.

**Tech Stack:** existing Python/PyTorch2.7 native identity/teaching code; owner-isolated Magnolia runtime.

**Spec:** `docs/eipm/PERSONALITY_INTERVENTION_LEARNING_DECISION_2026-10-25.md`.

## Constraints and review focus

Frozen Gemma/N1/N3 and objective parameters; all216 cases/40 groups per arm. Correction admission is additional to native batch admission. Keep unknown/forbidden coordinates, full source and semantic heads, and ordinary inference unchanged. No native checkpoint promotion. All dispatched older sources immutable. Same-value replay must preserve native outputs and gradients; exception paths restore hooks. Partial/full supplied-correction improvements cannot replace natural judgment and retention.

## Task1: explicit correction admission and native replay

Files: new `scripts/eipm/gemma_n0/conditional_teaching.py`, `tests/personality/test_conditional_teaching.py`.

- [x] Define and validate typed same-document TRAIN N1-to-N2 correction record bound to original frame/batch/case/review; independently pin records in owner plan. Retain all annotation provenance; no unknown facts or uncertainty1 admitted as corrections.
- [x] Implement scoped actual-model N1 replacement for only admitted conditional fields; no diagnostic-only API promoted into training. Preserve all masks/latents and restore on exceptions. Execute natural and second native losses separately so peak gradient memory is bounded.
- [x] Test real native output/gradient equality for sham; corrected gradients reach N2 only; records reject wrong case/frame/mask/unknown facts; unknown coordinates and rich state preserved; hooks cleaned on mid-forward exception.

## Task2: matched durable learning comparison and complete assessment

Files: new `scripts/eipm/gemma_n0/conditional_learning_experiment.py`, owner worker `magnolia_conditional_learning_20261025.py`, matching wrapper/tests.

- [x] Reuse original admission and proven group snapshot mechanics with a new plan/progress schema. Identical groups both arms; complete half-weight natural/replay objectives,40updates/216cases each. Save every completed group and verify resume equality in real public native fixtures.
- [x] Preserve existing owner loader, source/custody/394ancestry checks, original guard once first and nondestructive live checks; independently re-admit80 same-document N1 correction records against existing reviewed N1 batch manifests.
- [x] Assess both learned arms through the full236document/216objective ordinary route and original diagnostic families. Add complete broader full/partial correction contrasts without counting them as natural inference.
- [x] Review exact sources, test affected native mechanics and governance, pin owner plan and source files, submit once under mxrayan. Preserve all failed evidence, update FBM and existing PR100/handoff, then stop at the result dependency.
