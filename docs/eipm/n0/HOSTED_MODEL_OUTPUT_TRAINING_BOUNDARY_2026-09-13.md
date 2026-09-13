# Hosted-Model Output Training Boundary — 2026-09-13

**Status:** active lineage-safety rule  
**Scope:** permanent A.L.I.C.E.-native checkpoints and any artifact intended for later Fable/commercial reuse

## Why this rule exists

The current OpenAI terms applicable to ChatGPT/Services restrict use of Service Output to develop AI models that compete with OpenAI, subject to whatever separate agreement or permitted exception may apply to the account. Fable is intended as a consumer AI product, so A.L.I.C.E. should not build a commercializable training lineage that depends on ChatGPT-generated supervision without explicit contractual clearance.

This is a lineage-safety rule, not a judgment about model quality and not legal advice.

## Active rule

OpenAI/ChatGPT/Astra/Sol service output is **advisory-only by default** for permanent model development.

It may be used to discuss architecture, inspect failures, propose methods, or help the owner reason about the project, but it must not be silently inserted as training rows, preference labels, synthetic identity data, or teacher targets for a permanent checkpoint intended for Fable/commercial reuse unless the owner has a separate agreement or verified permission covering that use.

The N0 training code therefore requires an explicit curriculum manifest recording origin, rights/license review, content hash, and `training_authorized=true` before targeted curriculum training can run.

## Approved curriculum origins by default

A curriculum may be training-authorized when its provenance and applicable rights permit the intended use, for example:

- owner-authored material;
- deterministic transformations of appropriately licensed/public-domain source material;
- public datasets whose licenses permit the intended training use;
- outputs produced locally by a self-hosted model under a license/terms path that permits the intended model-development use;
- future native FBM output once FBM itself has a clean training lineage.

Every such corpus still requires its own manifest and source/rights record.

## Sol-authored seed generated on 2026-09-13

A small Sol-authored N0 curriculum seed was created during implementation before the current terms check was refreshed. It is retained only as a historical/advisory design example in Git history. It is **not training-authorized** and must not be used for a permanent Alice/Fable checkpoint under this default rule.

Its useful lesson is structural: semantic teaching should be expressed as context-conditioned candidate comparison with calibrated ties when multiple answers are genuinely supported. Equivalent training rows should be produced from an approved source path.

## FBM implication

The `fable-builder-model` branch may record high-level observable construction procedures for project continuity, but future FBM training data must carry its own origin/rights manifest. ChatGPT process traces are not automatically FBM training-authorized merely because they are useful documentation.

## No-bureaucracy interpretation

This rule is intentionally one manifest check, not a new qualification phase. Once a source/generator path is cleared, the build continues directly. The goal is to keep permanent Alice/Fable weights reconstructible and commercially clean while preserving the mainstreamed development loop.
