# N0 job 575986 — semantic localization v1 ready

**Date:** 2026-09-21  
**Status:** CPU/no-gradient diagnostic implemented and exact-head static-qualified; one owner Magnolia CPU run is next evidence  
**Diagnostic branch:** `alice-eipm-v1-n0-p2a-semantic-localization-v1`  
**Qualified head:** `8180ed500fcf2181d3a70f5a01481d00ebbbb24a`  
**Static qualification:** GitHub Actions run `35627255980` — SUCCESS  
**Source failure:** Magnolia job `575986` — valid P2A model/capability FAIL

## Purpose

The audit does not add another matcher or change N0 architecture. It tests why the final P2A frozen semantic authority collapsed.

It separates:

1. representation deficiency;
2. prompt/readout/task-geometry mismatch;
3. fusion/calibration loss.

## Exact evidence reused

- implementation source: `alice-eipm-v1-n0-frozen-semantic-authority-v3@0a8ac74aa6fa73259c03dc4d8754a42a6fde2307`;
- semantic checkpoint SHA: `6c2706984c0e05123c4d88ba456788fbd4c9e7f0fca41d43d9e575ac6e53bf43`;
- preserved job-575986 P2A result:
  - Production core: `0.4068181812763214`;
  - auxiliary seen: `0.2708333432674408`;
  - auxiliary holdout: `0.2604166567325592`;
  - held-out factor macro: `0.36666667064030967`.

The runner refuses to proceed if those metrics or the semantic checkpoint hash drift.

## Diagnostic matrix

The exact job-575986 examples are evaluated with:

- current P2A meta-prompt joint preference;
- teacher-native prompt/candidate joint preference;
- current semantic projection;
- current P2A principle alignment;
- teacher-native prompt/candidate principle alignment;
- current deterministic mixed-layer token evidence;
- every frozen hidden-state output separately with the same parameter-free local token matcher.

Per task it emits component accuracy, confusion matrices, layerwise token accuracy, best-layer diagnostic upper bound, current four-surface accuracy, teacher-native four-surface accuracy, teacher-native + best-layer upper bound, and an any-component per-row oracle upper bound.

The oracle quantities are explicitly diagnostic-only and cannot become production selectors.

## Governance

The package is:
- CPU-only;
- zero-gradient;
- optimizer-free;
- no model training;
- no TEST;
- no threshold changes;
- no automatic rerun;
- no Production P2 authorization;
- no semantic-backbone retraining authorization;
- no private identity gradient.

The new evidence root is:

`$ALICE_N0_WORKDIR/qsre-n0-p2a-semantic-localization-v1`

It must not already exist.

## Static qualification

Run `35627255980` passed:

- Python/Bash syntax;
- CPU-only and zero-gradient contract;
- exact job-575986 failure binding;
- all three localization hypotheses;
- anti-loop/no-hotfix contract.

## Next scientific action

Submit exactly one Magnolia CPU job from the qualified head.

After it finishes, interpret the first complete result before any architecture or training change.

No owner GPU run is authorized by this package.
