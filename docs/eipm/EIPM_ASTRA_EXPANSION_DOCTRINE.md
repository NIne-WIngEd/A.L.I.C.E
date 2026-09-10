# EIPM Astra Expansion Doctrine

**Status:** Owner-ratified working doctrine  
**Date:** 2026-09-10

## Objective

Use Astra as a temporary development-time generator to expand E-INF and A-SYN coverage before EIPM-v1 corpus freeze, subject to applicable usage terms. Astra output is never automatically canonical and is never E0.

## Generation policy

Astra should be driven by a behavioral coverage map rather than by a fixed row target. For each uncovered or weakly covered personality region, provide the smallest relevant evidence packet and request multiple competing behavioral hypotheses.

Generation should intentionally include:

- plausible Elaina-consistent behaviors;
- plausible but conflicting alternatives;
- conservative/uncertain alternatives;
- explicit historical UNKNOWN where evidence does not support a historical claim;
- A-SYN behavioral priors for runtime initialization when a practical behavior should exist despite historical uncertainty.

The purpose of UNKNOWN is epistemic accuracy. The purpose of A-SYN is behavioral completion. UNKNOWN must not become a permanent behavioral blank when an evidence-constrained starting prior can be constructed responsibly.

## Curation policy

Every generated candidate must be evaluated individually before entering the accepted training substrate. Review should check:

- support from E0 and accepted E-INF;
- contradiction with E0;
- unsupported specificity;
- generic-assistant or sycophancy contamination;
- duplication;
- relationship-context errors;
- public/private or intensity-context errors;
- historical-Elaina versus current-Alice confusion;
- provenance class;
- uncertainty calibration;
- whether the candidate contributes genuinely new behavioral coverage.

Rejected plausible alternatives should be preserved where useful as hard negatives for later EIPM judgment training.

## Global consistency

Individual plausibility is insufficient. Accepted candidates must also undergo a global consistency and coverage pass before corpus freeze. The pass should detect contradictions, overrepresented traits, underrepresented behavioral tails, near-duplicates, and missing combinations across context, relationship, emotional state, stakes, and social setting.

## Continuity

The expanded E-INF/A-SYN substrate gives Alice a broad starting behavioral base. After activation, governed A-EXP and Alice Continuity / Self mechanisms may revise non-E0 behavior as Alice develops through experience. E0 remains immutable.

## Training gate

Astra expansion and curation do not authorize weight generation. The owner final pre-weight review gate remains mandatory before any private EIPM gradient update.
