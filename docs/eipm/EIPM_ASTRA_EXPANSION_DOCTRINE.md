# EIPM Astra Expansion Doctrine

**Status:** Owner-ratified working doctrine  
**Date:** 2026-09-10  
**Clarification:** 2026-09-16 capability-limit audit

## Objective

Use Astra as a temporary development-time generator to expand E-INF and A-SYN coverage before EIPM-v1 corpus freeze, subject to applicable usage terms. Astra output is never automatically canonical and is never E0.

## Generation policy

Astra should be driven by a behavioral coverage map rather than by a fixed row target. For each uncovered or weakly covered personality region, provide the **smallest sufficient evidence packet** and request multiple competing behavioral hypotheses.

"Smallest sufficient" is an efficiency/provenance principle, not a context ceiling. The packet must expand whenever cross-source history, relationship context, temporal context, contradiction, wording nuance, or another evidence dependency could materially change the personality judgment. Evidence may never be truncated merely to fit a preferred packet size, context budget, or generation convenience.

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

**Plausible alternatives remain alternatives.** A candidate that is not selected as the preferred branch is not automatically a negative example. Competing or co-valid branches should remain available whenever evidence supports plurality. Hard-negative use is reserved for candidates that are actually contradicted, incompatible, provenance-invalid, or otherwise wrong for the supervised judgment.

## Global consistency

Individual plausibility is insufficient. Accepted candidates must also undergo a global consistency and coverage pass before corpus freeze. The pass should detect contradictions, overrepresented traits, underrepresented behavioral tails, near-duplicates, and missing combinations across context, relationship, emotional state, stakes, and social setting.

Global consistency must not erase genuine human variability. Context-dependent differences, uncertainty, tension between values, and multiple plausible reactions should be represented rather than normalized into one generic behavior when the source evidence does not justify such collapse.

## Continuity

The expanded E-INF/A-SYN substrate gives Alice a broad starting behavioral base. After activation, governed A-EXP and Alice Continuity / Self mechanisms may revise non-E0 behavior as Alice develops through experience. E0 remains immutable.

No current row count, coverage-map dimension, generation batch size, teacher context window, or corpus freeze is a permanent capability ceiling. If a measured fidelity hole remains or emerges, governed targeted expansion remains available.

## Training gate

Astra expansion and curation do not authorize weight generation. The owner final pre-weight review gate remains mandatory before any private EIPM gradient update. That gate validates provenance and lineage; it does not limit the eventual capacity of the personality model.
