# N0 Competency Registry Policy v0.1

**Purpose:** machine-readable definition of the public, identity-neutral competencies that A.L.I.C.E.'s native EIPM semantic substrate must demonstrate before private N1/N2 identity learning.

Registry: `competency_registry_v0.1.tsv`

## Boundary

This registry is **not** a dataset authorization list. Every named dataset is only a candidate until N0-R2 performs exact source/revision/license/terms/provenance review. The current `license_status=PENDING_N0_R2_EXACT_LICENSE_REVIEW` is intentional.

No candidate dataset may enter training merely because it appears in the registry.

## Why gates are not numeric yet

The registry deliberately does not invent fixed accuracy thresholds before baseline runs and human audit. For each competency, N0-R2/R4/R5 must establish:

1. clean/audited evaluation data;
2. human or authoritative-label reference where available;
3. compact encoder baseline;
4. qualified teacher baselines;
5. calibration baseline where the output is probabilistic;
6. perturbation/robustness baseline;
7. contamination check.

Only then should a numeric promotion gate be ratified.

## Evaluation rule

A model does not pass N0 because its mean score is high. Promotion requires a balanced competency vector with no material blind spot in personality-relevant semantics, pragmatics, social/emotional interpretation, literal grounding, uncertainty, ranking, or ACFP/graph alignment.

Particular emphasis is placed on paired competencies that can otherwise trade off destructively:

- pragmatics **and** literal grounding;
- social inference **and** stereotype/shortcut robustness;
- candidate preference **and** tie/plural-policy recognition;
- semantic compression **and** source-span attribution;
- confidence **and** calibrated uncertainty;
- structured-frame use **and** raw-text conflict detection.

## Dataset audit rule

The 2026 audit of SocialIQa, FauxPas-EAI, and ToMi is treated as a general warning: benchmark popularity is not evidence of validity. Candidate datasets must be sampled and manually inspected for ambiguity, duplication, implausible answers, label artifacts, and surface-form shortcuts before being used as gates.

## Contamination rule

`contamination_key` is a stable logical key. N0-R2 will map it to exact dataset file hashes / example hashes and will prevent those held-out examples and close derivatives from entering:

- public pretraining mixtures;
- teacher-generated curricula;
- tokenizer-development evaluation samples where leakage would matter;
- architecture tuning sets intended to remain final held-outs.

## A.L.I.C.E. private data rule

Private E0/E-INF/A-SYN material is **not** part of this registry. N0 must remain identity-neutral. Private data enters only after N0 target freeze and the final pre-weight review.

Later private train/eval splits must be grouped by underlying evidence/behavior family, not random rows, so E0 support clusters, A-SYN base policies, targeted gap families, and mechanical variants cannot leak across partitions.
