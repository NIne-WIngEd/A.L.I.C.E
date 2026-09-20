# N0 QSRE reality audit implemented and CPU run authorized

**Date:** 2026-09-20  
**Experiment branch:** `alice-eipm-v1-qsre-t2-operator-learning`  
**Authoritative state:** `alice_n0_latent_pool_stage_state_v0.56.json`

## Why this batch exists

Job 575934 exposed a correlated implementation/validator defect: malformed doubled-dollar Bash parameter expansion existed in the recovery scripts and the CI expected the same malformed source spelling. The recovery path remains revoked and fail-closed.

The follow-up work is intentionally one coherent CPU/no-gradient audit rather than another train/fix loop.

## Static qualification

Workflow:
- `n0-qsre-reality-audit-static-contract`
- run `35536054506`
- conclusion: SUCCESS
- qualified source core: `31d758b7d1734911f332375f0c264f35cf193abc`

Initial workflow attempt `35535990783` failed only because the GitHub runner imported QSRE modules before CPU torch was installed. It produced no model evidence and did not alter scientific source.

## Implemented audit tracks

1. **Curated realistic public operator corpus**
   - 56 manually authored public-N0 rows
   - natural entity-rich phrasing
   - six seeded relations
   - SOURCE/TARGET reversals
   - path composition
   - plurality/aggregation
   - reliability and temporal arbitration
   - ordinary non-relational queries
   - underspecified relational queries
   - private identity data: false
   - training_allowed: false

2. **Shallow shortcut attacks**
   - majority
   - length-only
   - family determinism diagnostic
   - query-only unigram Naive Bayes
   - lexical cue heuristics
   - evaluates T2 TRAIN -> T2 DEV and T2 TRAIN -> realistic audit distribution

3. **Frozen semantic layer sufficiency**
   - exact frozen N0 semantic checkpoint
   - all 17 hidden states
   - no-gradient nearest-centroid probes
   - relation-first, relation-sequence, role, operation, control, and full signature
   - reports best-layer accuracy and normalized lift over majority
   - paraphrase versus operator-counterfactual cosine diagnostics

4. **Behavioral runtime parity**
   - executes Bash parameter expansion rather than grepping source
   - verifies repo/workdir/PYTHONPATH transfer inside actual uDocker runtime
   - imports QSRE modules
   - executes superseded recovery scripts and requires fail-closed exit 90

5. **T2 -> frozen T1 totality fuzz**
   - 480 combinations in one batched executor call
   - 3 controls x 5 operations x 4 roles x 4 relation patterns x focus present/absent
   - impossible relational PATH_FOLLOW without focus must be scored invalid and canonicalized for execution
   - no provisional learned operator combination may crash evaluator execution

6. **Single precommitted architecture decision**
   - decision rules frozen before Magnolia semantic-probe output
   - possible outcomes:
     - REOPEN_FROZEN_SEMANTIC_REPRESENTATION_BEFORE_T2
     - KEEP_QSRE_T2_OPERATOR_FAMILY_REBUILD_T2_CURRICULUM_FROM_REALISTIC_PUBLIC_LANGUAGE
     - KEEP_CURRENT_T2_OPERATOR_FAMILY_PREPARE_ONE_REALISTIC_T2_EXPERIMENT
     - T2_OPERATOR_INTERFACE_FIRST_PRINCIPLES_REVIEW
   - no outcome automatically authorizes training

## Magnolia runtime

One CPU node job only:
- partition: node
- CPUs: 8
- RAM: 32 GB
- time: 4 h
- GPU: false
- output root: `$ALICE_N0_WORKDIR/qsre-reality-audit-v0.1`
- output root may not be deleted/reused for a rerun

Submit from repository root:

`sbatch scripts/eipm/n0/magnolia_cpu_n0_v02_qsre_reality_audit_v0_1.sbatch`

After completion, inspect:
- scheduler/accounting
- stdout
- stderr
- final combined decision
- all six output hashes

No T2/T3/GPU/TEST/private-identity work is authorized before that result is interpreted.
