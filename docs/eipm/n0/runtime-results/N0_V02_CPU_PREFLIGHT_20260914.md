# N0 v0.2 CPU Preflight — 2026-09-14

Status: **PASS**

Command executed on Magnolia login node:

```bash
bash scripts/eipm/n0/magnolia_cpu_n0_v02_preflight.sh
```

The run was CPU-only. It did not request a GPU, did not authorize training, and did not create any private identity gradient.

## Exact construction result

- model: `alice-n0-semantic-v0.2`
- exact trainable parameters: **136,594,435**
- auxiliary semantic/rationale/ranking head parameters: **537,475**
- planned reference: 140,000,000 parameters
- approved construction envelope: 110M–180M
- MLM corruption probability: 0.30
- private identity gradient: false

The exact model therefore lands inside the intended compute-efficient production class. No architecture resize is justified before training evidence exists.

## Fixed readiness suite

- base cases: 43
- competencies: 43/43
- prompt paraphrases per base: 2
- compiled invariance-scored cases: **256**
- eval-only: true
- training-authorized: false

The previously estimated 258 compiled cases was corrected before this run. The exact compiled count is 256 because a pairwise case has only two distinct candidate rotations.

## Teacher state

- governed public teacher rows currently available: 128
- minimum before full principle-aware multi-objective training: 1,000
- private identity data: forbidden in N0

The 128 rows remain useful seeds, but must not be repeated indefinitely to simulate scale.

## Corpus state

At the time of this preflight the corpus plan was not activated. Source rights/schema verification therefore remains a separate gate before tokenizer or training materialization.

After the preflight, the planned mixture was refined so `stackv2_edu_filtered` is treated as code/software rather than educational Q&A, while LibreTexts, PressBooks, and OERCommons provide actual instructional/explanatory material. The activated manifest must bind the refined plan SHA and exact Hugging Face source revisions.

## Decision

1. Preserve this PASS as the architecture construction result.
2. Do not run the old N0 v0.1 step-2500 continuation.
3. Do not allocate a P100 for N0 v0.2 yet.
4. Resolve exact source revisions and inspect row-level `metadata.license` / `metadata.provenance` for the refined v0.2 mixture.
5. Freeze an activated source manifest only after manual license review of the probe output.
6. Build deterministic source-balanced tokenizer/corpus tranches after activation.

No private N1 gradient is authorized by this result.
