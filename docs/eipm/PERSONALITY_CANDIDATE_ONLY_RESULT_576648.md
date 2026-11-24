# Completed candidate-only diagnostic 576648

Magnolia job 576648 completed `0:0` in 54m36s under `mxrayan`, at immutable
`7b82149b58a75b6a096e57f3fe812e34dd19fa54`. Existing-runtime compatibility passed
16 tests; the actual Torch 2.7.1+cu118 consumer then verified original controls,
the original step-128 sampler checkpoint, and all 256 matched TRAIN updates.
Only the complete closed PUBLIC feature package from 576635 was used. One
explicitly artificial zero source removed variable context; original candidate
banks, pools, seeds and AdamW recipe remained matched. This is a learnable-prior
diagnostic, not a personality repair or a causal account of the full scorer.

| Scorer | TRAIN correct / 56 | DEV correct / 16 | Sentence-unseen DEV correct / 14 |
| --- | --- | --- | --- |
| Freshly trained candidate-only | 4 | 0 | 0 |
| Original full learned | 9 | 1 | 1 |
| Fixed semantic control | 15 | 4 | 3 |

The candidate-only initial model already had the same top-1 counts. Its TRAIN
cross entropy moved from 2.959412 to 2.959038, and DEV from 2.664755 to 2.664519;
these are aggregate matched-row comparisons, not a comparison of different
first/last sampled minibatches. Candidate-only selected a TRAIN-positive
description on all 16 DEV rows, versus 15/16 for the original full model and
11/16 for fixed semantics. Familiarity association and weak candidate-only fit
do not prove that the full model ignores context, that exposure explains its
failure, or that Gemma's personality caused it. Calibration cannot repair wrong
ordering. DEV is already observed developmental evidence.

Static TRAIN pools contain 56 positive and 1,472 negative slots. The actual
verified 256-draw sampler gives 256 positives and 7,160 negatives. All eight DEV
descriptions have zero TRAIN exposure on either loss side. All 16 parameter
blocks had finite observed gradients through all updates; the source-role
anchors had zero loss gradients with the artificial zero source, while AdamW
decay still changed their values. This is expected degeneracy, not evidence of
a disconnected full-model gradient. Export/reload was exact and all 72 candidate
permutations had zero aligned-score difference. Complete banks were unchanged.

## Exact actual evidence

The byte-identical original receipts are preserved in
`evaluation/eipm/gemma_n0/receipts/`:

| Artifact | File SHA-256 | Canonical receipt SHA-256 |
| --- | --- | --- |
| `public_candidate_only_fit_576648.json` (622,088 bytes) | `eb9a234425154ec1260afabd1a165a70bf31a1d6ffd2ee7c408efc28137b88ec` | `d0d35932f0155a987388ffaa1c7882db51008cc8d1d7ba5949c3fa13954d52f4` |
| `public_candidate_only_plan_576648.json` (55,356 bytes) | `901e2d3dccc467947f5c1dd1eeaef17d120fc521a4c4bf51f90cae853ac8b9bb` | `e485905eb75ccf5a28b33b59e7bdba5f312ace0ea7121be02798dc407445e96e` |

Actual exported readout SHA-256:
`b81a98816d63a691043ea077d9555c7c5c42036d6a2cc7ed0fa391e31b3b5b1c`.
Remote evidence is under
`/homes/01/mxrayan/rayan-compute/rayan-personality/runs/public-candidate-only-576648/evidence/`.
Slurm batch MaxRSS was 6,134,344 KiB; the separate process peak was
6,395,555,840 bytes. Neither is publisher model memory.

Root independently checked original byte pins, canonical seals, exact Git code
bindings, all 216 per-row scorer results and 204 aggregate/strata groups,
positive/negative exposure, finite-update accounting and false authority flags.
The local replay does not repeat the 2.7 GB source-byte rehash or the sampler
reconstruction in Torch 2.7.1; those are actual allocated-consumer observations.
The readout tensor was not downloaded for a second local export reload.

No publisher weights were modified or loaded; no publisher forward, private
identity gradient, FINAL payload, installation, Kaggle job or paid GPU was used.
N0 approval, personality qualification, repair selection and Gemma neutrality
remain false. N1-N3/EIPM governing targets are not supplied by relation labels.
The next bounded comparison and its primary-method rationale are recorded in
`PERSONALITY_SEMANTIC_REPAIR_RESEARCH_AND_CONTROL_PLAN_2026-10-02.md`.
