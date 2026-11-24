# Actual public semantic learning result: Magnolia 576600

This experiment completed under `mxrayan` in 1h 54m 20s, with immutable code
`f85a9833c5f5af6b8359bd63f4391be7e4db1744`. It trained a first-party readout
above fixed pretrained Gemma features. It did not train Alice's identity,
modify publisher weights, open FINAL data or qualify personality N0.

The complete original receipt is preserved at
`evaluation/eipm/gemma_n0/receipts/magnolia_576600_public_semantic_experiment.json`.
Its exact file SHA-256 is
`12b583048ddf343e5724b92678b43bc7e247b46aa5f016653f9825e7da0d49df`;
canonical receipt SHA-256 is
`dd34d72006cbbfe86b748ad34e34dfe95dd1ff74631f36bc9345c4722410429f`.
The original precommitted plan file SHA-256 is
`6b565c38f969ceee2143f28fc29f760ec84c6f2c8cdc53dc29fd953ac3f93270`.

## What the result establishes

All 152 planned complete inputs produced genuine detached BF16 banks of all
49 states at width 3840. The longest input had 130 tokens, with no truncation.
Logical feature bytes were 2,790,036,480; this excludes serialization and runtime
allocations. Actual Linux process peak RSS was 25,803,636,736 bytes, including
the source-model stage; it is not GPU memory or Slurm's sampled batch maximum.

The learned readout has 291,442 parameters. First-party optimizer and both
recorded CPU RNG domains resumed exactly at step 128 through step 256;
export/reload reproduced the result. Candidate permutation was equivariant on
the 16 checked DEV rows. These are execution and interface findings.

| Readout | TRAIN correct / rows | DEV correct / rows | Sentence-unseen DEV correct / rows |
| --- | --- | --- | --- |
| Untrained learned architecture | 3 / 56 | 0 / 16 | 0 / 14 |
| Fixed semantic control | 15 / 56 | 4 / 16 | 3 / 14 |
| Trained readout | 9 / 56 | 1 / 16 | 1 / 14 |

The trained mean cross entropy decreased relative to the untrained architecture:
TRAIN 2.9594 to 2.4240, DEV 2.6648 to 2.5455. Top-choice accuracy still trails
the fixed semantic control. There is some optimization progress, but this
operating recipe does not establish adequate fit or semantic transfer. It is
not evidence that Gemma features are unusable, and it does not justify approving
N0 or adding larger architecture without diagnosis.

## Limits and next measured decision

The deterministic subset uses one TRAIN row for each of 56 relation families,
and two DEV rows for each of eight different families. Selected TRAIN candidate
pool sizes were 4/8/16/32/56; selected DEV sizes were 3/7/12/48. This subset did
not cover every candidate count admitted by the source. The full source's
normalized sentence-only TRAIN/DEV overlap is 149, and two selected DEV sentences
occur in original TRAIN. The separate 14-row sentence-unseen result retains the
complete source metric rather than dropping inconvenient rows.

Each of the eight held-out target descriptions is absent as a TRAIN positive,
while DEV pools can contain descriptions that were TRAIN positives. Preference
for familiar descriptions is therefore a concrete shortcut hypothesis requiring
measurement. The uniform-random expectations depend on each actual pool size;
the small DEV sample does not support a strong significance claim from 1/16.
DEV is now model-selection evidence, not unopened final evaluation.

The helper-wording diagnostic had zero learned top-choice flips on eight rows
per variant. That observation cannot establish removal of Gemma personality:
the learned base predictions were mostly wrong, prefixes change token positions,
and no source/host identity conflict or full personality action was evaluated.

The next step reuses the sealed complete feature banks: reproduce original
metrics, measure deterministic constant-source and within-split source-substitution
ablations, inspect target familiarity and pool/slot strata, and report learned
depth weights. These substitutions are destructive controls with unchanged
annotation, not truthful entity-role reversal or invented inverse-relation gold.
No additional Gemma forwards or private targets are needed for that diagnosis.
A baseline-preserving learned comparator is only a candidate after these checks.

Full personality N0 still needs the separately reviewed role coverage, actual
capability evidence, relevant inherited-judgment interference tests and residual
reporting. N1 conditional meaning, complete N2 judgment and N3 calibration remain
governing parts of the full Alice/Fable v1 build.
