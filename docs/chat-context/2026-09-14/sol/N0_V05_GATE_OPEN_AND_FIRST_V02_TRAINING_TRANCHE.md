# N0 v0.5 Gate Open → First Native v0.2 Training Tranche

**Date:** 2026-09-14  
**Build branch reference at handoff:** `alice-eipm-v1-build@5eff0f56ec77ee4f68e843917eb1b3e02089808f`

## Owner direction

Stay on the direct build/teaching path. Voice is the primary expected interaction mode, so expressive delivery is part of personality rather than a late TTS concern. Sol may generate needed synthetic teaching/completion directly. Do not restart external-agent or infrastructure qualification loops.

## Public teacher gate

Magnolia runtime audit of the generated v0.5 bank passed:

- 1,020 unique public teacher rows;
- 51 competencies;
- every competency exactly 15 train + 5 dev scenarios;
- 43 core + 8 voice competencies;
- 82 reusable v0.4+ principle tags;
- v0.5 coverage wave = 637 rows = 490 train + 147 dev;
- coverage gate true;
- full multitask gate true;
- failures none;
- private identity data false;
- private identity gradient false.

The historical 2,500-row readiness target remains a later quality target, not a blocker for the first bounded learning tranche. Further teacher expansion should now be failure-driven or target difficult cross-competency behavior rather than blindly filling a quota.

## Voice identity state

Private voice overlay v0.2 remains outside Git payloads and is reserved for N1/N2. Its metadata state is:

- 32 selected E0 voice-relevant anchors;
- 77 curated E-INF voice/style rows;
- 542 existing curated A-SYN voice/communication candidates;
- 96 structured A-SYN delivery policies;
- all 32/32 selected E0 voice anchors represented;
- 52/77 relevant E-INF anchors directly used;
- 0 new E-INF generated only for volume.

N0 remains public and learns generic spoken-expression reasoning only.

## First N0 v0.2 weight tranche

The build now contains the actual multi-objective trainer and single-job orchestration for the first permanent v0.2 public weight update.

Model:
- native random initialization only;
- 136,594,435 parameters;
- 16 x 640 encoder;
- no v0.1 weight initialization;
- 48k governed v0.2.1 tokenizer;
- first sequence length 512.

First segment:
- 500 optimizer steps;
- checkpoint/evaluation at 250 and 500;
- 2 x P100;
- MLM microbatch 1/process;
- 8 MLM microbatches per optimizer step;
- teacher batch 2/process;
- fp16;
- LR 3e-4;
- 100-step warmup within a 10,000-step scheduler horizon;
- objective weights 0.70 MLM / 0.10 preference / 0.10 rationale alignment / 0.10 semantic contrastive.

The trainer keeps DDP gradient-bearing work inside the wrapped model forward. Conditional MLM/teacher paths use unused-parameter detection; the final MLM accumulation microbatch synchronizes before the teacher backward so MLM prediction-head gradients are not lost.

Evaluation inside the same job:
- random-init baseline before optimizer step 1;
- fixed core candidate-order/paraphrase invariance suite;
- fixed voice-expression suite;
- deterministic held-out public MLM diagnostic;
- repeats at step 250 and step 500;
- raw MLM loss is diagnostic, not a promotion criterion.

No network acquisition and no private gradient are part of this job.

## Active Magnolia paths

- repo: `$HOME/rayan-compute/rayan-eipm-main`
- public corpus: `$HOME/rayan-compute/rayan-n0/n0-v02/tokenizer-corpus-v0.2.1-offline`
- tokenizer: `$HOME/rayan-compute/rayan-n0/n0-v02/tokenizer-v0.2.1`
- teacher runtime: `$HOME/rayan-compute/rayan-n0/n0-v02/teacher-bank-v0.5`
- first tranche output: `$HOME/rayan-compute/rayan-n0/n0-v02/first-tranche-v0.1`

## Next execution boundary

Before pulling, restore the locally sed-ed audit script so Git can fast-forward cleanly:

`git restore scripts/eipm/n0/audit_n0_v02_teacher_bank.py`

Then pull `alice-eipm-v1-build` and submit:

`sbatch scripts/eipm/n0/magnolia_p100x2_n0_v02_first_tranche.sbatch`

When the job completes, inspect `first_tranche_summary.json`, checkpoint receipt(s), and training log. Continue from the healthy checkpoint; do not restart or requalify passed infrastructure.
