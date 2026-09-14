# N0 v0.2 Preflight PASS and Source Activation Next

Date: 2026-09-14

## Durable state

N0 v0.2 CPU-only construction passed on Magnolia.

Exact construction:
- model `alice-n0-semantic-v0.2`
- parameters: 136,594,435
- auxiliary heads: 537,475
- 30% span MLM
- fixed readiness base: 43 competencies
- compiled invariance suite: 256 cases
- current governed teacher rows: 128
- full principle-aware multitask floor: 1,000 rows
- private identity gradient: false
- GPU training authorized: false

N0 v0.1 step-1000 remains the pathfinder baseline. Do not run the old step-2500 continuation.

## Source-plan correction after preflight

Dataset-card review found that `stackv2_edu_filtered` is a code corpus despite its name. It was moved from educational Q&A/explanation to software/code. Actual instructional sources were added: LibreTexts, PressBooks, and OERCommons.

Refined planned mix now has 22 candidate sources and remains `planned_not_activated`.

No source can enter tokenizer/model training until exact HF revision, text/id schema, row-level license and row-level provenance are probed and frozen.

## New implementation

- `scripts/eipm/n0/probe_n0_v02_sources.py`
- `scripts/eipm/n0/magnolia_login_n0_v02_source_probe.sh`
- source probe resolves immutable revisions and samples `metadata.license` / `metadata.provenance`
- probe never authorizes training; exact license strings still require manual review

N0 v0.2 contrastive teaching was also improved: same-principle examples can now be treated as multiple positives rather than accidental in-batch negatives.

## Immediate next action

Run the source schema probe from Magnolia login node after pulling `alice-eipm-v1-build`. Do not use `sbatch` and do not allocate a GPU.

Review the resulting `source-probe.json`; then freeze an activated source manifest conservatively from exact observed license strings and immutable revisions.

After source activation: build deterministic source-balanced tokenizer/corpus tranches. Magnolia should not hold the full low-billions-token corpus resident; tranches must be rebuildable and lineage-bound.
