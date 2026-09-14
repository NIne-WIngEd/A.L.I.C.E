# N0 v0.2 Design Implemented — GPU Hold

Date: 2026-09-14

## Decision

The frontier audit redirected the production N0 path away from extending the 352M N0 v0.1 pathfinder. N0 v0.1 step-1000 remains preserved as a baseline and must not initialize v0.2.

N0 v0.2 is now implemented in Git at the design/infrastructure level. No v0.2 GPU training has been authorized or started.

## Implemented on alice-eipm-v1-build

- `configs/eipm/n0/alice_n0_semantic_v0.2.json`
  - native random initialization
  - 16 layers, hidden 640, 10 heads, FFN 2560
  - ~140M reference parameter class pending exact CPU construction
  - 30% span MLM
  - explicit multi-objective weights: MLM 0.70, candidate preference 0.10, rationale alignment 0.10, contrastive 0.10
  - no private identity gradient

- `configs/eipm/n0/public_corpus_v0.2.plan.json`
  - balanced Common-Pile-derived mixture across 8 semantic categories
  - no source >12.5%
  - row-level rights and provenance required
  - plan remains non-active until exact revisions/schema/license values are frozen

- `src/alice_personality/n0/v02_objectives.py`
  - teacher rationale validation
  - rationale compatibility examples
  - binary rationale-alignment loss
  - symmetric contrastive loss
  - normalized weighted objective combiner

- `src/alice_personality/n0/v02_model.py`
  - shared native semantic backbone
  - MLM head
  - candidate preference head
  - semantic/rationale projection heads
  - principle-alignment logits

- fixed evaluation:
  - `evaluation/eipm/n0/n0_v02_fixed_readiness_base_v0.1.jsonl`
  - 43 unseen base content cases, one per competency
  - independent prompt paraphrase for every base case
  - deterministic candidate-order expansion
  - eval-only / training-authorized=false
  - compiler and dedicated fixed evaluator implemented

- CPU-only preflight:
  - `scripts/eipm/n0/preflight_n0_v02.py`
  - `scripts/eipm/n0/magnolia_cpu_n0_v02_preflight.sh`
  - exact model build and parameter count required before GPU
  - preflight explicitly does not authorize GPU training

- public teacher curriculum contract:
  - rationale must supply a gradient
  - current 128 public rows remain seeds
  - full multi-objective phase waits for >=1000 distinct high-quality principle rows
  - target before N0 readiness review >=2500, without filler generation

- compute envelope:
  - 2x P100 only for durable learning/evaluation
  - max 4h allocation units
  - architecture review after <=24 P100 GPU-hours
  - continuation based on fixed capability gain per GPU-hour, not MLM loss alone

## Safety against accidental old-path spend

`scripts/eipm/n0/magnolia_p100x2_n0_semantic_growth.sbatch` now refuses to run by default. It requires explicit `ALLOW_N0_V01_SEMANTIC_GROWTH=1` and is labeled retrospective ablation only.

## Immediate remaining gates before any N0 v0.2 GPU job

1. Run CPU-only v0.2 preflight in the validated Debian udocker runtime and record exact parameter count.
2. Verify/freeze exact revisions, schemas, row-level license values, and provenance fields for the v0.2 source candidates.
3. Acquire/materialize a balanced v0.2 public corpus and train a new v0.2 tokenizer with receipts.
4. Keep the fixed benchmark completely excluded from tokenizer/model/teacher training.
5. Phase A may then begin as durable 30% span-MLM public foundation training at sequence length 512.
6. Full multi-objective principle teaching waits for the governed teacher bank to reach the 1000-row quality gate.

No private N1 gradient authorization has been granted.
