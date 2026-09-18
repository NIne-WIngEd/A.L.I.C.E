# N0 v0.2 Production Build Plan

**Date:** 2026-09-14  
**Status:** implementation active; GPU training held until CPU/data gates pass  
**Replaces as production candidate:** N0 v0.1  
**Preserves:** N0 v0.1 step-1000 as pathfinder baseline

## Decision

N0 v0.1 proved the end-to-end training path, checkpoint lineage, public/private boundary, frozen semantic probes, and failure-driven teaching loop. It also exposed the production bottleneck: a ~352M random-init encoder was receiving too little and too narrow public pretraining for its size.

N0 v0.2 keeps the EIPM architecture and changes the production semantic foundation to spend scarce compute on more useful learning per GPU-hour.

No N0 v0.1 checkpoint initializes N0 v0.2. N0 v0.2 remains native random initialization.

## 1. Architecture

Canonical config: `configs/eipm/n0/alice_n0_semantic_v0.2.json`

Reference design:

- 16 bidirectional transformer blocks;
- hidden size 640;
- 10 attention heads, 64 dimensions per head;
- FFN size 2560 with GeGLU;
- pre-norm;
- RoPE;
- alternating local/global attention;
- 512-token local window;
- full attention every fourth layer;
- 8k position capacity;
- tied token embeddings;
- 48k byte-fallback BPE;
- expected parameter class ~140M, with the exact count required from CPU model construction before any GPU job.

Why this operating point: the EIPM is a semantic/judgment representation and policy component, not a world-knowledge generator. Under 2x P100, this native backbone buys substantially more token exposure and better capability gain per GPU-hour than spending the same budget on unused dense width. This is an operating point for the current semantic checkpoint lineage, not a capability ceiling.

### Capability-first architecture contract

N0 is being built as the full production personality-model foundation, not as a deliberately reduced pilot model. The complete N0 path is heterogeneous: semantic representation -> structured state -> evidence graph -> source-anchored cross-context fusion -> adaptive multi-view latent workspace. Diagnostic experiments must exercise that real path unless the experiment is explicitly about one isolated component.

No current number in this document is a permanent limit on model capability. Hidden width, depth, context capacity, attention pattern, expert count, graph width/depth, latent slots, memory capacity, or module count may expand when capability evidence requires it. The current values are resource-aware operating points for the active lineage.

Architecture selection is capability-first and frontier-informed. Prefer architectures and kernels that improve quality per unit compute, memory bandwidth, latency, or serving cost without removing representational capacity. Efficiency is an optimization objective only among designs that preserve the required personality fidelity and capability. Compute scarcity is not permission to make a permanent architectural compromise.

The semantic backbone also is not required to remain a generic transformer if a stronger encoder or hybrid design becomes justified. Future revisions may introduce better attention layouts, conditional computation, recurrence/memory, or other frontier mechanisms. Such changes must preserve provenance, public/private boundaries, and reproducible evaluation.

## 2. Tokenizer

N0 v0.2 gets a new 48k byte-fallback BPE trained from the broader v0.2 public mixture. The v0.1 tokenizer remains part of the v0.1 lineage and is not silently reused as the production tokenizer.

Tokenizer rules:

- public N0 data only;
- no private identity text;
- source-balanced sampling;
- no single source above 12.5% of tokenizer samples;
- same special-token IDs: PAD=0, UNK=1, CLS=2, SEP=3, MASK=4;
- receipt must bind source manifest, tokenizer artifacts, Git revision, and private=false.

## 3. Public corpus

Plan: `configs/eipm/n0/public_corpus_v0.2.plan.json`

The mixture deliberately adds reference, research, educational Q&A, conversation/spoken text, long-form books, software discussion, government/legal deliberation, and news/expository material.

The plan is intentionally **not activated yet**. Every candidate source must first have its exact dataset revision, split, text field, row-level license field/allowed values, provenance field, and ID field verified and frozen into an activated manifest.

No convenience fallback is allowed that drops row-level rights/provenance checks.

The corpus pipeline should prepare enough unique text to support a low-billions-token regime if learning curves justify it. A low-billions number is capacity planning, not a success target. Training stops or changes whenever fixed generalization gain per GPU-hour stops justifying continuation.

## 4. Learning objectives

N0 v0.2 is not an MLM-only production design.

### 4.1 Span MLM

- 30% dynamic span corruption;
- mean span length 3;
- maximum span length 10;
- initial base weight 0.70.

This remains the broad language/context objective.

### 4.2 Candidate preference

- initial base weight 0.10;
- listwise preference with valid ties/plurality;
- trains the generic operation later required by N2: compare plausible interpretations/actions/responses without forcing one winner when multiple are supported.

### 4.3 Principle/rationale alignment

- initial base weight 0.10;
- rationale text is gradient-bearing;
- every preferred and nonpreferred candidate can be paired with the governing rationale;
- preferred candidates form positive candidate/rationale pairs and unsupported candidates form negatives.

Implemented in `src/alice_personality/n0/v02_objectives.py` and represented in `src/alice_personality/n0/v02_model.py`.

This corrects the v0.1 weakness where rationales existed in JSON but were not part of the trained input/objective.

### 4.4 Semantic contrastive objective

- initial base weight 0.10;
- symmetric contrastive alignment between semantic examples and their governing rationale/principle embeddings;
- initial temperature 0.05.

The objective should improve semantic geometry and paraphrase robustness rather than relying solely on token reconstruction.

### 4.5 RTD

Replaced-token detection remains an ablation candidate only. It is not active in v0.2. It may be introduced only if a controlled comparison demonstrates better fixed-capability gain per GPU-hour than the simpler objective mix.

## 5. Training phases

The objective mix is a production target, not permission to repeat the current 128 public teacher rows indefinitely.

### Phase A — native public foundation

Start from random initialization with the activated broad public corpus and v0.2 tokenizer. Train 30% span MLM at sequence length 512. This phase establishes broad language/context representations while the governed teacher bank is expanded offline.

### Phase B — principle-aware semantic shaping

Begin full multi-objective training only after at least 1,000 distinct governed public teacher rows exist. Target at least 2,500 high-quality principle-bearing rows before N0 readiness review. Continue expanding by failure category, not by mechanically paraphrasing the same easy pattern.

Use the 0.70 / 0.10 / 0.10 / 0.10 objective weights as an initial mixture, then change them only from fixed-suite evidence.

### Phase C — context-length growth

Do not pay long-context cost early when it does not improve capability. Move from 512 to 1024 and later 2048/4096 as measured operating points. Those values are not a maximum context ceiling. Continue extending context or add more efficient memory/retrieval mechanisms when the fixed suite shows capability that depends on longer or persistent context.

## 6. Fixed evaluation

Base suite: `evaluation/eipm/n0/n0_v02_fixed_readiness_base_v0.1.jsonl`

Compiler: `scripts/eipm/n0/compile_n0_v02_fixed_eval.py`

The initial frozen base has exactly one unseen content case for each of the 43 N0 competencies. Each base case includes an independently worded prompt paraphrase. The compiler combines:

- 43 content cases;
- 2 prompt forms per content case;
- up to 3 candidate-order rotations, limited by the number of candidates in that case.

The current v0.1 base compiles deterministically to **256 fixed scoring instances**. Most cases produce six variants; the pairwise RANK-01 case produces four. The compiler and CPU preflight compute this count from the benchmark rather than hard-code it.

The suite is explicitly eval-only and training-authorized=false. It must never be added to the teacher corpus.

Evaluation must report:

- overall top-1 supported accuracy;
- supported-set separation;
- mean margin;
- per-competency metrics;
- metrics by answer-order variant;
- metrics by paraphrase variant;
- full-invariance pass rate per base case.

Evaluator: `scripts/eipm/n0/evaluate_n0_v02_fixed.py`

This is the first production fixed gate, not the final N0 benchmark. Before N1 authorization, the benchmark must be expanded with genuinely new content, longer context, multi-turn carryover, ambiguity, calibration, social perspective, and structured-evidence cases. Expansion must create a new benchmark version rather than editing v0.1 after results are observed.

## 7. CPU-only preflight

Script: `scripts/eipm/n0/preflight_n0_v02.py`

Before any v0.2 GPU allocation, preflight must verify:

- config invariants;
- native/random initialization rule;
- 30% masking;
- mixture weights and source ceilings;
- fixed-suite integrity and deterministic invariance expansion;
- current teacher rows all contain rationale supervision;
- exact multi-objective model construction succeeds in the validated container;
- exact parameter count lies inside the approved 110M–180M envelope for this specific v0.2 semantic operating point; this envelope is not a permanent N0 or A.L.I.C.E. parameter ceiling;
- private identity gradient remains false.

Wrapper: `scripts/eipm/n0/magnolia_cpu_n0_v02_preflight.sh`

The preflight itself does **not** authorize GPU training.

## 8. GPU budget

Remote accelerator time is reserved for durable learning and material evaluation only.

Initial budget envelope after data/tokenizer/preflight gates:

- hardware: 2x Tesla P100 12GB;
- allocation unit: maximum 4 hours;
- one allocation = at most 8 P100 GPU-hours;
- first architecture review after no more than 3 durable allocations = 24 P100 GPU-hours;
- every allocation must resume a canonical checkpoint and produce another canonical checkpoint;
- no throwaway pilot checkpoints;
- no repeated runtime/DDP qualification unless a concrete failure invalidates the proven runtime assumptions.

The first durable v0.2 segment is sequence length 512. It must record tokens processed, wall time, throughput, objective losses, checkpoint hashes, and fixed-suite measurements before/after the segment where meaningful.

Continuation is allowed only when capability gain per GPU-hour is material. MLM loss by itself is not sufficient.

If three allocations show strong MLM improvement but little or no fixed semantic/generalization gain, stop adding generic pretraining and revisit data/objective composition. If the fixed suite improves while throughput remains healthy, continue from the same checkpoint.

## 9. N0 readiness for N1

N0 is not ready for private N1 merely because a loss threshold or one ranking score is reached.

Required evidence includes:

- stable gains on the frozen benchmark and its later versioned expansions;
- answer-order robustness;
- paraphrase robustness;
- uncertainty/tie behavior;
- semantic, pragmatic, causal, social, epistemic, ranking, temporal, and structured-alignment coverage;
- no evidence that a tiny head is merely memorizing the teacher set;
- acceptable capability gain per GPU-hour;
- owner review of material failure tails.

Only then does the separate private-gradient authorization question for N1 arise.

## 10. What remains unchanged

The frontier audit did **not** invalidate the EIPM architecture.

Keep:

- EIPM separate from downstream generation;
- N0 -> N1 -> N2 -> N3 stage roles;
- native A.L.I.C.E. weights;
- provenance and public/private boundaries;
- Elaina-derived core identity separate from host, relationship, and continuity state;
- multi-dimensional IDP rather than one scalar reward;
- owner review as final identity-fidelity authority.

N0 v0.2 is a more compute-efficient route to the same mission.
