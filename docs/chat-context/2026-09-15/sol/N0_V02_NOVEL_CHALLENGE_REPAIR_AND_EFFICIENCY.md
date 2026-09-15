# N0 v0.2 novel challenge, targeted repair, and efficiency constraint

Date: 2026-09-15
Actor: GPT-5.6 Sol

## Owner reminder ratified

A.L.I.C.E. must remain interactive. Capability is not sufficient if routine inference becomes slow enough to damage the product experience. The project must also avoid the MC10D failure mode in which external infrastructure, qualification, and repeated validation became the work instead of building A.L.I.C.E.

Ratified rule: infrastructure and benchmarks may guard an imminent model/data/architecture change, but they may not become the project. Repeated qualification without new A.L.I.C.E. capability work is forbidden.

## Novel challenge result

Magnolia job 575588 completed the frozen novel cross-competency challenge.

Step 250:
- top1 = 0.8125
- separation = 0.8125
- full invariance = 0.7916666666666666
- failed compiled examples = 27
- mean abs score on failed examples = 2.6294699237670427

Step 500:
- top1 = 0.7916666666666666
- separation = 0.7916666666666666
- full invariance = 0.7916666666666666
- failed compiled examples = 30
- mean abs score on failed examples = 16.842013311386108

Decision: step 250 is the only authorized repair parent. Step 500 is rejected as a repair parent. It is less accurate on the novel set and much more confident on wrong examples.

Observed repair families:
- ALIGN-01 + TEMP-01 event-role and temporal-state updates
- SEM-04 + RANK-03 entailment/quantifier fidelity and ranking
- PRAG-03 + SOC-03 indirect request and relationship context
- SEM-06 + SOC-02 reference ambiguity and belief update
- VOICE-05 + SOC-04 teasing/sarcasm/tone under social context

The challenge rows remain eval-only and are not training material.

## Actual A.L.I.C.E. work created from the failures

Build branch added a new targeted repair curriculum:
- `data/eipm/n0/n0_v02_targeted_repair_v0.1.jsonl`
- 30 public rows total
- 20 train
- 10 held-out repair dev
- independently authored analogues only; no copied challenge cases

Governance manifest:
- `data/eipm/n0/n0_v02_targeted_repair_v0.1.manifest.json`
- public semantic only
- private identity gradient false
- parameter growth forbidden
- context growth forbidden
- parent pinned to step 250

Repair trainer:
- `scripts/eipm/n0/train_n0_v02_targeted_repair.py`
- starts from the exact step-250 full-model checkpoint
- freezes bottom 12 of 16 backbone layers
- trains top 4 layers plus ranking/semantic/rationale heads
- no new serving-time adapter
- no parameter growth
- no context growth
- 120 optimizer steps, save at 40/80/120
- LR 3e-5
- loss mix: public MLM replay 0.15, targeted repair 0.55, governed teacher replay 0.30
- DDP scheduler horizon corrected with the existing `accelerated_scheduler_steps` rule before any run

Runner and scheduler:
- `scripts/eipm/n0/run_n0_v02_targeted_repair.sh`
- `scripts/eipm/n0/magnolia_p100x2_n0_v02_targeted_repair.sbatch`
- real model training on 2x P100; py_compile is only an immediate in-allocation fail-fast guard, not a separate qualification phase

## Efficiency architecture decision

Policy file:
- `configs/eipm/n0/n0_v02_efficiency_policy_v0.1.json`

Current ModernBERT-style bidirectional encoder is retained for N0 because it remains a strong encoder efficiency/quality Pareto choice. Do not switch architectures merely because newer sequence-model research exists.

Research watchlist, for measured ablation only if N0 becomes a real bottleneck:
- Gated DeltaNet
- Kimi Delta Attention / Kimi Linear hybrids
- Mamba-2 hybrids
- Titans-style learned long-term memory

Fast-path design rules:
- default normal-turn N0 context <=512 tokens
- no routine full-history re-encoding
- retrieve compact relevant state/memory first
- cache stable memory-item embeddings for retrieval
- cross-encode only a small final candidate set, default <=3
- long context is an escalation path, not normal-path behavior
- no external judge committee or iterative self-judge loop on routine turns
- no assumption that bidirectional models support decoder-style prefix KV caching
- P100 is a training route, not a production architecture constraint
- FlashAttention-3 is Hopper-specific and must not be assumed on P100

Provisional product latency targets pending reference deployment hardware:
- warm N0 p50 preferred <=200 ms
- warm N0 p95 max <=500 ms
- routine multi-second or multi-minute N0 latency is unacceptable

Efficiency is a promotion metric alongside capability and GPU-hour efficiency.

## Active build state

`alice-eipm-v1-build` head after this work:
`c16fabe7002a735e63d36f7a3b10b4dacd9db4f6`

Active config status:
`novel_cross_challenge_step250_selected_targeted_repair_ready`

Next compute action:
`targeted_repair_from_step250_only`

No broad continuation from step 250 or step 500 is authorized. No private identity gradient is authorized.

## Exact next execution

From Magnolia repo root after pulling build head:

```bash
sbatch scripts/eipm/n0/magnolia_p100x2_n0_v02_targeted_repair.sbatch
```

This is real A.L.I.C.E. weight work, not another infrastructure qualification pass.
