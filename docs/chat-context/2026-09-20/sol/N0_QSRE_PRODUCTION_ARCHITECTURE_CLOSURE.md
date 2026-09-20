# N0 QSRE production architecture closure

**Date:** 2026-09-20
**Experiment branch:** alice-eipm-v1-qsre-t2-operator-learning
**Architecture contract CI:** 35540416348 SUCCESS

## Correction

Owner identified that the project was still drifting toward a T2 v0.3 -> v0.4 repair chain before N0 architecture closure.

T2 v0.3 was therefore frozen **before any GPU run**. It remains only as an unrun static diagnostic prototype.

Authoritative state:
configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.59.json

All obsolete T2 v0.3 GPU routes now fail closed with exit 90:
- scripts/eipm/n0/launch_n0_v02_qsre_t2_schema_ordered_pipeline_v0_3.sh
- scripts/eipm/n0/run_n0_v02_qsre_t2_schema_ordered_train_v0_3.sh
- scripts/eipm/n0/magnolia_p100_n0_v02_qsre_t2_schema_ordered_v0_3.sbatch

## Why v0.3 was not sufficient

Source review found production gaps independent of the v0.2 benchmark failure:
- fixed relation cardinality in model configuration;
- fixed learned hop slots;
- early argmax destroying operator plurality;
- mutually exclusive operation class even though traversal and arbitration can compose;
- fixed applicability centers;
- continuous operator state discarded before T1 execution;
- T1 relation-ID embedding remains fixed semantic authority;
- type/domain/range information absent from schema grounding;
- schema glosses averaged too early;
- binder interface not closed before operator redesign.

Therefore another T2-only GPU run would not have demonstrated production N0 architecture quality.

## Production QSRE Core v1

Architecture authority:
docs/research/eipm-n0-qsre-production-architecture-closure-v1.md

Machine contract:
configs/eipm/n0/n0_v02_qsre_production_core_v1.json

W1-W15 responsibility map:
configs/eipm/n0/n0_v02_qsre_production_core_w1_w15_map_v1.json

Core requirements:
- runtime-variable relation schema with no learned relation-class table;
- schema token/multifacet semantics plus type/domain/range constraints;
- UNKNOWN separate from STOP;
- one shared iterative relation decoder, no hop-specific learned slots;
- sparse plural operator hypotheses instead of mandatory early argmax;
- composable traversal and arbitration factors;
- continuous applicability/uncertainty;
- continuous operator state reaches executor;
- adaptive zero/one/many support;
- executor consumes semantic relation state rather than fixed relation-ID embeddings;
- shared iterative node/edge execution cell;
- support-local structural readout;
- exact non-relational pass-through.

## Build discipline

The architecture is designed once against the full N0 workload before causal stages execute.

Causal stages remain sequential but use one fixed Production Core v1 interface:
P0 static/CPU closure -> P1 executor -> P2 operator -> P3 binder -> P4 end-to-end QSRE -> P5 fusion.

The eventual Magnolia workflow may chain those stages with afterok dependencies under one owner-side launch; failure stops the chain. No LR/step/width patch loops or owner micro-gates.

## Current authorization

Architecture design and later CPU/static mechanics are authorized.
Optimizer, gradient, GPU, TEST, private identity gradient, T3-before-P2, and production promotion remain closed.

Architecture contract workflow 35540416348 passed.
