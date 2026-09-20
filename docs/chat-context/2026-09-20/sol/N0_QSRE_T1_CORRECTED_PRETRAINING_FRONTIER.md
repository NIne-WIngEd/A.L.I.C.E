# N0 QSRE T1 Corrected Pretraining Frontier

Date: 2026-09-20

## Authoritative branch

alice-eipm-v1-qsre-t1-executor-training

HEAD:
d62e8935811a2fcce41ddc425a7fb9788f7f0aed

State:
configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.47.json

CI:
35501709024 — SUCCESS

## Why v0.46 eligibility was retracted

No T1 gradient run occurred.

The correction is pretraining validity work, not learned-model failure.

Two issues were found before spending the one allowed P100 run:

1. The v0.1 T1 curriculum allowed operation/family-specific answer-position shortcuts.
2. The previous relation-order forward test could pass because q_step entered every supported node update, even when a node was not on the active structural path.

## Corrected executor mechanics

PATH_FOLLOW now uses an explicit oracle focus frontier.

For each relation step:
- edge must be in oracle support;
- edge relation must match the requested relation step;
- edge SOURCE must be on the current frontier;
- only nodes incident to active edges update;
- next frontier is the set of reached TARGET nodes.

Final PATH_FOLLOW readout is restricted to the final reached frontier.

Dead paths fail closed.

T1 also rejects:
- direct field-only relational support;
- fractional/normalized oracle edge support.

Oracle T1 edge support is exact membership 0/1.

## Corrected curriculum

Contract:
configs/eipm/n0/n0_v02_qsre_t1_curriculum_contract_v0_2.json

Exact deterministic curriculum SHA-256:
155f62e92cf8c6bff71a3b7d8913927e48c950407985db5a164eadaa056fb063

Still:
- 504 rows;
- 9 families;
- TRAIN + DEV only;
- no private identity;
- no TEST.

Now causal pairs explicitly cover:
- SOURCE vs TARGET;
- requested relation A vs relation B;
- plural SOURCE vs plural TARGET;
- r1->r2 vs r2->r1 on the same graph;
- reliability swap;
- temporal swap;
- FALLBACK vs DEFER;
- outside-support distractor metadata intervention with invariant target.

Field and edge order are deterministically permuted by causal group.

## Real-cache preparation

Source cache:
query-edge-setwise-router-v0.1/training-preflight-v0.2/setwise_query_edge_train_dev_hidden_cache.pt

SHA-256:
5bcc4ecf0fb2b776f738e4cca881e6b03a68c81a96923a5b08821930d0ebe823

Only field_semantic is reused.

TRAIN fields map only to source TRAIN representations.
DEV fields map only to source DEV representations.
Paired causal rows receive bit-identical representations.

No semantic re-encoding is needed.
No parent global field scores enter T1.

## Trainer plumbing

Trainer:
scripts/eipm/n0/train_n0_v02_qsre_t1_executor_v0_1.py

Contract:
configs/eipm/n0/n0_v02_qsre_t1_training_contract_v0_1.json

Current contract intentionally has:
gpu_training_authorized=false

The trainer therefore refuses execution at v0.47.

## CI evidence

Run 35501709024 passed:
- syntax;
- v0.47 governance;
- corrected path/frontier mechanics;
- static T1 causal isolation;
- paired v0.2 curriculum;
- fake-cache preparation;
- trainer import;
- CPU backward smoke.

## Current next action

Run exactly one Magnolia CPU/no-gradient real-cache preparation.

Do not submit P100 yet.

Only after the real-cache preparation receipt passes may a new state authorize exactly one T1 P100 run.
