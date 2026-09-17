# N0 Endpoint Repair and Downstream Arbitration Handoff

Date: 2026-09-17

## Authoritative build state

Primary personality-model build branch: `alice-eipm-v1-build`.

Current build head: `021c5021a98104b35f9c8e94b19e48d21f25f132` (`repair(n0): stage endpoint-role read repair v0.2`).

The six build commits after the last continuity-branch update were:

1. `21e82e1a819d5081de6a987c1bbae14fff01fb7e` — fresh selector-repair curriculum.
2. `2d3791732b9f90cfadb1fde56ed8dcb286e98b22` — bounded selector-repair trainer.
3. `8578285dfbc7bfd39c503904c187b0fe4799218b` — idempotent prepare/train flow.
4. `153ae658b17258550b1522ac8722d374c3dc5a85` — Magnolia P100 selector-repair job contract.
5. `dc029a63eb6e6a5773eb6c7b650107ff3e4f1c5f` — selector failure localization and scaling policy.
6. `021c5021a98104b35f9c8e94b19e48d21f25f132` — endpoint-role read repair v0.2 implementation/state.

`configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.9.json` is the latest committed stage-state file at this head, but its status predates the later Magnolia endpoint-repair execution. Do not interpret `PENDING_PREP_AND_GPU` as the latest observed runtime state.

## Selector repair result

Seed: `20260917`.

Result SHA-256: `80968a34a9bb7dbf3ffa549b1410f45001451700bbdf6404ad8ec667f89288c6`.

Status: `FAIL_SELECTOR_REPAIR_HELDOUT_STOP_AND_LOCALIZE_NO_AUTOMATIC_HOTFIX`.

Heldout pair accuracy: `0.3958333333333333`.
Heldout row accuracy: `0.4895833333333333`.
Selected failed checkpoint: `step-00000160`.
Selected failed adapter SHA-256: `b5062d121dfae8ab925faed7376185773d1265ffa5c3a83dfe63d72a291b126c`.

Source-role queries were strong while target-role queries collapsed. The next defect was localized to `query_conditioned_relation_endpoint_role_read`. The failed selector adapter was not promoted and was not used as the parent for the endpoint repair.

## Endpoint-role repair v0.2

Training seed: `20260917`.

Parent adapter remained the original evidence-graph specialist step 80. Parent graph remained relation-repair step 80. The failed selector-repair adapter was excluded as a parent.

Trainable scope for the causal repair:
- `source_relation_pool_mlp`
- `directed_relation_pool_mlp`
- `pool_relation_embedding`
- `query_projection`
- `pool_query`

The evidence adapter, semantic parent, and structured parent remained frozen. Message passing and conflict-pool paths were frozen for this causal repair only. These freezes are not permanent architecture ceilings.

Fresh public synthetic curriculum: 8 families, 32 pairs per family, split 20 train / 6 dev / 6 fresh test pairs per family. The prior selector heldout rows, frozen latent challenge rows, and parent-value diagnostic rows were not reused for training.

Operating budget: 200 steps with checkpoints every 40. These are experiment budgets, not capability ceilings.

## Magnolia runtime result

Magnolia job: `575718` on `gpu001`.
Elapsed: `00:01:14`.
Selected checkpoint: `step-00000200`.
Selected graph SHA-256: `3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f`.

Fresh heldout endpoint behavior:
- pair accuracy = `1.0`
- row accuracy = `1.0`
- all 8 endpoint-role families = `1.0`
- mean graph target margin ≈ `+0.8701`

Replay behavior remained top-1 correct (`1.0`). Macro target-support mass moved only slightly from the preserved baseline `0.9831965565681458` to approximately `0.9810`. A stricter replay-preservation floor still caused the stored formal status `FAIL_ENDPOINT_ROLE_REPAIR_HELDOUT_STOP_AND_LOCALIZE_NO_AUTOMATIC_HOTFIX`. A later recovered conversation note reported an additional support-mass gate around `0.89568` versus `>0.90`; because the exact raw field name is not yet repo-bound, preserve that value as conversation-recovered rather than silently converting it into an exact receipt field. Another reported weak replay confidence was `temporal_previous ≈ 0.8612` while top-1 remained correct.

Interpretation: endpoint directionality was repaired. Do not tune the graph merely to cross a soft replay-confidence threshold when ranking is already correct. The next causal question is whether the repaired graph materially improves the already-frozen downstream latent frontier.

## Next experiment: frozen downstream arbitration

Run exactly one paired downstream arbitration:

- baseline arm: original graph parent;
- candidate arm: selected repaired graph `step-00000200` / SHA-256 `3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f`;
- same original/frozen evidence adapter in both arms;
- same frozen latent candidate step `360`, SHA-256 `503d4064df6258d3a1bc0edeae17888cbeb4fb67e07ad7e27042f785ed092ba4`;
- all downstream weights frozen;
- no retraining;
- no graph-threshold chasing;
- no challenge leakage;
- no automatic rerun or hotfix chain.

If the repaired graph improves the frozen latent frontier, it has downstream causal value. If downstream behavior remains materially unchanged despite perfect fresh endpoint discrimination, move the bottleneck downstream into fusion, parent pooling, latent conditioning, or readout. Do not scale latent capacity until the repaired parent signal is proven to reach the latent input and a capacity/expressivity bottleneck is actually localized.

## Architecture research rule

Architecture changes for A.L.I.C.E. must not be generic defaults. Before introducing or expanding a personality-model component, compare the intended mechanism against current frontier work and relevant competing systems in long-term memory, graph/relational memory, personalization, adaptive routing/mixture mechanisms, external-memory agents, and identity/persona modeling. Record which ideas are being adopted, rejected, or modified and why they fit A.L.I.C.E.'s provenance, identity-fidelity, deletion, auditability, and continual-learning constraints. Research evidence informs architecture choice; it does not override A.L.I.C.E.'s ratified provenance and owner-governance rules.

## Branch-continuity rule

After every substantive personality-model step, update the continuity branches before beginning the next model-changing step:

- `fable-builder-model`: record every build/training/evaluation step, seed, parent, checkpoint, hash, result, failure, and lesson. Unknown values must be explicit rather than guessed.
- `alice-context`: maintain a future-chat handoff containing the current scientific state, exact next action, blockers, and interpretation.
- `alice-telemetry`: record machine-readable runtime/status evidence and provenance quality.

Historical MC10 and old feature branches retain their own scientific lineage. Do not force-sync them merely to make branch heads look current. `alice-eipm-v1-downstream-arbitration` is the experiment branch for the next frozen arbitration and currently begins from build head `021c5021a98104b35f9c8e94b19e48d21f25f132`.

No private identity gradient is authorized yet. No production promotion is authorized. N0 is not complete.
