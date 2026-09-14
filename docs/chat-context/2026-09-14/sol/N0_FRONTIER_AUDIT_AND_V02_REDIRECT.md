# N0 frontier audit and v0.2 redirect — 2026-09-14

The owner paused the proposed step-1000 -> step-2500 Magnolia run and requested a deeper comparison against current frontier encoder, personalized-alignment, persona, reward-modeling and evaluation research.

Decision: the high-level EIPM plan remains strong, but the immediate N0 v0.1 continuation is **held**. Do not run `magnolia_p100x2_n0_semantic_growth.sbatch` unless the owner later explicitly reverses this decision.

Why:
- N0 v0.1 is ~352M random-init parameters but has only 16.384M seen tokens at step 1000; step 2500 would only be ~40.96M.
- Current comparable encoder work (ModernBERT/NeoBERT) is trained on trillion-token scale, and compute-efficient work still emphasizes much denser training signal.
- Current five-source corpus is too narrow relative to available rights-clean Common Pile sources.
- Current Sol curriculum contains rationales, but the ranker/collator ignores them and only learns preferred candidate indices.
- 57 held-out ranking cases are diagnostic, not a readiness gate; reward-model research warns ranking metrics can decouple from generation behavior.

Recommended production redirect:
1. Preserve N0 v0.1 step 1000 as pathfinder baseline.
2. Design N0 v0.2 around a native ~150–200M ModernBERT-base-class encoder for better capability-per-P100-hour.
3. Keep native random weights; use Sol/frontier teachers only through governed supervision/distillation, never inherited weights.
4. Broaden rights-clean corpus into low-billions-token-capable mixture while retaining per-row licensing/provenance.
5. Use modern ~30% span MLM plus semantic contrastive/listwise objectives; evaluate RTD only as an ablation.
6. Turn rationale/principle annotations into actual gradient-bearing supervision.
7. Build a much larger fixed held-out N0 benchmark including invariance, calibration, hard negatives, multi-turn/context tests.
8. N1/N2 remain modular: shared semantics + governed identity adapter/latent + multi-head identity decisions and context gating.
9. N3 must test end-to-end identity behavior through multiple interchangeable downstream generators with owner review.

No private Elaina gradient has been authorized by this redirect. Main remains untouched.
