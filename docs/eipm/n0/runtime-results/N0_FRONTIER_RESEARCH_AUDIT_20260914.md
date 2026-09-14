# N0 Frontier Research Audit — 2026-09-14

Status: **hold the step-1000 -> step-2500 GPU job pending N0 v0.2 redesign**.

## Bottom line

The EIPM architecture and stage separation remain sound. The current N0 v0.1 experiment has been useful as a pathfinder, but the immediate plan of extending the 352M random-init encoder from 16.384M to roughly 40.96M seen tokens with the same narrow corpus and MLM-only recipe is not the best use of scarce P100 compute.

The production N0 path should preserve native weights and provenance while becoming more compute-efficient and more principle/representation oriented.

## Frontier comparisons reviewed

- ModernBERT (ACL 2025): 149M/395M encoder family, 8k context, 2T-token training, 30% MLM masking, staged short-to-long context training.
- NeoBERT (2025): 250M encoder, 4k context, 2.1T-token training, 20% masking, modern depth/width and normalization choices.
- ELECTRA / DeBERTaV3 line: replaced-token detection shows materially better sample efficiency than classic sparse MLM in compute-constrained settings.
- SimCSE / LLM2Vec / modern embedding work: contrastive objectives materially improve semantic representation quality beyond plain token reconstruction.
- Jina Embeddings v3 and related adapter work: a shared semantic backbone plus small task-specific adapters/heads is an efficient architecture pattern.
- Common Pile v0.1: 8TB, 30 diverse public-domain/openly-licensed sources. The current five-source N0 mix is unnecessarily narrow.
- Personalized alignment (ACL 2025 survey), PAL (ICLR 2025), PAD (ICLR 2025), P-GenRM (ICLR 2026): modular shared representations plus personalized preference components are competitive and sample-efficient.
- Dynamic Persona Coherence (ACL 2026) and PersonaForge (ACL 2026): separating stable identity from adaptive psychological/interaction state is strongly supported; this matches the EIPM/continuity/relationship separation.
- Anthropic Persona Selection Model / Assistant Axis (2026): pretraining itself creates persona/archetype priors. This strengthens the rationale for keeping A.L.I.C.E. runtime identity weights native rather than inheriting a third-party pretrained persona space.
- Anthropic “Teaching Claude Why” (2026): demonstrations alone can overfit; teaching principles/reasons and using diverse data improves OOD generalization. Current N0 curriculum stores rationales but the collator/ranker does not train on them.
- Reward Model Selection Crisis (2025): preference-ranking accuracy can be weakly related to actual reward-guided generation behavior. N0/N2 promotion therefore cannot depend only on ranker accuracy.
- RMTBench / PersonaEval: persona evaluation should include multi-turn behavior and human/owner review; LLM judges are not reliable enough to be sole fidelity authority.

## What remains correct

1. Dedicated EIPM rather than putting personality inside the downstream generator.
2. Encoder/discriminative foundation for interpretation, comparison, evidence and policy decisions.
3. N0 universal semantics -> N1 governed identity representation -> N2 context-conditioned preference/decision -> N3 calibration/owner refinement.
4. Stable Elaina-derived identity separated from Rayan host state, relationship state, and A.L.I.C.E. continuity/adaptation.
5. Multi-dimensional IDP rather than one scalar reward.
6. Explicit uncertainty, ties/plurality, provenance, evidence links, and out-of-character/generic-assistant drift signals.
7. Native A.L.I.C.E. weights and no private identity gradient in N0.

## Problems in the current N0 v0.1 execution recipe

1. **Scale/data mismatch.** 352M random-init parameters are being trained on tens of millions of tokens. Models of comparable scale in the modern encoder frontier are trained on orders of magnitude more text.
2. **Corpus diversity.** The current five-source mixture is book/government/legal/PEP heavy and weak in broad contemporary semantic, conversational, pragmatic, and social language.
3. **Objective efficiency.** The current 15% span MLM is conservative. ModernBERT uses 30%, NeoBERT 20%, and discriminative/contrastive objectives offer better signal density for a representation model.
4. **Principles are not learned.** Sol curriculum `rationale` fields are currently metadata; `CurriculumCollator` encodes only prompt and candidates. The model is taught which candidate wins, not explicitly why.
5. **Evaluation is too small.** The 57-row expanded dev probe is useful diagnostically but cannot be an N0 readiness gate.
6. **Ranking is a proxy.** Later EIPM evaluation must test downstream behavior across interchangeable generators, not only candidate-ranking accuracy.

## Recommended N0 v0.2 path

- Preserve step-1000 N0 v0.1 as the pathfinder baseline; do not delete it.
- Prefer a native ModernBERT-base-class student (~150–200M) over the current ~352M production candidate under the available 2x P100 compute budget. The EIPM is a policy/representation model, not a general-purpose generator.
- Keep native random initialization. Use frontier models/Sol as teachers through governed labels, principles, hard negatives, semantic pairs and optional representation/logit distillation rather than importing their weights.
- Build a much broader rights-clean Common-Pile-derived corpus with per-row license enforcement. Prepare enough unique data for at least low-billions-token learning if capability curves justify it; do not hard-code a token target as a success condition.
- Raise MLM corruption into the modern range (initially test ~30% span MLM) and add semantic contrastive/listwise supervision. Treat ELECTRA-style RTD as an ablation candidate rather than automatically adding generator complexity.
- Make rationale/principle supervision a real objective. Candidate preference, principle/rationale alignment, entailment/contradiction, calibration and tie handling should all supply gradients.
- Expand the fixed N0 benchmark to hundreds/thousands of clean held-out cases with paraphrase invariance, candidate-order invariance, hard negatives, uncertainty, structured evidence, relationship/social reasoning, and multi-turn/context carryover.
- Use capability-per-GPU-hour and fixed held-out generalization to choose continuation. Do not use raw MLM loss alone.
- Only after N0 stabilizes should governed private N1 identity gradients begin.

## N1/N2/N3 architecture direction after N0

Keep a shared semantic backbone and add identity-specific adapters/latent state plus interpretable multi-head outputs. Candidate preference should be only one head. Other heads should cover stance, value priorities, relationship posture, emotional posture, communication/style controls, uncertainty/plurality, evidence/precedent linkage, and out-of-character/generic-assistant drift. Context-dependent gating can combine these dimensions without collapsing identity into one scalar.

N3 must include end-to-end tests through multiple replaceable downstream generators. Owner review remains the final fidelity authority.

## Immediate decision

Do **not** submit `magnolia_p100x2_n0_semantic_growth.sbatch` yet. Redesign and implement N0 v0.2 first, then spend GPU time on the compute-efficient production path rather than deepening the v0.1 pathfinder simply because its continuation launcher already exists.
