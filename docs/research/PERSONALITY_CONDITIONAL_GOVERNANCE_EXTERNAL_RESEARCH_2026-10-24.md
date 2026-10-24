# Conditional personality governance: external mechanism review

Research date: 2026-10-24. Adoption status: **research only; no architecture, objective, data, checkpoint, or training selection adopted**. Local source inspected at HEAD `5c2d04da66e31f5baa49fc417f644ee422fe4e5f`, alongside the current uncommitted conditional-route diagnosis. Upstream code was read, not executed. This is full ALICE personality research, not FloRA.

## Decision and evidence boundary

The question is how native N1 interpretation and rich personal state can reliably govern N2's contextual, multidimensional judgment while N0 remains a frozen semantic feature base. N3 calibration and the replaceable generator cannot take over that responsibility. Preserve source-person, host, relationship, and self permissions, semantic alternatives, explicit uncertainty, rich latent concepts, and the 38-family/192-facet coverage intent.

The current [diagnosis](../eipm/PERSONALITY_CONDITIONAL_ROUTE_DIAGNOSIS_2026-10-24.md) reports exact authentic/replay reproduction, 400 completed forwards with no updates, and unchanged custody. All five completed variants retain 25/80 context top sets and 0/40 jointly correct pairs. Selected reviewed N1 values therefore failed to repair N2 behavior. This is not proof that concepts are absent, full N1 is correct, attention dilutes them, or a particular loss or architecture is responsible. The native stage trace is the appropriate next discriminator.

The inspected [model](../../src/alice_personality/identity/model.py) adds projected applicability/uncertainty and source-weighted support/disconfirmation messages to rich concepts, then puts contextual views, full source tokens, and query tokens into candidate attention. Candidate comparison, normalization, feedforward, and scalar/semantic readouts follow. A correction must preserve this responsibility, not reduce personality to four scalar fields.

## Mechanisms and transfer limits

### 1. Learn useful intervention response: CEM and IntCEM

[CEM (NeurIPS 2022)](https://arxiv.org/abs/2209.09056) mixes positive and negative input-dependent embeddings using a predicted concept probability. Changing that probability retains richer information than a Boolean bottleneck. [IntCEM (NeurIPS 2023)](https://arxiv.org/html/2309.16928) trains on correction trajectories, combining concept prediction, task prediction before/after corrections, and a learned intervention-selection policy. Its paper distinguishes these edits from interventions in a causal structural model. Merely exposing a scalar field or supervising it accurately does not train the consumer to use its correction.

The [official CEM code](https://github.com/mateoespinosa/cem/blob/d67716cf8435961b41087ec884f92c74e475d890/cem/models/cem.py#L234) replaces selected probabilities before embedding mixing. [IntCEM code](https://github.com/mateoespinosa/cem/blob/d67716cf8435961b41087ec884f92c74e475d890/cem/models/intcbm.py#L605) constructs trajectories, recomputes downstream task loss, and combines that with concept/policy losses. Configurations can supervise only the trajectory endpoint or intermediate states. Its policy uses task labels during training to select useful interventions; copying that policy would introduce unnecessary machinery here.

**ALICE fit:** if the stage trace shows that native conditional information reaches a trainable judgment path but existing supervision fails to teach useful use, adapt the intervention-aware *training principle* inside native N2. Use only independently admitted training facts and existing governing targets. Retain native uncorrected examples and the rich latent/source route. No external teacher, fixed concept list, or test-time oracle is required by this limited adaptation.

**Cost/assumption:** extra native N2 forwards; trustworthy same-context correction labels; a correction distribution that resembles the intended use. Diagnostic reviewed labels are not automatically training data. CEM's per-concept fixed-size embedding machinery is not a suitable replacement for ALICE's open semantic representations.

### 2. Preserve a distinct conditioning route: Flamingo / OpenFlamingo

[DeepMind's Flamingo (2022), section 2.2 and appendix A.1.2](https://arxiv.org/html/2204.14198) inserts separate cross-attention to conditioning features and adds its output through a learned tanh gate. A zero gate preserves the original model at initialization. This is a useful architectural mechanism for connecting a frozen base and learned conditioning; its vision-language results do not establish an ALICE cause.

[OpenFlamingo's independent open-source reproduction](https://github.com/mlfoundations/open_flamingo/blob/655f693fbfa04cd6e9a987d960654624d48917cf/open_flamingo/src/helpers.py#L236) implements masked cross-attention, separate attention/feedforward gates, and residual addition. The gate can become negative. Its media-position mask is specific to images and must not substitute for ALICE authority masks. Mathematically, a zero gate initially blocks gradients into branch parameters while permitting a gate gradient; that is a training consideration, not evidence of failure.

**ALICE fit:** only if measured effects are lost in mixed-memory reading, a separately masked read of authorized contextual concepts/source relations could contribute a learned residual to candidate states. Keep query/full-source attention and rich evidence. Do not force an attention quota, fixed number of concepts, or hard-coded candidate preference.

**Cost/assumption:** extra attention and checkpoint-version handling; the structured representation must contain useful information. A separate branch can still learn to be ignored. If condition information survives to candidate states and the readout discards it, this mechanism addresses the wrong location.

### 3. Separate judgment dimensions and context-dependent use: ArmoRM

[ArmoRM (June 2024; Findings EMNLP 2024)](https://arxiv.org/html/2406.12845) learns multiple objective predictions from frozen representations, then trains a prompt-conditioned gating network while freezing the objective regressors. Its published implementation uses 19 objectives and ultimately scalarizes them for pairwise reward. The backbone already comes from a reward-trained model, which is a material mismatch with ALICE's neutral N0 boundary.

[Official stage-two source](https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/fc39179f5a9a4dd75047a4c0311672905b9d9a04/armo-rm/stage-2_train.py#L43) is explicit: an MLP produces temperature-scaled objective weights and a learned scale; training scores the weighted objective vector using pairwise binary loss. The useful idea is auditable separation of *what the candidate implies* from *how the current context changes its relevance*, not the published scalar reward or objective taxonomy.

**ALICE fit:** if the stage trace finds a substantial appropriate contextual change immediately before the scalar/semantic heads but little governing change afterward, inspect head conditioning and native per-family objectives. A narrowly scoped first-party factorization could retain the full governing packet, uncertainty, co-validity and contraindication outputs while learning contextual interactions. The existing N2 already has multiple heads; adding another set without locating a deficiency is unjustified.

**Cost/assumption:** existing labels must identify the dimensions/context interactions; fixed linear combinations may miss nonlinear tradeoffs and intransitive/context-dependent preferences. Do not adopt 19 dimensions as ALICE's ceiling, import a reward-model backbone, or replace the packet with a scalar reranker.

### 4. Infer uncertain personal context: Variational Preference Learning

[VPL (August 2024)](https://arxiv.org/html/2408.10075) infers a distribution over a user's latent context from multiple preference comparisons and conditions prediction on that latent. Its LLM procedure freezes precomputed context embeddings but also describes LoRA training on the target reward path; it is not blanket evidence that every semantic component can remain frozen with identical performance.

[Authors' latent/decoder code](https://github.com/WEIRDLabUW/vpl_llm/blob/fdfdfa767752b5af7f89f2e6ab6989c1953062c9/hidden_context/vae_utils.py#L11) uses pair encoding, attention over the user's comparison set, mean/log-variance prediction, reparameterized sampling, and concatenation with candidate representations. The trainer includes a `fixed_llm_embeddings` path, pairwise preference loss and KL regularization. [Training entrypoint](https://github.com/WEIRDLabUW/vpl_llm/blob/fdfdfa767752b5af7f89f2e6ab6989c1953062c9/hidden_context/train_llm_vae_preference_model.py#L601) creates LoRA adapters. No upstream runtime compatibility was tested; the inspected trainer call also supplies mask keywords absent from the displayed `VAEModel.forward` signature, so this is not a verified drop-in package.

**ALICE fit:** a later native N1 uncertainty/latent-learning comparison if evidence implicates missing personal context, with each authority stream kept separate and explicit facts retained. Pooling host, source-person and relationship preferences into an anonymous user latent would violate the contract. Its scalar decoder and anonymous preference populations are not the target architecture.

**Cost/assumption:** enough grounded same-person histories, identifiable context, and genuine uncertainty calibration. A variational latent may collapse or learn shortcuts. There is no present evidence that replacing ALICE's existing latent representation is necessary.

## Frontier cautions that change interpretation

[MixCEM, ICML 2025, revised August 2025](https://arxiv.org/html/2504.17921) argues that incomplete concepts require residual information, but that residual information can obstruct corrections under distribution shift. It combines global concept embeddings with dynamic residuals and scales residual contribution using concept uncertainty. Its [official source](https://github.com/mateoespinosa/cem/blob/d67716cf8435961b41087ec884f92c74e475d890/cem/models/mixcem.py#L205) implements entropy-based residual scaling, residual dropout and a task loss on fully corrected global embeddings. Transfer lesson: test residual interaction; do not assume all non-explicit information is a harmful shortcut. Direct adoption would impose fixed per-concept embeddings and rely on concept uncertainty as an OOD signal. Neither is established here, and uncertainty must never modify permission.

[Harmful Partial Interventions and Benign Leakage (15 October 2026 preprint)](https://arxiv.org/html/2610.21750v1) shows why strong fully corrected endpoints can coexist with harmful partial corrections. Section 4 penalizes positive task-loss differences between adjacent correction sets, stopping gradients through predicted concepts while retaining residual/head gradients. The appendix details matched synthetic and frozen-CLIP CUB experiments. The paper claims supplementary anonymized code, but no accessible linked source was verified in this review; treat it as a recent paper-level result, not code-validated adoption evidence.

For ALICE, such a monotonic penalty is at most a secondary retention constraint: a model that ignores every correction also receives zero penalty. It cannot solve insensitivity by itself. Apply any eventual experiment only to verified corrections within one situation, never partner-context swaps, newly contradictory evidence, or arbitrary uncertainty edits. Test actual useful judgment gains separately.

## Company personality comparison: useful diagnostics, unsuitable ownership

[Anthropic Persona Vectors (July/August 2025)](https://arxiv.org/html/2507.21509) derives trait directions from contrasting model activations and uses them to monitor or steer behavior. The extraction process employs a frontier model to generate prompts, questions and a rubric. [Extraction source](https://github.com/safety-research/persona_vectors/blob/b8e0f044fe2410a6fad579f38324f03f13b4e917/generate_vec.py#L40) filters positive/negative examples and computes mean activation differences; [steering source](https://github.com/safety-research/persona_vectors/blob/b8e0f044fe2410a6fad579f38324f03f13b4e917/activation_steer.py#L73) adds scaled vectors through hooks.

Its transferable diagnostic idea is to ask whether a measured internal change predicts and causally affects behavior. Its pretrained assistant traits, generated rubrics and generator-layer steering cannot supply ALICE's grounded N1/N2 personality. No retrieved public system demonstrated ALICE's complete authority-separated, open semantic, generator-independent governing contract; this is a limit of this review, not a claim that no such system exists.

## Bounded adoption decisions after the native stage result

These are alternative follow-up experiments, not a tournament or an instruction to launch training. Select only the mechanism supported by the first demonstrated departure, or retain uncertainty if none is distinguished.

| Stage finding | One controlled experiment | Reject the proposed explanation if |
| --- | --- | --- |
| Useful conditional effects propagate, but governing learning remains weak | Native N2 baseline versus matched intervention-aware native task learning, same eligible data/update budget and frozen N0; only authorized development correction subsets | Only oracle-patched behavior improves, natural native pair correctness does not; irrelevant edits acquire influence; retention or authority behavior worsens |
| Effects disappear at mixed-memory reading | Baseline versus one separately masked structured-conditioning residual, retaining rich source/query reading and all output families | Larger branch activations do not improve natural context-pair judgment, or gains depend on bypassing masks/full evidence |
| Effects survive to candidate states but disappear in scoring | Baseline versus a native context-conditioned head correction using existing targets and semantic labels | It only changes loss/score scale, does not recover appropriate preference changes, or damages co-validity/contraindication and other head families |
| N1 lacks useful personal contextual representation | Native N1 baseline versus one justified latent-learning adjustment, preserving explicit facts and separate authority streams | It averages identities, collapses uncertainty, improves only familiar examples or needs generator/teacher authority |

Every experiment must preserve original failures and use independent qualification separate from reviewed diagnostic material. Measure jointly correct context pairs, per-context top-set transitions and target-change direction, plus the full packet and retained capabilities. Include unchanged replay, authorized relevant corrections, irrelevant corrections, unavailable-fact masks, natural partner conditions and authority-negative controls. Report individual transitions, not aggregate counts alone. A public synthetic mechanics check cannot establish native personality learning.

One source-level invariant merits measurement now: `conditional_memory` divides each support/disconfirmation message by `weights.sum(2).clamp_min(1)`. Uniform positive rescaling of a concept/channel's source weights cancels from the normalized message when both sums are at least one. The separate scalar mean path still carries magnitude. This is an algebraic property, **not a demonstrated cause**; distinguish source-weight composition from total-mass changes before blaming downstream attention.

## Source custody and construction record

Code pins were resolved from the public GitHub commit API on 2026-10-24: CEM/IntCEM/MixCEM `d67716cf8435961b41087ec884f92c74e475d890` (2025-10-10); OpenFlamingo `655f693fbfa04cd6e9a987d960654624d48917cf` (2023-10-11); ArmoRM `fc39179f5a9a4dd75047a4c0311672905b9d9a04` (2025-04-24); VPL `fdfdfa767752b5af7f89f2e6ab6989c1953062c9` (2024-08-21); Persona Vectors `b8e0f044fe2410a6fad579f38324f03f13b4e917` (2026-04-22). Paper mechanisms and actual source were compared; versions were not inferred from search snippets.

Reusable procedure: validate native failure and controls; locate the exact learned transformation; compare a small number of relevant external mechanisms with code and assumptions; propose one falsifiable change only after localization; verify natural behavior and retained governing capability. The artifact contains public research and public aggregate evidence only. It creates no admitted training examples, learned builder weights, model qualification, or production activation. No model code, private data, remote job or dependency was changed.
