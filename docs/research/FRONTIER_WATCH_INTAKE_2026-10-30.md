# Frontier Watch intake — 2026-10-30

**Decision.** After reading current A.L.I.C.E. main, Fable main, current MFM/FBM/continuity branch heads, the mandatory `FRONTIER_WATCH_FULL_FABLE_V1_BASELINE_2026-09-27.md`, the October 29 intake, selected memory architecture and current execution plan, **six newly published or newly surfaced primary papers qualify for future tracking**, with one additional research item corroborating existing design. The strongest *new* deltas are task-graph-transitive/cumulative disclosure control, proof-carrying independently attested agent claims and approval, and interference-aware **sequential** parametric unlearning. **No architecture-invalidating evidence, current selected implementation blocker, changed current training recipe, production authority migration, infrastructure tournament, reduced v1 foundation, or new consumer release gate.** All first-release full-cognitive-plane and hardware rules remain unchanged.

## Source and execution check (before paper judgments)

| Component | Verified head | Change since 2026-10-29 |
|---|---|---|
| A.L.I.C.E. `main` | `a1fe0a65df1d4143955eb1d3b35eb4ac590e7555` | None |
| Fable_Sleight `main` | `f8e9062ebd657f18c3a571cedc1206889d7fd53a` | None |
| MFM `research/mfm-v1-licensed-base-20260930` | `5653736f545c9c1c8d5d8006af0f1fa152eed9b1` | None |
| FBM `research/fbm-v1-licensed-base-20260930` | `bcd462b5bb93f41a56f79b5b63e3337c8e89bb45` | None |
| `alice-context` | `945245bbee4168fff46f863455a83bb219063017` | None |
| Graphify continuity | `a06f61eafd91d04fb7ec7c1de3ab8a9ed7071b15` | None |
| Research parent | `research/frontier-watch@8e825527e12b8a06d1ea2d0fb2120ae6aa6acaac` | October 29 intake |

Read: mandatory full-v1 baseline; frontier-watch execution guardrail; October 29 intake and preceding research deltas in daily continuity; main `docs/ALICE_PHASE2_REPLACEMENT_AND_FABLE_V1_EXECUTION_PLAN_2026-09-27.md`, `docs/MEMORY_ARCHITECTURE_V4.md`, `docs/FABLE_PERSONAL_DEVELOPMENT_ARCHITECTURE.md`; Fable main README. The October 29 note references later selected Fable execution profile and related continuity notes. The selected placements are KurrentDB/JetStream, XTDB v2, encrypted content-addressed evidence, LadybugDB/NebulaGraph, Qdrant, source-native/live reads, Temporal, L1/Valkey, PyTorch+Accelerate; these are selected **not** a tournament.

**Actual critical path:** checkpoint-10 MFM gave **0/9 valid native source-grounded complete formations** despite improved teacher-forced losses. Sixteen additional full-nine optimizer updates toward checkpoint 26 were dispatched; no new committed terminal native formation is verified. The personality conditional-judgment failure is likewise not shown repaired. No paper here directly diagnoses or repairs the measured MFM early lexical-token competition. Stay on the current evidence-driven training and independent evaluation path.

## 1. Task-Graph Contextual Integrity (TG-CI) — disclosure before downstream propagation

**Primary:** Wu, Delattre, Cao, *Judging Before Propagation: Task-Graph Contextual Integrity for Multi-Agent Privacy*, arXiv:2610.39532v1, submitted **Oct 29 2026**. https://arxiv.org/html/2610.39532

**Class:** **Future challenger** to the Mission/egress/federation disclosure boundary; materially new *qualification dimension*, **not a demonstrated current contract defect**.

**Method / implementation:** Each item has data subject, attribute and value. A live Multi-Agent Task Graph carries delegated tasks and permitted transfer edges. The privacy controller evaluates proposed disclosures in the context of (i) recipient's actually delivered prior information and (ii) intended propagation to future downstream recipients. It updates a conservative transaction-local authorization footprint and separately maintains actual-delivery history. **Full-Scope** denies an item unless all anticipated downstream edges are admissible; **Auto-Region** admits the item to a maximal permitted rooted region and places semantic filtering at its frontier. Optional local MiniLM embedding prefilter plus DeBERTa bidirectional entailment detects re-expression across a forbidden edge. Uses LangChain and Mem0 only as evaluation scaffolding; neither is necessary for Fable.

**Experiments and distinctions:** MAPLE-Bench consists of **144 synthetic cases** from 48 four-agent workflows under three topologies; each case has 30 memories, five multiple-choice questions and direct/cumulative/transitive authorization labels. In primary Haiku-task / Luna-judge assignment, mean unauthorized delivery events per run: no protection **15.84**, local-history CI **10.47**, per-edge online CI **6.11**, auto-region **1.49**, full-scope **0.62**. Thus reductions versus local-history are about **85.8%/94.1%**. Distinct inappropriate items exposed to compromised agents: 1.58 local-history versus 0.35/0.16. Utility (five-question accuracy): 83.26% local-history, 80.00% auto-region, 77.92% full-scope; **96.08%/93.58% relative** in that primary configuration. Other agent/judge pairs have larger declines (e.g., auto-region 68.40% vs 78.06% local-history), so don't universalize abstract retention. Mean privacy-judge usage **133.1k/144.5k tokens** for the two variants in primary experiments, plus local semantic compute: **not a cheap default runtime**. Paper does not provide a clean independent component ablation for recipient history vs propagation, but compares no-protection/local-history/per-edge/graph-aware modes and all three topologies.

**Already covered:** owner/privacy egress and permission gates, Experience vs bitemporal Claims, host/source-person separation, trusted Mission Graph/durable workflows, model/provider boundaries, federated devices, inference leakage (InferPriv, October 28), I-HITL/Concordat. **Genuinely new:** an *individually admissible* memory can become unsafe when combined with what recipient already knows **or** when passed further to another planned worker. This is an end-to-end task topology + recipient cumulative-information invariant. A graph score never grants source authority or host consent.

**Limits/compatibility:** Task graph is assumed **static** for the full task; dynamic replanning is explicitly left open. All memory and agent messages must pass through mediator. Out-of-band/covert encoding, unmonitored channels, arbitrary semantic transformations, and live real-human privacy norm elicitation are not solved. Judge makes mistakes, benchmark synthetic, substantial token cost. Rebuild scopes on permission revocation, Mission route change, new recipients or corrected/deleted source Claims; preserve source-person and owner custody, receipt lineage and rollback. Do not transfer private state across products/hosts or replace Claim Authority.

**Action:** Future focused test **when real delegated Mission-to-provider/agent disclosure is being built**: a piece safe for worker A but not downstream B; two individually safe items whose joint disclosure is forbidden; Mission replanning/revocation. Prove both prevention and legitimate completion. No new framework, no present barrier, no 144-case platform benchmark.

## 2. PCM — prove delegated claims with independent receipts

**Primary:** Ghosh, *PCM: A Proof-Carrying Message Protocol for Verifiable Multi-Agent Communication*, arXiv:2610.39109v1, submitted **Oct 29 2026**. https://arxiv.org/abs/2610.39109 ; specification and tests linked from paper.

**Class:** **Future challenger** for inter-agent and inter-device authority/evidence exchange; not a production deficiency shown in current A.L.I.C.E.

**Method:** A sender includes append-only Merkle-log inclusion proofs for attestable events and delegated grants. A signer **outside the agent's control** vouches for source events. Recipients independently verify the proof, signer, evidence type, nonces, replay protection and narrowing caveat chain. Message claims have three-valued `proven/unproven/failed` judgments; unverified agent text can never assert authoritative user approval or test completion. Four claim classes and independently implemented verifier conformance vectors make the contract portable.

**Evidence/negative ablation:** Author-created benchmark of **926 messages**, including **712 deliberate false claims** across 16 attack classes; the PCM verifier mistakenly accepts **10/712** lies (**1.4%**), **all stale genuine proofs** that no longer describe current state. Plain unverified A2A accepts 712/712 and modeled six alternative protocols accept 504–712. Two verifier implementations agree on all 926; ~1 ms verification. **Important:** author's H1 zero accepted false claims **FAILED** and >=99% honest acceptance **FAILED** (76/78 = 97.4%). Claimed security properties are conditioned on trusted external witness, a properly monitored log and freshness assumption. Competing protocol results are specification-modeled under paper's threat taxonomy rather than measured interoperable deployments. Benchmarks are in-process and narrow.

**Already covered:** canonical Experience ledger and evidence contracts; Claim Authority; provenance/corrections; owner-delegated Mission authority; cross-device transport via NATS JetStream; process/workflow receipts. **Novel:** operational trust-boundary check between model-authored *assertion* and third-party **verifiable receipt** of an observed event or authentic permission. A model claiming "the tests passed" may cite only historical run; it must not turn that into current pass. This reinforces October 20 Tether dependency validity and October 27 I-HITL.

**Compatibility / cost:** An evidence receipt attests an event; it does **not** independently settle present Claim validity, legal authorization or real host identity. Attach source identity, version/time, exact artifact/action digest, consent scope, policy head and expiry; current status rechecked at consumption. Revoked source invalidates derived permission/Claim/skills and any cached action eligibility. Add signer/log and audit storage only when necessary; ~1 ms verifier alone omits log writing, signatures, network/federation and operational keys. Not a warrant for blockchain, replacement event store, or irreversible key governance changes.

**Action:** In future **actual** multi-agent execution/federation, one targeted false approval / stale test-success / revoked delegation with independent proof and authoritative current-head recheck. No current implementation changes.

## 3. COP — deletion requests can undo one another in learned weights

**Primary:** Belieni, Erthal, de Souza da Silva, Mesquita, *Preserving Utility in Continual LLM Unlearning via Orthogonal Projections*, arXiv:2610.40483v1, submitted **Oct 29 2026**. https://arxiv.org/abs/2610.40483

**Class:** **Future challenger** for **parametric** personal memory unlearning and governed model evolution. Not yet a current MFM defect.

**Method / assumptions:** Continual Orthogonal Projection (COP) caches a capped low-rank basis of prior forget-set activation subspaces for each layer. On later deletion requests, projects new parameter gradients onto orthogonal complements, reducing interference with already-forgotten sets. It is objective-agnostic for gradient-based unlearning (GA, NPO, SimNPO, IDK) and avoids storing earlier full forget data and gradient histories. Its formal statement is **first-order** non-interference at the local linear approximation; no behavioral, exact, or influence-echo guarantee.

**Evidence and cost:** TOFU and RWKU experiments with Llama 1B/3B and a sequence of five deletion requests; example task-5 mean retain utility on 1B GA **0.159 without COP vs 0.461 with COP**, IDK **0.037 vs 0.465**, and 3B GA **0.224 vs 0.566**, with forgetting-utility tradeoffs varying by objective and rank. Ablations compare methods with/without projection and subspace rank; some settings retain utility while weakening forgetting, so do not claim universally stronger deletion. Extra activation passes / per-layer SVD or low-rank basis extraction, QR/projection matmuls, memory proportional to per-layer protected rank and additional training compute. Repeated deletions may saturate the available subspace and obstruct useful subsequent updates.

**Already covered:** cross-layer deletion from raw evidence through indexes/caches/replicas/datasets/weights, content-addressed model lineage, governed training, rollback, MUTE influence-echo testing and repeated correction/deletion eligibility. **Novel:** sequential deletes may conflict **in parameter space** even after each single deletion appears successful; one deletion may reintroduce earlier forgotten content or destroy general ability. A candidate update-geometry method may mitigate it.

**Authority/identity/rollback:** Keep proof of exact source-person/host origin, authorized deletion request, dataset and model checkpoints and ancestor lineage. Do not conflate lower average loss or local orthogonality with removal of user-identifying outputs, recomposition from related memories, re-exposure through source-native retrieval or reintroduction after backup restoration. Preserve meaningful current Claim history only to extent legal/authorized. Evaluate post-deletion behavior vs never-exposed controls and explicit retain utility at the model version active in service. No current training objective change.

**Action:** Only once real model-weight deletion is required and *measured* sequential deletions interfere, evaluate COP vs selected retrain/lineage process on a small repeat-delete sequence including influence-echo and rebuild/rollback. Do not add an unlearning project to current MFM lexical-failure critical path.

## 4. EpiAgent — learned epistemic beliefs become active planning state

**Primary:** Jin et al., *EpiAgent: Epistemic Self-Play for LLM Agents*, arXiv:2610.40227v1, submitted **Oct 29 2026**. https://arxiv.org/abs/2610.40227

**Class:** **Future challenger** for source-person/social/relationship/world models and Mission planning.

**Method:** Learned epistemic-update stage represents uncertain beliefs about another actor's private information; action-planning stage uses those beliefs. Alternating self-play with an intrinsic next-observation prediction reward updates the two stages, rather than just injecting another person's assumed state as text. In modified two-player bargaining and Kuhn poker, compared prompting and no-dedicated-epistemic-training baselines; ablation mean reward in reported cases bargaining **0.321 vs 0.204**, poker **0.091 vs 0.031** with/without the dedicated training. Competitive/mixed games and held partner persona evaluation remain far from real longitudinal human relationships.

**Already covered:** separate host/source-person/assistant-self and relationship models; uncertain social-state hypotheses; causal/world/mission evidence; October 29 PIP-ToM *perspective-bounded* evidence. **Novel:** a *trained* loop turning probabilistic estimates of the other participant into behavior changes and later prediction-error learning; not just improving factual recall.

**Governance/cost:** Inferred private mental states remain provisional social models with subject attribution and confidence, **never** canonical Claims about someone's true beliefs or consent. Revoke if contradicted/deleted; log observation/prediction/actual consequence separately. Self-play and training compute substantial; game reward may incentivize manipulation. Human permission and outcome ethics separate from win-rate. **Action:** Future small, ethically authorized personal-judgment test, not a current training modification.

## 5. Social re-entry — finding past conversation is not permission or ability to re-enter it

**Primary:** Wan et al., *Finding Is Not Returning, Returning Is Not Reconstructing: A Multi-Dimensional Evaluation of Agent Social Re-entry*, arXiv:2610.39781v1, submitted **Oct 29 2026**. https://arxiv.org/abs/2610.39781

**Class:** **Future challenger** to the relationship/Mission proactive-recollection evaluation boundary.

**Method/evidence:** Six LLM configurations with shared-snapshot conversations; independent metrics for Finding dormant interactions, choosing autonomously to Return, and reconstructing context accurately. Resurfacing a prior thread can make target identification easy; memory increases target-directed responses, yet **does not reliably improve the accuracy of reconstructed history**. Strong caveat: constructed shared-snapshot tasks are not months of real host interaction; no proof of consent or appropriate timing, and metrics can reward intrusive engagement.

**Already covered / new:** proactive memory, host relationships, Mission scheduling, source-native evidence and current Claim validation; earlier ProMemAgent and proactive-alert scale paper. **New** is distinguishing *accessibility*, *appropriateness/eligibility to interrupt*, and *evidence-faithful continuation* as three separately failed capabilities. **Action:** When owner-authorized proactive social tasks are operational, evaluate all three alongside permission, abstention, stale-context and false intrusion. Don't implement autonomous outreach solely on this result.

## 6. ThinkJev — calibrated test-time compute for typed decisions

**Primary:** Ding et al., *ThinkJev: One Thought, Every Budget for Calibrated Typed Decisions*, arXiv:2610.37732, **Oct 27 preprint newly surfaced in Oct 30 screening**. https://arxiv.org/abs/2610.37732

**Class:** **Conditional future challenger** for native typed-judgment/serving efficiency. Distinct from previously tracked Jev-Mem.

**Method:** One hybrid-thinking model produces System-1 option probabilities without thought and System-2 probabilities after an incremental thought; reuse prefix/KV cache instead of two independent full calls. Calibrate a fast-path acceptance threshold against labeled dev decisions to bound observed risk; a single thought gives answers at every shorter token budget. At target 5% fast-path error on 8,359 evaluated Qwen3.5-27B decisions, **24.5%** can exit fast with **3.9%** actual error. An early-stopping thought policy achieves **81.6%** vs **82.7%** always-think accuracy with **72%** of thought tokens. Controlled closed-choice arithmetic/counting/date decisions; calibration may drift with subject and context; measured cost and KV retention must be verified on target hardware.

**Already covered/new:** adaptive MRM/working memory and typed native judgment, model-serving efficiency; new specific calibrated *per-decision compute allocation* using shared model state, not an ontology or a new personality model. Authority/provenance checks still hard-gate any low-latency answer; fast confidence is not truth, owner authorization or deletion. **Action:** Revisit after native judgment qualifies **and** real latency/compute bottleneck exists, not for current 0/9 MFM malformed native generations or a separate GPU campaign.

## Corroboration / screened out

- Microsoft, *Long-Term Agent Memory through Complementary Curation and Consolidation* (https://arxiv.org/abs/2610.39313, Oct 29): structured derived memory with episodic source, consolidation and answer comparison reinforces existing Curator + canonical evidence/source-native approach. LoCoMo results vary by generator and full-context can outperform compressed memory. **Already covered/corroboration**, not a new cognitive plane or database decision.
- *MUSE: Multi-Level Memory with Selective Runtime Injection for Scientific Coding Agents* (arXiv:2610.38332): procedural memory use boundaries in narrow coding task; overlaps MemCryst/HaMo/DiscernMem. **Already covered**.
- *HEDGE: Gated Hypergraph Memory for Scalable Multi-Agent Debate* (arXiv:2610.38902): alternative memory representation for debate; no directly unexpressible associative relation in existing Cognitive Multi-Graph. **Future challenger only on actual current workload blocker**, otherwise not actionable.
- Do not relabel October 20–29 studies (including MUTE, REALM, AdaMM, StateWise, memory tax, PIP-ToM, DiscernMem, Concordat) as new.

## Disposition

**Do not reopen physical stack. No architecture-invalidating evidence. No production/main semantic change. No PR required; this is an intake-only branch note.** Complete the already running MFM native-learning and personality judgment work. Future-only findings remain nonblocking. The first consumer gate stays `full_personal_cognitive_foundation_after_f11`, and every logical cognitive plane remains regardless of hardware. The strongest future integration check is *one recipient-history + downstream-propagation leak*, *one falsified or stale delegated approval/result*, and *one repeated parametric deletion interference*, each only when its actual execution surface exists.
