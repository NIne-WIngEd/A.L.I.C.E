# Frontier Watch intake — 2026-10-31

**Disposition.** Three newly surfaced late-October primary research papers expose material *future* mechanisms for A.L.I.C.E./Fable: evidence-gated weight learning, joint routing of experience and learning mechanism, and active cognitive context checkpointing. Two earlier October papers are newly surfaced writer/evaluation methods. **No current architecture-invalidating evidence, existing selected-store contract failure, or justification to change active MFM/personality training.** Do not open a backend tournament, create a smaller cognitive tier, or turn a paper's model count, method library, memory size, benchmark, or runtime into a release ceiling.

## Required baseline and execution inspection before judgments

Read actual `docs/research/FRONTIER_WATCH_FULL_FABLE_V1_BASELINE_2026-09-27.md` on this research branch and preceding October 30 and October 29 intakes; checked A.L.I.C.E. main owner plan, Memory v4.1 and personal development architecture; Fable main README and updated Oct 10 execution profile; current MFM/FBM diagnostic commits and continuity.

| Source | Verified head before intake |
| --- | --- |
| A.L.I.C.E. main | `a1fe0a65df1d4143955eb1d3b35eb4ac590e7555` |
| Fable_Sleight main | `f8e9062ebd657f18c3a571cedc1206889d7fd53a` |
| MFM engineering (`research/mfm-v1-licensed-base-20260930`) | `5653736f545c9c1c8d5d8006af0f1fa152eed9b1` |
| FBM engineering (`research/fbm-v1-licensed-base-20260930`) | `bcd462b5bb93f41a56f79b5b63e3337c8e89bb45` |
| N0 full-envelope | `5e29f7f69ba4a5d031c7036639b67bebbcdc0bd2` |
| alice-context | `945245bbee4168fff46f863455a83bb219063017` |
| Graphify context | `a06f61eafd91d04fb7ec7c1de3ab8a9ed7071b15` |
| frontier-watch before intake | `590391f93ac9666f9c6ce4f396342a0cf3bbe5ad` |

Current real learning issue: MFM checkpoint-10 native evaluation **0/9 complete grounded formations** (8 alignment, 1 parse), with early lexical-task-token competition despite better teacher-forced loss. Sixteen additional full-nine updates toward checkpoint 26 were dispatched, but **no newer committed terminal learning/qualification result**; personality contextual judgment also remains unqualified. No cited new paper tests or remedies that specific failure, so don't disturb its evidence-based next action.

Full Fable v1 remains the complete transferable personal foundation with canonical Experience, bitemporal Claim Authority, raw/source-native/episodic/graph/vector/multimodal/perceptual and host/source-person/relationship/self/causal/skill/mission states, activation/parametric memory, MRM/Recollection/Lifecycle Curator, governed deletion, multi-device, Mission Graph, Workspace, learning and rollback. Selected physical placements unchanged: KurrentDB + JetStream, XTDB v2, encrypted content-addressed evidence, LadybugDB/NebulaGraph, Qdrant, Temporal, L1/Valkey, PyTorch+Accelerate. F4–F11 internal; first consumer release predicate remains `full_personal_cognitive_foundation_after_f11`.

## 1. Surprise Alone Learns Lies — novel learning-admission failure

**Primary:** Metales and Grabenheinrich, *Surprise Alone Learns Lies: Evidence-Gated Continual Learning for LLMs from Streaming Claims*, arXiv:2610.39840v1, **submitted Oct 29 2026**. https://arxiv.org/html/2610.39840

**Class:** Future personal-model/parametric-memory **learning-qualification challenger**, not a demonstrated current MFM defect.

**Method:** For each incoming atomic claim, compare (i) novelty/surprise (student negative log-likelihood), (ii) disagreement with **frozen base model** prior, (iii) independent verifier, and sometimes (iv) unrelated-fact regression rollback. The student is a Qwen2.5 1.5B/3B or Llama 3.2 3B frozen base plus rank-16 LoRA. Incoming claim chunks of 16 train for eight AdamW steps on accepted and replayed facts, with a base KL regularizer and reversible optimizer snapshots. A distinct source-recovery route considers contrary claims only with externally verified support. Experiment includes invented people, known-fact CounterFact lies and legitimate factual changes.

**Evidence and ablations:** Surprise-only internalized **84–93%** of streamed lies, including **99–100%** of lies contradicting known facts; random same-budget selection similarly failed. Prior-conflict screening distinguished known lies from truly novel claims (AUC **0.78–0.83**) but not from **legitimate corrections** (AUC **0.53**), and saved **43%** of verifier calls; adaptive adversaries can evade it. Unrelated-fact stability checks reduced overall learning without filtering targeted poison. Independent verification works only to the extent the judge and its sources are trustworthy: with forged matching evidence for **30%** of lies, the evidence-route policy was **worse than verifier-only**. Research runs reportedly feasible on one 16GB GPU for 5–12 minutes, but this is a small atomic-fact study rather than long-horizon personal training.

**Already covered:** A.L.I.C.E. preserves canonical source evidence separate from governed Claims, subject attribution, source trust, correction and revocation lineage, dataset/model versions and rollback. **Genuinely new:** direct controlled evidence that *novelty*, *model conflict*, *general-regression safety*, and even *apparent source corroboration* are separate non-equivalent signals for parametric update admission. A source that can be planted is not independent verification.

**Governance/risks/cost:** Do not train weights from an attractive but unadjudicated contradiction, nor suppress an authorized correction because the frozen prior disagrees. Bind each training assertion to exact subject, actual source custody, authoritative Claim/time/version, consent, revocation, data/model influence and rollback. Verifier calls and replay incur cost; evaluation doesn't show cross-layer deletion or real attackers with arbitrary modalities. **Action:** At the actual governed weight-promotion boundary, one focused mixed stream containing genuine novel evidence, authorized correction, false contrary assertion and forged corroboration; measure accepted train examples and later behavior, not just QA or regression loss. **No current MFM recipe change.**

## 2. Routed Continual Learning — route experience *and* learning mechanism together

**Primary:** Philip et al. (FAIR/Meta), *Routed Continual Learning*, arXiv:2610.38796v1, **submitted Oct 28 2026**. https://arxiv.org/html/2610.38796

**Class:** Future MRM / Cognitive Recollection / procedural-learning **resource-allocation challenger**.

**Method:** For each task, a model router chooses an existing experience cluster or opens a new one, binding the cluster to a memory strategy from a fixed experimental library (raw trajectory ICL, extracted strategy, structured numerical state, etc.). A Context Memory Adapter renders appropriate state into the agent context and updates only its selected cluster after the run. Routing and the representation choice happen jointly, rather than relying on one growing global memory.

**Evidence and ablations:** Four LLMs, five matched rollouts, **398 heterogeneous interleaved tasks** (301 CL-Bench / six kinds + 97 banking). Against global ICL, routed CL improves normalized reward by roughly **0.15–0.30 of attainable score**, with estimated API cost **65–72% lower**. GPT-5.6 Sol global ICL **0.168**, routed-ICL **0.387**, routed-CL **0.464** (ground-truth assignment 0.478). Gemini 3.7 Flash routed ICL 0.211 vs full routed CL **0.213**: much of the gain can come from clustering, not switching memory systems. Baselines separate no-history, single global ICL, routed ICL and routed library. Experimental stream is small and task-oriented; LLM routing, summaries and update state cost extra. No year-scale cross-host memory, correction, multimodal or deletion evidence.

**Already covered:** Full Fable has MRM scheduling over multiple cognitive forms, diverse retrieval methods, domain/project/mission views, activation and procedural memory. **Genuinely new:** explicit empirical case for jointly deciding **which prior experiences share context** and **which learning representation to use**, including cost-effectiveness on mixed workloads. Fixed five-strategy library and finite task count are benchmark specifics, not an ontology or ceiling.

**Governance/risks:** Cluster scope cannot accidentally merge hosts, source-persons, products or permission domains. A routed summary is only a projection; consult canonical Experience/Claim authority, temporal validity, deletion/rollback and source-native evidence. **Action:** Preserve for future MRM scheduling if actual heterogeneous task interference or cost requires it; first diagnose the observed misrouting against existing selectors. **No new store, benchmark matrix, or model architecture.**

## 3. CORAL — agent-initiated Workspace checkpoint and episode reset

**Primary:** Zhu et al., *Don't Lose the Thread: Empowering Long-Horizon LLM Agents with Cognitive Resource Self-Allocation*, arXiv:2610.40350v1, **submitted Oct 29 2026**. https://arxiv.org/html/2610.40350

**Class:** Future conditional **Mission Graph / Cognitive Workspace / harness improvement**.

**Method:** Agent-callable operations checkpoint progress and allegedly verified facts into a retained task summary; a separate operation starts a clean reasoning episode using that state. Supervised examples plus multi-episode policy optimization teach agents when to checkpoint/compact/resume, rather than indefinitely appending conversation context. The system is not permanent memory deletion or a new cognitive tier.

**Evidence and ablations:** Text-only GAIA (103 tasks, eight sampled trajectories per task) on Qwen3-8B: ReAct baseline about **19.4%**, checkpointing prompted 20.4%, trained SFT **41.6%**, trained RL **42.8%**; **6–7x fewer input tokens** vs ReAct in measured setup. BrowseComp test of 150 improves strong external agent **22.7% to 30.7%**. Training ablations: full SFT 41.6%; minus math examples **39.9%**, minus incorrect trajectories **39.3%**, matched-size random subset **36.5%**. Additional RL improvement is modest; no evidence of broad generalization to personal longitudinal workflows. Training adds synthetic checkpoint supervision and policy rollouts. Tool calls, context summarization and reset cost runtime compute but can reduce future input tokens.

**Already covered:** Mission Graph/durable workflow progress, Cognitive Workspace, working memory, source-native lookup, verified outcome receipts and original canonical Experience. **Genuinely new:** training an agent to **self-schedule active context reset** while resuming from a bounded durable task checkpoint.

**Authority/risks:** Self-declared “verified fact” is not Claim Authority. If a checkpoint carries an incorrect, revoked or stale premise and wipes its counterevidence, the restarted agent can propagate an error. Persist checkpoint provenance, source references, policy head and active permissions; on restart revalidate current Claims and action eligibility. Erasing the ephemeral context must never delete canonical raw evidence or Experience. **Action:** Only when a real long-Mission bloat/recovery cost appears, test one stale-checkpoint correction and valid progress recovery; do not replace Temporal, truncate cognition, or interrupt current model work.

## 4. Hy-Memory-30B-A3B — observational process credit for memory writing (newly surfaced older Oct paper)

**Primary:** Fei et al. / Tencent Hunyuan, *Hy-Memory-30B-A3B: Attributed Reinforcement Learning for Long-Term Memory Writing*, arXiv:2610.22196v1, **submitted Oct 15 2026; surfaced in this intake**. https://arxiv.org/html/2610.22196

**Class:** Future trained MFM/write-policy challenger; not a new physical stack nor specific remedy for checkpoint-10 failure.

**Method:** On-policy distillation across multiple write protocols, then **writer-only** memory RL using per-chunk recall probes, traps for over-retention, cross-chunk answer questions whose evidence is attributed to source chunks, a bank-growth penalty and chronological policy-written bank carry-forward. Frozen reader/judge helps isolate writer changes from answer-model improvements.

**Evidence:** Fixed **20% benchmark subsamples**, four-harness macro-average **55.49** versus **54.62** Claude Opus 5; held-out EverMemOS **56.89** vs **54.04**. Eval includes Mem0/MemOS/Hy-Memory/EverMemOS. Reward artifact uses 57,616 chunks, about 164,576 positive probes, 8,599 traps and 60.8k multi-chunk questions. RL trains ~3B active MoE on **16 nodes x 8 H20 GPUs (128 GPUs)**, costly and not justified for current private serving. Measured gains are modest, model- and benchmark-specific; reward probes can miss authority violations and true deletion requirements.

**Already covered / new:** MFM formation bundles and evidence citations already planned; distinct future option is **writer-attributed outcome supervision** preserving bank state across windows rather than broadcast final QA reward. No canonical Claim grants from writer; maintain subject/consent/provenance, source invalidation, deletion and model rollback. **Action:** Research-only reference for future observed write-error attribution; no present RL project or MFM objective edit.

## 5. FEB-30 — intrinsic memory-state evaluation (newly surfaced older Oct paper)

**Primary:** Toma et al., *Beyond Downstream QA: Rethinking How We Evaluate Agent Memory*, arXiv:2610.22209v1, **submitted Oct 15 2026; surfaced in this intake**. https://arxiv.org/html/2610.22209

**Class:** **Already covered / corroboration** of independent canonical-state vs downstream-response evaluation, with a useful future focused diagnostic.

**Method:** 30 fact-evolution trajectories / six categories / three difficulties, with known source/authored evolution and inspection of a memory system after *each* ingestion, alongside its later QA answers. Includes multi-author histories, supersession and temporality; 11 runs of a representative extract/embed/classify/synthesize implementation.

**Evidence:** Intrinsic vs extrinsic mean absolute disagreement **0.168 (16.8 points)**; **43.6%** of trajectories diverge by more than 0.15. Observed memory-operation classification only **20.1%**, with **22.2% false updates** and 41.5% intrinsic multi-author attribution accuracy. One system + largely synthetic trajectories: not proof A.L.I.C.E. has these issues. The design explains why good final answers or lower teacher-forced loss cannot independently qualify memory formation.

**A.L.I.C.E. implication/action:** Already separates canonical evidence, bitemporal Claim transition, formation proposals, and downstream output; eventual qualification should inspect **actual proposal/Claim create-update-reinforce-supersede state**, source-person ownership and rollback independently of QA. Only focused test on a real failing transition when the current native MFM works. No duplicate benchmark program.

## Screened, no current action

- *Context Compaction for World Models*, arXiv:2610.38980, Oct 28: context selection extends retrieval in ~14M-param synthetic room-generation model up to 4x context; narrow spatial video benchmark, not a compelling different Fable serving mechanism and cannot replace canonical perceptual history.
- *Memory-Centric Continual Learning for AI Agents*, arXiv:2610.29357, Oct 21: broad conceptual taxonomy/learning loop already covered by Fable; not an experimentally established new blocker.
- Previously tracked late-October Task-Graph Contextual Integrity, PCM stale proof, COP sequential unlearning, MemCalib/JitMem/REALM/AdaMM/MUTE, and other previous deltas are **not** new findings here.

## Execution/disposition

No architecture-invalidating evidence, no real selected placement blocker, no right to introduce a fixed ontology/store count, no unambiguous current correction to production semantics. The three recent papers strengthen future decisions about safe parametric learning admission, heterogeneous memory scheduling and active context resets. Current engineering priority remains **complete and independently assess MFM's ongoing native formation continuation** and **resolve measured personality contextual judgment**, keeping authority and lineage gates intact. All findings are notes, not release gates; no production/main change or PR requested.
