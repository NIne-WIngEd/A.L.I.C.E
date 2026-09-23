# Memory Formation Model: independent foundation

**State:** Research-active contract and evaluation tooling. No trained MFM
weights, production inference, full Stage G gold corpus, or Q0–Q5 certification
is claimed.
**Base:** owner-ratified memory identity architecture and Stage G qualification matrix.

## What is underway

The MFM is a learned interpreter of experiences with a reusable, host-neutral
formation capability and an instance-specific adaptation path. It proposes claims,
episodes, preferences, goals, corrections, deletions, contradictions, temporal
scope, and uncertainty. It cannot grant a claim authority, accept a deletion,
rewrite source-person history, or choose a database. The deterministic memory
gate performs those decisions. This work is separate from N0/N1 personality
training and from the existing Phase 2 candidate-staging implementation.

## FBM builds the personal entity, including MFM

The Fable Builder Model (FBM) is the shipped construction capability, not just a
personality-model trainer. Starting with a user's authorized corpus, FBM must
interpret and attribute sources, preserve provenance and unknowns, plan coverage,
generate and critique appropriate synthetic training cases, construct and
evaluate the instance's personality/identity, user/host, memory-formation,
assistant-self and relationship capabilities, connect them to the governed memory
fabric and native judgment path, and continue the governed development loop after
activation. The components can have different weights, representations and
stores; FBM must make their interfaces and version lineage coherent rather than
requiring a user to hand-build or hand-label each component. Where a host has no
separate source-person corpus, FBM must not invent one. A.L.I.C.E. has the extra
Elaina source-person axis; ordinary Fable has a user and a developing Fable self.

The user-facing installation flow is authorization of personal data followed by
FBM construction and evaluation of the personal entity. The product must supply
the host-neutral builder competence, training/update machinery, deterministic
authority, storage/retrieval and serving interfaces. A particular person's raw
data and adapted weights stay within that person's authorized custody. The
builder cannot assume an external teacher, paid API, manually curated gold
corpus, or ideal-self questionnaire at deployment. Sparse or contradictory data
remains uncertain; construction must not fabricate a lived history to claim full
personal fidelity. Normal experiences and their observed outcomes feed subsequent
governed updates to the appropriate model or projection. MFM may reuse learned
representations from FBM, but FBM's build orchestration, MFM's runtime proposals,
the independent authority gate, and model promotion have distinct responsibilities.

The existing `fable-builder-model` branch describes a **working** personality-
focused architecture, not a finalized limit on FBM's duties. This section records
the required MFM integration boundary; it does not implement FBM or take over
the parallel builder/personality work.

`src/cognitive_kernel/formation_contracts.py` establishes a backend-neutral
`FormationContextPacket` and `MemoryProposalBundle`. They bind product and host
scope, evidence roles and modalities, model and context digests, epistemic
status, and typed proposals. Values are references so private evidence does
not enter public repository fixtures. The model's cited evidence must be in
the actual context packet; outside-source text cannot become an authenticated
owner statement just because its words say so. Source authentication and
canonical acceptance remain responsibilities of the deterministic authority
gate, which must check the real registered evidence and actor.

`src/cognitive_kernel/formation_evaluation.py` assesses a candidate's output
against frozen gold cases. It records misses, unsupported additions, and named
critical failures separately. It can evaluate abstention. A good aggregate
does not override a critical identity or provenance failure. These contracts
and checks do not imply a trained model or an integrated runtime.

## Destination behavior traced to the A.L.I.C.E. comic

The 64-page *The Second Mind* storyboard is a fictional destination, not proof
of a current implementation. Its key formation jobs are:

| Story beats | MFM must learn to propose | Separate authority or serving work |
|---|---|---|
| Pages 18–21: anger, uncertain guesses, invented rehearsal, host learning | distinguish observation, direct statement, inference, hypothetical training, source-person evidence, and host subject | authenticate source; keep Elaina canon and Rayan learning separate |
| Pages 25–26: linked priorities and missions | goals, tradeoffs, dependency and temporal context | Mission Graph adjudication and planning |
| Pages 32–35: advice, override, mixed outcome, later lesson | decision rationale, action, outcome, changed assumption, relevant provenance | decision authority, retrieval, native judgment |
| Pages 36–39: repeated pattern, negotiated norm, learned routine | repeated observations, tentative pattern, relationship norm, procedural-skill candidate | ask before adopting a norm; test skill before promotion |
| Pages 40–42: new model and changed goals | portable, temporal, source-linked episode and host-state proposals | preserve entity identity across model replacement |
| Pages 49–56: devices, forgetting, calibration, evolving preference | multi-device provenance, deletion request, calibration evidence, revised preference | custody, deletion/unlearning, serving, calibration and authority |
| Pages 57–64: decades of shared history | assistant-self and relationship observations distinct from host and source-person history | selective retrieval and independent judgment |

The MFM is not the whole companion. It must provide evidence-linked material
from which the host, relationship, assistant-self, episode, mission and skill
planes can develop. The native personal judgment path described in
`docs/FABLE_PERSONAL_DEVELOPMENT_ARCHITECTURE.md` remains a separate causal
qualification. Storing a pattern does not prove it changes judgment.

## Learning and acceptance program

The destination is a learned, context-conditioned, multimodal formation
component with its own evaluated weights. It consumes authorized experience
plus a scoped Formation Context Packet and proposes typed, evidence-citing
semantic changes. The semantic taxonomy is versioned and expandable; the
current names are an initial contract surface, not a lifetime limit on memory
or modality. Text, audio, image, video, code, action and sensor evidence are
eligible. Encoders, model size, fusion, training data mixture and hardware
placement are selected by capability evidence, not an arbitrary cap.

Training data must distinguish original observations from model-generated
reconstructions, direct owner speech from outside-source quotation, historical
versus current truth, and private source-person material from host and current
assistant experience. Preserve source linkage, dataset lineage, exact
exclusions, temporal splits, model/checkpoint hashes, and deletion influence.
Synthetic continuation can stress the system; it cannot be relabeled as
Elaina's or Rayan's actual history. Private data stays in authorized custody.

### Synthetic data and gold evidence

| Origin | MFM/FBM use | Provenance and evaluation boundary |
|---|---|---|
| Licensed or permissioned host-neutral material and fabricated fictional users | Train source attribution, temporal revision, contradiction handling, outcomes, abstention, cross-modal grounding and proposal formation; cover rare and adversarial cases | Mark as third-party or fictional; never identify a fictional case as an actual user's past |
| Procedurally generated sequences and controlled counterfactuals | Vary speakers, times, evidence quality, decisions, corrections, relationships and outcomes; train contrasts and stress-test the complete formation-to-judgment loop | Retain generation recipe, seed, parent case and label rationale; split held-out evaluations by underlying source and generator family |
| Private Elaina evidence (E0) and evidence-constrained E-INF/A-SYN completion | Train and evaluate A.L.I.C.E. identity distinctions and, where authorized, source-person versus host formation | E0 is attested source; E-INF and A-SYN are inferred/synthetic and never become Elaina historical evidence; no private identity bytes or weights ship in Fable |
| Private Rayan corpus and synthetic Rayan-life sequences | Train and evaluate A.L.I.C.E. host, relationship and memory formation, plus simulated long-lived outcomes | Keep actual Rayan observations distinct from fictional continuations; never transfer his data or derived private weights to another user |
| Later per-user authorized evidence and outcomes | FBM creates an instance-specific training/evaluation substrate and revises its stack under governance | Record consent, subject, source, time and derivative lineage; use held-out real evidence and feedback to assess personal fidelity |

Synthetic examples teach general formation operations and probe failure modes;
they cannot certify that a specific personal claim is true or that a model knows
a real person. Gold for personal fidelity is adjudicated from independent actual
evidence, corrections and observed outcomes. Do not grade a generator only on
its own invented answers or allow variants of a source case to cross into the
held-out set. Preserve all generated-source lineage for later correction and
deletion propagation. User review is useful for disputed personal interpretations
but is not a required manual annotation workload for every Fable installation.

Frozen behavioral qualification must include extraction, multi-session
reasoning, temporal updates, supported abstention, long-running decision and
outcome lineage, relationship and skill formation proposals, model replacement,
source collisions, prompt injection, and deletion/revocation requests. Evaluate
formation quality both alone and through the deterministic gate, projections,
retrieval and later native judgment. Isolate interventions in host, relationship
and assistant-self state and show a causally relevant change in future judgment.
Report critical errors individually and never hide them in average scores.

Relevant research starting points: [Generative Agents](https://arxiv.org/abs/2304.03442)
separates observation, reflection and planning;
[LongMemEval](https://arxiv.org/abs/2410.10813) tests extraction, multi-session
reasoning, temporal updates and abstention;
[LongMemEval-V2](https://arxiv.org/abs/2605.12493) adds workflow knowledge and
environment failures; [A-MEM](https://arxiv.org/abs/2502.12110) studies
dynamic linking and revision. These motivate qualification coverage; they do
not establish that one architecture or benchmark passes A.L.I.C.E.'s gates.

## Next independent implementation work

1. Build the context planner that selects relevant registered evidence across
   memory planes and loads authorized content behind the packet's references.
   Compare learned planning to simpler mechanisms on frozen cases.
2. Build full gold semantic decomposition and stress cases for source-person,
   host, relationship, continuity, and world domains. Keep real owner material
   in its authorized private custody domain. Never copy it into Friday.
3. Train and compare learned formation candidates with frozen evaluations for
   temporal changes, contradictions, ambiguity, identity collisions, injections,
   corrections, deletion requests, and abstention. Record dataset and model
   lineage. Do not use N0 weights as a prerequisite.
4. Qualify complete context-planner -> MFM -> deterministic gate -> projection
   behavior against the owner-ratified Stage G matrix. Evaluate engines on
   capability, license, owner custody, and operational cost without declaring
   a permanent backend or assigning a production authority prematurely.

The existing Stage G candidate inventory remains unchanged. Adding or removing
mandatory qualification candidates requires its separate owner-ratified
amendment. This branch changes no current production routing or authority.
