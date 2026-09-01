# A.L.I.C.E. Stage G / G+ Memory Architecture Handoff

**Purpose:** Handoff for the A.L.I.C.E. Stage G working chat.  
**Date:** 2026-08-23  
**Scope:** Records the newly clarified lifelong-memory architecture direction, what Stage G must prove, what must wait until the new post-G stage ("G+"), Phase 2 governance during the work, and how this changes the implementation of future Phases 6–15 without changing the frozen top-level roadmap.

---

## 1. Core conclusion

A.L.I.C.E. must **not** be designed as one infinitely growing chat window or as an LLM with an "unlimited token" context.

The correct model is:

> **A finite-context model connected to a persistent, scalable, addressable lifelong memory system that dynamically retrieves only the memories relevant to the present interaction.**

Alice may eventually have years or decades of data, but the LLM should never load the full lifetime on every turn.

Instead:

```text
                 A.L.I.C.E. LIFETIME
                         │
        ┌────────────────┼────────────────┐
        │                │                │
     Claims          Experiences       Evidence
        │                │                │
        └────────────────┼────────────────┘
                         │
                  Canonical Memory
                         │
          ┌──────────────┼──────────────┐
          │              │              │
       Semantic       Relational      Temporal
        Index           Graph          Index
          │              │              │
          └──────────────┼──────────────┘
                         │
                 Memory Retrieval
                         │
             small relevant memory set
                         │
                         ▼
                    finite LLM context
```

The important distinction is:

- **Lifetime storage can be very large.**
- **Working context stays finite.**
- **Retrieval reconstructs the relevant slice of the lifetime when needed.**

This is the architecture that makes decades-long memory computationally plausible.

---

# 2. "Unlimited memory" wording must be precise

Do **not** describe Alice as literally having infinite or unlimited memory.

Use this instead:

> **A.L.I.C.E. has persistent, scalable, addressable lifelong memory independent of the underlying LLM context window.**

Or, in plain language:

> Alice does not need to keep her whole life in working context in order to remember it.

A retained memory may be old, cold, compressed for indexing, or absent from current context, but it must not become permanently inaccessible merely because it is old.

---

# 3. Two separate memory guarantees

This distinction is now important enough to treat as a permanent architectural concept.

## 3.1 Recallability Guarantee

> **Every retained canonical memory must remain technically addressable and recoverable regardless of whether it is currently present in working context or a fast retrieval projection.**

This is primarily an engineering/information-retrieval guarantee.

If a fast semantic lookup misses a memory, Alice should be able to widen the search:

```text
fast semantic path
        ↓
entity / relationship path
        ↓
temporal path
        ↓
graph expansion
        ↓
keyword / full-text path
        ↓
source / canonical-store recovery
```

"Harder to retrieve" is acceptable.

"Silently inaccessible despite still being retained" is not.

## 3.2 Attention Guarantee

> **Alice should proactively attempt to surface relevant memories without requiring the host to explicitly ask her to remember them.**

Example:

```text
Host:
"Everyone wants me to go back there next month.
I don't really want to."
```

Alice should be capable of surfacing an old relevant experience even though the host never said:

```text
"Search my memory."
```

This second guarantee is not deterministic in the same way as canonical addressability.

It is a cognitive retrieval / attention problem.

---

# 4. Why this is possible even if a base LLM does not have the capability

Alice is a **system**, not merely a naked LLM.

The system can have capabilities the underlying model does not natively possess.

```text
              BASE LLM
                  +
        Canonical memory store
                  +
             Qdrant
                  +
              Neo4j
                  +
        Temporal/source indexes
                  +
       Memory formation/governance
                  +
      Memory attention/controller
                  =
               A.L.I.C.E.
```

The lifetime should live in Alice's external memory architecture, not be embedded only inside one model's context or weights.

Therefore the underlying LLM can later be upgraded or replaced without erasing Alice's lifetime.

---

# 5. Existing architecture remains directionally correct

Do **not** replace the current memory foundation.

The existing direction remains sound:

- canonical Claim Store / Claim Fabric;
- Experience Ledger;
- Raw Buffer / Workspace;
- object/evidence storage;
- Memory Formation Manager;
- Memory Gate / Authority controls;
- Qdrant semantic projection/index;
- Neo4j relationship/graph projection;
- provenance;
- correction;
- deletion;
- rollback;
- rebuildable projections/indexes;
- strict separation between canonical state and derived retrieval infrastructure.

The key principle remains:

> **Indexes are not the source of truth.**

If Qdrant is rebuilt, memories must survive.

If Neo4j is rebuilt, memories must survive.

If the embedding model changes, canonical memory must remain available for re-embedding.

If a summary is wrong, Alice must be able to return to the underlying evidence.

---

# 6. The major missing layer: Memory Attention / Retrieval Controller

The current foundation is stronger at **possessing memory** than at **remembering intelligently**.

A future dedicated layer must decide:

> **What should Alice remember right now?**

Conceptual architecture:

```text
USER MESSAGE / CURRENT STATE
            │
            ▼
   Situation Interpretation
            │
    ┌───────┼────────┐
    │       │        │
 entities  goals   timeline
 people    mission  emotion
 context   thread   uncertainty
    │       │        │
    └───────┼────────┘
            ▼
     Retrieval Planner
            │
 ┌──────────┼────────────────────────────┐
 │          │          │        │        │
semantic   entity    temporal   graph   episodic
search     search     search    walk    patterns
 │          │          │        │        │
 └──────────┴──────────┴────────┴────────┘
                    │
                    ▼
              candidates
                    │
                    ▼
                 reranker
                    │
                    ▼
            context allocator
                    │
                    ▼
                   LLM
                    │
          "Need more memory?"
              /           \
            yes            no
             │              │
       deeper retrieval    answer
```

The controller will eventually need to reason over factors such as:

- semantic relevance;
- entity identity;
- relationship relevance;
- temporal relevance;
- causal relevance;
- current mission;
- current active thread;
- recency;
- importance;
- emotional significance;
- unresolved events;
- frequency;
- authority;
- confidence;
- contradiction;
- host-specific relevance;
- long-term goals;
- retrieval cost;
- latency;
- context budget.

This should become a major first-class architectural component.

---

# 7. Important efficiency principle

Alice must **not** perform exhaustive lifetime retrieval on every message.

Example retrieval budget:

```text
LOW MEMORY NEED
"lol"
→ current context / hot state only

MEDIUM MEMORY NEED
"what did she say?"
→ recent + entity + relationship retrieval

HIGH MEMORY NEED
"why do I keep making this same mistake?"
→ deep episodic + causal + graph + long-horizon retrieval

EXPLICIT ARCHIVAL SEARCH
"find exactly what I said about this in 2028"
→ exhaustive multi-index / source-aware retrieval
```

Cold memory means "not constantly loaded," **not forgotten**.

Possible memory temperature model:

```text
HOT
current conversation
active mission
current people/events
        │
        ▼
WARM
recent experiences
frequently used memories
unresolved threads
        │
        ▼
COLD
older autobiographical memory
        │
        ▼
ARCHIVE
raw/source evidence
```

All retained canonical memory remains recoverable.

---

# 8. Associative activation must eventually exist

Memory retrieval cannot rely only on vector similarity.

One memory can trigger related memories:

```text
NVIDIA
   ↓
open-source contribution
   ↓
2026
   ↓
AMD contribution
   ↓
career strategy
   ↓
GPU / quantitative / ML goals
```

This suggests controlled graph-based associative activation.

However, graph expansion must be bounded.

The future controller must decide:

- which edge types to traverse;
- how far to traverse;
- how much activation decays by hop;
- how salience changes the search;
- when to stop expansion;
- how to prevent graph explosion.

---

# 9. Active unresolved threads should become first-class memory state

Some items should remain mildly salient without requiring full lifetime search:

- pending interview;
- unfinished project;
- unresolved disagreement;
- unanswered question;
- future deadline;
- promise;
- waiting-for-response state;
- active research problem;
- current goal;
- current relationship issue;
- pending decision.

Conceptually:

```text
ACTIVE THREADS

A.L.I.C.E. memory migration       0.95
current interview preparation    0.83
open-source contribution         0.61
old resolved topic               0.08
```

This enables statements such as:

```text
"I'm nervous about Tuesday."
```

to resolve against currently active context before launching a huge historical search.

---

# 10. Two retrieval modes are required

## Automatic Recall

Fast and proactive.

Alice decides what old memory is probably relevant to the current interaction.

```text
current message
    ↓
context interpretation
    ↓
fast multi-path retrieval
    ↓
top relevant memories
```

## Deliberate Recollection

Alice detects that the first retrieval is insufficient.

```text
"I think something from the past matters here."
        ↓
broader semantic search
        ↓
temporal expansion
        ↓
entity / graph expansion
        ↓
raw/source search if needed
```

This distinction should eventually be explicit in the architecture and evaluation.

---

# 11. Stage G must NOT become the full Memory Attention implementation stage

The full Memory Attention / Retrieval Controller should be built **after Stage G**, in a dedicated post-G stage currently referred to as **G+**.

Reason:

Stage G's job is to qualify the underlying memory substrate.

If G simultaneously introduces:

- new retrieval planner;
- salience manager;
- active-thread system;
- associative activation;
- context allocator;
- multi-pass recollection;
- new agent loop;

then a failure becomes difficult to localize.

Example failure:

```text
User:
"I don't want to go back there."
```

Possible causes would become:

- memory not stored;
- memory stored incorrectly;
- embedding failure;
- entity resolution failure;
- graph failure;
- temporal indexing failure;
- planner failure;
- salience failure;
- reranking failure;
- context allocation failure;
- LLM reasoning failure.

That is too much architectural movement inside one qualification stage.

Therefore:

> **Stage G validates the substrate and measures cognitive retrieval weaknesses. G+ builds the cognitive retrieval machinery.**

---

# 12. Revised Stage G mission

Stage G should answer:

> **Can Alice's memory substrate preserve, route, relate, correct, delete, rebuild, and recover long-horizon memory reliably?**

Stage G must continue validating:

- canonical memory persistence;
- Claim Store / Claim Fabric behavior;
- Experience Ledger behavior;
- Object Store / raw evidence behavior;
- Raw Buffer / Workspace;
- Memory Formation Manager;
- Memory Gate / Authority Manager;
- Qdrant;
- Neo4j;
- storage routing;
- routing conflicts;
- cross-layer retrieval;
- temporal retrieval;
- correction;
- supersession;
- deletion propagation;
- rollback;
- rebuildability;
- provenance;
- authority;
- long-horizon retrieval;
- combined-system behavior;
- realistic LifeSim data.

Known prior Stage G requirements already include testing memory layers separately and together under complex synthetic real-life-person data.

---

# 13. New requirement to add inside Stage G: cognitive retrieval baseline

Stage G should **not implement G+**, but it should establish the baseline that G+ must improve.

Add a dedicated section to Stage G evaluation:

> **LONG-HORIZON COGNITIVE RECALL BASELINE**

Test at least these three levels.

## 13.1 Explicit Recall

Example:

```text
"What happened with Maya at Tahoe in 2019?"
```

The query names the event/entity/time.

Expected: high success.

## 13.2 Indirect Recall

Example:

```text
"Why might I not want to go back to Tahoe?"
```

The query does not name the original event.

Expected: relevant historical event should become retrievable.

## 13.3 Spontaneous Recall Candidate Test

Example:

```text
"Everyone wants me to go skiing there next month.
I don't really want to."
```

No explicit request to remember anything.

Expected during Stage G:

The system does **not yet need a full cognitive controller**, but the test should measure whether the relevant historical memory appears anywhere in the candidate set and where it ranks.

Record data such as:

```text
relevant memory exists                YES
canonical memory intact               YES
semantic path found it                YES/NO
entity path found it                  YES/NO
graph path found it                   YES/NO
temporal path found it                YES/NO
candidate set contains it             YES/NO
candidate rank                        #__
entered final working context         YES/NO
retrieval latency                     __
retrieval cost                        __
false candidates                      __
```

This gives G+ a measurable starting point.

---

# 14. Proposed Stage G retrieval metrics

Stage G should add or preserve metrics such as:

- Exact Recall;
- Entity Recall;
- Temporal Recall;
- Semantic Recall;
- Graph Recall;
- Cross-Layer Recall;
- Indirect Recall;
- Spontaneous Candidate Recall@K;
- Relevant@K;
- MRR / rank of ground-truth memory where useful;
- False Recall Rate;
- retrieval latency;
- retrieval cost;
- candidate-set size;
- deep-recollection fallback success;
- index rebuild consistency;
- canonical-source recovery success.

## Critical new metric

### Spontaneous Candidate Recall@K

Definition:

> Without explicitly naming the old memory, did the Stage G retrieval substrate place the genuinely relevant historical memory among the top K candidates?

This is the bridge between substrate qualification and cognitive retrieval.

Example:

```text
Before G+:
important memory rank = 37

After G+:
important memory rank = 2
→ enters working context
→ Alice can use it
```

---

# 15. Stage G vs G+ boundary

Use this conceptual boundary:

```text
STAGE G
════════════════════════════════════════
"Can Alice preserve and reach her past?"

Prove:
- memory exists
- memory survives
- memory remains canonical
- memory is addressable
- memory can be recovered
- memory corrections work
- memory deletion/rollback works
- indexes are rebuildable
- cross-layer retrieval works
- long-horizon memory remains reachable
- baseline spontaneous candidate retrieval is measured

                    ↓

STAGE G+
════════════════════════════════════════
"Can Alice know when her past matters?"

Build:
- Memory Attention Controller
- Situation Interpreter
- Retrieval Planner
- Salience Manager
- Active Thread Manager
- Associative Activation
- Retrieval Budgeter
- Candidate Reranker
- Context Allocator
- Multi-pass Recollection
- Cognitive recall evaluations
```

Short form:

> **G proves Alice cannot lose access to her past.**

> **G+ teaches Alice when her past matters.**

---

# 16. Phase 2 governance during G and G+

Phase 2 remains the governing/canonical authority throughout Stage G and throughout G+.

Do **not** retire Phase 2 merely because G or G+ passes.

Core rule:

> **New capability does not imply new authority.**

The candidate architecture may:

- operate in shadow mode;
- retrieve from candidate memory structures;
- run evaluation;
- produce candidate behavior;
- compare against Phase 2;
- prove new cognitive retrieval capability.

But Phase 2 remains the released authority until the existing migration/cutover sequence completes.

Current conceptual sequence:

```text
Phase 2 = GOVERNING / CANONICAL
        │
        ▼
Stage G
substrate qualification
        │
        ▼
Stage G+
memory cognition qualification
        │
        ▼
Phase 2 STILL GOVERNING
        │
        ▼
Stage H
bounded canary authority
        │
        ▼
Stage I
full canonical transition candidate
        │
        ▼
Stage J
compatibility + rollback qualification
        │
        ▼
OWNER ACCEPTANCE
        │
        ▼
Phase 2 retirement / new canonical authority
```

Do not use:

```text
G+ passed
→ Phase 2 retired
```

That is explicitly not the intended governance model.

---

# 17. Stage G current known technical context

The Stage G working chat should preserve awareness of the existing memory-validation program.

Known prior context includes:

- Stage G must validate complex synthetic real-life-person data.
- Layers must be tested separately and together.
- G2 LifeSim v2 previously used:
  - canon = 128;
  - synthetic = 43;
  - total = 171;
  - hashes verified.
- Live validation previously exercised Neo4j and Qdrant.
- Known live environment included:
  - Docker 29.6.2 aarch64;
  - Neo4j 2026.06.0-ubi10;
  - Qdrant v1.18.3.
- Prior Stage G live evaluation verified mutation/correction/deletion-style changes in Neo4j and Qdrant and recorded final-state hashes.
- Phase 2 remains canonical until the later cutover/rollback qualification sequence completes.
- Stage G exists to qualify the candidate matrix before closure, not to silently replace Phase 2.

Treat exact current repository state, branch, head SHA, generated evidence, and completed sub-gates as runtime facts that must be re-read from the active Stage G repo/handoff before executing new work.

---

# 18. G/G+ will pull forward future-phase work

Completing G and especially G+ will inevitably implement some capabilities that were originally expected later.

This is **not roadmap failure**.

The frozen A.L.I.C.E. roadmap defines top-level capability domains.

It should not be interpreted as:

> "Capability X may not exist until Phase X."

Instead:

> **Phase X is the point by which capability domain X must be complete, integrated, evaluated, and satisfy its exit criteria.**

Earlier phases may create foundations or even fully qualify pieces of later capability if required by sound architecture.

---

# 19. Do NOT renumber or replace Phases 6–15

The top-level roadmap should remain frozen.

Current high-level destinations remain:

```text
Phase 6
Cognitive Control Plane / Inspector / UI / Voice

Phase 7
Universal Integrations / Capability Fabric / Multimodal Perception

Phase 8
Autonomous Memory / Reflection / Procedural Learning

Phase 9
Cognitive Core

Phase 10
Planning / Curiosity / Proactive Agency

Phase 11
Computer Use / Autonomous Coding / Skill Synthesis

Phase 12
Scientific Discovery / Simulation / Formal Reasoning

Phase 13
Continual Model Adaptation / Self-Training

Phase 14
Operating Environment / Embodiment

Phase 15
Generalized Platform / Agent Federation / Frontier Research
```

G/G+ should change **internal implementation plans and inherited capability status**, not top-level numbering.

---

# 20. Expected impact on each future phase

## Phase 6 — mostly unchanged

Phase 6 still owns:

- cognitive control plane;
- inspector;
- correction UI;
- memory/belief/goal inspection;
- permissions;
- override;
- rollback;
- shutdown;
- voice;
- desktop/mobile surfaces;
- explanation views.

G+ may make some inspector contracts richer because Phase 6 can expose:

- why a memory was recalled;
- which retrieval path found it;
- memory rank;
- salience;
- active-thread source;
- context allocation decision.

But Phase 6 still has substantial unique work.

---

## Phase 7 — mostly unchanged

Phase 7 still owns:

- files;
- email;
- calendars;
- repositories;
- APIs;
- services;
- images;
- audio;
- video;
- telemetry;
- sensors;
- capability routing;
- multimodal perception.

G/G+ mainly gives Phase 7 a stronger destination for acquired information.

It does not replace Phase 7.

---

## Phase 8 — biggest overlap / biggest internal re-baseline

Original Phase 8 includes:

- automated memory formation;
- consolidation;
- belief revision;
- source-trust learning;
- skills;
- compression;
- forgetting;
- training candidates;
- reflection;
- procedural learning.

G/G+ will likely pre-complete or substantially advance:

- long-horizon retrieval;
- semantic/graph/temporal retrieval;
- memory attention;
- retrieval planning;
- salience;
- associative recall;
- active-thread retrieval;
- cognitive recall evaluation;
- context allocation.

Therefore Phase 8 should **inherit** these capabilities if they are already proven.

It must not rebuild them simply because they were originally placed in Phase 8.

Phase 8 can then focus more strongly on:

```text
"What deserves to become memory?"
"What did Alice learn?"
"How do experiences consolidate?"
"How do beliefs change?"
"How do repeated behaviors become skills?"
"What should be compressed or archived?"
"What becomes a model-training candidate?"
```

Likely result:

> Phase 8 becomes more about autonomous learning and memory evolution than basic memory retrieval.

---

# 21. Phase 9 — meaningful overlap, but still substantial work

G+ may advance prerequisites involving:

- situation interpretation;
- temporal relation;
- causal retrieval;
- host/context modeling;
- salience;
- metacognitive retrieval;
- attention.

But Phase 9 still owns much broader cognitive structures:

- world model;
- user model;
- self model;
- social model;
- causal model;
- uncertainty;
- identity continuity;
- independent judgment;
- broader metacognition.

Important distinction:

> G+ builds **just enough cognition to retrieve memory intelligently**.

> Phase 9 turns those mechanisms into a broader cognitive architecture.

---

# 22. Phase 10 — some foundations arrive early, core work remains

Earlier work already introduced Mission Graph foundations.

G+ may add:

- active threads;
- unresolved concerns;
- salience;
- memory-aware mission context.

But that is not the same as:

- planning;
- simulation;
- curiosity;
- proactive research;
- long-running mission execution;
- specialist-agent coordination;
- resource-aware initiative.

Therefore Phase 10 remains a major independent stage.

---

# 23. Phases 11–15 remain largely distinct

G/G+ should not materially eliminate their purposes.

## Phase 11
Computer use, autonomous coding, reusable skills, repository agency.

## Phase 12
Scientific discovery, numerical/symbolic tools, formal reasoning, simulation, evolutionary search.

## Phase 13
Personal rankers, routers, adapters, LoRA/QLoRA, self-training, challenger/champion evaluation, canary/rollback.

## Phase 14
Persistent operating environment, desktop/mobile/edge/device integration, embodiment.

## Phase 15
Generalized platform, federation, ecosystem, frontier research.

They simply inherit a stronger memory/cognitive substrate.

---

# 24. Required action after G+

After G+ closes, perform a formal:

> **POST-G+ ROADMAP CAPABILITY RECONCILIATION**

Do **not** redesign the top-level roadmap.

Instead map every relevant future capability to one of:

```text
NOT STARTED
FOUNDATION EXISTS
PARTIAL
ADVANCED
QUALIFIED
INHERITED / COMPLETE
NEEDS EXTENSION
NEEDS REPLACEMENT
```

Example structure:

```text
PHASE 8

Automated memory formation       PARTIAL
Long-horizon retrieval           QUALIFIED
Memory attention                 QUALIFIED
Salience                         QUALIFIED
Associative retrieval            QUALIFIED
Reflection                       NOT STARTED
Belief revision                  PARTIAL / NOT STARTED
Procedural learning              NOT STARTED
Training candidate pipeline      PARTIAL
```

Then when Phase 8 officially begins:

```text
IF capability already qualified by G/G+
→ inherit evidence
→ do not rebuild

IF partially qualified
→ extend

IF untouched
→ build normally
```

This prevents duplicate engineering.

---

# 25. Roadmap principle to preserve permanently

Use:

> **The roadmap defines capability destinations, not artificial prohibitions on when enabling work may begin.**

A capability developed early must still be audited against the later phase's broader requirements.

Example:

```text
G+ builds causal retrieval
        ↓
Phase 9 later audits it against the full causal-model requirements
        ↓
reuse / extend / replace based on evidence
```

Do not duplicate a proven component merely to satisfy sequencing aesthetics.

Do not silently claim an entire later phase is complete merely because one of its capabilities was pulled forward.

---

# 26. What the Stage G chat should do next

The Stage G chat should **not immediately build G+**.

It should:

1. Continue the current Stage G substrate qualification plan.
2. Preserve Phase 2 governing authority.
3. Add the long-horizon cognitive recall baseline to Stage G.
4. Add explicit/indirect/spontaneous candidate-recall scenarios.
5. Record candidate ranks and retrieval paths, not merely pass/fail.
6. Record latency/cost and false-retrieval behavior.
7. Ensure canonical recovery remains possible independent of fast indexes.
8. Finish Stage G without introducing the full Memory Attention Controller.
9. At Stage G closure, produce a clean requirements/evidence handoff for G+.
10. Implement G+ as a dedicated memory-cognition qualification stage.
11. Keep Phase 2 governing throughout G+.
12. After G+, run the post-G+ roadmap capability reconciliation.
13. Then return to the remaining governed roadmap work without duplicating already-qualified capabilities.

---

# 27. Non-goals for Stage G

Stage G should **not** attempt to fully implement:

- generalized world model;
- full user model;
- full self model;
- full social cognition;
- full planning;
- curiosity;
- proactive agency;
- autonomous computer use;
- self-training;
- full Phase 8 autonomous learning;
- full Phase 9 cognition;
- full Phase 10 agency;
- complete Memory Attention Controller.

Stage G may expose weaknesses or create interfaces needed by these future systems, but should not become a catch-all cognitive phase.

---

# 28. Non-goals for G+

G+ should also remain bounded.

G+ exists to solve:

> **Intelligent memory attention and recollection.**

It should not silently become:

- Phase 8 in full;
- Phase 9 in full;
- Phase 10 in full.

If G+ needs a narrow capability from those future domains in order to make retrieval work, implement the minimum principled version, document the pull-forward, and mark it for future phase reconciliation.

---

# 29. Core evaluation philosophy

The memory system is **not complete** merely because these pass:

```text
Store memory ✓
Retrieve by ID ✓
Vector search ✓
Graph traversal ✓
Delete ✓
Correct ✓
Rollback ✓
```

Real memory quality also requires tests like:

```text
User:
"I don't want to go back there."

Ground truth:
An old experience is highly relevant.

Question:
Did Alice's retrieval machinery recognize that memory as important
without the test explicitly naming it?
```

This is the transition from:

> **Can Alice possess memory?**

to:

> **Can Alice remember intelligently?**

Stage G measures the baseline.

G+ attacks the cognitive problem.

---

# 30. Final governing summary

```text
A.L.I.C.E. IS NOT:
one infinite chat
one giant prompt
one vector DB
one summary of a lifetime

A.L.I.C.E. IS:
finite working context
+
persistent canonical lifetime
+
multiple rebuildable retrieval projections
+
provenance / authority / correction / rollback
+
long-horizon recovery
+
future memory attention and recollection controller
```

And the program sequence is:

```text
PHASE 2 GOVERNING
        │
        ▼
STAGE G
Memory substrate qualification
+ cognitive recall baseline
        │
        ▼
STAGE G+
Memory attention / intelligent recollection
        │
        ▼
POST-G+ ROADMAP CAPABILITY RECONCILIATION
        │
        ▼
continue governed roadmap
        │
        ▼
H → I → J → owner acceptance
        │
        ▼
only then retire Phase 2 authority
```

---

# 31. Hard decisions established by this handoff

1. **Do not model Alice as an unlimited-token chat.**
2. **Do not load lifetime memory into every interaction.**
3. **Canonical memory must remain recoverable even when absent from working context.**
4. **Fast indexes/projections must never become the sole surviving representation of memory.**
5. **Spontaneous recall is a cognitive attention problem, not only a database problem.**
6. **Stage G validates the substrate and establishes the cognitive-recall baseline.**
7. **The full Memory Attention / Retrieval Controller belongs after G, in G+.**
8. **Phase 2 remains governing during G and G+.**
9. **G+ completion does not itself authorize Phase 2 retirement.**
10. **Top-level Phases 6–15 remain frozen and are not renumbered.**
11. **G/G+ may pull forward later-phase capabilities when technically necessary.**
12. **Future phase implementation plans must be reconciled after G+ to prevent duplicate work.**
13. **Phase 8 is expected to change the most internally after G/G+.**
14. **Phase 9 will inherit meaningful prerequisites but still has substantial independent work.**
15. **Phase 10 may inherit Mission Graph/active-thread foundations but still owns real agency.**
16. **Phases 11–15 remain largely distinct and benefit from the stronger memory substrate.**
17. **The success criterion is eventually not merely "memory exists" but "Alice remembers the right thing at the right time."**

---

## End of handoff

This document is intended to be provided directly to the active Stage G A.L.I.C.E. chat before further Stage G planning or execution so that the new memory-architecture decisions are incorporated without collapsing Stage G and G+ into one uncontrolled implementation stage.
