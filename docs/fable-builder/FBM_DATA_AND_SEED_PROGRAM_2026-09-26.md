# FBM data, seed, and qualification program

**Status:** proposed build contract; no FBM weights or consumer qualification claimed. 2026-09-26.  
**Inputs reviewed:** `fable-builder-model@6a787d72`, `alice-eipm-v1-n0-full-envelope-foundation-build-v1@4270bfa2`, A.L.I.C.E. `main@75db33c5`, `research/frontier-watch@2b9a9311`, `research/mfm-foundation-20260923@efbc6b3a`, and Fable Sleight `main@63bc85f`.

## Decision

FBM is the **host-neutral builder of a connected personal foundation**, not merely a personality-substrate generator. Its first-release target includes source interpretation, the identity/personality role, MFM, host, relationship, assistant-self, governed memory and Experience Ledger interfaces, native judgment, and the conversation handoff. The five personal roles do not imply exactly five weight files. A consumer does not need to bring hand-labeled gold examples or a separate source-person corpus.

The present branch is an excellent **process notebook**, but it is not yet a trainable, generalizing FBM seed. Its compact N0 traces describe operations and lessons, and `training/fbm/bootstrap/sol_curriculum_seed_v0.1.jsonl` is an N0 teacher curriculum. Neither alone supplies diverse input/output/feedback pairs for all builder roles or proves cross-user transfer. Treat process traces as procedure seeds and provenance records; build a separate supervised/evaluable corpus from authorized input-output transformations and observed outcomes.

## What the shipped seed must contain

| Seed/data class | What it teaches or initializes | Authority and split rule |
| --- | --- | --- |
| **Reusable builder machinery** | Versioned schemas, source adapters, provenance and authority rules, training/evaluation recipes, model assembly, rollback, egress controls, and a host-neutral FBM checkpoint or other demonstrated equivalent. | The checkpoint is shared builder competence, never an individual's memory or identity. Software rules remain independently enforceable. Record exact pretrained/fresh-init lineage. |
| **Identity-neutral foundation material** | Language, social meaning, evidence interpretation, uncertainty, relations, candidate comparison, temporal and provenance semantics for N0 or its successor. | Public/licensed or otherwise authorized packs with source IDs, license, hashes, exclusions, contamination checks, and target-specific adequacy. No Elaina/Rayan gradients. User selection can reveal an inadequate build. |
| **Builder-operation examples** | Raw item to attributed evidence; evidence to scoped hypotheses/UNKNOWN; coverage and synthetic policy; curriculum selection; model assembly; failure localization and repair. Include rejected proposals and why. | Fictional independent people, permissioned data, and sanitized A.L.I.C.E. *methods*. Split by person/source family, generator family, and scenario lineage. A method-only trace is not a gold output pair. |
| **Longitudinal formation cases** | Source collision, changing host goals, repeated patterns, competing explanations, corrections, deletion, contradictory outcomes, relationship and assistant-self revision. | Controlled fictional histories with exact generator/seed/parent lineage; independent real outcomes when authorized. Never relabel simulations as lived experience. |
| **Instance evidence** | The user's own initial host/personality evidence, and a distinct source-person axis only where actually provided, as for Elaina in A.L.I.C.E. | Local authorized custody; preserve speaker, subject, audience, time, modality, exactness, original source, permission, and derivative IDs. A user's aspiration is evidence, not an automatic identity command. |
| **Interaction/outcome records** | Whether a native verdict, expression, action, correction, or update helped in context; first-pass API draft successes and failures. | Versioned Experience Ledger plus consent and subject scope. An approving reaction alone is not objective truth or a blanket reward. Keep actual outcome separate from hoped outcome and simulated rehearsal. |
| **Independent qualification sets** | Tests that the assembled entity behaves differently for genuinely different evidence and remains stable when irrelevant state changes. | Freeze before optimization; isolate users, source families, time, and synthetic generators. Keep sealed final and personal owner-reviewed fidelity cases apart from builder-generated labels. |

Every training/evaluation example needs a source-role and subject map. Generic classes are direct evidence, evidence-constrained inference, synthetic behavioral teaching, unknown, actual experience, and assistant-self observation. A.L.I.C.E. additionally distinguishes Elaina E0, Rayan host evidence, relationship evidence, and A.L.I.C.E. experience. No Elaina or Rayan private payload, embedding, adapted weight, or reconstructable derivative enters the shared builder seed or another consumer build.

**External teacher boundary.** Owner-authorized Sol examples can remain in the A.L.I.C.E./FBM development lineage with their origin manifest. They are training material, not permanent deployment dependencies or independent historical truth. The builder must pass held-out tests without a paid teacher or manual expert at consumer installation. Resolve rights and portability per source before distributing any shared pack or weights; a training permission for a project does not by itself license redistribution.

## From trace notebook to learning corpus

For each material build operation, keep the existing compact trace, then add a private or safe fictional **case record** where available:

1. Input state: source/evidence IDs, subject and permission map, artifact versions, candidate alternatives, and the *actual* evidence accessible to the operation.
2. Target: structured proposal/action, abstention or escalation, and accepted/rejected alternatives. Separate a human/authority-verified target from an automated heuristic or model-generated guess.
3. Rationale *as checkable features*: supporting and conflicting evidence IDs, scope, validity time, uncertainty, disconfirmation condition, and deterministic authority decision. No hidden chain-of-thought is needed.
4. Outcome: which proposed intervention was executed, what changed, independent validation and later feedback, regressions, compute/latency, and whether the case remained unresolved.
5. Lineage: data rights, model and evaluator versions, source/derivative hashes, split group, generator family/seed, parent case, and correction/deletion dependencies.

Collect successful and failed transformations. Include **do nothing, retain raw, defer, ask, and reject** as valid targets. Do not optimize a builder on self-generated answers graded by its own judge. Independent checks can be deterministic authority invariants, exact source attribution, withheld observed interactions, source-person/owner review for genuinely ambiguous identity cases, and causal downstream behavior. User review is a selective fidelity signal, not a required labeling job for every installation.

## Build sequence and interfaces

1. **Source intake.** Obtain scoped authorization; enumerate and fingerprint raw sources; deduplicate while retaining originals; authenticate speaker/subject and separate quoted outside material. Record permissions for processing, training, sharing, and egress independently. Reject unsupported modalities rather than pretending they were interpreted.
2. **Evidence and hypothesis layer.** Create typed evidence units linked to original spans. Generate conditional competing interpretations, counterevidence, UNKNOWN, and explicitly non-historical synthetic *policies*. Govern promotion with the existing independent memory/authority gate.
3. **Host-neutral competence.** Train/evaluate reusable FBM operations on distinct fictional/permissioned users and A.L.I.C.E. methods. Use a simple deterministic or smaller learned operator where it qualifies; escalate uncertain, novel, and authority-sensitive operations. Establish competent cold-start and sparse-data abstention before personal assembly.
4. **Instance assembly.** Select adequate outside N0 source packs and build or initialize the identity-neutral foundation with true weight lineage. Compile instance-specific teaching sets without crossing subject axes. Form the identity/personality, MFM adaptation, host, relationship, and self capabilities only to the extent supported. Connect evidence/Claim/Experience, retrieval/context, governed revision, and native judgment. A relationship or assistant-self with no lived history starts with a bounded prior, not invented shared memories.
5. **Closed-loop qualification.** Verify source-to-verdict-to-expression-to-outcome causality. Hold the generator/provider fixed while intervening on one user, identity, relationship, or self state at a time; then swap providers. Require evidence-linked changed judgment where relevant and invariance elsewhere. Test conclusion-matched but wrong voice, disagreement, perspective, and evidence bounds. Check first-pass API behavior, bounded correction, latency, multi-turn egress, and honest offline fallback.
6. **Governed updates.** MFM proposes; authority gates decide; FBM diagnoses failures and proposes dataset/model updates separately. Version every update, shadow-evaluate it, propagate correction/deletion to derivatives, and retain a rollback path. An ordinary memory event cannot silently rewrite source identity.

The **identity decision packet** is a versioned interface: stance/action/abstention, evidence IDs actually opened for this invocation, scoped reasons and uncertainty, subject perspective, relationship context, expression obligations, prohibited assertions, state/model versions, and egress allowance. The conversation capability checks both verdict and way of speaking. FBM prepares and tests this contract; it does not hand the final decision to an external API.

## Research deltas and where to apply them

| Finding reviewed | FBM and personal-model implication | Current N0 decision |
| --- | --- | --- |
| Personalized RewardBench, arXiv:2604.07343 | Add equal-general-quality personal preference pairs and compare static ranking with downstream verdict/response fidelity. A raw profile and a structured evidence packet are separate baselines. | Carry to N1/N2 and conversation evaluation; no topology change. |
| Agent Zero, consolidation path dependence, and REALM, summarized in `research/frontier-watch` | Capture evidence actually opened, test different consolidation orders, allow retain-raw, and keep usage-aware retrieval a reversible projection. | No Claim/Experience authority transfer or N0 rewrite. |
| MemCalib, arXiv:2609.24259; Jev-Mem, arXiv:2609.23986 | Train/evaluate per-proposition IGNORE/BOUND/CONTROL use after retrieval, measure overuse and underuse separately, and challenge frequent bounded decisions with a lightweight controller. | Future context/MFM/N1/N2 and downstream arbitration qualification; do not add labels to N0 solely to follow a paper. |
| LGM, arXiv:2609.18461; RPMem, arXiv:2609.23466 | Challenge structured retrieval with query-conditioned latent views and portable parametric adaptation only when evidence IDs, source-exact recovery, correction/deletion and rollback survive. | Later personalization/context challenger. No N0 architecture mutation. |

These are research directions or qualification obligations, not claims that a paper already solves Fable's source authority or user ownership. The canonical main-branch memory-use and personal-development contracts retain authority.

## Gates and immediate N0 decision

**FBM gate A: data viability.** Produce a rights-audited multi-person case manifest; count *independent source families*, supported modalities, complete input/target/outcome examples, critical negative/abstain cases, and held-out families. A large trace count cannot substitute for this.

**FBM gate B: operation transfer.** Freeze cross-person and generator-held-out tasks. Measure extraction/provenance, uncertainty, wrong-speaker defenses, alternative quality, targeted repair success versus regression, cost, and failures individually. Compare deterministic/rule and learned baselines.

**FBM gate C: assembled instance.** Prove the full experience-to-judgment loop, subject isolation, state-only causal interventions, conversation voice, source removal/rollback, and adequate consumer hardware/runtime profiles. Fix thresholds before using the results as a release claim. An incomplete sparse-data build must say which capability is unqualified.

**N0 now.** The current registered N0 full-envelope topology and public mixture remain the correct *identity-neutral* frontier for the personality-model foundation. It trains semantic, evidence, relational, candidate-comparison, public judgment, long-context and runtime-schema competence; it cannot itself certify Elaina voice, host-specific aspiration, relationship learning, memory calibration in live use, or the Fable conversation loop. Do not contaminate N0 with private data or retune its frozen gates to satisfy the product story. Preserve the downstream native arbitration boundary and add personalized voice/decision and outcome-learning evaluations in their own later stages.

The pasted Magnolia job `576166` failed exit 92 in six seconds because `full-envelope-gpu-memory-v1` already exists. It returned **no GPU measurement**. Inspect that root's receipt/manifest and any running writer first. Preserve it. If it is an incomplete previous attempt, choose a *new* versioned output root and rerun the exact-head, no-gradient dry run on `4270bfa2`; do not delete or overwrite evidence. If the old root already contains a valid exact-head pass, verify all required cases and use that receipt. GPU projection is diagnostic, never optimizer/training authorization. Proceed to gradients only after the separate exact-head CPU/data, GPU-memory and training-authorization gates all pass.

## Next implementable FBM slice

Build a small, complete fictional multi-person fixture spanning a direct statement, a quoted outside statement, a conflicting later event, a hypothetical rehearsal, a correction/deletion, and an observed outcome. Store exact source IDs and split groups. Compile typed FBM cases and test the builder's proposed identity/MFM/host/self/relationship interfaces against deterministic authority and causal judgment interventions. Use A.L.I.C.E. traces as recipes and failure labels, then expand by observed failure and independent source families. This is the first evidence that FBM can *build*, rather than only document, another person's entity.
