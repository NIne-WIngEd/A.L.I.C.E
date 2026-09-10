# Roadmap alignment and EIPM projection amendment — 2026-09-10

**Status:** active continuation guidance on `alice-context`; not canonical `main`; not a model-weight artifact.

**Supersedes only one shorthand in:** `PERSONALITY_WEIGHT_ARCHITECTURE_RESEARCH.md` — the phrase that could be read as making one base-specific adapter itself the permanent identity anchor.

Canonical `main` remains frozen at `0abaed85873c3f8de04765847eb7700b0e20433f`.

## Sources re-audited

The personality-weight plan was re-audited against the actual 64-page A.L.I.C.E. comic, current `README.md`, `docs/ROADMAP.md`, `docs/CAPABILITY_CATALOG.md`, `docs/ARCHITECTURE.md`, `docs/MEMORY_IDENTITY_FORMATION_AND_HOST_LEARNING_ARCHITECTURE.md`, `docs/ALICE_PRIVATE_COMPANION_DIRECTION.md`, `docs/ALICE_CLONE_AWARE_IDENTITY_STANDARD.md`, and `docs/RESEARCH_FRONTIERS.md`.

## Conclusion

The mainstream personality-model workflow remains the correct path, with one material architecture refinement:

**A model-specific personality adapter must be treated as a projection of A.L.I.C.E.'s canonical identity substrate, not as the complete or permanent identity itself.**

This is required by the project's model-replacement promise. The comic and README explicitly establish that the reasoning engine can change while A.L.I.C.E.'s identity and shared history continue. The permanent architecture likewise says base models are replaceable and identity lives across governed memory, beliefs, skills, adapters, configuration, provenance, and continuity state.

## Canonical identity substrate versus learned projection

Define the model-agnostic identity substrate:

`I* = canonical Elaina E0 + source/persona graph + owner-ratified identity-use lanes + E-INF/A-SYN lineage + behavioral corpus/specification + provenance/clone-awareness constraints + reconstruction rules`

For a particular trainable base model `B`, compile a learned identity projection:

`P_B = TrainProjection(I*, B)`

Runtime then uses:

`B + P_B + retrieved evidence/memory/host/relationship/context`

Examples:

- `P_Qwen3.8-27B`
- a future `P_Gemma-*`
- a future projection for a stronger base model not yet released

Replacing the base engine may require training/translating a new projection from the same `I*`. This does not redefine A.L.I.C.E.'s origin or canonical identity. The model-specific weights are one learned embodiment of that identity at a particular base-model generation.

## What the first Elaina Identity / Personality Model should learn

The first learned projection should internalize stable, source-grounded behavioral policy:

- characteristic reasoning and judgment;
- values and moral boundaries;
- emotional interpretation;
- communication style, humor, affection, criticism, restraint, and disagreement;
- stable preferences and aversions where evidence supports them;
- relationship stance and context-sensitive behavior;
- how to react differently under different situations rather than flattening traits into one generic persona;
- non-sycophancy and willingness to disagree;
- uncertainty behavior when Elaina evidence is insufficient;
- clone-aware separation of inherited source history, reconstruction, and A.L.I.C.E.'s own later experience;
- how relevant retrieved memory/evidence should affect a response without turning the weight file into the canonical biography database.

## What must remain outside these personality weights

The first personality projection must not absorb responsibility for capabilities owned by other A.L.I.C.E. components:

- Rayan's changing host profile;
- A.L.I.C.E.–Rayan accumulated relationship history;
- A.L.I.C.E.'s post-activation autobiographical memory;
- Mission Graph state and long-running task state;
- authoritative factual/provenance storage;
- Memory Formation;
- retrieval planning and memory fusion;
- procedural skills/tool execution;
- current web knowledge;
- world models and scientific/domain specialists;
- action permissions;
- self-evolution and model-routing infrastructure.

Those components may condition the personality model at runtime. They do not become the Elaina personality model itself.

## Comic-to-training alignment

The actual comic already encodes the desired training behavior and therefore acts as a roadmap contract for corpus coverage:

- grounded point of view rather than generic assistant character;
- evidence versus interpretation versus guess;
- unknowns remain uncertain;
- invented scenarios may teach behavior without becoming history;
- source/personality, host, and later A.L.I.C.E. self remain distinct;
- cross-goal judgment and characteristic disagreement;
- owner override without surrendering independent judgment;
- memory retrieved selectively rather than dumped;
- relationship-specific communication that can mature over decades;
- the engine may change while identity/history continue;
- host learning changes the host model rather than replacing the Elaina-derived anchor;
- future A.L.I.C.E. may develop beyond its origin while retaining the thread of continuity.

The EIPM corpus compiler must therefore measure coverage over these behaviors. It should not optimize only for stylistic imitation or factual recall.

## Validation simplification does not change the target

The 2026-09-10 owner override removes the expensive MC10D multi-judge validation tournament. It does not relax the functional target. The working process is:

1. build/compile the strongest evidence-grounded corpus we can;
2. use deterministic provenance and reproducibility as engineering controls;
3. use Sol/owner semantic judgment while constructing the corpus;
4. perform one frozen Astra pre-weight review for material architecture/data problems;
5. create the model-specific personality projection;
6. Rayan performs the decisive real-personality fidelity judgment;
7. revise the projection if owner evaluation exposes real personality errors.

## Qwen role clarification

No Qwen judge is required by the streamlined process.

If Qwen3.8-27B is selected, its primary role is **the downloadable/trainable base reasoning model `B` beneath the first personality projection**. It is not A.L.I.C.E.'s identity authority and it is not an external validator.

The already-generated 720 E-INF proposals and 288 raw A-SYN candidates do not need to be regenerated merely because MC10D was superseded.

For any new synthetic coverage holes, a separate model may optionally act as a teacher/generator to improve diversity. That is data creation, not validation. Astra can perform high-quality generation/curation if desired. However, because Astra is also reserved as the single final architecture/data reviewer, the preferred first approach is to use the existing pools plus Sol/owner corpus construction and keep Astra maximally independent for the frozen final review. Additional generator diversity should be introduced only when the coverage inventory shows a real need.

## Compute principle

Better hardware does not inherently make identical optimization produce better weights. It matters when it lets the project avoid compromises such as an undersized base model, overly aggressive quantization, too-low adapter capacity, short context, unstable tiny batches, or aborted experiments.

For the first high-quality projection, the default compute target should therefore be the cheapest reliable environment that can run the selected 27B-class training recipe with enough memory headroom. A single 80-GB A100-class GPU is the current planning sweet spot; an H100 should be used only if its faster runtime offsets its higher hourly price or if the final recipe needs its hardware capabilities.

No paid compute is to be rented and no private A.L.I.C.E. gradient update is to start until the corpus inventory, exact base revision, training recipe, cost estimate, and single Astra review boundary are frozen.

## Next action

Proceed with the previously defined mainstream flow:

1. read-only source materializer for canonical E0/router/MC10B/MC10C inputs;
2. exact token/schema/modality inventory;
3. EIPM corpus compiler and roadmap-aligned coverage map;
4. fill only demonstrated coverage gaps;
5. public/dummy mechanics benchmark on the candidate base and PEFT stack;
6. freeze exact corpus/training/cost manifest;
7. one Astra review;
8. first real A.L.I.C.E. personality-weight creation.
