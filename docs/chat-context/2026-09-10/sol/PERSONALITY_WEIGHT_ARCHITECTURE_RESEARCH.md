# A.L.I.C.E. Personality-Weight Architecture Research — 2026-09-10

**Status:** working research recommendation; not canonical `main`; not final Astra-reviewed training authority.

**Parent owner override:** `OWNER_OVERRIDE_PERSONALITY_MODEL_MAINSTREAMING.md`

Canonical `main` remains frozen at `0abaed85873c3f8de04765847eb7700b0e20433f`.

This note records the technical continuation after the owner superseded the validation-heavy MC10D path. It does not authorize a new semantic-validation tournament. It does not create A.L.I.C.E. model weights.

## 1. Current assets recovered

### Canonical Elaina evidence

- 128 / 128 source records are owner-ratified E0 gold.
- 290 / 290 canonical semantic units are owner-ratified E0 gold.
- The global source registry historically reconciled those 290 units plus 11 supplemental/context entries for 301 registered entries.
- Existing overlapping EIPM routing lanes are directly useful to the new training-data compiler:
  - `direct_identity_supervision`: 135 canonical units
  - `conditional_identity_supervision`: 195
  - `episodic_behavior_supervision`: 93
  - `context_only_conditioning`: 221
  - `style_voice_supervision`: 28
  - `exclude_from_identity_loss`: 37
- These counts overlap by design. Context-only or excluded-from-identity-loss material is not discarded; it remains conditioning/provenance context rather than direct personality loss.

### Completion frontier already generated

- MC10A froze 399 candidate packets against 151 unique E0 units.
- 65 packets were E-INF eligible and 24 packets were A-SYN eligible under the then-current frontier construction.
- `alice-mc10b-live` records a completed 60-packet frontier with 720 raw E-INF proposals.
- `alice-mc10c-live` records 288 raw A-SYN candidates across 24 packets. The later MC10D continuation operated on 287 effective candidates after downstream filtering.

Under the 2026-09-10 owner override these are no longer waiting for a multi-judge tournament. They remain provenance-marked raw material for the training-data compiler. E-INF does not become E0. A-SYN does not become historical or lived Elaina/A.L.I.C.E. memory.

## 2. External research synthesis

The strongest current evidence points away from a monolithic biography-in-the-weights design.

### Persona grounding

- **ThinkPersona (ACL 2026)** trains individualized role behavior from Persona Graphs encoding life trajectories, values, relationships and events, then creates grounded Question–Reasoning–Answer examples. This maps closely to A.L.I.C.E.'s existing structured E0 / graph architecture.
- **PsyMem (TACL 2026)** explicitly trains role responses to align with controlled memory rather than relying on implicit model knowledge or simple retrieval alone. This supports keeping personality weights stable while teaching them how to respond to retrieved memory.
- **PD-LLM / Trait Activation in Silicon (ACL 2026)** identifies `Personality Inertia`: ordinary post-trained assistants tend to fall back to a sanitized helpful-assistant persona. It uses situation-conditioned LoRA control to recover context-sensitive trait expression. For A.L.I.C.E. the lesson is to train across real situations and tensions rather than flatten Elaina into a static trait paragraph.
- Anthropic's **Persona Selection Model**, **Assistant Axis**, and **Persona Vectors** work provides convergent evidence that post-training selects/refines a persona from a broader latent character space and that persona behavior can drift. For A.L.I.C.E. these techniques are useful as diagnostic/drift instrumentation, not provenance authority.

### Weight adaptation

- **LoRA Without Regret (Thinking Machines, 2025)** reports that LoRA in supervised post-training is substantially stronger when applied to all relevant weight matrices, especially MLP/MoE matrices; attention-only LoRA underperforms. A.L.I.C.E. should therefore not default to a q/v-only adapter.
- Hugging Face PEFT currently supports LoRA, rank-stabilized LoRA (`rsLoRA`), DoRA, PiSSA and other initializers. rsLoRA permits higher-rank capacity without the original rank-scaling pathology. DoRA can improve learning capacity/stability, but adds training overhead and must be proven against the exact selected base architecture.
- Hugging Face TRL supports assistant-only/completion-only SFT loss. This is particularly important for A.L.I.C.E.: E0/E-INF/provenance/context can appear in the input without being treated as target text; gradients can be restricted to the desired A.L.I.C.E. response.

### Preference optimization

- DPO remains the conservative preference-stage choice: simple, stable and no learned reward model / PPO loop.
- SimPO is a credible lower-memory alternative because it is reference-free and length-normalized.
- Persona-tailoring research shows that hard rejected examples should represent plausible alternative people / generic-assistant behavior, not merely obviously bad responses.
- C-BPO (ACL 2026) is a useful future owner-feedback method if the owner supplies binary `Alice-like` / `not Alice-like` labels rather than response pairs.

### Synthetic data

- Recursive synthetic-data research shows that indiscriminately replacing real data with generated data can collapse tail behavior. A.L.I.C.E. must keep E0 as a persistent high-authority anchor rather than letting generations of synthetic descendants replace it.
- ACL 2026 work on multi-source synthetic data and diversity optimization supports using multiple teacher families and explicit diversity selection instead of one generator producing a narrow synthetic monoculture.
- PerSyn-style teacher routing suggests generating each kind of synthetic behavior with the teacher best suited to that prompt instead of requiring every teacher to answer every prompt.

### Industry / competitor pattern

- Character.AI publicly exposes layered persistent memory (`Story Memory`, `Facts`, pinned/auto memories) outside the character model. This is consistent with A.L.I.C.E.'s separation of identity weights from dynamic memory.
- Replika likewise describes layered memory and gradual host-personalization rather than one immutable prompt or one biography-only model.
- Mistral Forge's 2026 proprietary-model workflow combines organization-specific data, post-training and model/version lineage rather than retraining general intelligence from scratch. The applicable lesson is `strong generic foundation + private high-signal specialization + traceability`.

## 3. Recommended A.L.I.C.E. learned architecture

Do **not** train a foundation model from scratch and do **not** make one checkpoint the canonical biography/memory database.

Use:

`B = strong post-trained/open base reasoning model`

`P = stable Elaina Identity / Personality adapter`

Runtime behavior is `B + P`, conditioned by separately authoritative state:

`G = Elaina persona/source graph`

`M = A.L.I.C.E. lived memory (A-EXP/A-SELF)`

`H = Rayan Host Model`

`R = A.L.I.C.E.-Rayan relationship state`

`C = retrieved/selective context`

The Stage-G personality artifact is **P**. It should internalize stable behavioral policy: voice tendencies, preferences, values, judgment patterns, emotional interpretation, relationship stance, situation-conditioned reactions, ambiguity behavior, tensions and how to use retrieved evidence/memory.

It should **not** be the authoritative store of historical Elaina facts. The provenance graph/claim fabric remains the authority. Ordinary host data may update H/R/M but must not silently rewrite P.

### Initial adapter topology

Start with **one unified Elaina identity adapter**, not a bank of generic Big-Five/DIAMONDS trait adapters. A single real person's evidence is richer and more context-dependent than a generic trait decomposition. Borrow PD-LLM's situation-aware *data design* first. Consider modular trait/situation adapters only if the first real owner-evaluated EIPM shows measurable personality inertia that a unified adapter cannot fix.

Activation-vector techniques remain optional diagnostics / runtime stabilization tools. They do not replace the primary learned identity adapter.

## 4. Training-data compiler

The old MC10D simulation machinery is not the new pipeline. Build a new `EIPM corpus compiler` whose purpose is training, not validation.

The compiler should produce these example families:

1. **E0-grounded persona-graph examples** — relevant E0 nodes/episodes/relations + situation/question -> desired A.L.I.C.E. response. ThinkPersona-style grounding is the closest current research analogue.
2. **Direct style/voice examples** — only from evidence already routed as style/voice supervision or strongly supported source text.
3. **Conditional behavior examples** — same underlying person, different relationship/state/situation -> different appropriate reaction. Preserve contextual tensions instead of averaging them away.
4. **Memory-conditioned examples** — stable personality adapter + different retrieved memory packets -> contextually appropriate response, PsyMem-style.
5. **Multi-turn continuity examples** — personality stays stable while new conversation state changes the response.
6. **Uncertainty/provenance examples** — when source support is insufficient, A.L.I.C.E. must not invent a historical Elaina fact; E-INF and A-SYN remain represented truthfully.
7. **Hard preference pairs** — chosen Elaina/A.L.I.C.E.-aligned response versus plausible but wrong alternatives: generic assistant, excessive sycophancy, wrong relationship stance, flattened neutral behavior, caricature/extreme trait behavior, unsupported biography, or context-insensitive response.
8. **General-capability replay** — a small licensed instruction mix may be retained so specialization does not unnecessarily destroy useful base capabilities.

Synthetic expansion should be coverage-driven. Do not pick an arbitrary 50k/100k target first. Build a coverage tensor over behavior/situation/relationship/evidence-use dimensions, generate where holes exist, deduplicate/diversify, then stop when meaningful holes have been filled.

Every generated training row keeps an external manifest containing provenance class, parent E0/E-INF/A-SYN IDs, source hashes, generator/model identity, generation prompt identity, transformation recipe and corpus version. Model weights are never trusted to self-report this provenance.

## 5. Weight-creation recipe

### Base checkpoint

Start from an **instruction/post-trained model**, not a raw pretraining checkpoint, unless later evidence strongly justifies rebuilding broad instruction behavior. A.L.I.C.E.'s private corpus is high-signal but far too specialized to recreate a modern general assistant/agent post-training stack from scratch.

Current candidates:

- **Qwen3.8-27B** — preferred high-capacity target if final training may use rented/modern 48–80GB-class GPU compute. Apache 2.0, 27B dense, current Qwen generation, flexible thinking, long context. Its hybrid Gated DeltaNet / gated-attention architecture means exact PEFT target modules and quantized training compatibility must be dry-run proven before private training.
- **Gemma 4 12B IT** — preferred constrained-compute first-build candidate. Apache 2.0, 12B dense, official current Hugging Face / QLoRA ecosystem, and substantially smaller weight footprint.
- **OLMo 3.1** — strongest reproducibility/control baseline because code, checkpoints, data/post-training details are unusually open. 32B is expensive; 7B variants are useful for mechanics/control experiments.
- **Mistral Small 4** and **Llama 4 Scout** are less attractive first EIPM targets because total model footprint / MoE complexity is high relative to the private specialization task.

No final base is frozen until the owner's allowed compute envelope is known.

### Adapter method

Default research recommendation: **4-bit base + high-capacity language-backbone all-linear rsLoRA**.

- Target relevant language-model linear matrices, including MLP paths rather than attention-only adapters.
- Freeze vision/audio-specific components unless a later multimodal EIPM objective explicitly needs them.
- Determine exact rank, alpha, learning rate, sequence length and batch/gradient accumulation empirically on dummy/public data. Do not copy a generic rank from a blog and call it A.L.I.C.E.-optimal.
- DoRA/PiSSA may be compared in a tiny non-private mechanics/optimization dry run if supported. They are optional, not prerequisites.

### Training stages

**Stage P1 — Grounded SFT**

Train with completion/assistant-only loss over the compiled E0/E-INF/A-SYN-grounded demonstrations. Source/provenance/context lives in the prompt/context; desired A.L.I.C.E. behavior is the gradient target.

**Stage P2 — Preference specialization**

Train on hard chosen/rejected identity pairs. Use DPO as the conservative default. Use SimPO if the reference-model memory burden is materially limiting and the exact implementation is proven.

Do not use verifiable-reward RL as the main personality objective. Personality fidelity is not an objectively verifiable math/code reward.

**Optional identity mid-training / continued pretraining**

Do not assume this is useful. Consider it only after measuring the actual raw E0 token volume and source format. If the private corpus is small, grounded SFT is more appropriate and less likely to overfit/memorize source wording.

## 6. Exactly one Astra review boundary

Engineering dry runs on public/dummy text are permitted before Astra. They are not semantic validation and create no A.L.I.C.E. personality weights.

Before the first private A.L.I.C.E. gradient update, freeze one review bundle containing:

- exact E0/E-INF/A-SYN corpus manifest and hashes;
- provenance/sampling/coverage policy;
- representative transformed training examples;
- hard-negative construction policy;
- exact base revision/tokenizer/chat template;
- exact adapter target-module inventory and PEFT method;
- exact SFT and preference-stage configurations;
- checkpoint/rollback/versioning plan;
- compute benchmark / expected memory envelope.

Give that frozen bundle to Astra once with the owner-directed question:

> Given the goal of producing the most Elaina-faithful A.L.I.C.E. identity/personality weights possible, is there any material technical flaw in this frozen corpus or weight-creation architecture that should be fixed before the first A.L.I.C.E.-data gradient update?

Fix only material issues. Freeze the corrected manifest. Then create the first real personality weights.

After training, the owner is the decisive personality-fidelity judge. Owner feedback can drive an explicit new adapter revision; this is not a return to the old multi-judge MC10D validation tournament.

## 7. Immediate next work

Do not resume MC10D.

The next executable work is:

1. materialize the canonical private E0 / router / MC10B / MC10C artifacts in one read-only training-source workspace;
2. inventory actual private token counts, dialogue/source modalities and raw-candidate payload structure;
3. implement the EIPM corpus compiler and coverage map;
4. generate/transform only the missing training behaviors;
5. run a **public/dummy-only** base-model + PEFT mechanics benchmark for the selected candidate base;
6. freeze the single Astra review bundle;
7. Astra once;
8. first A.L.I.C.E. gradient update and training.

One owner decision is still required before the final base/model training configuration can be chosen without guessing: whether the first real EIPM build may use bounded rented cloud compute (for example, a modern ~48–80GB GPU) or must remain within current free/Kaggle/Magnolia-class compute.

## 8. Sources consulted for this research snapshot

Primary/current sources included Qwen3.8 model documentation; Google Gemma 4 documentation/model card; Ai2 OLMo 3.1 model documentation; Mistral Small 4 and Forge documentation; Hugging Face Transformers/PEFT/TRL documentation; Thinking Machines' `LoRA Without Regret`; DoRA / rsLoRA; DPO / SimPO; ACL/TACL 2025–2026 papers ThinkPersona, PsyMem, Trait Activation in Silicon, C-BPO, persona-tailoring preference work, PerSyn and VOYAGER; Nature's recursive synthetic-data collapse study; Anthropic Persona Vectors / Assistant Axis / Persona Selection Model; and current Character.AI / Replika public memory documentation.
