# MFM native foundation architecture v0.1

**State:** proposed architecture and qualification plan. No native foundation
checkpoint, multimodal processor receipt, optimizer step, independent FINAL, or
MFM capability has been measured. This design implements the owner-directed
[native lineage decision](MFM_NATIVE_LINEAGE_DECISION_2026-09-29.md); the former
Gemma staging and paid-training commands are retired.

## Destination and lineage

Train a reusable, host-neutral semantic and sensory foundation **from freshly
initialized first-party parameters** on admitted public or otherwise permitted
base-data packs. Initialize the specialist Memory Formation Model (MFM) from
that first-party checkpoint. Its job is to read authorized experience and
context, then emit evidence-bound `MemoryProposalBundle` entries. FBM may use
the qualified shared native checkpoint when building an individual Fable,
record eligible base-pack selections or additions, and perform governed
instance adaptation. It must keep the person's private biography in their own
source custody and memory, not in a redistributed shared checkpoint. A shared
native seed alone does not establish that an instance is a personal model;
source-grounded construction and causal behavior must be evaluated.

Every trainable component on the required formation path has fresh-init
ancestry: language weights, visual/audio/video encoders, fusion, retrieval
policy, output decoder and any learned tokenizer or media codebook. Reuse
published architecture *mechanics* and non-learned decoders such as ffmpeg,
but do not import pretrained parameters, adapters, merged checkpoints or
cached feature embeddings from Gemma or another model into core MFM. Optional
external feature tools and frontier teachers have separately declared
lineage. Neither becomes a hidden required runtime dependency or Claim
authority. A teacher's generated candidate is attributed training material;
it is not an independently adjudicated historical fact.

## Stage A: host-neutral public-data foundation

| Component | Fresh-init representation and training objective | Source fidelity |
| --- | --- | --- |
| Text, code and structured records | Train a tokenizer on admitted data and a scalable sequence backbone with causal language modeling, span reconstruction, record/schema parsing, temporal-order and cross-record consistency objectives. Code and tables retain field and row boundaries rather than becoming unlabeled prose. | Preserve source IDs, byte offsets, document structure, language, timestamps and duplicate families outside the model. Token-to-byte maps support later evidence pointers. |
| Images | Train a patch-based visual encoder from pixels with masked reconstruction and source-aligned image/text objectives. Include documents, handwriting, scenes and visually ambiguous cases as admitted data permits. | Keep original bytes, orientation, page and region coordinates. OCR may supply a parallel, attributed observation but cannot replace the image input. |
| Audio | Train a waveform or spectrogram encoder with masked acoustic and admitted speech/transcript alignment objectives; include nonspeech events, turn boundaries and speaker evidence. | Keep sample rate, channel, interval and speaker hypotheses bound to original audio. ASR output is another attributable interpretation, not the waveform. |
| Video | Train a temporal visual encoder over variable-resolution clips with masked temporal modeling, event order, motion and admitted video/text/audio alignment. Share compatible visual primitives with images while preserving temporal features. | Keep frame/time correspondence and synchronized soundtrack. Sampling is adaptive and reversible; no fixed one-frame-per-second, 60-frame, or 30-second semantic limit. Reopen original intervals for fine evidence. |
| Fusion | Cross-modal alignment and a hierarchical, source-aware latent workspace combine modalities and source metadata. Train contrastive/alignment, matching, temporal order and source-span localization where trustworthy targets exist. | Encode subject, speaker, source role, observed time, recorded time and ancestry as distinct inputs. A semantic match never upgrades evidence authority. |

This is a **scalable model family**, not a fixed parameter or context size.
Candidate implementations should compare hierarchical source/window encoding
with a latent cross-attention workspace and learned source selection. Cross-source
fusion can reopen original chunks when a compressed view is insufficient.
Historical coverage grows through indexed, authenticated retrieval and
iterative reading rather than claiming one finite context window contains an
unbounded life. The source registry and read receipts decide which bytes may
enter; model attention and summary fidelity must be measured separately.

Pretraining packs need exact source/version/hash, license and attribution,
commercial model-training and model-distribution eligibility, privacy and
removal rules, deduplication, language/modality coverage, and a contamination
exclusion list. Merely being downloadable or labeled public is insufficient.
Pack selection and owner additions are recorded by FBM. Private Rayan/Elaina
facts cannot enter the portable Stage A weights. A changed mixture produces a
new versioned checkpoint, never an undocumented rewrite of lineage.

## Stage B: evidence-pointer formation specialist

Initialize the Stage B backbone and sensory encoders from the qualified Stage A
checkpoint. Add newly initialized source-selection, event segmentation,
cross-history fusion and structured proposal heads. Bind training to the
existing `FormationContextPacket`, registered source reads, exact digests and
`MemoryProposalBundle` contract. The model should learn:

- **Fast formation:** distinguish actor, recipient, subject and source-person
  roles; detect event boundaries, direct observations and no-proposal cases;
  propose `propose`, `defer`, `retain_raw` or `abstain` by scope.
- **Slow consolidation:** revisit episodes across time; propose corrections,
  contradictions, changing preferences, patterns, relationship norms, skills,
  decision rationales and outcomes. Preserve competing interpretations and
  uncertainty. Re-run from source ancestry after correction or deletion.
- **Grounded decoding:** emit kind, domain, epistemic status, subject,
  source/target IDs, validity interval, confidence and an anchored semantic
  value. Prefer a typed constrained decoder with explicit source-pointer and
  span/region/interval heads over free text that merely names a citation.
- **Source-native anchors:** text/code/table byte span and field; image page
  and region; audio channel/speaker and time interval; video time/frame region
  plus linked audio interval. Version these typed locator schemas and verify
  coordinate and byte bounds against the opened original. The current v1.5
  `FormationEvidenceAnchor.locator` is an opaque string and temporal
  granularity is only `instant`/`day`; neither alone proves semantic grounding
  or covers all temporal uncertainty. Evolve the contract compatibly before
  claiming fine-grained sensory or uncertain-time support.

Joint supervised losses should cover structured proposal generation,
source/anchor pointer accuracy, subject and role classification, event boundary,
valid-versus-recorded time, contradiction/target linkage, scoped disposition,
abstention/calibration and duplicate/idempotent outcomes. Add contrastive
counterfactuals where a source, subject, timestamp, permission or correction is
changed while the other evidence stays constant. Weight critical false-memory,
wrong-subject and deletion failures visibly in selection criteria; do not bury
them in an average. Outcome-driven policy learning is a later option only with
authentic, permissioned outcomes and a separate held-out test. Training loss
or a syntactically valid bundle is not authority.

The existing 49,819 synthetic cases are frozen **training-only formation
supervision**. They do not teach broad language or raw-media perception from
scratch, independently verify their own labels, or certify unseen host and
generator families. Admit broader permissioned histories and independently
adjudicated multisource/multimodal targets. Use owner-authorized frontier
teachers for attributed candidate examples or comparisons, then separate
teacher generation from human/independent review and sealed grading. Protect
the FINAL split from both the optimizer and iterative model selection.

## Runtime authority and qualification

The fast path records immutable experience, authenticates sources and proposes
time-sensitive formation. The slow path consolidates and may revise proposals
against original sources. The independent gate decides Claim writes,
correction, revocation and deletion; accepted projections feed the selected
episode, graph, vector and personal-state stores. Future retrieval and native
judgment must demonstrate an observed causal effect. Neither the foundation
nor MFM may infer consent from content or transform a generated reconstruction
into owner history.

Evaluate Stage A separately on unseen text/structured semantics, raw image,
audio and video grounding, temporal order, source localization, languages and
long-history retrieval. Evaluate Stage B on held-out people, source systems,
duplicate families, generator families and time windows. Measure extraction,
multi-session reasoning, temporal revision, contradictory evidence, deliberate
abstention, subject/provenance errors, deletion influence, calibration and
multimodal anchors. Compare capable extraction and long-context baselines on
the **same opened evidence**. Then test gate acceptance, replay, projections,
later retrieval, native judgment, changed-state interventions and a second
independent FBM host. Publish every critical failure class. A tool receipt,
one backward pass, a synthetic training loss and a stage-specific benchmark
cannot stand in for the governed full loop.

No fixed 12B configuration, clip length, frame rate, source-count, parameter
size or deployment device is made a capability ceiling. Real runs still have
finite windows and hardware. Record the exact tokenizer/processor/model/data
hashes, initialization audit, tokens and media hours, optimizer topology,
peak memory, full-step throughput, restart/export integrity and energy/cost
before pricing a full fit. The amount of eligible training data and compute
needed to reach the requested capability is **unknown**. Dense language
pretraining's rough `6 × parameters × tokens` FLOP relation is only a planning
order of magnitude and excludes sensory training, repeated epochs and system
overheads. Measure scaling and held-out competence before committing to a
size or declaring success. The old Gemma CPU and GPU receipts cannot qualify
this native route.

## Research transfer and its limits

- [Common Pile v0.1](https://arxiv.org/abs/2506.05209) reports an 8 TB
  openly licensed/public-domain collection and fresh 7B language training on
  1–2 trillion tokens. It establishes a plausible public-data route, **not**
  this project's pack rights or MFM competence.
- [Compute-optimal language scaling](https://arxiv.org/abs/2203.15556)
  connects model size and training tokens. The 49,819 formation rows cannot
  replace base pretraining at comparable scales.
- [Perceiver IO](https://arxiv.org/abs/2107.14795) motivates variable-modality
  input to structured output through a latent workspace;
  [Chameleon](https://arxiv.org/abs/2405.09818) demonstrates early fusion of
  text and image tokens. Neither directly validates this source-governed
  specialist or prescribes inherited weights.
- [VideoMAE](https://arxiv.org/abs/2203.12602) motivates fresh video
  pretraining without an ImageNet checkpoint. The
  [Whisper paper](https://proceedings.mlr.press/v202/radford23a.html) reports
  680,000 hours of weakly supervised speech for robust multilingual ASR;
  that scale is context for the audio challenge, not a required MFM target.
- [LongMemEval](https://arxiv.org/abs/2410.10813) isolates extraction,
  multi-session reasoning, time, updates and abstention;
  [Memory-R1](https://arxiv.org/abs/2508.19828) motivates learned memory
  operations but fine-tunes existing LLMs. Their results do **not** prove
  native MFM formation or permissioned personal memory correctness.

Source-of-truth scope and the full-loop completion criteria remain in
[`MFM_FULL_CAPABILITY_BUILD_PLAN_2026-09-29.md`](MFM_FULL_CAPABILITY_BUILD_PLAN_2026-09-29.md)
and the [native lineage decision](MFM_NATIVE_LINEAGE_DECISION_2026-09-29.md).
