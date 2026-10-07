# MFM next-route design review — 7 October 2026

Independent research and architecture-constraint review. This document changes no
model, corpus, admission, job or authority. The completed failures remain failures.
The immediate recommendation is to build the two shared substrates below while
holding the architecture decision until its mechanism and resource assumptions
are concrete. No further fit on the same 441 cases follows automatically.

## Governing constraints and the selected implementation

[The licensed-base direction](../MFM_V1_LICENSED_BASE_DIRECTION_2026-09-30.md)
permits exact, traced pretrained/base Gemma ancestry and derivatives for V1. It
requires separately initialized, learned formation-specific weights with their
own objective and demonstrated causal contribution. The base supplies general
competence; the independent Claim gate retains memory-write authority. Fine-tuning
a base and serving its free-form answers alone is expressly insufficient.

[The selected specialist path](../MFM_V1_SPECIALIST_TRAINING_PATH_2026-09-30.md)
is narrower: frozen Gemma source states feed a fresh projection, six-layer
formation decoder, token embedding and vocabulary head. It explicitly does not
use the pretrained language head for answers or training. The current calibrated
decoder and its task-conditioned successor retain that choice. A lexical import
or full-LM producer would be a new versioned route, not an unchanged continuation
or a hidden fallback. The broader licensed-base decision permits considering such
a successor; it does not prove one works or retroactively change this recipe.

The current source processor encodes the complete scoped context and original
opened evidence, escapes textual media markers, and rejects overflow. Gemma
features are target-free. The bank retains processed token IDs, states and masks,
but the learned decoder consumes states and mask, not a token-to-original-byte
copy map. Targets include the full learned formation decision. The compiler
allocates only new local bookkeeping identities; existing external references,
uncertainty sharing, subjects, times and semantics remain learned content.

The full capability remains proposals, uncertainty, source/subject/time binding,
episodes, relationships, correction/deletion, prior-state evolution, native media
and accepted-memory usefulness. A successful string or compiler is not that role.
The build doctrine allows a justified architecture migration and rejects blind
retries, weakened tests and a new standing qualification tournament.

## What the verified failure supports

Read-only job 576984, immutable code `d3977bcbd2179144f5e2398163d7e84c9bd58a32`,
has externally pinned analysis seal
`350ac0ad3fa8f340db5596473f89cec026ff3c25827e2323bc6b50b0ef9de1d0`.
Independent checking reconciled both raw hashes, all 882 case/task rows, 41
retained cases, 24 nested code bindings and the complete training objective.
The maximum per-case loss difference was 4.30e-7; objective 1.707743900 matches
sealed progress 1.707743879. Task-bound late cache/reference checks reached EOS
and parser neighborhoods, with maximum absolute difference 1.05e-5.

All following top1 figures use complete gold prefixes; none are free-generation
semantic accuracy. Formation key top1 is 83.4%/94.0%/63.4% for diverse train/longitudinal train/dev;
value-text top1 is 33.9%/71.1%/10.8%. Source reconstruction keys are about 98%,
but source text is 23.7%/41.1%/11.1%. All 534 rare training source-text positions
are incorrect; the 393 positions still rare in the combined two-task frequency
basis are also incorrect. All 202 dev source-reference positions with no positive
label exposure in either task are incorrect. Teacher-prefix EOS top1 is 100% in
every cohort. The retained formation outputs still yield zero positive grounded
proposals: 35 JSON failures, one grounding failure and two empty valid decisions.

At the first author departure, known training targets are often rank two under a
correct prefix. All nine dev first departures concern the same never-positive
source-reference token; its median rank is 65,590 and probability 2.20e-8. Zero
source changes predictions substantially. Foreign sources change paired ranks
and probabilities, but leave the same dev greedy choice. Equal cohort averages
here reflect the deterministic permutation of a common-prefix cohort, not absent
case-level response.

This rejects cache malfunction, EOS-only repair and an exclusively erroneous-
rollout-prefix account of those authored departures. It supports incomplete
conditional sequence/lexical learning and failed open-reference emission under
this construction. It does not establish inadequate Gemma encoding, inevitable
scratch-model failure, absence of source dependence, semantic correctness of the
author target, or auxiliary-task interference. Weighting, task mixture, task
vectors and Adam reset changed together. Later rollout can still amplify errors.

## Mechanism audit

| Candidate | Compatibility and useful contribution | Strongest objection / limit | Concrete falsifier |
|---|---|---|---|
| Licensed lexical geometry with fresh native producer | Compatible with the broad V1 direction; explicitly changes the selected initialization/head recipe. Reuse the pinned vocabulary embedding basis while learning native dimension bridges, decoder and formation decisions. It addresses the independent random lexical interface without delegating memory authority. | An embedding matrix is not an autoregressive language model. Gemma width 3840 and specialist width 768 differ. Projection, normalization, scale, tied geometry and context conditioning are real learned interfaces; arbitrary slicing or copying rows does not establish transfer. Sparse teaching remains sparse. | On new public conditional examples, a measured bridge still cannot generate low-frequency/held-out lexical identities available in its source, or free generation gains vanish under source changes. A small teacher-CE improvement without lexical transfer rejects the claimed benefit. |
| Full pretrained LM inside producer plus fresh native formation heads | Permitted in principle as a declared licensed successor if separately learned native heads actually determine the full formation proposal and the Claim gate remains independent. It supplies much more language competence than embedding reuse. | A base answer wrapped by a serializer or cosmetic head violates the stated separate-MFM boundary. Native decisions must influence the actual generation/selection path, not merely annotate its answer afterward. It changes the resource, cache, source-conditioning and inherited-behavior surface substantially. | Removing or swapping learned native heads leaves memory decisions unchanged, or the assembled producer cannot outperform the required base/base-fine-tune controls on memory responsibilities. Fluent but unsupported proposals also falsify success. |
| Typed source/context/prior-memory pointer | Compatible with either lexical route. Learn selection from complete caller-authored inventories; resolve identities through a strict compiler. It avoids requiring a global vocabulary head to invent arbitrary existing IDs. Source-span copying is a separate channel requiring exact alignment. | A pointer does not choose the right memory, interpret uncertainty, author paraphrases or understand evolution. Missing prior-memory inputs, target-derived inventories and wrong typing can make a apparently successful selector meaningless. Processed source tokens include instructions and context as well as evidence. | Novel identity renaming and inventory-order changes break selection, distractors attract unsupported actions, or complete exact IDs improve while positive semantic decisions and value text remain failed. The last result rejects a pointer as a sufficient repair. |
| Full typed AST/event decoder | Compatible if it learns every semantic node and relation and compiles losslessly to the complete formation contract. Deterministic serialization can remove incidental JSON syntax from the prediction problem without deciding meaning. | A tiny fixed template or a few classifiers would narrow MFM. Full coverage includes variable-cardinality proposals, dispositions, time intervals, uncertainty, shared/new/local versus existing references and free semantic values. Syntax validity can conceal systematic semantic failure. | A full-contract round trip loses a field or relation, or the learned decoder emits valid trees but fails subject/time/evidence/uncertainty/correction choices and positive grounded behavior. A codec pass alone supplies no learning evidence. |
| Broad fresh conditional pretraining | Compatible with the fresh specialist route and future first-party foundation destination. It can teach the conditional lexical/sequence path before full formation specialization. Public source-only tasks can supply useful learning without personal biography. | No large, ready, rights-admitted current-contract corpus has been established. The current 47,399 source labels per epoch are not broad pretraining. Frozen Gemma states versus jointly learned encoder/decoder pretraining are materially different assumptions. Data, export storage and accelerator throughput must be measured, not inferred from a paper. | A finite public stage improves reconstruction while held-source transfer or full formation regresses, or the measured resource budget cannot provide meaningful lexical coverage. That rejects the declared stage, not all scratch architectures. |

These mechanisms can complement one another. The table is a constraint audit,
not a request to train five competing systems or to declare each indispensable.

## Primary mechanisms and transfer limits

[Rothe et al., §3](https://aclanthology.org/2020.tacl-1.18.pdf) pair a pretrained
encoder with a random decoder in BERT2RND, but explicitly share pretrained
embeddings. Their sentence-fusion experiments use millions of examples; the
smallest subset is 45,000. This supports investigating lexical reuse, not claiming
that our independently random head or 441 examples replicate that result. Their
architectures also fine-tune representations; our frozen-source transfer is a
different hypothesis.

[The pinned Transformers 5.17.0 Gemma implementation](https://raw.githubusercontent.com/huggingface/transformers/v5.17.0/src/transformers/models/gemma4/modeling_gemma4.py)
ties `lm_head.weight` to the language-model token embedding, scales token input
embeddings, and applies the configured output softcap. Reusing this tensor's
geometry in a 768-wide native decoder would not reproduce the original LM
computation. Input/output bridges and calibration must be specified and tested.

[Pointer-generator §2.2](https://aclanthology.org/P17-1099.pdf) combines learned
vocabulary probability with source attention through a learned gate. The
[pinned Fairseq implementation](https://raw.githubusercontent.com/facebookresearch/fairseq/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/examples/pointer_generator/pointer_generator_src/transformer_pg.py)
scatters attention using explicit source token IDs. Our bank's processed IDs are
not automatically an original-evidence alignment map. A typed registry pointer
and a textual copy channel solve different identity problems.

[RAT-SQL §4.3–4.4 and §5.4](https://aclanthology.org/2020.acl-main.677.pdf)
learn alignment and schema-node selection inside an AST decoder. The paper also
reports that its simple token pointer fails on multi-token values. This supports
typed identity selection while warning against treating it as complete value
generation. Its database schemas are available inputs; it does not justify
inventing unavailable MFM prior-state candidates.

[Yin and Neubig, §3–4](https://aclanthology.org/P17-1041.pdf) generate grammatical
structure with learned rule actions and variable terminals with generation/copy
actions. Syntax and lexical leaves remain distinct learned responsibilities.
[Text2Event §2–3](https://aclanthology.org/2021.acl-long.217.pdf) instead reuses
T5 language competence for reversible event structures and restricts mentions to
input spans. Its extraction design is not a full memory-evolution model; its
span-only output constraint cannot be imposed on MFM paraphrases and inferences.

[MASS §3–4](https://proceedings.mlr.press/v97/song19d/song19d.pdf) jointly
pretrains encoder and decoder, masks decoder-visible context to encourage source
dependence, and uses large monolingual corpora. It supports conditional teaching
as a mechanism. It does not predict a sufficient budget for a frozen Gemma source
plus our fresh decoder, and its corpus rights cannot be inherited as MFM admission.

## Concrete next work, shared by the credible routes

1. Build a separately versioned caller-input manifest and read-only typed
   registry from the original context, opened exact bytes and explicitly
   available prior memory. Include complete candidate records and distractors,
   subject/scope/type/digest bindings, stable source addressing and the distinction
   between existing identities and newly allocated local slots. Never derive an
   inventory, candidate order or expected action from a teacher decision. Missing
   claims/scopes/prior memory are explicit input gaps. Registered IDs in source
   bytes and unavailable external identities are different cases.
2. Stage actual rights-cleared public conditional material and tokenize complete
   input/target sequences under a versioned recipe. Preserve original bytes,
   code/record boundaries, generator/source splits, removals and target
   derivation. Source reconstruction/span tasks teach copying and conditioning;
   genuine paired or authorized derived material is needed for paraphrase and
   semantic transformations. Historical N0/public-menu and legacy 49,819-case
   listings are leads, not admitted payloads or full-role targets.
3. Record the lexical-interface design and resource plan before adopting it.
   Lexical reuse is the strongest small successor hypothesis to examine because
   it directly changes the demonstrated random lexical interface while preserving
   fresh native semantic decisions. It remains a hypothesis until actual bridges
   and conditional transfer are measured. Do not prematurely add projected
   embeddings, a pointer, grammar constraints or another private 441-case fit.

The first two items build useful real inputs and teaching substrate now. They do
not create a new qualification program. Existing structural/lineage checks and a
bounded actual backward/reload/resource check belong inside the chosen build.
The original invalid outputs, matched controls and whole-role evaluation remain
the before/after evidence; the nine repeatedly inspected dev cases cannot become
independent qualification or the optimization target.

For the formation successor, measure actual target-free positive grounded
proposals and complete semantics, not only first tokens, exact author wording,
loss, EOS or valid JSON. Use source/identity counterfactuals whose correct outcome
is observable, include competing candidates and legitimate paraphrases, and retain
the original JSON/codec/grounding failures. Missing native media and accepted-
memory consumer outcomes remain work, rather than negative labels or neutral
defaults. Required specialist/base controls establish native contribution when a
capability claim is made; they are not a prerequisite tournament before building
the missing substrate.

## Resource and custody limits

The publisher lexical matrix has 262,144 × 3,840 = 1,006,632,960 elements: about
2.01 GB in BF16 or 4.03 GB in FP32. The existing fresh 262,144 × 768 matrix has
201,326,592 elements. Replacing one with the other is not a shape-compatible
checkpoint migration. Keep the reused tensor immutable where possible and count
bridges, trainable weights, gradients, Adam moments, activations and full-vocabulary
projection separately. Training the entire FP32 lexical matrix would add roughly
16.1 GB for weights, gradients and two Adam moments alone. These are arithmetic
floors, not a tested fit estimate.

The complete base is about 23.9 GB of BF16 weights before runtime state. A
full-LM-plus-native-head route cannot inherit the current 267M CPU decoder's
capacity or cache receipts. Nor can a public-bank export be called cheap: at
3840-wide BF16, one million retained source-token states alone occupy about
7.68 GB before masks, IDs and metadata. Pack size and export throughput need an
observed bound, with complete sequences and no silent truncation.

Public-only lexical construction may use an available free accelerator if its
actual device, dtype, storage and complete forward/backward/reload path fit. This
review makes no assertion that a suitable Kaggle allocation is available or that
the 12B base fits a particular free device. Cached public features can avoid
placing the base and learner on one accelerator; generating them still has a
measurable cost. A changed feature recipe or precision needs its own lineage.
Private 441-case source, features, teacher labels, learned checkpoints and optimizer
state stay in the protected owner path. A public learner receipt cannot qualify
private training, and no private material is uploaded to make a free route fit.

## Decision boundary

Build the caller/reference substrate and actual public conditional material.
Withhold architecture adoption until the proposed lexical, selection and
serialization interfaces are concrete and their falsifiers are measurable.
Prefer an explicitly declared lexical-competence successor with a genuinely
native full-semantic producer over a hidden base-answer fallback or a larger
unchanged epoch loop. This is a reasoned direction for the next build, not a
claim that frozen embeddings, copying, ASTs or a pretrained LM already repair MFM.
