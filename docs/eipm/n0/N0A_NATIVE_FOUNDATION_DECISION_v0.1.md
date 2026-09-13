# N0A Native Foundation Decision v0.1

**Date:** 2026-09-13  
**Status:** active build decision; implementation starts here  
**Model:** `alice-n0-semantic-v0.1`

## Objective

Build the first real A.L.I.C.E.-native semantic foundation checkpoint. This is not a teacher/architecture qualification artifact. It is intended to become an actual ancestor of the permanent Elaina Identity / Personality Model unless later observed failures justify replacement.

## Architecture selected

Use a native bidirectional transformer encoder with modern encoder mechanics:

- 24 transformer blocks;
- hidden width 896;
- 14 attention heads (64 dimensions/head);
- feed-forward width 3584;
- GeGLU feed-forward blocks;
- pre-normalization;
- rotary positional embeddings;
- alternating local/global attention;
- tied token embedding / masked-token output weights;
- no inherited third-party model weights.

Approximate parameter scale is ~350M parameters. This is an initial build scale, not a ceiling or ratified optimum. Increase capacity only if actual learning/fidelity failures indicate a capacity bottleneck.

## Attention/context policy

Initial training uses mixed sequence lengths up to 4096 tokens to avoid spending scarce compute on long context before the model has learned basic semantics.

The architecture remains compatible with later 8192-token continuation. Long-context continuation is performed only when the real Alice workload shows material benefit.

Local-attention blocks use a 512-token window. Every fourth block is global attention so information can propagate across the whole sequence.

## Tokenizer decision

Start with a 48,000-token byte-fallback BPE tokenizer trained only from approved public N0 text.

Reasons:

- large enough for efficient English/mixed technical text;
- small enough to avoid excessive embedding parameters;
- byte fallback guarantees representability of names, unusual spellings, code fragments, and multilingual material;
- no private Elaina text needs to enter the public/native tokenizer build.

This is a build choice, not a tournament winner. If real tokenizer diagnostics later show pathological fragmentation on Alice inputs, repair the tokenizer or introduce a governed private extension rather than running a broad tokenizer competition now.

## Public corpus decision

Use only sources marked training-allowed in the N0 public-source manifest. Common Pile / Comma material is the primary broad language source when the individual source/license metadata permits training. Sources with unresolved content-rights questions remain excluded from training even if useful for research/evaluation.

The source manifest remains a safety/provenance control, not a separate qualification phase.

## N0 teaching objectives

### Foundation objective

Dynamic masked-language modeling is the main public-corpus objective.

Use span-aware masking so the model must reconstruct phrases and relationships rather than isolated easy tokens only.

### Sol-authored curriculum

Ordinary corpus learning is supplemented with explicit teaching material for:

- paraphrase and semantic equivalence;
- contradiction and negation;
- temporal ordering;
- causal relations;
- reference/coreference;
- implicature and indirect meaning;
- presupposition;
- sarcasm/irony where context supports it;
- politeness and social intention;
- emotion and affect interpretation;
- relationship-sensitive interpretation;
- uncertainty and literal grounding;
- candidate ranking/judgment;
- structured context and ACFP-like field alignment;
- evidence/relation graph alignment.

Sol is the primary development-time curriculum generator. These examples teach general understanding. Elaina-specific identity targets remain governed by the private identity substrate in N1.

## Training loop

```text
approved public source shards
        -> deterministic normalize/filter/dedup
        -> 48k tokenizer
        -> native MLM pretraining
        -> compact held-out semantic checks
        -> Sol-generated targeted curriculum
        -> re-check failures/regressions
        -> continue training/repair
```

Do not wait for a large external qualification program before the first checkpoint.

## Checkpoint policy

The first meaningful checkpoint is a real model artifact, not a smoke-test-only artifact.

Checkpoints must record:

- source manifest revision;
- tokenizer hash;
- model config hash;
- training code revision;
- token count seen;
- optimizer/scheduler state where resumable;
- RNG seeds;
- hardware/runtime metadata;
- checkpoint SHA256;
- compact held-out results.

Training continues while capability improves materially. There is no arbitrary fixed token-count finish line.

## N0 -> N1 transition

N1 begins when the N0 model can reliably represent the semantic, pragmatic, uncertainty, relationship, and ranking distinctions needed to learn from the curated private personality substrate.

The transition is based on direct observed competence, not agreement from an external judge committee.

The owner still authorizes the first private identity-gradient step.

## Parallel FBM capture

Every material N0 build step produces a compact process trace on `fable-builder-model` describing the reusable transformation, constraints, and observed result. Alice development remains primary; tracing must not block training.
