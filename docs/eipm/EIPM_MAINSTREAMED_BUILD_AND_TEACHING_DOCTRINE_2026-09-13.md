# EIPM Mainstreamed Build and Teaching Doctrine

**Date:** 2026-09-13  
**Status:** owner-directed active working doctrine  
**Precedence:** where this document conflicts with the earlier staged N0 qualification sequence, this document controls the active build path.

## Problem being corrected

The earlier N0 research plan began to recreate the failure pattern seen during MC10: substantial time and compute were being spent qualifying datasets, teachers, judges, architectures, and scale choices before producing the next real A.L.I.C.E. capability.

Validation is still required. Provenance, privacy, held-out evaluation, checkpoint integrity, regression testing, and source authority remain mandatory. What is removed is the assumption that every design decision needs a separate external tournament or qualification stage before development can continue.

## Active development rule

> Build A.L.I.C.E. directly. Validate inside the build loop. Escalate only when a concrete failure, ambiguity, or capability gap justifies it.

The active loop is:

```text
prepare substrate
    -> build/train the next native capability
    -> evaluate Alice directly
    -> classify concrete failures
    -> generate targeted teaching/repair material
    -> retrain/update
    -> regression + fidelity check
    -> repeat
```

## Teacher authority

Sol is the primary development-time teacher and curriculum generator for the current EIPM build.

Sol may generate:

- semantic and language-understanding teaching examples;
- pragmatics and implied-meaning exercises;
- social/emotional interpretation cases;
- temporal and causal reasoning cases;
- ranking and judgment examples;
- uncertainty and literal-grounding cases;
- ACFP/structured-context exercises;
- persona/evidence graph alignment tasks;
- contrastive candidates;
- hard negatives;
- targeted repair examples after observed model failures.

Sol is **not** the authority on Elaina identity truth.

The division is:

> Sol teaches the model how to understand. Canonical Elaina evidence teaches the model who to be. Rayan remains final owner authority.

For identity-specific teaching, targets must be derived from the governed E0/E-INF/A-SYN substrate and its provenance rules. Generic Sol preferences must not silently become Alice personality labels.

## External models

Astra, Qwen, Claude, Gemini, reward models, rerankers, or other systems may be used as optional specialists when a concrete need appears. They are not standing prerequisites, mandatory judges, or members of a permanent committee.

Examples of justified external use include:

- producing diverse alternative candidates when Sol output is too homogeneous;
- solving a demonstrated linguistic or reasoning gap;
- investigating an architectural failure that remains unclear after direct debugging;
- performing a high-risk audit before a major irreversible training step.

No broad teacher tournament is required by default.

## Active N0 sequence

The previous R1/R2/R3/R4/R5/R6/R7-style qualification chain is no longer the required execution path.

The active sequence is:

### N0A — finalize buildable native foundation

Choose the native semantic architecture, tokenizer, and legally usable public corpus using the research already completed. Additional comparison is performed only where a specific unresolved choice blocks implementation.

### N0B — build preprocessing and curriculum

Implement the public-corpus preprocessing path and generate targeted Sol-authored teaching material for competencies that ordinary corpus learning does not cover well.

### N0C — train the first real Alice-native semantic checkpoint

Produce an actual permanent native checkpoint rather than another qualification artifact.

### N0D — evaluate and repair

Use held-out competency cases, regression tests, direct Sol analysis, and Rayan review. Diagnose actual failures and generate targeted repair material.

### N1 — identity learning

Begin governed Elaina-derived identity training using the curated private substrate once the native semantic foundation is competent enough for the task and the owner authorizes the private-gradient step.

## What remains mandatory

Mainstreaming does not weaken the following:

- exact source/dataset/checkpoint hashes;
- explicit source and provenance class;
- legal/license screening for training corpora;
- train/dev/held-out separation;
- private-data custody and no public leakage;
- E0/E-INF/A-SYN distinction;
- source/evidence-family split before mechanically related expansions;
- deterministic preprocessing where practical;
- checkpoint and resume integrity;
- compact hidden fidelity and regression sets;
- owner review before promotion of a major identity checkpoint;
- no ordinary runtime process silently rewriting the core identity model.

## Parameter scale

No fixed size target or ceiling exists. Prior 100M–400M references are historical working guesses only.

Do not run a large parameter-count tournament merely to produce a chart. Start with a credible architecture and scale. Train it. Increase capacity only when observed learning/fidelity failures indicate that capacity is a real bottleneck.

## Public source manifest

The existing N0 public-source/license manifest remains useful as a safety and provenance artifact. It is no longer a phase gate that must grow into an exhaustive dataset qualification program before training work can proceed.

A source must still be legally suitable before it is used for training.

## Fable process capture

The same operations used here to build A.L.I.C.E. are now being captured on the dedicated `fable-builder-model` branch. The objective is to train a future host-neutral Fable Builder Model that can perform source interpretation, E-INF/A-SYN generation, coverage planning, curation, teaching, evaluation, and targeted repair without depending on hosted Sol/Astra access for each consumer installation.

Process capture must remain lightweight and must not slow the A.L.I.C.E. build materially.
