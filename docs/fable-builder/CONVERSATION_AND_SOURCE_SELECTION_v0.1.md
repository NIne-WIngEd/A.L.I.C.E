# FBM consumer build and conversation handoff — v0.1

**Status:** first-release research contract, 2026-09-25. No runtime implementation or qualification is implied.  
**Scope:** `fable-builder-model` workstream and A.L.I.C.E.-first transfer to Fable Sleight.

## Builder scope

FBM must eventually construct and connect the qualified consumer personal foundation: identity/personality, Memory Formation, host, relationship, and assistant-self capabilities, together with evidence, provenance, memory, Experience Ledger, context, and governed revision infrastructure. These five names identify starting roles, not a settled neural-model count. Some roles may require multiple learned components or structured stores.

For a new user, the builder must start from that user's authorized evidence. Identity-neutral semantic foundation training (N0 or a successor) also needs public, licensed, or user-provided eligible source data and controlled synthetic teaching examples. Provide a user-facing source selection and addition flow, a provenance/license eligibility manifest, an adequacy check, and a compute estimate. Excluding sources is permitted, but a foundation that cannot qualify must fail transparently. Disallow transferring Elaina or Rayan private content. Record whether each personal weight file really began at fresh initialization; a third-party checkpoint with user tuning does not qualify as personal weights trained from scratch.

The formation model is responsible for preparing personal components and their test data. It does not itself become the source person, the live conversation voice, or an independent memory authority. Observed outcomes must enter governed personal-state revision so that later relevant judgments change.

## First-release conversation dependency

The native EIPM/host/relationship/self/memory path and ACFP context assembly must produce a versioned identity decision packet *before* downstream generation. Beyond stance, the packet must carry supported reasons, uncertainty, relationship context, and expression obligations that can distinguish the entity's voice from a generic assistant that merely agrees with the conclusion.

A first-party local conversation capability manages each turn. It derives an API request from the packet and the minimum permitted abstract task context, receives a candidate from a replaceable external language service, checks the draft using the five personal roles and existing memory/governance architecture, restores private references locally, and accepts or issues a bounded focused correction. There is no extra sixth memory/judge model and no external provider acting as Fable's judgment authority.

The normal path should succeed on the first provider response. FBM should capture teaching, evaluation, and hard-negative examples where fluent outputs match the conclusion but miss the identity, relationship-specific voice, disagreement intent, or evidence bounds. It should capture first-pass failures and targeted repair cases without storing private payloads in this public branch. The first-party comparison, expression, and retry topology must be decided by implementation evidence rather than naming an orchestrator a model.

The local encoder/decoder is a privacy and task translation boundary. It is not a claim that an external text API can understand ciphertext or learn nothing from abstract semantics. Maintain local placeholder maps; respect permissions and visible egress controls; reject or require specific authorization when useful external processing needs sensitive content. Repeated-call leakage must be evaluated. Offline mode preserves the local foundation but cannot silently promise API-grade language generation.

## Qualification before product transfer

Demonstrate distinct state causes distinct verdict and expression with the provider held fixed; preserve behavior under provider replacement. Deliberately feed drafts that are superficially agreeable but wrong for the entity and confirm they are rejected. Measure first-pass acceptance, corrective-call frequency, tail latency, user-recognizable voice, egress privacy across turns, and honest fallback. Thresholds should be set and tested before release.

The consumer-facing version is [Fable Sleight's first-release plan](https://github.com/NIne-WIngEd/Fable_Sleight/blob/main/docs/FIRST_RELEASE.md). The main-branch architectural authority is [Fable Personal Development Architecture](https://github.com/NIne-WIngEd/A.L.I.C.E/blob/main/docs/FABLE_PERSONAL_DEVELOPMENT_ARCHITECTURE.md). A.L.I.C.E. must prove the transferable capability before Fable can claim it.
