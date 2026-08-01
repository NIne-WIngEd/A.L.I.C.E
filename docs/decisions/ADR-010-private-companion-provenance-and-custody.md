# ADR-010 — Private Companion Provenance and Custody

**Status:** Accepted
**Date:** 2026-07-31
**Decision owner:** MK Rayan

## Context

A.L.I.C.E. has an owner-ratified private companion purpose backed by highly sensitive source, directive, person-model, and training material. Existing public/private boundaries prohibit exposing this material in Git, but Phase 5 needs neutral contracts for provenance, learning eligibility, deletion, and evaluation.

## Decision

Public Git stores only opaque `PX-*` identifiers, neutral schemas, custody rules, validators, and synthetic fixtures. Private source, plaintext meanings, owner models, examples, keys, ciphertext, decrypted material, and training artifacts remain in owner-controlled storage. Canonical source, inference, reconstruction, correction, and evolved identity are distinct typed states.

Phase 5 builds lineage and custody substrate; later phases may perform evaluated private learning and parametric adaptation. This restriction protects custody and truthfulness and is not a permanent capability ceiling.

## Consequences

- encrypted private persona payloads are not treated as public-repository-safe;
- key and ciphertext use separate custody and rotation;
- generated reconstruction cannot silently become historical fact;
- private companion data never seeds Friday or shared-kernel distributions;
- deletion and rollback lineage apply to future derived artifacts.

**Baseline superseded:** `07e95a85d27b0c08b08ab857c6d9b75cdf8a6446` where conflicting.

## Owner clarification — clone-aware identity target

The private companion purpose is specifically the highest-fidelity achievable reconstruction of the owner-designated source person's personality and mindset. "Inspired by" or generic companion behavior is insufficient. The reconstruction must know and disclose that it is an AI clone rather than the literal original person. This clarification strengthens the capability objective while retaining provenance, custody, uncertainty, and product-isolation rules.
