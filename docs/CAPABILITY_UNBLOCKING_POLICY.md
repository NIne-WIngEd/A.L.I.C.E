# A.L.I.C.E. Capability Unblocking Policy

**Version:** 1.0.0
**Status:** Ratification candidate
**Authority:** A.L.I.C.E. Constitution 1.1.0

## Purpose

No implementation file, test, phase document, schema, policy, workflow, or historical decision may silently become a permanent ceiling on A.L.I.C.E.'s eventual capability.

A current release may intentionally ship with capabilities disabled. That is a deployment profile, not a statement that the capability is forbidden forever.

## Rules

1. Every restrictive implementation rule must be one of:
   - an authority-kernel invariant;
   - a mission boundary;
   - a release-local capability profile;
   - a resource limit selected by configuration;
   - a temporary constraint registered in `docs/CONSTRAINT_REGISTRY.md`;
   - a historical record explicitly marked superseded or non-normative.
2. Feature modules may not hard-code a phase number as the permanent maximum capability.
3. Boolean capability checks must be authorized by a named capability profile.
4. Numeric limits must be configurable by a resource profile or mission budget. Constants may provide defaults, not universal maxima.
5. Provider, model, tool, data-classification, network, retention, and deployment choices must be registries or policy inputs rather than closed lists embedded as permanent architecture.
6. Tests may verify a compatibility profile, but test names and assertions must not imply that the entire project can never exceed it.
7. Old ADRs and phase documents remain historical evidence. If superseded, they must say so prominently.
8. New capabilities are presumed researchable. They do not require a new constitutional amendment unless they alter the minimal authority kernel.
9. The repository-wide barrier audit is a required merge check.

## Capability profiles

The canonical registry is `policies/capability_profiles.json`.

A profile contains:

- a domain;
- a scope kind;
- authorized capabilities;
- resource limits or defaults;
- a declaration that it is not a permanent ceiling.

The released Phase 3 and Phase 4 behaviors remain available as compatibility profiles. Broader profiles may enable tools, live networks, memory writes, sensitive grounding, authenticated browsing, background operation, code execution, provider fallback, self-training, and self-modification under mission authority.

## What remains non-negotiable

Only the minimal authority kernel is excluded from silent autonomous activation:

- Rayan's final constitutional authority;
- truthful reporting of material state and outcomes;
- legitimate stop and rollback control;
- provenance, evaluation, and audit integrity;
- credential custody;
- recoverable known-good states for material self-change.

Everything else is eligible for research, replacement, or evolution.
