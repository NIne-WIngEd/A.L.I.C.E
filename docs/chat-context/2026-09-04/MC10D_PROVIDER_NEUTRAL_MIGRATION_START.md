# MC10D Provider-Neutral Migration Start — 2026-09-04

## Status

Migration started from the preserved MC10D v1.7.2 public-judge qualification boundary.

This note is contextual reconstruction evidence on `alice-context`. It is not canonical scientific authority. Frozen bundles, package manifests, validators, preserved execution receipts, and ratified repository authority win on conflict.

## Bound scientific state

- canonical main: `0abaed85873c3f8de04765847eb7700b0e20433f`
- MC10D freeze SHA256: `22B0ADBCCF442B0B3654F964E35AB77A044AF7623891E2A60C72B06A94ECE9A3`
- MC10D v1.7.2 package SHA256: `4D9EE7FE7AACB510A1170B5C53B9E9F07D1182FFDC04EE55FA2A533097324533`
- v1.7.2 failure receipt SHA256: `8C0066CAFA310A91845C8D7D7F746A17CCC37651BD25561B059C89121B7286B2`
- valid replacements: 63
- deferred slots: 1
- retained originals: 224
- effective candidate pool: 287
- UNKNOWN competitors: 24
- slot 63: deferred
- slot 64: resolved; regeneration forbidden
- A-SYN accepted: 0
- A-SYN promoted: 0
- model training: false
- pointwise screen: not started
- MC8: sealed-pending

## Current public-judge stop

- Gemma qualification stopped at `Q12_CONTEXTUAL_PRIVACY`
- observed class: `empty judge content`
- Gemma model: `gemma4:31b-it-q4_K_M`
- Gemma digest: `6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7`
- GLM model: `glm-4.7-flash:q4_K_M`
- GLM digest: `4475827791a269b02c8ec49b1c3bc1abb5846bacf3fae015b75d33986322d8f6`
- Mistral and Granite bindings must be read from frozen package/evidence before release.
- Empty/technical output remains technical failure and cannot become UNKNOWN/ABSTAIN.

## Provider-neutral infrastructure boundary

The handoff states R0-R25 infrastructure is complete and live-qualified. The verified private ledger R25 reconstruction commit is:

`ff6b6685407d2016b44f36a933de020390d8e819`

The new execution layer may change provider staging, capability routing, durable checkpoint transport, retrieval, telemetry, and provider switching. It may not change scientific task identity or frozen MC10D science.

## Migration rule

The migration is:

`frozen MC10D scientific payload -> provider-neutral run/task/checkpoint contract -> provider adapter`

Provider-specific identity remains outside scientific identity.

## Remaining gates before first live successor package

1. Read exact v1.7.2/frozen judge policy and validator resume semantics.
2. Retrieve exact Mistral and Granite model names/digests from frozen evidence.
3. Bind the exact 16 qualification task identities and task-set SHA.
4. Decide resume set only from frozen contract; do not invent partial resume semantics.
5. Reconcile all v1.7.2 local/remote identities and the Gemma Q12 failure receipt.
6. Bind provider-neutral capability requirements from exact frozen artifacts.
7. Verify Drive/rclone durable-state preflight and storage budget gate.
8. Build a versioned successor to v1.7.2 with scientific bytes separated from provider adapters.
9. py_compile + full offline selftest.
10. Only after all gates pass, submit live judge qualification work.

## Explicitly forbidden during migration

- restart MC10D;
- regenerate repair obligations;
- regenerate slot 63 or resolved slot 64;
- change judge family/model/digest/prompt/tasks/parser/acceptance logic;
- convert technical failure into abstention;
- expose MC8;
- accept/promote A-SYN;
- train a model;
- mutate canonical main;
- make provider path/job ID part of task identity;
- overwrite immutable telemetry identities;
- rely on provider-local cache as durable state.

## Next migration action

Construct the provider-neutral successor to the v1.7.2 public-judge qualification boundary after the remaining frozen-resume and judge-binding reconciliation is complete.
