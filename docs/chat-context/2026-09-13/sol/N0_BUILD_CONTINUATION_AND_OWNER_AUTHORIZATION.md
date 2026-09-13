# N0 Build Continuation and Owner Authorization — 2026-09-13

**Status:** active continuation pointer  
**Purpose:** supersede the stale “research-only / teacher qualification” next-action language in the earlier 2026-09-13 handoff without deleting historical context.

## Owner correction retained

The owner explicitly states that the project has permission to use Sol/ChatGPT-produced teaching material for A.L.I.C.E. and Fable model development. The Sol teaching seeds must remain part of the governed training lineage rather than being removed because of a generic hosted-model-origin assumption.

A.L.I.C.E. and the OpenAI assistant are not treated as the same system. The EIPM remains an A.L.I.C.E.-native identity/judgment component with its own architecture, weights, evidence authority, memory boundaries, and runtime role.

Sol is the primary development-time teacher for general semantic, pragmatic, social, emotional, epistemic, ranking, and structured-alignment competencies. Canonical Elaina evidence remains the authority for Elaina-specific identity targets. Rayan remains final authority on fidelity.

## Mainstreamed build rule

Do not recreate MC10 as a teacher/judge qualification bureaucracy. Build A.L.I.C.E. directly. Keep integrity, provenance, held-out evaluation, regression, and source-authority checks inside the actual build loop. Escalate only when a concrete failure or ambiguity warrants it.

The active loop is:

```text
public semantic corpus
  -> Alice-native tokenizer
  -> Alice-native random-init N0 semantic model
  -> owner-authorized Sol curriculum
  -> candidate-ranking / judgment training
  -> per-competency + row-level failure evaluation
  -> targeted Sol repair curriculum
  -> regression
  -> repeat only while failures justify more teaching or capacity
  -> N1 governed Elaina identity learning after explicit private-gradient authorization
```

## Parameter scale correction

The approximately 350M N0 configuration is an initial engineering reference, not a global model-size requirement or ceiling. The obsolete 300M-to-400M preflight/test gate has been removed. Scale changes are driven by observed capability or efficiency evidence.

## Build state after this continuation

`alice-eipm-v1-build` tip after the current implementation pass:

- `102771b397b0d3ce9125f3b0908e0c1dfb45c8a1`

Material changes now present on that branch:

- the 60-row owner-authorized Sol N0 curriculum remains active and provenance-bound;
- N0 curriculum rows receive structural validation before training;
- the native tokenizer is loaded directly from the guaranteed `tokenizer.json` artifact rather than relying on generic model-directory discovery;
- candidate evaluation distinguishes top-1 correctness from **supported-set separation**, which matters when multiple alternatives are legitimately preferred;
- development metrics are broken down by competency rather than only one aggregate number;
- `evaluate-curriculum` emits metrics, row-level predictions, and failure rows for the next targeted teaching pass;
- N0 preflight records actual versus planned parameter count but does not reject a model merely for falling outside an arbitrary global range.

`fable-builder-model` tip after parallel process capture:

- `838efd3afef6438b3abcaec2292c18bc335854e5`

FBM now contains a reusable builder event for this failure-driven teaching loop in addition to the preserved Sol bootstrap seed.

## Immediate operational pointer

The next expensive step is no longer another research/qualification round. It is the real N0 public-data execution path:

1. materialize a corpus smoke sample;
2. train/verify the native tokenizer;
3. run forward/backward preflight;
4. materialize the approved public corpus;
5. train the first resumable public N0 MLM checkpoint;
6. train the owner-authorized Sol curriculum ranker;
7. run `evaluate-curriculum` and use concrete failure families to produce the next teaching batch.

No private E0/E-INF/A-SYN gradient is authorized by this continuation note. That transition remains an explicit owner action. Private curated payloads remain outside public Git.

## Help boundary

Sol can continue the repository implementation independently. Rayan is only needed when execution reaches infrastructure that is not directly accessible from the chat environment, such as launching the real GPU run on Magnolia/Kaggle/another approved machine, or when an explicit private-gradient authorization is required.
