# Owner Clarification — Exclusive A.L.I.C.E. Personality Weights

**Date:** 2026-09-10
**Status:** owner clarification; supersedes any wording in the same-day personality-weight research note that implies a third-party foundation checkpoint is a permanent part of A.L.I.C.E.'s identity/personality model.

## Owner constraint

A.L.I.C.E. is being built as part of a future startup/product family. Persistent A.L.I.C.E. model components and learned identity artifacts should be exclusively A.L.I.C.E.-owned rather than depending on a third-party vendor's model weights as a permanent embedded component.

A third-party/open-weight model such as Qwen may be used as an **offline tool or teacher** for tasks such as synthetic-data generation, corpus transformation, comparison, or mechanics experiments, provided its weights are not part of the resulting A.L.I.C.E. entity/model artifact.

Astra may likewise be used as the single external pre-weight architecture/corpus reviewer and, when useful, as an offline reasoning/data-generation tool. Astra is not an A.L.I.C.E. runtime weight dependency.

## Architectural correction

Do not describe the target EIPM as `Qwen + Alice LoRA` if that combination is intended to be a permanent A.L.I.C.E. personality model. A LoRA trained on Qwen still requires or derives from Qwen's underlying weights and therefore does not satisfy the owner's exclusivity requirement.

The target distinction is:

- **A.L.I.C.E.-native learned personality component:** weights/artifacts created for A.L.I.C.E. and owned as part of the project.
- **Replaceable reasoning engines / external teachers:** Qwen, Astra, future frontier models, or specialist systems may assist generation, reasoning, evaluation, or execution but are not the identity artifact itself.
- **Canonical model-agnostic identity substrate:** E0, structured source/persona evidence, E-INF, A-SYN, provenance, clone-awareness rules, behavioral specifications, and training lineage remain portable and independent of any external model family.

The exact A.L.I.C.E.-native personality-model architecture is to be selected after source/corpus inventory and technical research. It may use a custom model trained from scratch or another architecture that does not embed third-party model weights. The priority is highest achievable Elaina fidelity while preserving startup ownership, portability, and later multi-model A.L.I.C.E. architecture.

## Consequence for the workflow

The data workflow remains valid:

`E0 -> structured identity evidence/persona graph -> E-INF -> A-SYN -> coverage-driven training corpus -> one Astra pre-weight review -> A.L.I.C.E.-native personality weight creation -> owner fidelity judgment`.

What changes is the final weight target. Qwen is no longer assumed to be the permanent base of the EIPM. If used at all, it is a tool/teacher unless the owner later explicitly authorizes a licensed third-party base as a persistent A.L.I.C.E. component.
