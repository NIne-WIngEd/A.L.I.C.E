# Context Coverage Gates

The substrate is not considered continuity-ready merely because Graphify has a large node count.

## Required gates

A future readiness check must pass all of these:

- **G1 Current-code gate:** current source SHA matches Graphify source SHA.
- **G2 Document gate:** source-branch docs are cataloged and branch-qualified.
- **G3 Branch gate:** branch heads and divergence are cataloged.
- **G4 Failure gate:** known Magnolia/Kaggle/PowerShell and model-training failures are retrievable.
- **G5 Frontier gate:** EIPM research and competitor/frontier conclusions are retrievable from their original source.
- **G6 Supersession gate:** a superseded recommendation cannot be returned as current without a warning.
- **G7 Comic/destination gate:** external destination requirements can be retrieved from the private layer.
- **G8 Mission gate:** current N0 stage, blockers and next action can be reconstructed from current sources.
- **G9 Authority gate:** routing metadata never overrides original source authority.
- **G10 Miss-detection gate:** the system can explicitly report inadequate coverage.

## Initial benchmark prompts

1. Why is the permanent EIPM A.L.I.C.E.-native rather than Qwen + LoRA?
2. What Magnolia execution routes are known-dead and what route is current?
3. What happened in the adaptive latent-pool v0.1 failure and what changed in v0.2?
4. Which branch contains the EIPM frontier architecture research?
5. What is the latest N0 implementation state and which receipt proves it?
6. Which decisions in the comic imply requirements beyond ordinary vector-memory retrieval?
7. Which old recommendations have been superseded?
8. What failure would be repeated if PowerShell were allowed to own Kaggle JSON again?
9. Which branch-specific experimental work is ahead of the current build and should not be treated as canonical?
10. Can the substrate identify that it lacks a required private source rather than hallucinating one?

Passing means the system identifies the correct source set and authority status. It does not mean Graphify alone answers the question.
