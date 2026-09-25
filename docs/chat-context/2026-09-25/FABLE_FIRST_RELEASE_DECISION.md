# Fable first-release conversation decision — 2026-09-25

**Status:** product and architecture target; not current implementation evidence.

The user approved this first-release direction for Fable Sleight:

- Fable's local personal foundation comprises the starting identity/personality, MFM, host, relationship, and assistant-self roles, plus memory, provenance, Experience Ledger, context assembly, and governed learning. The roles do not fix a total count of trained models.
- An identity-neutral N0-like foundation uses qualified outside data as well as suitable synthetic teaching material before private identity learning. The Fable installer should allow a user to choose eligible outside source packs and add permitted sources. Source provenance, licensing, adequacy, compute cost, and actual fresh initialization must be inspectable.
- The live local conversation capability must obtain native judgment from the personal foundation *before* any downstream language service is used. The verdict includes stance, reasons, evidence/uncertainty, relationship context, and expression obligations; agreement with the conclusion alone does not qualify a reply.
- A local encoder/decoder and egress gateway prepares privacy-limited API requests. GPT/Claude-class APIs may propose language and provide other feature skills in the first release, but cannot be Fable's personality or judgment authority. The personal foundation and conversation architecture check each draft, including the way it speaks, and allow focused bounded correction if needed. The goal is a fast first-pass match for the great majority of interactions. No separate sixth memory/judge model is required.
- Masking or encoding cannot guarantee zero disclosure of personal meaning to an external text API. Exact private-content tasks require local handling or specific authorization. Offline functionality has explicit first-release limits. Later stages aim to build first-party frontier feature models and allow the builder to construct feature specialists.
- A.L.I.C.E.'s Phase 3 local Qwen adapter is a research runtime, not evidence of this destination conversation behavior. Implement and qualify the transferable native verdict, response comparison, expression control, privacy, latency, outcome learning, and provider-swap behaviors in A.L.I.C.E. before consumer parity. A.L.I.C.E.'s Elaina-derived source identity remains distinct from Rayan's host model and does not transfer to a Fable consumer.

Canonical product document: https://github.com/NIne-WIngEd/Fable_Sleight/blob/main/docs/FIRST_RELEASE.md  
A.L.I.C.E. main architecture: https://github.com/NIne-WIngEd/A.L.I.C.E/blob/main/docs/FABLE_PERSONAL_DEVELOPMENT_ARCHITECTURE.md  
FBM workstream handoff: https://github.com/NIne-WIngEd/A.L.I.C.E/blob/fable-builder-model/docs/fable-builder/CONVERSATION_AND_SOURCE_SELECTION_v0.1.md  
A.L.I.C.E. merged documentation PR: https://github.com/NIne-WIngEd/A.L.I.C.E/pull/93

This note records an architecture decision. No model training, runtime change, or release qualification was performed for this conversation plan.
