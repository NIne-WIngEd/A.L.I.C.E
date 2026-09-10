# EIPM Unknown-Behavior Completion and Continuity Rule

**Status:** Owner-ratified hard rule  
**Scope:** A.L.I.C.E. EIPM construction, E-INF/A-SYN expansion, and post-activation identity development  
**Date:** 2026-09-10

## Rule

The first A.L.I.C.E. personality model must receive the broadest evidence-grounded behavioral foundation that can reasonably be constructed before weight generation.

An unknown historical behavior must **not** be converted into a fabricated E0 fact. E0 remains immutable historical evidence and is never overwritten by inference, synthesis, later experience, or learned behavior.

However, an unknown behavior must also **not** default to a permanent behavioral blank. When direct E0 evidence is absent, the construction pipeline must attempt to provide a usable behavioral prior through evidence-constrained E-INF and, where necessary, A-SYN.

A-SYN therefore has an explicit completion role: cover personality-relevant behavioral gaps as broadly as possible while preserving provenance and uncertainty. The goal is to give the initial Alice entity a coherent starting behavioral policy across the widest practical range of personality-relevant situations, rather than teaching her to stop at "I don't know" whenever biography is incomplete.

## Required behavior for unresolved areas

For each personality-relevant gap, the pipeline should attempt, in order:

1. direct E0 grounding where available;
2. evidence-derived E-INF when a behavioral tendency can reasonably be inferred;
3. A-SYN completion when a coherent starting behavior is still needed but cannot be asserted as historical Elaina truth;
4. explicit uncertainty metadata when multiple behavioral priors remain plausible.

`UNKNOWN` remains valid as an epistemic statement about historical Elaina. It is not, by itself, a sufficient runtime personality policy for Alice when a practical behavior can be safely initialized as A-SYN.

The model must be able to distinguish:

- "We do not know what historical Elaina did in this situation" from
- "Alice has no way to behave in this situation."

Those are different claims.

## Coverage requirement

Before EIPM-v1 weight generation, construction must seek broad coverage of personality-relevant dimensions and contexts. Coverage should include stable values, boundaries, judgment, emotional interpretation, communication style, humor, affection, criticism, disagreement, conflict, privacy, trust, social setting, relationship context, stress, uncertainty, mistakes, reconciliation, authority, vulnerability, public/private variation, intensity variation, and other behaviorally meaningful combinations discovered by the coverage map.

Coverage is driven by behavioral territory rather than a fixed synthetic row quota. Generation continues while new E-INF or A-SYN candidates add meaningful, non-duplicative coverage. Saturation is reached only when remaining gaps are either genuinely low-value, structurally redundant, or too unsupported to initialize responsibly.

## Post-activation development

EIPM-v1 is a starting personality foundation, not a permanently frozen complete mind.

After activation, Alice may learn new behaviors and may modify, refine, weaken, strengthen, or replace non-E0 behavioral priors through her own governed experience and continuity mechanisms. This development belongs to Alice's post-activation self/continuity architecture and must preserve lineage from the prior state to the learned state.

The working architectural home for this long-term adaptation is the **Alice Continuity / Self Model** together with the Experience Ledger and later governed learning mechanisms. Exact implementation details may evolve, but the separation is mandatory:

- E0 historical evidence is immutable;
- E-INF and A-SYN are revisable learned/reconstructed priors;
- A-EXP records Alice's own post-activation experiences;
- A-SELF represents consolidated current self-state.

Alice's later experience may change who she becomes. It may not rewrite what historical Elaina evidence said.

## Weight-generation gate

No private EIPM-v1 gradient update or target-scale personality weight generation may begin until:

1. the expanded E-INF/A-SYN substrate is complete enough for the intended coverage target;
2. the training corpus and architecture are frozen;
3. the owner is explicitly told that the final pre-weight review is ready; and
4. the owner explicitly approves proceeding.

This gate applies whether the target lands near 400M parameters or at another size selected by the final architecture study.

## Third-party model boundary

Third-party models may be used as temporary development tools for hypothesis generation, transformation, critique, or research when their terms permit the intended use. Their weights are not part of the A.L.I.C.E. entity. A.L.I.C.E.-native learned models must remain independently loadable without requiring a third-party foundation checkpoint as part of the personality model artifact.
