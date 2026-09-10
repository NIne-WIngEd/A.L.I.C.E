# EIPM Astra Saturation Expansion Protocol

**Status:** Owner-directed execution protocol  
**Date:** 2026-09-10  
**Scope:** Pre-curation E-INF/A-SYN expansion for EIPM-v1

## No hard row ceiling

Astra generation must not be constrained by a fixed maximum number of E-INF or A-SYN proposals.

There is no quota and no preferred round number. Generation should continue while meaningful, non-duplicative personality coverage remains available.

The valid stopping criterion is behavioral coverage saturation, not row count.

Astra should stop only when repeated coverage passes show that remaining high-value personality regions are already represented, structurally redundant, or too unsupported to initialize responsibly. Remaining unsupported regions must be documented explicitly.

## Quality over single-response completion

The expansion must not be compressed, simplified, or stopped early merely to fit within one model response.

The generation job may span multiple Astra responses. Astra should maintain durable checkpoints between responses. Completed proposal partitions should be preserved and verified before continuation.

If a response boundary occurs before saturation, Astra should:

1. persist proposal files and coverage state;
2. persist a machine-readable progress checkpoint and hashes;
3. state that the job is incomplete;
4. identify the next uncovered behavioral region;
5. resume from the checkpoint after owner continuation rather than restarting or regenerating completed work.

The final proposal package should be produced only after genuine saturation and final validation.

## Coverage objective

The goal is the broadest useful starting behavioral foundation that can be responsibly derived from E0, E-INF, and provenance-preserving A-SYN.

UNKNOWN remains valid for historical uncertainty. It does not require Alice to remain behaviorally blank. A-SYN should provide practical starting priors for unresolved personality-related behavior whenever this can be done without fabricating historical Elaina truth.

Coverage should prefer breadth before density. New proposals should add a materially distinct context, boundary, trade-off, intensity regime, relationship condition, emotional regime, social setting, public/private distinction, value conflict, uncertainty condition, or behavioral branch rather than merely paraphrasing existing rules.

## Authority boundary

Astra has proposal-generation authority only.

It may not mutate E0, accept its own proposals, promote provenance, create Alice lived memory, authorize training, create model weights, or write generated candidates into the canonical EIPM substrate.

Independent Sol curation remains required after generation. Owner review remains required before any private EIPM gradient update or target-scale weight creation.
