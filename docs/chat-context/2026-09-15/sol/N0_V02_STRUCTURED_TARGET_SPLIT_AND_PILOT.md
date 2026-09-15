# N0 v0.2 structured-state target split and pilot

Date: 2026-09-15

The first structured-state preparation completed with 16 tests passing and 3,039 compiled examples from the governed 1,020-row public teacher bank: 2,279 train, 760 dev, 2,001 negative compatibility labels, 1,038 positive labels, 51 competencies, and no compiler-generated text.

Before any structured-state gradient, a modeling contradiction was found: v0.1 used the rationale text as both the semantic-alignment target and the compatibility target. Negative candidates would therefore be pulled toward the same rationale they were trained to reject.

This was fixed before gradient work:
- curriculum v0.2 represents semantic target as the exact context/candidate text pair;
- governed rationale is a separate compatibility target;
- compatibility BCE supports the train-set negative/positive ratio as positive-class weight;
- v0.1 compiled artifact remains preserved and is not activated.

One bounded public pilot is authorized:
- semantic parent: ratified repair step080, frozen completely;
- semantic vectors are precomputed once and cached;
- trainable branch: 1,656,064 parameters only;
- 240 steps, saves at 80/160/240;
- no private data or private gradient;
- selection: grouped preferred-candidate top1, then balanced compatibility, semantic alignment, then earlier checkpoint;
- no automatic production promotion.

If the pilot passes, ratify the smallest useful structured checkpoint and continue to graph/evidence alignment or fusion. Do not enter repeated structured tuning by default.
