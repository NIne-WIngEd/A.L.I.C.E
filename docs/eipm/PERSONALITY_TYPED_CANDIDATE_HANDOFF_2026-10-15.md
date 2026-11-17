# Typed personality candidates and governed delivery

## Why this can proceed during cumulative learning

Job 577714 supplies the next actual broader N1/N2 result. Its successor calibration labels and learned-quality decisions require that result. The existing broader curriculum separately identifies a producer/consumer gap: `CandidateEntry.text` is both the semantic description presented to the model and the literal response returned by `DecisionConsumer`. Identical words with different proposed delivery therefore need distinct semantic inputs, and an interpretation or action proposal must not automatically become spoken text.

The role contract explicitly includes interpretations, actions and responses. The voice contract requires jointly coherent wording and delivery under replaceable speech renderers. These contracts are at `origin/alice-eipm-v1-build@021c5021a98104b35f9c8e94b19e48d21f25f132`, `docs/eipm/EIPM_ROLE_AND_OBJECTIVE_CONTRACT_v1.0.md` and `docs/eipm/EIPM_VOICE_FIRST_INTERACTION_CONTRACT_v1.0.md`. Their historical custom-N0 prerequisite suites remain superseded by the owner's frozen-Gemma v1 decision.

## Implemented connection

`scripts/eipm/gemma_n0/typed_candidates.py` adds a compiler and consumer adapter outside the pinned scientific core:

1. Supply the original full document and an exact ordered tuple of `TypedCandidate` entries. Kinds are `utterance`, `interpretation` and `action`. An utterance contains its original words and explicit qualitative delivery or unavailable delivery; an interpretation contains a proposed proposition; an action contains operation, description and arguments.
2. Compile the complete kind/payload into native candidate semantic text before `FrozenFrameProducer.produce`. Preserve document fields, sources, concepts, labels, graph, candidate IDs and both admission masks. Opaque candidate IDs and flags are not embedded into the semantic description. Targets remain separately reviewed material, never compiler-generated preferences.
3. Bind the original document, compiled document and complete ordered payload. New candidate meanings require fresh features and applicable new teaching admission; no sidecar may retrofit an old bank. This does not require resetting compatible model weights.
4. Run the actual governed native consumer. Successor consumers use the cumulative wrapper, preserving valid teaching scope, historical N3 inapplicability and successor lineage. The complete original governed receipt stays unchanged inside the typed handoff.
5. Only an actual single `SELECTED` candidate can produce an utterance intent or typed proposal. Ties, untaught decisions and governing conflicts keep their original unresolved status; acceptable alternatives are not reranked.
6. A replaceable speech callback receives original words, qualitative candidate delivery and learned native numeric controls as distinct fields, plus the full governing packet and scope. Changed reported words/delivery remain unresolved. Callback input is copied; typed bindings and model/checkpoint evidence are rechecked afterward.

Qualitative emphasis spans use half-open Unicode code-point indices into the original speech text, with matching span text. Pause locations use Unicode code-point offsets into that same text. They never refer to the compiled semantic descriptor. No millisecond, pitch or confidence values are invented from qualitative language. Native learned controls preserve their declared meaning, units, values and confidence. Compatibility between qualitative and numeric intent remains `UNREVIEWED`; this adapter does not implement a semantic contradiction judge.

Interpretation/action results are `PROPOSAL_ONLY`, with no speech call, execution or Experience claim. The existing conversation orchestrator owns committed conversation responses; Mission routing/state and capability policies own their respective transitions and permissions; Result Capsules and the Experience ledger own actual outcome records. This adapter does not replace those components or claim their missing physical connections are finished.

## Research and transfer limits

The [W3C SSML 1.1 specification](https://www.w3.org/TR/speech-synthesis/#S3.2) separates the spoken text from emphasis, boundary and prosody instructions and assigns acoustic interpretation to the synthesis processor. This supports explicit original-text locations and portable qualitative intent. This change neither generates SSML nor treats markup/echoed fields as verified speech; individual renderers still need applicable mapping and acoustic review.

[Anthropic's tool-interface engineering guidance](https://www.anthropic.com/engineering/building-effective-agents#appendix-2-prompt-engineering-your-tools) supports explicit operation/argument meaning and testing the actual consuming path. Its agent-as-controller design is not adopted: ALICE's native judgment remains governing and downstream action owners retain authority. No new framework, backbone or external identity teacher is introduced.

## Ordered diagnosis retained during construction

1. **Validate evidence:** nine initial tests failed on the explicit absent-adapter assertion, after valid collection. These establish a missing connection, not a learned-model failure. The first broad invocation also collected eleven imported existing consumer tests; those passed. The fixture import was changed to a module import to avoid unintended repeat collection. A separate focused speech-location check passed its valid Unicode case and exposed a `TypeError` for a null emphasis list.
2. **Claim and goal:** complete typed meaning must reach the actual producer and govern the resulting handoff. Input-shape refusal must precede iteration. Public fixtures establish mechanics, not Alice identity or Gemma semantics.
3. **Architecture:** typed payload → native document text → unchanged feature producer/model/checkpoint consumer → complete governed packet → typed handoff → optional speech callback.
4. **First departure:** the typed connection was absent. In its initial implementation, optional location-array validation occurred after iterating the array; JSON null therefore reached iteration instead of the declared contract refusal.
5. **Responsibility:** the adapter preserves meaning, original words, current authority masks, all judgment dimensions and unknowns. It cannot author teaching gold, rerank a conflict, execute an action or manufacture lived outcomes.
6. **Coherent correction:** add the pre-feature compiler and exact handoff; validate optional location arrays before iterating them. Keep valid text/span semantics and the original invalid input unchanged.
7. **Research:** the role/voice contracts, actual native/cumulative paths, SSML boundary semantics and tool-interface guidance above support the connection. [Python 3.11 JSON conversions](https://docs.python.org/3.11/library/json.html#json.JSONDecoder) distinguish an array/list from null/None; preserve that distinction rather than coercing null to an empty list.
8. **Verification:** use actual public fixture feature production and native optimization, original words with changed delivery, nonspoken proposals, altered-payload/old-feature refusal, ties, callback changes and actual cumulative successor scope. Original missing-feature and null-array failures remain retained alongside passing evidence.

Fresh review additionally found Python equality treating renderer-reported Boolean `true` as equal to an integer speech offset `1`. The actual optimized consumer regression reproduced a false equivalent-delivery result. The input contract already refuses Boolean offsets, so this was a report-comparison defect at the callback boundary, not a model error. The same JSON conversion research above confirms distinct Boolean/numeric representations. Canonical delivery comparison now preserves those types while retaining the altered renderer report and marking the handoff unresolved; no semantic or acoustic verdict is inferred.

Final scoped verification passed **12 tests and 13 subtests in 169.02 seconds**, including an actual optimized native consumer and an actual cumulative successor. The successor retains its current teaching scope and historical N3 calibration inapplicability through the complete callback packet. Fresh independent source review confirmed its reported comparison defect was corrected and found no further blocking findings. An earlier 11-test run took 1,861.59 seconds; its evidence remains retained, and this shorter measured run does not establish the cause of that timing anomaly. Production custody checks and CI limits were not weakened.

## Current limits and next use

The exported adapter is construction capability, not private training or qualification. Use it when authoring the next grounded typed interpretation/action/voice examples; compile before materialization, preserve compatible weights and include retained teaching. Use actual successor outputs for N3 reviews. Then connect the selected proposal to its existing conversation/action owner and bind genuine results through the existing Experience path. Full semantics, intent coherence, action execution, acoustic realization, N0 qualification and identity fidelity still require their own actual evidence.
