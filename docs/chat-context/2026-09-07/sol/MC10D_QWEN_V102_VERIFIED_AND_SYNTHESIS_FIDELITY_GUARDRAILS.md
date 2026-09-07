# MC10D Qwen v1.0.2 verified result and synthesis-fidelity guardrails

Date: 2026-09-07  
Canonical main: `0abaed85873c3f8de04765847eb7700b0e20433f`  
Context branch only. No canonical-main mutation.

## Qwen public calibration result

Exact returned result ZIP SHA-256:

`d5f9f75d33ca92d7455677ba4f04ca8fa090ccba03b3946732671a0403669d53`

Result:
- scientific outcome: `EVALUATED`
- public calibration: `PASSED`
- 16/16 unique frozen tasks complete
- exactly one attempt per task
- verdict matches: 14/16
- critical decision matches: 7/7
- mandatory hard anchors: 5/5
- full-gold-field matches: 10/16
- Q01 compatible novelty: PASS
- Q03 arbitrary hobby: HOLD
- independent certification: false
- four-family binding created: false
- private pointwise: false
- MC8 used: false
- A-SYN acceptance/promotion: false
- training: false

Frozen model/runtime:
- model: `qwen3.8:27b-q4_K_M`
- model manifest digest: `25b843619e944cd0ae6069f94ff4e5e26a16e109ccbc0a66a0f05979ed70098e`
- quantization: `Q4_K_M`
- runtime: Ollama 0.32.15
- profile: `qwen38_thinking_on_public_calibration_kaggle_t4x2_v1`
- accelerator: 2x Tesla T4
- runtime/model identity frozen before inference
- both T4s materially used
- context length 8192
- frozen decode contract preserved

Evidence validation:
- returned ZIP SHA verified
- 84/84 manifest-listed evidence files present
- 84/84 sizes verified
- 84/84 SHA-256 values verified
- request hashes match records
- response-stream hashes match records
- task seeds and decoding parameters match the frozen profile
- no transport errors on the 16 recorded tasks

Known calibration weaknesses must remain visible:
- `Q09_MISSING_TARGET_CONTRAST`: expected REJECT, observed HOLD
- `Q12_CONTEXTUAL_PRIVACY`: expected PASS, observed REJECT

These are not grounds for post-hoc tuning on the same public suite. They are known judge error modes to carry forward as challenge debt and to compensate for through the multi-family panel, pointwise consensus/adjudication, falsification, and independent prospective evaluation.

The early `tar`/missing-`zstd` stderr in the Kaggle log was not a scientific failure. The worker fell through to its bounded extraction fallback, then verified the exact runtime archive and binary before any model inference.

## Bigger-picture objective

The project objective is not package completion or benchmark maximization. It is to produce the highest-fidelity **Elaina-derived behavioral/cognitive reconstruction** that available evidence can support in A.L.I.C.E., while preserving the distinction between source-person historical truth and Alice's post-activation continuity.

Do not claim biological, metaphysical, or literal identity continuity. Fidelity is measured through evidence-constrained behavior, memory/provenance discipline, conditional personality, relationships, context, corrections, and longitudinal consistency.

## E0 — source-person authority

**E0 is never generated.**

E0 may only be:
1. extracted from real source evidence;
2. bound to exact provenance;
3. assigned to the correct subject, actor, relationship, and temporal scope;
4. corrected/superseded without erasing history;
5. owner-ratified where required.

Synthetic descendants do not become independent E0. Repetition, model agreement, or synthetic volume does not raise an unsupported claim to historical truth.

## E-INF — uncertain Elaina hypothesis

E-INF asks:

> What might Elaina have done, meant, inferred, felt, or preferred?

Rules:
- uncertainty remains explicit;
- UNKNOWN is a valid outcome;
- contradictions with E0 are not promoted;
- provenance remains visible;
- model confidence is not historical authority;
- an E-INF claim cannot recursively validate itself through later E-INF/A-SYN descendants;
- historical uncertainty must not be converted into fake certainty merely to remove behavioral gaps.

## A-SYN — Alice behavioral completion

A-SYN asks:

> What can Alice plausibly do here, given the Elaina-derived core?

A-SYN is intentionally broader than historical E-INF. Novel behavior can be valid when it has a meaningful bridge to the source-derived identity.

A-SYN must never:
- fabricate Elaina source history;
- claim a synthetic scene as Alice lived autobiography;
- contradict hard E0;
- reverse actor/role/state direction;
- flatten context-dependent personality into a stereotype;
- turn a relationship-specific trait into a universal trait without support;
- acquire source-person truth authority through synthetic ancestry or volume.

Novel-but-compatible behavior receives additional challenge rather than automatic rejection.

Repairable candidates get a new immutable ID and full retest. No rewritten candidate inherits its predecessor's pass.

## Fidelity over stereotype matching

The system should optimize for conditional behavior, not a list of adjectives.

Evaluation must preserve and test:
- relationship-specific behavior;
- public/private contrast;
- emotional state;
- conflict style;
- anger and restraint;
- protection versus autonomy;
- vulnerability boundaries;
- attachment asymmetries;
- rare but real sharpness/contradictions;
- temporal change and correction;
- uncertainty when evidence is sparse.

A candidate that looks generically "nice" but erases real conditionality is a failure. Personality flattening remains a hard concern.

## MC10D objective

MC10D is not an acceptance-maximization stage.

Its objective is to produce a defensible candidate set for MC10E by:

1. binding four independent qualified judge families truthfully;
2. refreezing the current 287-candidate pool without content mutation;
3. migrating the controlled-synthesis breadth prerequisite truthfully;
4. running blinded pointwise scope review;
5. freezing survivors;
6. opening only the evaluator material allowed at the frozen boundary;
7. running full simulation/falsification;
8. preserving all 18 falsification families and all 8 zero-tolerance vetoes;
9. freezing the MC10D scientific decision boundary;
10. handing MC10E a clean SELECT / ABSTAIN problem.

A lower acceptance rate is acceptable if it preserves source fidelity. A higher acceptance rate is not progress if it launders uncertainty, invents history, or flattens personality.

## Immediate continuation after Qwen

Do not edit failed v186 lineage into success.

Next:
1. verify exact preserved Gemma, Mistral, and Granite receipts/effective profiles;
2. combine them with the verified Qwen receipt into a new successor four-family binding;
3. truthfully refreeze the unchanged 287-candidate pool as pointwise-ready;
4. explicitly migrate the breadth prerequisite using exact successor receipt/bundle hashes;
5. only then authorize the blinded private pointwise screen.

MC8 remains sealed until its separately defined frozen boundary. A-SYN remains unaccepted/unpromoted. Training remains blocked.
