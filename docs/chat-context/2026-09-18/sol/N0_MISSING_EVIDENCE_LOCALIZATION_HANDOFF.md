# N0 Missing-Evidence Localization Handoff

**Date:** 2026-09-18  
**Status:** final frozen challenge is an immutable valid FAIL; one fresh non-challenge causal-path localization run is ready; no rerun, threshold change, training, promotion, scaling, or N0 completion authorized

## Current experiment frontier

`alice-eipm-v1-missing-evidence-localization @ da6fbe80a00f6463d28d3ce29300e1c2943003ef`

Stable build remains:

`alice-eipm-v1-build @ 021c5021a98104b35f9c8e94b19e48d21f25f132`

The localization frontier is 44 commits ahead and 0 behind the stable build. It is experimental, not canonical promotion.

## Final frozen challenge evidence

Magnolia job `575794` completed successfully as an execution with exit code `0:0`.

Execution revision:

`ae1be5f811ceea772d138b35191278b8a304d023`

Model conclusion:

`FAIL_STOP_AND_LOCALIZE_WITHOUT_AUTOMATIC_HOTFIX`

Only failed frozen gate:

`counterfactual_family_min_drop`

Failing family:

`missing_structured_evidence`

Observed family mean target drop:

`-0.009834006428718567`

Frozen minimum:

`0.005`

All other frozen gate checks passed.

Result SHA-256:

`a2b8d62a28daf546d337eb36cef711eb29821caa3a292b2439c8a0c2cdfaf09d`

Execution receipt SHA-256:

`31a20c0f8281f15d060e85df9f457bbaadc1cff12ea6b995dca6d319a0529c0e`

Do not rerun this challenge. Do not alter its frozen thresholds.

## Correct failure semantics

The family name is easy to misread.

For `missing_structured_evidence`:

- structured view is already unavailable;
- semantic prose deliberately does not identify which record is current;
- evidence/relation view is uniquely necessary;
- a directed `corrects` relation identifies the current value;
- the failed counterfactual removes evidence view 2 before parent-cache construction and before cross-context fusion.

Therefore the localization target is:

`evidence relation -> graph read/pooling -> fusion -> adaptive latent pool`

Do not open a structured-encoder repair from the family name.

## Endpoint repair remains useful

The selected step-200 repaired evidence graph is:

`relation-endpoint-repair-v0.2/training/step-00000200/evidence_graph_dual_endpoint.safetensors`

SHA-256:

`3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f`

It achieved perfect fresh endpoint-role discrimination and later improved two frozen downstream arbitration metrics with no harm.

The final challenge still reproduced the historical `missing_structured_evidence` absolute target-cosine failure. Therefore the endpoint repair has real downstream value but does not resolve this separate path.

The graph remains unpromoted while localization is open.

## Current latent candidate

Step 360:

`503d4064df6258d3a1bc0edeae17888cbeb4fb67e07ad7e27042f785ed092ba4`

Weights unchanged.

Not ratified.

No private identity gradient.

## Fresh localization

Authoritative branch state:

`configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.10.json`

Builder:

`scripts/eipm/n0/build_n0_v02_missing_evidence_fresh_localization_v0_1.py`

Diagnostic:

`scripts/eipm/n0/diagnose_n0_v02_missing_evidence_path_v0_1.py`

Runner:

`scripts/eipm/n0/run_n0_v02_missing_evidence_path_localization_v0_1.sh`

Magnolia job wrapper:

`scripts/eipm/n0/magnolia_n0_v02_missing_evidence_path_localization_v0_1.sbatch`

### Fresh data contract

32 rows / 16 relation-flip pairs.

No frozen challenge rows are reused.

No prior diagnostic rows are reused.

Within each pair:

- raw text identical;
- query identical;
- field text and values identical;
- availability/reliability identical;
- only `corrects` relation direction changes;
- correct answer changes with the relation source endpoint.

Structured is unavailable. Semantic prose is non-answering. Evidence is uniquely answer-authoritative.

### Diagnostic trace

The same fresh pairs are traced through:

1. raw semantic negative control;
2. graph field selection;
3. raw target-field value geometry;
4. transformed graph target-field value geometry;
5. graph pooled state;
6. fusion evidence contextualization;
7. fusion semantic contextualization / transfer;
8. latent best-slot value readout;
9. latent pooled readout;
10. proper pre-fusion evidence ablation.

The diagnostic also measures probe-independent pair separation caused solely by the relation flip at graph, fusion, and latent stages, including a permutation-invariant latent-slot-set distance.

This prevents weak full-sentence semantic cosine from being mistaken for a module-capacity failure.

## Pre-GPU checks

Workflow:

`N0 Missing Evidence Localization Contract Check`

Latest CI run:

`35378274405`

Result:

`SUCCESS`

Passed:

- syntax;
- fresh-data isolation;
- exact final-failure trigger binding;
- causal instrumentation contract;
- runner safety;
- no training flags;
- no Git mutation;
- udocker environment forwarding.

## Exact next action

Run exactly one fresh Magnolia P100 localization job from:

`da6fbe80a00f6463d28d3ce29300e1c2943003ef`

Do not make a model-changing patch before its result.

## Decision discipline after localization

If graph selection itself fails:
- localize within evidence relation read; one causal repair only.

If graph selection is correct but graph pooled state loses relation identity:
- localize graph pooling/readout before touching fusion or latent.

If graph signal reaches fusion but fusion loses it:
- localize fusion preservation/routing.

If graph and fusion preserve it but latent does not:
- localize latent conditioning/readout; only then consider whether capacity is implicated.

If relation-sensitive internal state survives graph -> fusion -> latent and removing evidence destroys that separation, while the historical absolute-cosine gate remains negative:
- preserve job 575794 as an immutable FAIL;
- treat its scalar as insufficient as a sole capability signal;
- do not retrain the model merely to satisfy that misleading scalar;
- move forward through an explicit ratification/readiness decision based on the valid causal evidence.

No automatic hotfix chain in any case.

## Standing rules

- N0 is the full production personality-model foundation, not a reduced pilot.
- no current parameter count or shape is a permanent capability ceiling;
- efficiency may not silently remove required capability;
- scale only after a valid localized capacity/expressivity bottleneck;
- one causal change per failed model gate;
- infrastructure failure is not model evidence;
- failed results and receipts remain immutable;
- frozen challenge rows and fresh diagnostic rows are not training data;
- no private Elaina identity gradient at N0;
- no production promotion while the current localization question is unresolved.
