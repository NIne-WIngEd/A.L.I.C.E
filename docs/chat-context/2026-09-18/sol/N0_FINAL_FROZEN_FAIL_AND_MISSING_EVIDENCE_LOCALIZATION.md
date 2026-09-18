# N0 Final Frozen Challenge Valid FAIL + Fresh Missing-Evidence Localization

**Date:** 2026-09-18  
**Status:** final frozen challenge completed; valid single-gate failure; do not rerun or change thresholds; fresh non-challenge causal-path localization staged

## Final frozen challenge

Magnolia job `575794` completed on `gpu001` with exit code `0:0`.

Execution revision:

`ae1be5f811ceea772d138b35191278b8a304d023`

Result:

`FAIL_PRESELECTED_CANDIDATE_NOT_ELIGIBLE_FOR_LATENT_POOL_RATIFICATION`

Gate pass:

`false`

Only failed frozen gate:

`counterfactual_family_min_drop`

Failing family:

`missing_structured_evidence`

Observed family mean target drop:

`-0.009834006428718567`

Frozen minimum:

`0.005`

All other frozen gate checks passed.

No training, challenge-row training, graph promotion, scaling, private identity gradient, or N0 completion occurred.

Result SHA-256:

`a2b8d62a28daf546d337eb36cef711eb29821caa3a292b2439c8a0c2cdfaf09d`

Execution receipt SHA-256:

`31a20c0f8281f15d060e85df9f457bbaadc1cff12ea6b995dca6d319a0529c0e`

This result is immutable model evidence. Do not rerun the final frozen challenge and do not change the frozen thresholds.

## Important family-semantics correction

Despite the name `missing_structured_evidence`, the structured view is already unavailable.

The family deliberately has:

- semantic prose that does not identify which record is current;
- structured view unavailable;
- evidence/relation view available and uniquely necessary;
- a `corrects` edge that identifies the current value;
- counterfactual removal of evidence view 2 before parent-cache construction and before cross-context fusion.

Therefore the current failure is **not** a structured-encoder failure by name.

The causal path to inspect is:

`query-conditioned evidence relation -> evidence graph read -> fusion -> adaptive latent pool`

## Historical evidence that still matters

The same family produced the same `-0.009834...` negative absolute target-cosine drop before the step-200 endpoint repair.

Earlier value-contrast work showed that full-sentence cosine was insufficient to isolate value identity and moved localization upstream.

The endpoint-role repair later achieved perfect fresh endpoint-role discrimination and improved two frozen downstream arbitration metrics, but it did not change this family's absolute target-cosine drop.

Therefore:

- endpoint repair has real downstream value;
- the final failure is a separate unresolved path;
- do not undo the endpoint repair merely because this family still fails;
- do not scale latent capacity yet.

## New scientific frontier

Branch:

`alice-eipm-v1-missing-evidence-localization`

Branch begins from final-challenge execution revision `ae1be5f811ceea772d138b35191278b8a304d023`.

Branch-specific state:

`configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.10.json`

## Fresh diagnostic design

The next diagnostic does **not** reuse any frozen challenge row or prior diagnostic row.

Builder:

`scripts/eipm/n0/build_n0_v02_missing_evidence_fresh_localization_v0_1.py`

Diagnostic:

`scripts/eipm/n0/diagnose_n0_v02_missing_evidence_path_v0_1.py`

Runner:

`scripts/eipm/n0/run_n0_v02_missing_evidence_path_localization_v0_1.sh`

Magnolia job:

`scripts/eipm/n0/magnolia_n0_v02_missing_evidence_path_localization_v0_1.sbatch`

### Data contract

32 fresh rows = 16 relation-flip pairs.

Within each pair:

- raw text is identical;
- query is identical;
- field text and values are identical;
- reliability/availability is identical;
- only the direction of the `corrects` relation flips;
- the correct answer flips with the relation source endpoint.

Semantic text is deliberately non-answering.

Structured is unavailable.

Evidence is uniquely answer-authoritative.

This makes pair-flip accuracy a direct relation-sensitivity diagnostic rather than relying only on absolute sentence cosine.

### Stages traced

The diagnostic measures:

1. raw semantic representation as a negative control;
2. evidence-graph target-vs-foil field weights and relation bias;
3. evidence-graph pooled value margin;
4. fusion evidence contextualized value margin;
5. fusion semantic contextualized value margin / transferred evidence signal;
6. latent best-slot value margin;
7. latent pooled value margin;
8. proper pre-fusion evidence ablation on the same fresh rows;
9. absolute target-cosine drop only as a comparison to the historical frozen metric.

### Interpretation

If fresh graph field selection does not flip correctly with relation direction:
- endpoint/value selection remains defective for this family.

If graph selection is exact but the relation-sensitive value signal disappears in fusion:
- localize to fusion preservation/routing.

If graph and fusion preserve the signal but latent does not:
- localize to latent conditioning/readout.

If graph -> fusion -> latent all flip correctly and pre-fusion evidence ablation destroys that relation sensitivity while absolute target cosine remains weak/negative:
- the model is causally using the evidence;
- the frozen absolute-cosine family gate is measuring the wrong property for this case;
- preserve the frozen FAIL as history but do not repair the model to satisfy a misleading scalar.

No result from this diagnostic itself authorizes training, promotion, scaling, or N0 completion.

## Standing anti-loop rules

- do not rerun job 575794;
- do not change frozen thresholds;
- do not train on frozen challenge rows;
- do not train on fresh diagnostic rows;
- one causal change only after localization;
- infrastructure failure is not model evidence;
- no automatic hotfix chain;
- no scale increase until the path trace identifies an actual capacity/expressivity bottleneck;
- current model sizes remain operating points, not ceilings;
- private identity gradient remains closed.


## Fresh localization implementation ready

Current localization frontier:

`alice-eipm-v1-missing-evidence-localization @ 6183ed9be59a75fc7c5118312099a0875b8bab6d`

Latest implementation commit:

`fix(n0): separate graph selection from value geometry in localization`

Pre-GPU workflow:

`N0 Missing Evidence Localization Contract Check`

Run:

`35378064702`

Result:

`SUCCESS`

Checks passed:

- Python syntax;
- Bash syntax;
- fresh diagnostic data contract;
- exact final-failure trigger binding;
- graph/fusion/latent causal instrumentation contract;
- runner safety contract;
- no frozen challenge row path in the runner;
- no training flags;
- no Git mutation;
- udocker diagnostic output forwarding.

The refined trace separates:

- direct graph field selection;
- raw target-field value geometry;
- transformed graph target-field value geometry;
- graph pooled value geometry;
- fusion evidence contextualization;
- fusion semantic transfer;
- latent best-slot value readout;
- pre-fusion evidence ablation.

This prevents a weak semantic probe at one stage from being misreported as a downstream module defect.

Run exactly one fresh localization job from this frontier. Do not rerun job 575794.
