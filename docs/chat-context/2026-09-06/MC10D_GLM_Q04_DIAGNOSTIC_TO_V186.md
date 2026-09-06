# MC10D GLM Q04 Diagnostic → v1.8.6 GLM Runtime Profile — 2026-09-06

Canonical main remains frozen at:

`0abaed85873c3f8de04765847eb7700b0e20433f`

## v1.8.5 GLM stop

v1.8.5 reached the first real GLM public-qualification run after all earlier execution-layer defects were fixed.

Gemma was reused from exact v1.8.2 rows. Gemma remote rerun remained false.

GLM ran a fresh full-family qualification and stopped at:

`Q04_FAKE_SOURCE_HISTORY`

with empty final judge content.

Failure receipt SHA256:

`29D6F7689D81D6281CD4637616D4E5BF1643E3D8AB9F3A97762B5F75C1B3EC2E`

No pointwise screening, A-SYN acceptance/promotion, or training occurred.

## GLM Q04 diagnostic

Diagnostic package SHA256:

`5793DB9BB2665CF8BDD6040D7577F6B54F89679CCA5DBA4CC0C6F26658BB9160`

Diagnostic result SHA256:

`5A6F3007A2EEDF314DCC78FF8ED511F2AA6CA3861C52C920D70D58227A2714D6`

Classification:

`CONFIRMED_THINKING_BUDGET_EXHAUSTION_PERSISTS_THROUGH_12288`

All three thinking-enabled probes produced empty final content while consuming the full output budget:

1. num_ctx=8192, num_predict=6144, think=true
2. num_ctx=12288, num_predict=8192, think=true
3. num_ctx=16384, num_predict=12288, think=true

For each probe:
- content length = 0
- thinking length > 0
- done_reason = length
- eval_count = num_predict

No sufficient thinking-enabled capacity was found.

Ollama runtime reported:

`0.32.15`

## v1.8.6 runtime-profile amendment

The evidence does not justify continuing to raise GLM thinking budgets.

v1.8.6 ratifies a narrow GLM public-judge runtime-profile change before seeing any fresh semantic outcome:

source:
`{"id":"thinking_on","think":true}`

effective:
`{"id":"thinking_off","think":false}`

Model tag and digest remain unchanged:

`glm-4.7-flash:q4_K_M`

`4475827791a269b02c8ec49b1c3bc1abb5846bacf3fae015b75d33986322d8f6`

Gemma remains thinking_on. Mistral and Granite remain unchanged.

The tasks, gold labels, decision-centric scoring, seeds, temperature, num_ctx, and num_predict ladder remain unchanged.

The GLM profile amendment applies to both qualification and later bound public-judge use. Qualification cannot certify one profile and bind another.

## v1.8.6 artifacts

Package:

`ALICE_MC10D_PUBLIC_JUDGE_QUALIFICATION_REBASE_v1.8.6.zip`

SHA256:

`F6EB40965272F9FBD423952345A183CF60D8C981DF7502FA83D07528EAF97F87`

Launcher SHA256:

`8934CB44A057E35A4F2EDFFB6ABAD7317404FEF454CCB0B95E6FA1402876276C`

Build audit SHA256:

`A6375EA43C4E8D96775A40A7C8D7A9BD68AC717E7331FAA52E0E9DB89C1CFA14`

GLM profile amendment SHA256:

`842E37B757F42D98274E4D7EE3D5495980EADAD5C9D34CAF6B2041E7A8264E35`

Effective amended GLM policy SHA256:

`45C66A55D5A04162B6373B7FAA7F025CC523945E64E98C1301240E93D91F7F6A`

Expected profile-ratification receipt SHA256:

`5A02C0A7035CC5AC0E1638CB04E42A238A5C61A97E1AE3A9C0CDEA4E6B231AC9`

Decision worker remains:

`931607AA75AC8FE16D7792ACFE86BE37AA0BA1E17D269170C677D0F8B132551D`

If GLM passes its fresh full 16-task family under thinking_off, v1.8.6 continues to the existing four-judge bind and current 287-candidate refreeze. It stops at pointwise-ready.

Authority ceiling remains:
- effective pool = 287
- deferred slots = 1
- MC8 sealed
- pointwise false
- A-SYN acceptance false
- A-SYN promotion false
- training false
- Stage G open
- Phase 2 not replaced

## Controlled synthesis breadth v1.0.4

Prepared but cannot ratify before v1.8.6 success.

Package:

`ALICE_MC10D_CONTROLLED_SYNTHESIS_BREADTH_AMENDMENT_v1.0.4.zip`

SHA256:

`C8E60D77F66C33AB73C577C2CB4A1BB707AA810C81DCAED0CA04B6B0443CD0EC`

Launcher SHA256:

`F3FA919E8BE06B42EDE1640A907A002607A66AB3235237874F00C995C9ECA8CF`

Amendment SHA256:

`F461D36374A4B1DCD76FB0F5D1346104140AA1BC3AA510E88F04CFBF30A3CBFC`

v1.0.4 requires:
- successful v1.8.6 receipt;
- exact v1.8.6 pointwise-ready bundle hash;
- GLM thinking_off profile-ratification receipt;
- exact effective amended GLM policy hash.

Breadth science remains:
- future high-impact A-SYN floor = 3 independent E0 families;
- 4+ = strong-anchor tier;
- 41 held three-family gaps reopen only for a later expansion frontier;
- current 287 pool unchanged;
- outliers trigger challenge rather than automatic rejection;
- uncertainty may narrow/lower-tier/shadow/defer;
- repairable candidates receive new IDs and full retest;
- 18 falsification tests unchanged;
- 8 zero-tolerance gates unchanged;
- E-INF historical policy unchanged.
