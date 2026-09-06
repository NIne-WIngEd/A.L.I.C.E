# A.L.I.C.E. MC10D Public Judge Qualification Rebase v1.8.6

v1.8.6 is a governed successor to v1.8.5 after the GLM Q04 diagnostic proved that the thinking-enabled profile consumed the entire generation allowance at 6,144, 8,192, and 12,288 tokens while producing no final content.

## Narrow runtime-profile amendment

Only GLM's qualified runtime profile changes:

- from `{"id":"thinking_on","think":true}`
- to `{"id":"thinking_off","think":false}`

Unchanged:

- GLM model tag and digest
- Gemma profile and preserved qualification evidence
- Mistral/Granite bindings
- all 16 public tasks
- all gold labels
- decision-centric scoring
- Q01/Q03 and hard anchors
- seed formula / temperature / num_ctx
- `num_predict` ladder 2048/4096/6144
- current 287-candidate pool
- MC8 firewall
- no A-SYN acceptance/promotion/training

The profile amendment is ratified before any fresh GLM semantic result is observed. Ratification alone grants no qualification authority.

A fresh full 16-task GLM family then runs under `think=false`. If GLM passes, the same launcher performs four-judge binding with GLM's amended profile recorded in the binding and performs the unchanged 287-candidate refreeze. Pointwise screening still does not start.

Profile amendment receipt SHA-256: `5A02C0A7035CC5AC0E1638CB04E42A238A5C61A97E1AE3A9C0CDEA4E6B231AC9`
Effective policy SHA-256: `45C66A55D5A04162B6373B7FAA7F025CC523945E64E98C1301240E93D91F7F6A`
