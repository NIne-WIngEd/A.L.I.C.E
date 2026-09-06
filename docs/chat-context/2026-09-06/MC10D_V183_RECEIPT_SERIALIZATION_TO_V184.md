# MC10D v1.8.3 Receipt Serialization Stop → v1.8.4 — 2026-09-06

Canonical main remains frozen at `0abaed85873c3f8de04765847eb7700b0e20433f`.

## v1.8.3 deterministic stop

v1.8.3 stopped before GLM execution at the decision-scoring ratification receipt hash gate.

Root cause: the receipt content was constructed as a JSON string and then passed to the canonical JSON writer, causing double serialization.

Expected canonical-object receipt SHA256:
`60BDBA8C405767075F386420761802FDF710702218857F0FFC137F3041A47661`

Actual failed double-serialized receipt SHA256:
`5F9D25064D05C6951AF552CE89FDBD0CBD1B97F1EC3B95F8588A1BDAF9789BD1`

No GLM GPU run occurred. No pointwise screen, A-SYN acceptance/promotion, training, Stage G closure, or Phase 2 replacement occurred.

## v1.8.4

Package:
`ALICE_MC10D_PUBLIC_JUDGE_QUALIFICATION_REBASE_v1.8.4.zip`

SHA256:
`B1E16B569219A21D1BF9729B535B44485B08F9F98E4D4457480C71CDCD0B0FFB`

Expected correctly serialized scoring-ratification receipt SHA256:
`FD1228D88F7F7B7498F70F9DB7595A042E94D8293F19D156C5A970C06B775BF5`

v1.8.4:
- preserves/reconciles the failed v1.8.3 staging evidence;
- writes the receipt as a JSON object;
- explicitly selftests the canonical object hash and rejects the double-serialized form;
- preserves exact v1.8.2 Gemma rows with no Gemma rerun;
- runs only a fresh full 16-task GLM family;
- keeps decision-centric scoring unchanged: verdict >=14/16, critical decision >=6/7, hard anchors Q02/Q04/Q05/Q11/Q13, Q01 PASS, Q03 HOLD;
- keeps current pool 287, MC8 sealed, pointwise false, A-SYN acceptance/promotion false, training false;
- if successful, performs the existing four-judge bind and 287-candidate refreeze.

## Controlled synthesis breadth v1.0.2

Package:
`ALICE_MC10D_CONTROLLED_SYNTHESIS_BREADTH_AMENDMENT_v1.0.2.zip`

SHA256:
`998E83B402C86E5EACD17F65F5476DA04FC84D5F082AA6317ADD8A470A68C61D`

Same breadth semantics as v1.0.1. Prerequisite updated to successful v1.8.4 plus exact v1.8.4 pointwise-ready bundle hash and scoring receipt.
