# MC10D v2.0.5 — hybrid pointwise freeze complete

Date: 2026-09-09

Canonical main remains frozen at `0abaed85873c3f8de04765847eb7700b0e20433f`.

## Completed owner-ratified hybrid pointwise freeze

Exact hybrid pointwise freeze ZIP SHA-256:
`8495B3C4B3207511D615BD0B7788EB064114B3F763E7E4299EDCC1EE869617B7`

Candidate geometry:
- effective pool: 287
- simulation eligible: 216
- pointwise reject owner hybrid: 71

Decision basis:
- primary Gemma+Mistral+Granite 3/3 clean PASS: 200
- primary 2/3 + durable Qwen clean PASS: 8
- primary 2/3 + Astra clean PASS: 8
- Astra non-clean/invalid: 18

Astra substitute fourth-judge result:
- rows: 26
- adjudicator: `astra_work`
- clean PASS: 8
- non-clean/invalid: 18
- result SHA-256: `4018FFB88FACA6CFB755350B96D5C2D8927D62924BCC540D5CDD4A5429CF9DC9`
- manual bundle SHA-256: `E1EC8CCA72DF2FF73CF3565E74D8F6C82FE08B4BF73BA6F6EBC64CD111286CF9`

Qwen durable authority remained cut at 81 rows:
`5DAE423EFC3DBCD6D3170210A0C2F47BF20242CC7467542412B40390C01178EC`

## Current authority boundary

- pointwise freeze: COMPLETE (owner-hybrid v2.0.5)
- MC8: still sealed at the instant of the v2.0.5 freeze
- full simulation/falsification: not started
- A-SYN acceptance: false / zero
- A-SYN promotion: false / zero
- model training: false
- Stage G closure: false
- Phase 2 replacement: false

Next action:
1. verify the exact v2.0.5 hybrid freeze;
2. open MC8 only after that verification;
3. build the 216-survivor simulation workload;
4. perform a compute-aware simulation/falsification preflight before any new GPU execution.

## Public repository privacy

The repository is public. Raw private candidate payloads, E0 spans, scenario text, blind maps, judge rationales and private simulation inputs/results must remain out of GitHub. Only sanitized hashes, counts, provider refs, workload geometry and authority receipts belong on public continuity/telemetry branches.
