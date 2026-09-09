# MC10D v2.0.5 — Astra fourth-judge result received

Date: 2026-09-09

Canonical main remains frozen at `0abaed85873c3f8de04765847eb7700b0e20433f`.

## Astra substitute fourth-judge result

Input bundle SHA-256:
`E1EC8CCA72DF2FF73CF3565E74D8F6C82FE08B4BF73BA6F6EBC64CD111286CF9`

Astra result SHA-256:
`4018FFB88FACA6CFB755350B96D5C2D8927D62924BCC540D5CDD4A5429CF9DC9`

Adjudicator: `astra_work`

Rows: 26
- exact clean PASS: 8
- non-clean/invalid: 18
- PASS verdicts: 8
- HOLD verdicts: 6
- REJECT verdicts: 12
- valid rows: 23
- invalid rows: 3

## Owner-ratified hybrid pointwise geometry

Pre-Astra:
- primary 3/3 clean PASS -> direct pass: 200
- primary 2/3 + durable clean Qwen -> pass: 8
- known fail before Astra: 53
- manual fourth-judge cases: 26

Astra:
- clean pass: 8
- non-clean/invalid fail: 18

Expected final pointwise geometry:
- SIMULATION_ELIGIBLE: 216
- POINTWISE_REJECT_OWNER_HYBRID: 71
- total: 287

The 12 primary-3/3 candidates with observed Qwen dissent remain direct-pass under the owner's explicit 3/3-primary rule. Two of those had Qwen hard-block-style dissent; those flags must remain visible as challenge metadata downstream.

## Current boundary

A v2.0.5 deterministic local successor is prepared to independently revalidate the exact manual bundle and Astra result, recompute the full 287-candidate geometry, and create an immutable hybrid pointwise freeze.

Before v2.0.5 executes:
- pointwise hybrid freeze: absent
- MC8: sealed
- full simulation/falsification: not started
- A-SYN acceptance/promotion: zero/false
- training: false

## Public privacy boundary

The repository is public. Raw private candidate payloads, E0 spans, Astra rationales, and raw judge outputs are not committed. Only hashes, counts, authority state, and sanitized execution metadata belong on public branches.
