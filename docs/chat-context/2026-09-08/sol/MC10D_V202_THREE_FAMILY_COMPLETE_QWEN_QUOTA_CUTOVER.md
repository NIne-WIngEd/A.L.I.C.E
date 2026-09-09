# MC10D v2.0.2 — three-family completion and Qwen quota cutover

Date: 2026-09-08

Canonical main remains frozen at `0abaed85873c3f8de04765847eb7700b0e20433f`.

## Observed execution state

Owner-observed pointwise state:
- Gemma: 287/287 complete.
- Mistral: 287/287 complete.
- Granite: 287/287 complete.
- Qwen first private-pointwise checkpoint: 81/287, result SHA-256 `5DAE423EFC3DBCD6D3170210A0C2F47BF20242CC7467542412B40390C01178EC`.
- Qwen continuation currently running under exact ref `mkrayanyan/alice-mc10d-pw-qwn-7b7d6e4f972e`.
- Pointwise freeze is not yet created.
- MC8 remains sealed.
- Full simulation/falsification has not started.
- A-SYN acceptance/promotion remain zero/false.
- Model training remains false.

## Read-only short-circuit audit

At the 81-row Qwen checkpoint:
- 287 total candidates.
- 200 were clean PASS across Gemma+Mistral+Granite.
- 87 were already non-clean in at least one of those three families.
- Of the 200 three-family clean-PASS candidates, Qwen had covered 57.
- 45/57 were Qwen clean PASS.
- 12/57 were Qwen non-clean or invalid.
- 143 three-family clean-PASS candidates still lacked Qwen at that snapshot.
- 63 of the 206 full-order remaining Qwen calls were provably irrelevant to exact four-family simulation eligibility because one of the other three families had already made those candidates non-eligible.

## Owner direction under remaining GPU quota

The owner does not want another week-long defer cycle. A successor decision rule is being reviewed that avoids imputing missing evidence and avoids retrying already decided failures.

The architecture review must distinguish:
1. candidates already unanimously clean PASS across Gemma+Mistral+Granite;
2. candidates where a fourth judgment can still alter a majority/hybrid decision;
3. candidates already unable to meet the successor threshold regardless of a fourth vote;
4. missing-Qwen cases requiring a separately identified, blinded owner-side/Astra adjudicator if such substitution is ratified.

Any manual/Astra adjudication must:
- remain blinded to real candidate IDs, generator identity, and prior judge outcomes;
- use the exact frozen pointwise evidence/rubric fields;
- be recorded truthfully as a substitute adjudicator, never relabeled as Qwen;
- preserve all zero-tolerance provenance/identity/reality vetoes;
- never open MC8 before the successor pointwise decision freeze.

## GitHub privacy boundary

This repository is public. Raw private pointwise payloads, candidate evidence, rationales, personal E0 spans, and private judge outputs must NOT be pushed to GitHub. Only sanitized hashes, counts, provider refs, authority receipts, and non-sensitive execution summaries belong on public continuity/telemetry branches.

