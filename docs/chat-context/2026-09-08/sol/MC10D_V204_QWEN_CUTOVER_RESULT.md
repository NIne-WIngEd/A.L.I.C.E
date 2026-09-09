# MC10D v2.0.4 — immediate Qwen cutover result

Date: 2026-09-08

Canonical main remains frozen at `0abaed85873c3f8de04765847eb7700b0e20433f`.

## Verified local cutover result

The networkless v2.0.4 cutover completed successfully from durable local evidence only.

Primary pointwise completion:
- Gemma: 287/287
- Mistral: 287/287
- Granite: 287/287

Qwen credited authority:
- durable rows: 81/287
- first checkpoint result SHA-256: `5DAE423EFC3DBCD6D3170210A0C2F47BF20242CC7467542412B40390C01178EC`
- later long-running Qwen continuation intentionally excluded from v2.0.4 authority

Hybrid audit counts:
- primary 3-clean: 200
- primary 2-clean: 80
- primary 1-clean: 7
- primary 0-clean: 0
- primary hard-block: 50
- Qwen observed: 81
- Qwen clean: 60
- Qwen non-clean valid: 21
- Qwen invalid: 0
- direct pass primary 3/3: 200
- pass 2/3 primary + Qwen clean: 8
- fail 2/3 primary + Qwen non-clean: 3
- manual fourth judge required: 26
- fail cannot reach 3/4: 7
- fail primary hard-block: 43
- primary 3/3 with Qwen dissent ignored under owner rule: 12
- primary 3/3 with Qwen hard-block ignored under owner rule: 2

Manual bundle:
- rows: 26
- SHA-256: `E1EC8CCA72DF2FF73CF3565E74D8F6C82FE08B4BF73BA6F6EBC64CD111286CF9`

Current boundary:
- pointwise freeze: absent
- MC8: sealed
- full simulation/falsification: not started
- A-SYN acceptance/promotion: zero/false
- model training: false

Next action:
- run the 26-row blinded substitute fourth-judge adjudication in Astra/Work if available;
- merge those results with the 208 already-passing candidates;
- create the owner-ratified hybrid pointwise freeze;
- only then open the post-pointwise MC8 boundary and proceed to simulation/falsification.

## Public GitHub privacy boundary

The repository is public. Do not commit raw private pointwise payloads, candidate text, E0 spans, rationales, or raw judge outputs. Only sanitized counts, hashes, provider refs and authority receipts belong on public continuity/telemetry branches.
