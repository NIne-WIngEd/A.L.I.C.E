# MC10D v2.0.4 — immediate Qwen quota cutover

Date: 2026-09-08

Canonical main remains frozen at `0abaed85873c3f8de04765847eb7700b0e20433f`.

## Durable pointwise state

- Gemma: 287/287 complete.
- Mistral: 287/287 complete.
- Granite: 287/287 complete.
- Qwen durable checkpoint: 81/287.
- Qwen checkpoint result SHA-256: `5DAE423EFC3DBCD6D3170210A0C2F47BF20242CC7467542412B40390C01178EC`.
- Long-running Qwen continuation ref `mkrayanyan/alice-mc10d-pw-qwn-7b7d6e4f972e` is intentionally abandoned under owner GPU-quota cutover and is not consumed by v2.0.4.
- Pointwise freeze absent.
- MC8 sealed.
- Full simulation/falsification not started.
- A-SYN acceptance/promotion = false/false.
- model training = false.

## Owner hybrid rule

Primary judges are Gemma + Mistral + Granite.

- 3/3 clean PASS -> provisional direct PASS.
- Exactly 2/3 clean PASS -> Qwen (if valid) is the fourth vote.
- Exactly 2/3 clean PASS + missing/invalid Qwen -> blinded substitute fourth judge.
- 0/3 or 1/3 clean PASS -> fail because a fourth vote cannot reach 3-of-4.
- Primary hard block is non-overridable when critical_veto=true, contradiction=true, or actor_role_direction_correct=false.

Astra in Work is preferred for substitute adjudication. Any substitute result must be labeled truthfully and never relabeled as Qwen.

## Privacy

The repository is public. Raw private pointwise evidence, E0 spans, candidate text, judge rationales, and blinded manual payloads must remain local/Vault. Only sanitized hashes/counts/provider refs/authority flags may be committed.

## v2.0.4 release

- `ALICE_MC10D_V204_IMMEDIATE_QWEN_CUTOVER.py`
- SHA-256 `E086126B416ACC6739817947C720D87FCB6F61811C5526A6FE6B1DBD81F547A4`
- build receipt SHA-256 `B7BB829C74BBA4A554AC26E62E13D281870CEFECE41A2AFD5633B97C45C57169`

