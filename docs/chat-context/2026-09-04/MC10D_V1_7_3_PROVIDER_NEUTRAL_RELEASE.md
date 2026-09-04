# MC10D v1.7.3 Provider-Neutral Judge Continuation Release

**Status:** released for execution; not yet executed.

This file is a context/archive checkpoint on `alice-context`. It is not canonical scientific authority. Frozen bundles, manifests, validators, preserved receipts, and ratified repository authority win on conflict.

## Release artifacts

- package: `ALICE_MC10D_PROVIDER_NEUTRAL_JUDGE_CONTINUATION_v1.7.3.zip`
- package SHA256: `F7E815B759FDCD3F1C3D32664589A147C90486D9B074C8A20E9B7EB730D5C483`
- launcher: `Start-ALICEMC10DProviderNeutralJudgeContinuationV173.ps1`
- launcher SHA256: `3F8C3ADC7261DF84B550405CCDC427BA963B9E115C88500EDB978931E08045F7`
- build audit SHA256: `CF0463D04E8105DAF3BA8ABCE39F5F6A553CA25B0E6CCAADE9DD76F1C3ECD49A`

## Bound predecessor

- canonical main: `0abaed85873c3f8de04765847eb7700b0e20433f`
- MC10D freeze SHA256: `22B0ADBCCF442B0B3654F964E35AB77A044AF7623891E2A60C72B06A94ECE9A3`
- parent v1.7.2 SHA256: `4D9EE7FE7AACB510A1170B5C53B9E9F07D1182FFDC04EE55FA2A533097324533`
- v1.7.2 failure receipt SHA256: `8C0066CAFA310A91845C8D7D7F746A17CCC37651BD25561B059C89121B7286B2`

## Preserved scientific state

- valid replacements: 63
- deferred slots: 1
- retained originals: 224
- effective candidate pool: 287
- UNKNOWN competitors: 24
- slot 63 deferred
- slot 64 resolved; regeneration forbidden
- A-SYN accepted: 0
- A-SYN promoted: 0
- model training: false
- MC8 sealed-pending
- pointwise screen not started

## Frozen judge bindings

Gemma:
- `gemma4:31b-it-q4_K_M`
- `6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7`

GLM:
- `glm-4.7-flash:q4_K_M`
- `4475827791a269b02c8ec49b1c3bc1abb5846bacf3fae015b75d33986322d8f6`

Mistral:
- `mistral-small3.2:24b-instruct-2506-q4_K_M`
- `5a408ab55df5c1b5cf46533c368813b30bf9e4d8fc39263bf2a3338cfa3b895b`

Granite:
- `granite4.1:30b-q4_K_M`
- `3f3e5df8a021439fd6f867a0e526bdc303cac79c811201cb6bac193298cb9fcd`

## v1.7.3 execution boundary

The frozen qualification contract does not authorize Q12-only Gemma resume. A valid family qualification remains a complete 16-row result. Therefore:

1. Gemma runs the complete frozen 16-task qualification.
2. GLM runs the complete frozen 16-task qualification.
3. Both independent family jobs are submitted before polling either, allowing provider-level concurrency.
4. Each successful family is independently validated and made content-addressed Drive-durable before it may be reused.
5. If one family passes and the other fails, the passing family is preserved and the failed family is not blindly rerun.
6. Only after both pass does the unchanged v1.7.2/v1.7.0 chain bind all four judges and refreeze the 287-candidate current pool.
7. The package stops at pointwise-ready. It does not start the pointwise scope screen.

## Provider route

- provider: Magnolia
- partition: `node`
- accelerator: none
- minimum scientific RAM: 32 GiB
- adapter requests 48 GiB when live inventory supports it, otherwise 32 GiB
- up to 48 CPUs per exclusive family job
- P100 route not used because the frozen artifacts exceed the qualified 12 GiB route
- unauthorized A100 route not used
- Kaggle GPU submission absent from v1.7.3

## Frozen runtime

- Ollama v0.32.15
- runtime archive SHA256: `50539C5FE9BF85887733355098DCDB266B433CB8C73FA180713417E9ED6E42BB`
- binary SHA256: `EB99A47AAD366636488EBD9C163A9180254DFFCFDFE359939F9AABC36E2399C8`

## Provider-neutral infrastructure bindings

- R13-R17 package SHA256: `BCA2FD9BBDB55A7F74F71538D33B0F5FF7568F6B54D05CA0130E5A8BBF914E23`
- R18-R25 package SHA256: `77E788CA33F443ED984E91B1EB4D641793D8A346C4D49D1904C77AD2B7664AA0`
- MC10D opaque workload SHA256: `F1F79C15EA3052E420AF769E806F9DDBA148C9057671749482A03F353741F6CC`

## Release validation

- ZIP CRC PASS
- deterministic byte-identical rebuild PASS
- clean extraction PASS
- py_compile PASS
- outer package manifest verification PASS
- exact path-adapter reversibility PASS
- full package selftest PASS
- no Kaggle GPU submission in v1.7.3
- no automatic retry
- no model substitution
- no slot-64 regeneration
- no A-SYN acceptance/promotion/training
- no pointwise screen
