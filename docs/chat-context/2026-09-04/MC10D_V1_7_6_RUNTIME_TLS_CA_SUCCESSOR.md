# MC10D v1.7.6 Runtime TLS CA Successor — 2026-09-04

## Status

Continue from the exact v1.7.5 provider-runtime failure. Do not rerun v1.7.5.

This note is contextual reconstruction evidence on `alice-context`. Frozen packages, validators, receipts, and canonical repository authority win on conflict.

## Preserved scientific state

- canonical main: `0abaed85873c3f8de04765847eb7700b0e20433f`
- repair state: 63 replacements / 1 deferred / 287 effective candidates
- slot 63: deferred
- slot 64: resolved; regeneration forbidden
- A-SYN accepted: 0
- A-SYN promoted: 0
- model training: false
- pointwise screen: not started
- MC8: sealed-pending
- scientific workload SHA-256: `F1F79C15EA3052E420AF769E806F9DDBA148C9057671749482A03F353741F6CC`
- neutral task-set SHA-256: `7D91C291A65A61D16D5CC0623D58B49D856944F8466FEAD8DBB866EC26B867AA`
- neutral run ID: `RUN-3DCE428B707EB452860F751F`

## v1.7.5 runtime failure

Provider runtime job:

- Slurm job ID: `574947`
- job name: `rayan-mc10d-v175-runtime-860f751f`
- node: `node019`
- state: `FAILED`
- exit: `1:0`
- elapsed: 11 seconds
- Gemma submitted: false
- GLM submitted: false
- model pull started: false
- judge inference started: false

Private compute ledger commit:

`9ea6aba5a1b62b72b6304e6a9b0c9bffceaa4a95`

The application log proves the exact failure:

`ssl.SSLCertVerificationError: certificate verify failed: unable to get local issuer certificate`

This is a Magnolia Python CA-trust/runtime transport failure. It is not a scientific qualification failure, model failure, Slurm failure, or judge result.

## v1.7.6 successor

Package:

`ALICE_MC10D_PROVIDER_NEUTRAL_JUDGE_CONTINUATION_v1.7.6.zip`

SHA-256:

`04A93C51721D383A54F82F66E083906104A46E1F5A99A1BFC5E323A3005C425C`

Launcher:

`Start-ALICEMC10DProviderNeutralJudgeContinuationV176.ps1`

SHA-256:

`B9B3CBD4C2EE994089532906C366B82C31EFBE69C2A6A33924FE15670AEC53B5`

Build audit SHA-256:

`77C9FAAD1B2423704FF5834845459CB66C450328CB9279E62B9FA6123175D0B4`

## TLS correction

v1.7.6 preserves TLS verification.

It selects the first readable PEM CA bundle from:

1. `/etc/pki/tls/certs/ca-bundle.crt`
2. `/etc/ssl/certs/ca-certificates.crt`
3. `/etc/ssl/cert.pem`

Then exports:

- `SSL_CERT_FILE`
- `REQUESTS_CA_BUNDLE`
- `CURL_CA_BUNDLE`

Python HTTPS explicitly uses:

`ssl.create_default_context(cafile=os.environ["SSL_CERT_FILE"])`

Forbidden/inactive:

- no `CERT_NONE`
- no unverified SSL context
- no `PYTHONHTTPSVERIFY=0`
- no `verify=False`
- no `--insecure`
- no `curl -k`

The same CA environment is inherited by the Ollama process.

## Required reconciliation before new provider runtime attempt

v1.7.6 requires:

- exact v1.7.5 local failed runtime state;
- source job ID `574947`;
- source job state `FAILED`, exit `1:0`;
- no v1.7.5 Gemma job;
- no v1.7.5 GLM job;
- exact source application log and telemetry receipt;
- exact SSL issuer-chain failure text;
- no completed pinned runtime cache.

Only after those checks may v1.7.6 create a fresh provider runtime attempt.

Neutral scientific run/task identities remain unchanged.

## Intended continuation

If runtime staging passes:

1. submit full Gemma 16-task qualification;
2. submit full GLM 16-task qualification before polling either;
3. preserve any successful family result durably;
4. require both exact frozen validators to pass;
5. invoke unchanged v1.7.2;
6. invoke unchanged v1.7.0 current-pool refreeze;
7. bind Gemma + GLM + Mistral + Granite;
8. preserve effective pool 287;
9. stop at POINTWISE_READY.

Do not start the pointwise screen, expose MC8, accept/promote A-SYN, train, regenerate slot 64, substitute models, or use Kaggle GPU during this successor.
