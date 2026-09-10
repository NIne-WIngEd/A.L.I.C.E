# MC10D v1.2.3 launcher R1 local compile-gate repair

Date: 2026-09-09

The first v1.2.3 launcher stopped locally at the external compile gate before the offline regression and before the controller's live section.

Observed failure:
`SyntaxError: invalid syntax` at `cfile=str(out / (name + .pyc))`.

Root cause:
- the launcher passed a multiline Python program through PowerShell's native-command `-c` argument;
- Windows PowerShell argument marshalling stripped the embedded quotes around `.pyc`;
- package code was not implicated.

Remote/provider effects:
- Kaggle controller was not invoked;
- no Kaggle status/push/output call occurred from the v1.2.3 launcher;
- no GPU time was consumed;
- no Vault/Drive scientific state was mutated by this failed launcher attempt.

Repair:
- package remains byte-identical v1.2.3, SHA-256 `4BA1F35B77ADDA667EEEC67430C123A8D4545EA09D221154AA28430BFCDD51CD`;
- new launcher `Start-ALICEMC10DKaggleDurableFullMC10DV123R1.ps1`;
- new launcher SHA-256 `5722DCAE3A30D81DCEC18F3A7A2D3D0B4DA94E34924339F970BF4815063FB8E4`;
- compile program is now written as UTF-8 without BOM to an external temporary `compile_gate.py` and invoked as a script, eliminating native `-c` quoting ambiguity;
- extracted package selftest was independently rerun and remains 20/20 PASS.

Run rule:
- use the R1 launcher only;
- do not use `-NewQuotaWindow` in the current quota window;
- scientific/package authority is unchanged.
