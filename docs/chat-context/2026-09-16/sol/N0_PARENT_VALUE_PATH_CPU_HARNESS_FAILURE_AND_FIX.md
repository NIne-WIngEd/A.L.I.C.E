# N0 Parent Value-Path CPU Harness Failure and Fix

Date: 2026-09-16

The parent value-path preparation gate failed during pytest collection before diagnostic execution.

Observed failure:

`ModuleNotFoundError: No module named 'scripts'`

Root cause: the test imported `scripts.eipm.n0.diagnose_n0_v02_parent_value_path_v0_1`, but the udocker preparation contract intentionally places `scripts/eipm/n0` itself on `PYTHONPATH`. Therefore the diagnostic is a top-level script module in this environment, not a package member under `scripts`.

Interpretation:
- infrastructure/test-harness failure only;
- no parent diagnostic model execution;
- no GPU use;
- no gradient;
- no model evidence;
- no effect on the historical latent frozen challenge FAIL;
- no latent repair or scaling authorized.

Correction:
- test now imports `diagnose_n0_v02_parent_value_path_v0_1` directly;
- prep schema advanced to v0.1.1;
- prep receipt preserves this prior CPU collection failure explicitly;
- GPU runner now requires the v0.1.1 prep schema and hash-checks the corrected test file;
- same-revision gate remains mandatory.

Current next step: rerun the CPU preparation gate on the current `alice-eipm-v1-build` head. Only after it passes may the parent-path P100 diagnostic be considered.
