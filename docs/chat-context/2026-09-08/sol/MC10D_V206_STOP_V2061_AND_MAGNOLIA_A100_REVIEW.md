# MC10D v2.0.6 local extraction stop, v2.0.6.1 repair, and Magnolia A100 route review

Date: 2026-09-08

Canonical main remains frozen at `0abaed85873c3f8de04765847eb7700b0e20433f`.

## v2.0.6 observed stop

The local-only MC8/simulation preflight stopped before MC8 verification with:

`Stop:extract destination exists`

Root cause: exact v2.0.0 `safe_extract()` requires a non-existing destination, while v2.0.6 passed the already-created root of `TemporaryDirectory()`.

Scientific impact:
- none;
- no network calls;
- no remote inference;
- MC8 was not verified/opened by that failed run;
- simulation did not start;
- A-SYN acceptance/promotion remain false;
- training remains false.

Technical successor:
- `ALICE_MC10D_V2061_MC8_SIMULATION_PREFLIGHT.py`
- SHA-256 `A04F0F482F97FB411FCAD203FCFCEEF96D71B078C3941F7C71EB079F7EB023ED`
- extracts into a fresh child directory and otherwise keeps v2.0.6 semantics unchanged.

## Magnolia route correction

Earlier MC10D no-go decisions correctly rejected the proven P100 route as provider-equivalent to the exact 2x-T4 v2.0.0 worker.

A preserved Magnolia access survey also contains a materially different result:

`sbatch --test-only --partition=suliaoma --qos=normal --gres=gpu:a100:1 ...`

was accepted for node gpu003.

The separate `qos=bxmarg` test failed with `Invalid qos specification`.

Therefore the correct current statement is:
- P100 route: not qualified for current MC10D inference;
- A100 normal-QOS route: test-only admission observed, but actual allocation/runtime/model-profile equivalence not yet qualified.

A bounded actual-allocation probe is now prepared:
- `ALICE_MC10D_MAGNOLIA_A100_ROUTE_QUALIFICATION_v1.0.0.py`
- SHA-256 `DD88A963520FBC17816ECC4752908A8D044602567A7C7B8D098064C0C263A570`

It uploads no private A.L.I.C.E. payload, downloads no model, and performs no inference.

If actual A100 access is proven, Magnolia must still prospectively qualify exact Ollama v0.32.15 source-build compatibility and the frozen Gemma/Qwen/Mistral/Granite model profiles before private simulation/falsification is routed there.

If Magnolia qualification fails, Kaggle fallback must be checkpointed and compute-budget-aware rather than weakening MC10D scientific gates.
