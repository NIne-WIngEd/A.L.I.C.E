# ALICE v186 offline forensics v1.0.0

This package continues the accepted Astra audit from the actual v186 GLM semantic failure. It reads the already downloaded result, 16 response rows, runtime metadata and failure receipt. It does not start Kaggle, Magnolia, Ollama, SSH, training or candidate work.

The recorded source is:

`C:\ALICE_Vault\datasets\memory_stage_g2\alice.stage-g2.g2a.gold-semantic-decomposition.v1\audits\alice-mc10d-repair-qualify-refreeze-v1-3.work\judge_qualification\download-v186-glm-260906185551-e563ae\output`

## Run

Save the supplied ZIP and `Start-ALICEAstraForensicsV100.ps1` in Downloads. Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "$HOME\Downloads\Start-ALICEAstraForensicsV100.ps1"
```

The launcher verifies the ZIP hash, extracts into a new disposable run directory, verifies the file manifest, compiles the Python source, runs the offline tests, then collects and analyzes existing evidence. Python standard library only; Anaconda Python is preferred. The launcher works from its own directory and accepts Downloads duplicate ZIP names when the bytes match the exact expected hash.

It writes a new `ALICE_V186_FORENSICS_<id>.zip` in Downloads and prints `FORENSICS_RESULT_BUNDLE=...`. Return that ZIP and the terminal output to Astra. The result ZIP includes the existing public fictional response rationales for interpretation. No private candidate files or hidden MC8 files are collected.

The supplied launcher also publishes a compact typed/numeric summary to `alice-context` using existing local Git credentials. It uses a temporary isolated Git index, never changes the working main checkout, and never force-pushes. It preserves concurrent context edits and skips identical uploads. Raw rationales and local paths are excluded from public telemetry. Publication adds a receipt beside the immutable result ZIP's expanded directory; it does not rewrite the already created result ZIP.

## Outcomes

| Exit | Meaning | Next action |
|---|---|---|
| 0 | Existing failure evidence collected and its public summary published | Return the result ZIP and terminal output; review actual failed tasks before another model/protocol decision. |
| 75 | Collection succeeded; Git publication needs credentials/connectivity or remote-state review | Keep and return the result ZIP. Retry publication alone with `publish_context.py --run-dir <printed expanded directory> --repo-root C:\A.L.I.C.E-main`; no compute rerun is needed. |
| 76 | Required source evidence, lineage or consistency check failed | Return the terminal output. Do not rerun qualification to replace missing evidence. |

`mc10d_forensics.py` also supports `--vault-root`, `--output-root`, `--repo-root` and optional `--publish-context` for a later CPU environment. Output must be outside the source vault. Publication is the only network operation, and it targets the ALICE context branch. The 20 offline tests include a temporary local Git remote; no GitHub service is used by the tests.

## Interpretation

The effective rule is still 16 unique frozen tasks, at least 14 correct verdicts, at least 6 correct critical decisions, all five mandatory hard anchors, Q01 PASS and Q03 HOLD. Secondary gold fields are diagnostic under the recorded decision-centric amendment. The historical policy JSON still contains superseded full-gold fields; `EFFECTIVE_CALIBRATION_CONTRACT.json` makes that distinction explicit without changing ratification.

A successful collection verifies consistency of the recorded failure with the frozen tasks and supplied metadata. It is not a new model evaluation or proof that an arbitrary file is authentic merely because it contains a hash. Actual source bytes, exact failure hash, rendered-worker lineage, runtime binding, all row labels, task identities and counters are preserved and checked together. The reused public suite is calibration, not independent certification.

The runtime may have produced a response longer than its declared 500-character rationale schema limit; the old worker did not enforce that limit in its Python checker. Such a result is preserved and reported as a schema diagnostic. It does not silently change historical semantic gates.

The included failure receipt was reconstructed exactly from the supplied terminal JSON and its known SHA-256 verified. No GLM response rows were reconstructed. Tests generate clearly labeled synthetic rows in temporary directories; those are not experimental evidence.

No qualification binding is installed. v186 success and breadth-v104 eligibility remain unestablished. Gemma rows, canonical qualification directories, the 287 pool, 63 replacements and one deferred item remain intact.
