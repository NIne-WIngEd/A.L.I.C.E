# N0 Kaggle CPU Lineage Success and Magnolia Import — 2026-09-14

## Current state

The real bounded public N0 corpus/tokenizer lineage has now been built successfully on Kaggle CPU after Magnolia CPU compute nodes `node021` and `node013` both failed outbound DNS resolution to Hugging Face.

The successful transport used the restored known-good Kaggle pattern: tiny/local launcher boundary, Python-owned JSON and state, UTF-8 metadata without BOM, deterministic content-derived kernel identity, persisted push intent, one push only, exact-identity reconciliation, raw CLI receipts, verified output retrieval, and cleanup only after validated evidence.

Successful deterministic kernel:

`mkrayanyan/rayan-n0-cpu-v102-bbb0bb7a59d1`

Request SHA-256:

`bbb0bb7a59d164d076232df76395b4681845c76c39fecab778078782365318f0`

The kernel reached COMPLETE. Output retrieval passed. Local hash validation passed. Remote cleanup passed.

Verified public-lineage archive:

- name: `rayan-n0-v01-public-lineage.tar.gz`
- bytes: `130670111`
- SHA-256: `d8df5c0cab365cf55c66fc173d439f34c3d40a409ab45cada1c21cca5a02a182`
- lineage source revision: `cb23ac483d0ec5bcc9107f895ada0a6a9c8c34ba`
- transfer manifest SHA-256: `dbf547fd7fe09bc8513f23b8036696dadd511e62b3da658206606c0d4337e824`
- transfer receipt SHA-256: `35d1edd5f4f05437f1e3d52ac042cdcbaa89add24a621831f2eb08ad91972211`

The failed V101 PowerShell launcher is historical negative evidence only. It repeated the already-known Windows PowerShell 5.1 BOM regression and has been retired from the EIPM branch. Do not restore or reuse it.

## Immediate continuation

Next boundary is transfer/import, not more Kaggle or Magnolia network qualification.

1. Copy the verified Kaggle archive and transfer receipt from Windows to the owner-private Magnolia namespace.
2. Sync `alice-eipm-v1-build` in `$HOME/rayan-compute/rayan-eipm-main`.
3. Remove only the failed empty/partial real `corpus` state if it still exists. Do not overwrite a valid imported lineage.
4. Run `scripts/eipm/n0/import_external_public_lineage.py` targeting `$HOME/rayan-compute/rayan-n0/n0-v01`.
5. Require importer PASS before any P100 pilot.
6. After import PASS, submit `scripts/eipm/n0/magnolia_p100x2_mlm_pilot.sbatch`.

The importer verifies archive hash, manifest hash, every manifested file size/hash, corpus receipt hash, tokenizer receipt hash, tokenizer hash, and the no-private/no-training transfer flags before moving state into the canonical real lineage.

Do not rerun corpus materialization on Magnolia CPU. Do not rerun Kaggle merely because import or local retrieval has a problem; diagnose transfer/import directly.

No private E0/E-INF/A-SYN gradient is authorized yet.
