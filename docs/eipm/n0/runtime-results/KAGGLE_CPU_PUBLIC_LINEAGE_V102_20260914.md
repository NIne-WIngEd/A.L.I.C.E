# Kaggle CPU Public N0 Lineage v1.0.2 — 2026-09-14

## Result

The governed Kaggle CPU fallback completed successfully after Magnolia CPU compute nodes repeatedly failed outbound DNS resolution to Hugging Face.

Local controller:

- `scripts/eipm/n0/rayan_n0_kaggle_cpu_lineage_controller_v102.py`
- deterministic request SHA-256: `bbb0bb7a59d164d076232df76395b4681845c76c39fecab778078782365318f0`
- deterministic kernel ref: `mkrayanyan/rayan-n0-cpu-v102-bbb0bb7a59d1`
- Kaggle CLI: `2.2.4`
- accelerator: none / CPU-only
- internet: enabled
- private kernel: true
- private identity data: false
- model training: false

The kernel reached `RUNNING` and then `COMPLETE`. Output retrieval and local SHA-256 verification both passed. The remote kernel was deleted only after verified output retrieval.

## Produced lineage bundle

Downloaded archive:

`rayan-n0-v01-public-lineage.tar.gz`

- bytes: `130670111`
- SHA-256: `d8df5c0cab365cf55c66fc173d439f34c3d40a409ab45cada1c21cca5a02a182`
- lineage git revision: `cb23ac483d0ec5bcc9107f895ada0a6a9c8c34ba`

Transfer manifest:

- SHA-256: `dbf547fd7fe09bc8513f23b8036696dadd511e62b3da658206606c0d4337e824`

Transfer receipt:

- SHA-256: `35d1edd5f4f05437f1e3d52ac042cdcbaa89add24a621831f2eb08ad91972211`

Runtime receipt was also returned and validated as PASS by the local controller.

## Transport lessons applied

The first v1.0.1 direct PowerShell launcher repeated a previously solved Windows PowerShell 5.1 UTF-8 BOM failure in `kernel-metadata.json`. It failed locally during Kaggle metadata parsing before a valid remote submission. That launcher has been retired.

v1.0.2 restored the proven transport rules:

- Python owns JSON and dispatch state;
- metadata is UTF-8 without BOM and reparsed before push;
- runner syntax is compiled before push;
- staging is exactly one code file plus `kernel-metadata.json`;
- deterministic content-derived kernel identity;
- push intent is persisted before submission;
- one push only;
- delayed/ambiguous discovery does not authorize a second identity;
- exact identity is resumed after local interruption;
- raw Kaggle CLI stdout/stderr/exit codes are retained;
- output retrieval failure never causes compute repetition;
- archive hash is verified against the transfer receipt before cleanup;
- remote cleanup happens only after verified successful retrieval.

## Next action

Transfer the verified archive and transfer receipt from Windows to Magnolia. Import them with `scripts/eipm/n0/import_external_public_lineage.py` into:

`$HOME/rayan-compute/rayan-n0/n0-v01`

The importer must verify the archive SHA-256, embedded manifest, every manifested file hash/size, corpus receipt hash, tokenizer receipt hash, tokenizer hash, and all private/training flags before moving corpus/tokenizer state into the real lineage.

After import PASS, run the already-prepared 2×P100 200-step public N0 MLM pilot. No more corpus/network qualification is needed unless import or pilot exposes a concrete failure.

No private E0/E-INF/A-SYN gradient is authorized by this result.
