# N0 Magnolia DNS boundary and Kaggle CPU fallback — 2026-09-14

## Confirmed failure pattern

The real N0 public corpus bootstrap was attempted twice on Magnolia's CPU-only `node` partition.

- job `575538` ran on `node021` and failed in 10 seconds with `httpx.ConnectError: [Errno -3] Temporary failure in name resolution` at `HfApi.dataset_info(...)`.
- the failed corpus directory contained only `exact_dedup.sqlite3`; no shards and no corpus receipt were created.
- the dependent tokenizer job `575539` became `DependencyNeverSatisfied` and was cancelled.
- the partial corpus directory was deleted.
- job `575540` retried the same bounded corpus bootstrap with `node021` excluded and landed on `node013`.
- job `575540` failed with the same Hugging Face DNS/name-resolution error.
- dependent tokenizer job `575541` became `DependencyNeverSatisfied`.

Two distinct CPU compute nodes showing the same immediate external DNS failure closes the single-node hypothesis. Do not keep cycling Magnolia CPU nodes for Hugging Face acquisition.

The failure is an execution-environment network boundary, not an N0 corpus-materialization semantic or provenance failure.

## Active route correction

Network-dependent public corpus materialization and tokenizer creation move to Kaggle CPU-only. This does not consume Kaggle GPU quota. Magnolia remains the preferred actual training route on the already qualified 2xP100 path.

Tracked build artifacts added on `alice-eipm-v1-build`:

- `scripts/eipm/n0/kaggle_cpu_real_lineage.sh`
- `scripts/eipm/n0/import_external_public_lineage.py`
- `docs/eipm/n0/KAGGLE_CPU_REAL_LINEAGE_FALLBACK_2026-09-14.md`

The Kaggle exporter runs the existing governed `corpus-bootstrap -> verify-corpus -> tokenizer` stages and packages only the public corpus/tokenizer lineage. It creates a manifest and sidecar receipt with SHA-256 binding. Both declare no private identity data, no private gradient, no model training, and no weights.

The Magnolia importer refuses to overwrite an existing corpus/tokenizer lineage, validates archive and manifest hashes, validates every manifested file, validates the tokenizer hash against its receipt, and imports only into a clean target.

## Current owner-help boundary

Next external action is a Kaggle notebook with Accelerator=None/CPU and Internet enabled. Clone the public `alice-eipm-v1-build` branch and run:

```bash
bash scripts/eipm/n0/kaggle_cpu_real_lineage.sh
```

Return the final PASS line plus `transfer_receipt.json` and the corpus/tokenizer receipts, or transfer the generated archive to Magnolia and run the tracked importer.

Do not run the 2xP100 200-step MLM pilot until the imported real corpus and tokenizer receipts are verified.

No private E0/E-INF/A-SYN gradient is authorized or performed by this route.
