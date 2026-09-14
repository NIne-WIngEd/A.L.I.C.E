# Kaggle CPU fallback for the real N0 public lineage — 2026-09-14

## Why this route exists

Magnolia CPU corpus jobs failed before accepting any source rows because compute nodes could not resolve Hugging Face hostnames.

Observed failures:

- job `575538` on `node021`: `httpx.ConnectError: [Errno -3] Temporary failure in name resolution`
- job `575540` on `node013`: same failure at `HfApi.dataset_info(...)`

The second failure on a different CPU node is sufficient evidence to stop treating this as a single bad node. Do not keep cycling Magnolia CPU nodes for network acquisition.

These failures happened before the real corpus receipt was created. The partial corpus state contained only the dedup SQLite database and was deleted before retry. No private data and no model gradient were involved.

## Active split of responsibilities

Use Kaggle **CPU-only** for the network-dependent public corpus materialization and 48k tokenizer build. Then transfer the verified artifact to Magnolia. Magnolia remains the preferred route for the actual 2xP100 N0 training.

This keeps Kaggle GPU quota untouched while avoiding Magnolia's compute-node DNS boundary.

## Kaggle procedure

Create a Kaggle notebook with Accelerator set to **None / CPU** and Internet enabled. Run:

```bash
git clone --depth 1 --branch alice-eipm-v1-build \
  https://github.com/NIne-WIngEd/A.L.I.C.E.git \
  /kaggle/working/rayan-eipm-main

cd /kaggle/working/rayan-eipm-main
bash scripts/eipm/n0/kaggle_cpu_real_lineage.sh
```

The script uses the canonical bounded bootstrap of 100M accepted normalized characters per configured source unless overridden. It runs:

1. `corpus-bootstrap`
2. `verify-corpus`
3. `tokenizer`
4. transfer manifest creation
5. archive hashing

Expected outputs:

```text
/kaggle/working/rayan-n0-export/rayan-n0-v01-public-lineage.tar.gz
/kaggle/working/rayan-n0-export/transfer_manifest.json
/kaggle/working/rayan-n0-export/transfer_receipt.json
```

The transfer receipt declares that the package contains no private identity data, no private gradient, no model training, and no weights.

## Magnolia import

After the archive and `transfer_receipt.json` are copied to Magnolia, pull the latest build branch and import into a clean real lineage target:

```bash
cd "$HOME/rayan-compute/rayan-eipm-main"
git checkout alice-eipm-v1-build
git pull --ff-only

rm -rf "$HOME/rayan-compute/rayan-n0/n0-v01/corpus" \
       "$HOME/rayan-compute/rayan-n0/n0-v01/tokenizer"

python3 scripts/eipm/n0/import_external_public_lineage.py \
  --archive /path/to/rayan-n0-v01-public-lineage.tar.gz \
  --receipt /path/to/transfer_receipt.json \
  --target-workdir "$HOME/rayan-compute/rayan-n0/n0-v01"
```

Then verify the imported corpus inside the validated udocker runtime:

```bash
export ALICE_N0_WORKDIR="$HOME/rayan-compute/rayan-n0/n0-v01"
export RAYAN_UDOCKER_NVIDIA=0

bash scripts/eipm/n0/magnolia_udocker_exec.sh \
  scripts/eipm/n0/run_stage.sh verify-corpus
```

Do not run the 2xP100 MLM pilot until the imported corpus and tokenizer receipts have been inspected and verified.
