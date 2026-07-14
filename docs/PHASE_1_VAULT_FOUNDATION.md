# Phase 1 — Vault Foundation and Read-Only Inventory

The first source collection is approximately 27.227 GiB across 11,804 files and is classified `HIGHLY_SENSITIVE` by default.

## Controls implemented

- the vault must be outside the public Git repository;
- source and vault cannot contain one another;
- symlinks and Windows reparse points are not followed;
- metadata mode never reads file content;
- SHA-256 mode verifies that size and modification time did not change during hashing;
- exact duplicates require matching SHA-256 and file size;
- SQLite, JSON, and CSV manifests remain inside the private vault;
- no archive, MBOX, document, image, media, or `.dat` parsing occurs yet.

## Commands

```powershell
py scripts\init_vault.py --vault "C:\ALICE_Vault"
py scripts\inventory_dataset.py --source "C:\dataset_A.L.I.C.E" --vault "C:\ALICE_Vault" --mode metadata
```

After metadata inspection succeeds:

```powershell
py scripts\inventory_dataset.py --source "C:\dataset_A.L.I.C.E" --vault "C:\ALICE_Vault" --mode sha256
```

## Deferred

Archive extraction, MBOX parsing, `.dat` signature identification, text extraction, media transcription, OCR, AI classification, raw-file copying, and deletion are deferred until parser sandboxing and resource limits exist.
