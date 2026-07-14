# Phase 1 — Inventory Analysis and File-Signature Review

**Subphase:** P1.4  
**Status:** Initial implementation  
**Safety mode:** Read-only

## Purpose

This stage analyzes the latest completed SHA-256 inventory and produces private review reports.

It does not:

- extract archives;
- parse email messages;
- execute code or binaries;
- move files;
- create a quarantine copy;
- delete duplicates;
- send any file or metadata to a cloud service.

## Detection design

`puremagic` is used in header-oriented mode with deep scanning disabled. ZIP-compatible containers receive an additional structural inspection through Python's standard-library `zipfile` module.

ZIP inspection reads only the archive directory and records:

- member counts;
- compressed and uncompressed totals;
- compression ratio;
- encrypted member count;
- unsafe absolute, drive-qualified, or `..` paths;
- oversized members;
- Office Open XML structure for DOCX, XLSX, and PPTX.

No archive member is extracted.

## Install

```powershell
py -m pip install `
  -r requirements-dev.txt `
  -r requirements-phase1.txt
```

## Test

```powershell
py scripts\validate_phase0.py
py -m unittest discover -s tests -p "test_*.py" -v
```

## Run

```powershell
py scripts\analyze_inventory.py --vault "C:\ALICE_Vault"
```

## Private output

```text
C:\ALICE_Vault\manifests\exports\
├── analysis-summary-<RUN-ID>.json
├── analysis-files-<RUN-ID>.csv
└── analysis-review-<RUN-ID>.csv
```

The summary is aggregate-only. The CSV files contain private filenames and paths and must not be uploaded publicly.

## Recommendations

- `quarantine_recommended`: executable/script content, unsafe ZIP paths, or corrupt archives.
- `specialized_review`: mailbox, archive, opaque DAT/BIN, serialized model/pickle, or archive-risk metadata.
- `manual_review`: signature mismatch, extensionless identified content, or signature error.
- `pilot_candidate`: small text or document format suitable for the later controlled pilot.
- `metadata_only`: image/audio/video retained for metadata work only at this stage.
- `inventory_only`: no parser decision yet.

A recommendation is not an action. This stage never moves or deletes a file.
