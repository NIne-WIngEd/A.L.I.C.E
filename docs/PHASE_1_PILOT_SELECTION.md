# Phase 1.5 — Pilot Dataset Proposal

**Status:** Proposal generation implemented  
**Safety mode:** Read-only and review-required

## Purpose

The first ingestion experiment must remain small, reviewable, and diverse.
This stage proposes a private pilot manifest; it does not parse or copy any
personal file.

Default proposal:

- 120 file records total;
- 110 unique primary contents;
- five exact-duplicate control groups with two paths per group;
- balanced representation across supported text and document families;
- less than 2 GiB;
- every row marked `pending` until Rayan reviews it locally.

## Included families

- JSON and web manifests;
- HTML;
- CSV;
- TXT and Markdown;
- PDF;
- DOCX;
- XLSX;
- PPTX;
- ICS calendar exports;
- VCF contact exports;
- SRT subtitles;
- XML and Atom.

## Excluded from the first pilot

- executables and scripts;
- serialized model/pickle files;
- MBOX mailboxes;
- ZIP archives;
- opaque DAT/BIN files;
- images, audio, and video;
- empty files;
- anything carrying an analysis risk flag;
- likely identity documents, credentials, tax records, banking records, and
  medical records based on conservative private path screening.

Path screening is only a safety filter. It is not a content classifier and
must not be treated as proof about a document.

## Generate the proposal

```powershell
py scripts\propose_pilot.py `
  --vault "C:\ALICE_Vault" `
  --target 120 `
  --duplicate-groups 5
```

## Private outputs

```text
C:\ALICE_Vault\manifests\exports\
├── pilot-proposal-summary-<ID>.json
├── pilot-proposal-<ID>.csv
├── pilot-candidate-audit-<ID>.csv
└── pilot-manual-additions-<ID>.csv
```

Only the summary JSON is appropriate to paste into a development discussion.
The CSV files contain private paths and filenames.

## Review fields

The proposal CSV contains:

- `decision`: change locally to `approve`, `reject`, or `replace` later;
- `review_notes`;
- `known_contradiction_group`;
- `contains_identity_document`;
- `contains_credentials_or_secrets`.

No approval command exists yet. The next P1.5 step will validate the reviewed
CSV and create an immutable approved manifest without copying source files.
