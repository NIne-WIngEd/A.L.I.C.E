# Phase 1 — Safe Parser Registry and Verified Extraction

**Subphase:** P1.6  
**Input:** `C:\ALICE_Vault\raw\pilot-v1`  
**Output:** private derived text and provenance metadata  
**Cloud use:** none

## Purpose

P1.6 converts the approved immutable pilot objects into normalized text for the later chunking and retrieval stages.

The parser registry is machine-readable:

```text
policies/parser_registry.json
```

Every enabled parser declares:

- accepted family and extensions;
- dependency;
- maximum source size;
- subprocess timeout;
- maximum output characters;
- format-specific limits.

## Safety controls

- verifies every pilot object against its stored SHA-256 and size;
- rejects absolute paths, `..` traversal, symlinks, and Windows reparse points;
- requires the manifest family to match an approved extension;
- parses every unique content object only once;
- launches parsing in a separate Python process;
- applies a per-parser timeout;
- does not use a command shell;
- does not extract embedded files or ZIP members;
- does not execute macros or formulas;
- uses XLSX read-only/data-only mode with external links disabled;
- performs no OCR;
- skips vCard photo payloads and calendar attachments;
- caps extracted text and records truncation;
- writes output atomically;
- hashes every derived text file;
- records parser ID, registry digest, source hash, output hash, warnings, and limits;
- resumes only when existing outputs and metadata still verify.

The subprocess and proxy settings reduce accidental parser side effects, but they are not a complete operating-system sandbox. The approved pilot, signature analysis, strict limits, hashes, and private encrypted vault remain important security layers.

## Install

```powershell
py -m pip install `
  -r requirements-dev.txt `
  -r requirements-phase1.txt `
  -r requirements-extraction.txt
```

## Extract the pilot

```powershell
py scripts\extract_pilot.py `
  --vault "C:\ALICE_Vault" `
  --pilot-name "pilot-v1" `
  --fail-on-error
```

Private outputs:

```text
C:\ALICE_Vault\derived\pilot-v1\extracted\
├── text\
│   └── <SOURCE-SHA256>.txt
└── metadata\
    └── <SOURCE-SHA256>.json
```

Private run manifests:

```text
C:\ALICE_Vault\manifests\extractions\pilot-v1\
C:\ALICE_Vault\manifests\exports\pilot-extraction-summary-*.json
```

## Verify

```powershell
py scripts\verify_pilot_extraction.py `
  --vault "C:\ALICE_Vault" `
  --pilot-name "pilot-v1"
```

P1.6 is ready for chunking only when:

```json
{
  "error_count": 0,
  "ready_for_chunking": true
}
```

## Deliberately deferred

- OCR;
- audio/video transcription;
- XML parsing;
- email MBOX parsing;
- archive extraction;
- embedded Office objects;
- PDF attachments;
- formulas, macros, and executable content;
- cloud parsing;
- semantic chunking and embeddings.
