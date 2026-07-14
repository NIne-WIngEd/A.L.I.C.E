# ADR-002: Local Personal Vault with Cloud Model Training

**Status:** Accepted  
**Date:** July 14, 2026  
**Decision owner:** MK Rayan

## Decision

A.L.I.C.E. uses a hybrid architecture:

- source archives, raw vault objects, extraction, privacy scanning, provenance,
  memory review, and deletion controls remain local;
- cloud compute may be used later for approved fine-tuning or training;
- cloud training receives only an explicit export derived from approved pilot
  and memory records;
- the export must be minimized, redacted, versioned, encrypted in transit and at
  rest, and covered by a documented deletion/retention plan;
- credentials, identity documents, raw email archives, full social-media
  exports, and unreviewed highly sensitive records are excluded by default;
- cloud-trained weights do not become the canonical store for mutable personal
  facts; those facts remain in inspectable memory/RAG.

## Reason

Training compute and personal-data custody are separate concerns. Cloud compute
can accelerate later training without making the raw personal archive a cloud
dataset.
