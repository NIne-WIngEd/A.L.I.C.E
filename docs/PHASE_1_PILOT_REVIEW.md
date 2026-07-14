# Phase 1 — Pilot Human Review and Approval

**Subphase:** P1.5b  
**Status:** Human review gate  
**Data boundary:** Private vault only

## Purpose

The automated proposal is not final. This workflow requires Rayan to inspect
the selected content, approve or reject every record, identify contradiction
controls, and explicitly confirm that approved files do not contain identity
documents or credentials.

## Prepare the review sheet

```powershell
py scripts\prepare_pilot_review.py `
  --vault "C:\ALICE_Vault"
```

This creates a private CSV:

```text
C:\ALICE_Vault\manifests\exports\pilot-review-<PROPOSAL-ID>.csv
```

Open it locally:

```powershell
$Review = Get-ChildItem `
  "C:\ALICE_Vault\manifests\exports\pilot-review-*.csv" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

Start-Process $Review.FullName
```

## Columns Rayan may edit

- `decision`: `approve`, `reject`, or `pending`
- `review_notes`
- `known_contradiction_group`
- `contains_identity_document`: `yes` or `no`
- `contains_credentials_or_secrets`: `yes` or `no`

Do not change generated IDs, paths, hashes, family labels, roles, or sizes.

## Review rules

1. Every row must be inspected and changed from `pending`.
2. Approved files must explicitly mark both sensitive-content flags as `no`.
3. Keep both members of a duplicate-control pair approved, or reject both.
4. Label at least two contradiction groups with at least two approved files
   per group.
5. A contradiction group should contain records that disagree because one is
   outdated, superseded, corrected, or represents a changed plan/status.
6. The approved set must retain at least 100 items, eight format families,
   five source buckets, and the core JSON, HTML, CSV, PDF, and DOCX families.

## Validate review decisions

```powershell
py scripts\validate_pilot_review.py `
  --vault "C:\ALICE_Vault"
```

Exit code `0` means the pilot is ready. Exit code `2` means the printed
blocking errors must be corrected in the private review CSV.

## Finalize the pilot snapshot

```powershell
py scripts\finalize_pilot.py `
  --vault "C:\ALICE_Vault" `
  --pilot-name "pilot-v1"
```

Finalization:

- re-hashes every approved source file;
- aborts if any source changed;
- copies each unique content object exactly once;
- maps duplicate file records to the same content-addressed object;
- marks object copies read-only;
- preserves review and provenance manifests;
- does not modify source files;
- refuses to overwrite an existing pilot snapshot.

The finalized snapshot is private:

```text
C:\ALICE_Vault\raw\pilot-v1\
```

Nothing in the pilot snapshot may be committed to the public repository.
