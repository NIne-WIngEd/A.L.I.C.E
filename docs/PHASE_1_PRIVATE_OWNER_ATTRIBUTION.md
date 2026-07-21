# P1.11 — Private Owner Attribution

The grounded-response pipeline can retrieve a correct personal record yet still
refuse to answer a question such as "What research experience do I have?" when
the evidence never explicitly tells the language model that a resume belongs to
the vault owner.

P1.11 now uses a private owner-identity configuration stored outside Git:

```text
C:\ALICE_Vault\config\owner_identity.json
```

The public repository contains only generic attribution logic.

## Initialize

```powershell
py scripts\init_owner_identity.py `
  --vault "C:\ALICE_Vault" `
  --primary-name "<OWNER NAME>" `
  --alias "<ALIAS 1>" `
  --alias "<ALIAS 2>"
```

The attribution layer uses deterministic owner-name matching plus source-type
signals.

Relations:

- `owner_self_record`: high-confidence self-record, such as a resume whose
  filename/provenance contains an owner alias;
- `owner_related_record`: owner alias is present but self-record status is not
  established;
- `owner_account_record_unverified`: resembles an account export but lacks a
  deterministic owner-name match;
- `unknown`: no deterministic attribution signal.

Only `owner_self_record` is explicitly safe for the response model to treat as
describing the user's own roles, projects, education, experience, or
achievements.

This metadata does not alter retrieval ranking, source hashes, citations, or
benchmark labels.

Owner identity and aliases remain private and must never be committed to Git.
