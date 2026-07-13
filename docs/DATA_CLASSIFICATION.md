# A.L.I.C.E. Data Classification

**Version:** 1.0.0  
**Status:** Ratified for Phase 0  
**Owner:** MK Rayan

## Classification rule

Every stored object, memory, document, log, tool parameter, and model-context segment must have a data classification. When classifications conflict, the most restrictive applicable class controls.

## PUBLIC

Information intentionally approved for unrestricted public release.

Examples:

- public portfolio content;
- published papers;
- public GitHub source code;
- approved biography;
- public project documentation.

Controls:

- may be stored in the public repository;
- may be sent to approved external models;
- still requires factual accuracy and source integrity.

## INTERNAL

Non-public project material with limited harm if exposed.

Examples:

- architecture drafts;
- development notes;
- non-sensitive test fixtures;
- internal technical decisions;
- unfinished public-facing writing.

Controls:

- may be stored in private development systems;
- public release requires review;
- external processing is allowed only through approved providers.

## PRIVATE

Personal or project information that should not be publicly disclosed.

Examples:

- personal conversations;
- unpublished applications;
- personal schedules;
- private source documents;
- private contact details;
- unannounced project plans;
- ordinary personal memories.

Controls:

- encrypted at rest;
- access limited to approved A.L.I.C.E. components;
- excluded from the public repository;
- external disclosure requires a defined purpose and permission.

## HIGHLY_SENSITIVE

Information whose disclosure could create material emotional, financial, legal, academic, reputational, or safety harm.

Examples:

- detailed financial records;
- identity documents;
- immigration records;
- medical or mental-health information;
- intimate relationship history;
- precise location history;
- private legal records;
- unpublished proprietary research;
- deeply personal life events.

Controls:

- strongest available encryption;
- local-first processing by default;
- no logging of full content unless essential;
- purpose-limited retrieval;
- explicit authorization before external transmission;
- deletion must include derived indexes and caches;
- cloud use requires a recorded exception.

## SECRETS

Authentication material that must never enter ordinary AI memory or prompts.

Examples:

- passwords;
- API keys;
- access and refresh tokens;
- private keys;
- recovery codes;
- session cookies;
- database credentials;
- encryption master keys.

Controls:

- dedicated secret manager only;
- never committed to Git;
- never stored in embeddings;
- never included in conversational logs;
- redacted from errors;
- rotated immediately after suspected exposure;
- access granted only to the exact process that requires it.

## Default classifications

- unknown personal content: `PRIVATE`
- unreviewed life archive: `HIGHLY_SENSITIVE`
- generated logs containing personal context: `PRIVATE`
- credentials detected anywhere: `SECRETS`
- public-source web content: `PUBLIC`, while derived user profiles remain at least `PRIVATE`
