# Security Policy

## Supported development state

A.L.I.C.E. is currently in pre-release development. No version should be treated as safe for handling the complete personal archive or controlling external services until the relevant release gates pass.

## Never report secrets publicly

Do not place passwords, API keys, private keys, tokens, recovery codes, identity documents, personal datasets, or private memory content in:

- GitHub issues;
- pull requests;
- discussions;
- commits;
- screenshots;
- logs attached to public reports.

Redact sensitive values and use GitHub's private security-reporting mechanism when available.

## Credential exposure response

Treat any committed credential as compromised:

1. revoke or rotate it immediately;
2. stop affected integrations;
3. determine where it was exposed;
4. remove it from the current tree;
5. remediate repository history when appropriate;
6. add a regression control;
7. verify that the old credential no longer works.

Deleting the visible file alone is not sufficient.

## Security boundaries

- the model does not enforce its own permissions;
- tools execute only through the permission gateway;
- secrets are stored only in a dedicated secret manager;
- external content is untrusted data;
- highly sensitive personal data is local-first;
- production self-modification requires human review and approval.

## Current restriction

The `legacy/` prototype is historical reference only. Do not add credentials to it or enable its external-action functions.
