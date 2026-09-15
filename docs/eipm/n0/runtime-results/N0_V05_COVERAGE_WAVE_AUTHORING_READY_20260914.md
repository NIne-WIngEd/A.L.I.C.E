# N0 v0.2 Coverage Wave v0.5 — Authoring Ready

**Date:** 2026-09-14  
**Status:** deterministic public authoring package ready; no GPU/model training performed

After the v0.4b teacher-bank audit passed, the exact competency deficits were computed against the active curriculum contract of at least 15 train and 5 dev scenarios per each of 51 competencies.

Current bank: 383 rows.

Exact deficit:

- 490 additional train rows;
- 147 additional dev rows;
- 637 total additional rows;
- resulting bank size: 1020 rows.

This supersedes the looser `617 rows remaining to 1000` numerical observation because the per-competency coverage floor is stricter and therefore controls.

A deterministic Sol-authored public curriculum generator has been prepared for this wave. It contains competency-specific scenario generators, reusable principle tags, deterministic candidate-position rotation, fixed-eval semantic-item exclusion, and no private identity material. It generates a runtime registry and SHA-256-bound origin manifest before invoking the existing teacher-bank audit.

The authoring package SHA-256 is:

`21981AABAE58BD6F4E038C0B7E91B801DFF5089A8CE699D473E5F0A771E9FADD`

No private identity gradient is authorized and no model training has occurred.
