# Phase 1.11 — Verifier Failure Forensics

The ensemble diagnostic showed that no fixed Boolean combination improved
support precision above the best individual observed precision on the current
12-item holdout.

This diagnostic joins the private holdout bundle with the HHEM holdout
evaluation details and records:

- verifier decision vectors;
- human-label distribution for each vector;
- Qwen false positives;
- HHEM false positives;
- FEVER false positives;
- shared Qwen/HHEM false positives;
- unanimous false positives;
- private claim text and cited evidence for failure inspection.

Private claim and evidence text remain inside the vault. The exported summary
contains counts and identifiers only.

This step is diagnostic only and cannot change the production gate.
