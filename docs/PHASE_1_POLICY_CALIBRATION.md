# Phase 1 — Auto-Review Policy Calibration

**Status:** P1.5 decision-policy correction  
**Purpose:** Reduce false manual-review routing without weakening hard privacy controls.

## Why the previous run left all 120 files pending

The model layer succeeded, but the policy layer was too conservative:

- any truncated preview was blocked before semantic review;
- any Presidio high-risk label was treated as conclusive without score/context;
- every `highly_sensitive` label was blocked even though the pilot remains local;
- every non-empty contradiction topic was blocked even though one document
  cannot establish a cross-document contradiction by itself;
- the approval/rejection threshold remained 0.92.

## Calibrated rules

Hard manual-review blockers remain:

- deterministic secret pattern;
- identity-document pattern;
- prompt-injection pattern;
- explicit sensitive-topic pattern;
- context-supported high-risk Presidio entity;
- substantial third-party private data;
- financial, medical, legal/immigration, and relationship categories;
- extraction failure or no extractable text.

Changes:

- Presidio high-risk labels require both score >= 0.85 and supporting context.
- Truncated previews may be semantically reviewed; approval requires >= 0.92.
- Safe local categories may be approved even when the model labels them
  `highly_sensitive`, because the snapshot remains encrypted and local.
- Contradiction topics are retained as annotations for later cross-document
  comparison and are not blanket blockers.
- New model results use an 0.85 approval/rejection threshold.
- Raw semantic recommendation fields are preserved in future output.

## Targeted recalibration

The recalibration command reuses the latest successful semantic details. It
does not rerun all 115 unique contents.

It only sends previously unclassified items—primarily truncated previews and
Presidio-only false positives—to the local model.

```powershell
py scripts\recalibrate_auto_review.py `
  --vault "C:\ALICE_Vault" `
  --model "qwen3:4b-instruct"
```

Defaults:

```text
batch_size = 4
max_chars = 1200
approve_threshold = 0.85
reject_threshold = 0.85
Presidio score threshold = 0.70
Presidio blocking threshold = 0.85 plus context
resume = true
```

Results are checkpointed and written to private run-specific CSV and JSON files.
The canonical review CSV is updated only after the run-specific output exists.

Do not upload review CSVs, calibration details, source paths, or extracted text.
