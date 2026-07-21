# P1.11 — HHEM Calibration Bundle Selection Fix

The HHEM calibration evaluator searched for:

`judge-calibration-*.json`

That wildcard also matched newer derivative files such as:

`judge-calibration-evaluation-details-*.json`

Those derivative files contain human labels and judge outputs, but they do not
contain the original `claim_text` and `evidence_windows` required to build HHEM
premise-hypothesis pairs.

The evaluator therefore selected the wrong newest file and raised:

`ValueError: No calibration items had usable premise-hypothesis pairs`

This patch selects only files that:

- use `judge_calibration_bundle_schema_version == 1`;
- contain item dictionaries with `claim_text`;
- contain `evidence_windows`;
- are not `judge-calibration-evaluation-*` derivative files.

No model, threshold, human labels, or production verifier behavior changes.
