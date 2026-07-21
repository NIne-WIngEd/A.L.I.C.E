# P1.11 — HHEM Transformers 4 Compatibility Fix

The HHEM-2.1-Open repository uses custom `PreTrainedModel` code written against
the Transformers 4 API. The model's own `config.json` records
`transformers_version: 4.39.3`.

The A.L.I.C.E. HHEM dependency file accidentally allowed Transformers 5.x:

`transformers>=4.45,<6.0`

With Transformers 5.x, model loading reached the newer tied-weight finalization
path and failed because HHEM's custom class does not define the new
`all_tied_weights_keys` attribute.

This patch:

- changes the supported dependency range to `transformers>=4.45,<5.0`;
- adds a fail-fast runtime message when Transformers 5.x is detected;
- preserves the previous human-calibration bundle-selection fix.

It does not change HHEM scores, calibration labels, thresholds, or the
production A.L.I.C.E. response pipeline.
