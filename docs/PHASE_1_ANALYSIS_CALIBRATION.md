# P1.4 Calibration Patch

This patch recalibrates the initial file-signature analysis after the first
real dataset run.

Changes:

- Treat `.jpg`, `.jpeg`, and `.jfif` as the same JPEG/JFIF family.
- Treat `.ipynb` and `.webmanifest` as JSON-family formats.
- Treat `.atom` as an XML-family format.
- Record identified `.dat` and `.bin` contents as `opaque_identified`
  instead of ordinary extension mismatches.
- Record `.pth`/`.pt` files detected as ZIP containers as
  `serialized_container`; they remain under specialized review.
- Detect ISO Base Media files from the leading `ftyp` box before relying on
  a generic magic-number result.
- Downgrade `.xxx` signature results to `ambiguous` instead of trusting them.
- Record zero-byte files as `empty` rather than signature errors.
- Add regression tests for all of the above behavior.

The patch does not move, extract, execute, quarantine, or delete any file.
