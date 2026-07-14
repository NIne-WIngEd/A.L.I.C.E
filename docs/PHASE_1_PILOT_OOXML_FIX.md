# P1.5 Pilot OOXML Eligibility Fix

The first pilot proposal contained no DOCX, XLSX, or PPTX records because the
risk analyzer treated every ZIP-based package as a generic archive.

Office Open XML documents are intentionally ZIP-packaged and can be safely
classified as document candidates when their internal structure is recognized
as WordprocessingML, SpreadsheetML, or PresentationML.

This patch changes only recommendation logic:

- recognized `.docx`, `.xlsx`, and `.pptx` packages can become
  `pilot_candidate` records;
- literal `.zip` files and unclassified ZIP containers remain
  `specialized_review`;
- `.pth` and other serialized model containers remain specialized because
  their serialized-code flag is unchanged;
- unsafe, corrupt, encrypted, extreme-ratio, and oversized archives retain
  their existing protections.

The patch does not extract, copy, move, execute, approve, or delete any file.
After applying it, rerun inventory analysis and regenerate the P1.5 proposal.
