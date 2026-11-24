# Preserve dispatched artifacts through Git publication

1. The exact staged-byte check failed before commit or push. Original controller
   `work/publish_conditional_residual_20261104.py` and tool chunk6223c6 preserve
   the failure. This valid check did not run or change the Magnolia model.
2. Publication must reproduce the actual reviewed and dispatched inputs, so a
   mathematically equivalent JSON document with different bytes is insufficient.
3. The root `.gitattributes` uses `* text=auto`; Git normalized CRLF when staging.
4. Exact comparison isolated two artifacts: the owner plan contains one CRLF,
   and source-pins JSON contains32. All other staged files match their originals.
5. These inputs already have sealed hashes and one running job. Rewriting them
   would break the evidence and imply a different input from the dispatched one.
6. Preserve the original files and hashes. Disable text conversion only for
   these two exact artifact paths; recognize CRLF as line endings in their diff.
   Future generated pinned JSON should write explicit LF from the outset.
7. [Git's primary attribute documentation](https://git-scm.com/docs/gitattributes)
   defines `-text` as disabling checkin/checkout end-of-line conversion. This
   preserves the original byte contract without changing global configuration.
8. Re-stage these exact files and new metadata, compare every staged and
   committed artifact against its actual bytes, and verify the published head.
   The publication receipt records these checks. No job is resubmitted, source
   pin changed, or scientific test weakened.

The first attribute-aware staging attempt retained Git's already-cached blobs
because the worktree files had not changed. The exact-byte check again refused
publication. `check-attr` confirmed both paths were `text: unset`, and comparison
isolated the same two blobs. Git's documented explicit `add --renormalize` was
then applied only to those two paths under the new attributes. All21 staged
artifacts subsequently matched their actual bytes. Both failed controllers
remain preserved; no commit or push preceded this verification.
