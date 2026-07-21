# Phase 1.11 — Personal Claim Scope Regression Fix

The first personal-claim validity contract enforced a hard requirement that
every claim produced for a first-person query begin with "The user" or
"The user's".

That rule was too broad for the deterministic validation layer. It broke an
existing citation-cleanup unit test whose synthetic claim text is intentionally
subject-neutral ("Valid claim.").

The corrected boundary is:

- generation is still instructed to phrase personal claims self-containedly as
  "The user..." or "The user's...";
- deterministic validation rejects explicit third-party subjects such as
  author, reader, student, instructor, professor, or teacher when the query is
  personal;
- subject-neutral synthetic claims remain available to downstream independent
  validators and citation-contract unit tests.

This preserves the personal-018 textbook-author failure protection without
coupling unrelated citation validation to a stylistic prefix requirement.
