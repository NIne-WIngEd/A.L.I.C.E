# Phase 1.11 — Personal Claim Validity Contract

Failure forensics showed that factual-consistency models were being asked to
solve problems outside factual consistency.

Three recurring failure classes were identified:

1. **Personal subject/scope mismatch**
   - A personal benchmark question produced claims about an unrelated textbook
     author.
   - Such claims can be entailed by retrieved text while still being invalid as
     facts about the vault owner.

2. **Relative-time leakage**
   - A dated diary statement such as "training finished today" was copied into
     a timeless claim.
   - Source-relative words such as today/tomorrow/yesterday are rejected unless
     generation rewrites the event with an absolute date.

3. **Self-record claim-strength mismatch**
   - A resume/self-record statement claiming that work "accelerated R&D cycles"
     was treated as a verified impact claim.
   - P1.11 now prefers concrete roles, responsibilities, and actions and
     conservatively rejects selected promotional/causal impact language when
     the only cited provenance is an owner self-record.

The contract is enforced before the existing entailment gate. Factual-
consistency models remain useful for premise-hypothesis support, but are not
expected to determine source ownership, temporal anchoring, or whether a
self-authored impact statement has sufficient corroboration.

This change remains private, read-only, and offline. It does not change the
production gate by itself.
