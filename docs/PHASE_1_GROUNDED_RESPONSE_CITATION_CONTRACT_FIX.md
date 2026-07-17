# P1.11 Citation Contract and Contradiction Handling Fix

The first full grounded-response benchmark exposed two verifier-contract issues.

1. Structured output required citation values to be strings, but did not
   constrain those strings to exact package IDs. A model could therefore return
   `S1` while the verifier accepted only `[S1]`.
2. The verifier rejected a grounded answer whenever any retrieved source carried
   an unresolved contradiction label, even if that contradiction did not affect
   the answer.

This patch:

- builds a response JSON schema whose citation fields enumerate the exact
  package-local citation IDs;
- canonicalizes `S1`, `[S1]`, `1`, and simple equivalent forms to `[S1]`;
- normalizes bare inline `S1` citations to bracketed form when the ID exists;
- deterministically preserves every unresolved contradiction group in the
  response package;
- no longer forces `answer_type=contradictory_evidence` merely because an
  unrelated contradiction exists somewhere in the retrieved context;
- advances the evaluation checkpoint key to v3 so failed v2 cases are not
  incorrectly resumed.

All deterministic verification and read-only guardrails remain enabled.
