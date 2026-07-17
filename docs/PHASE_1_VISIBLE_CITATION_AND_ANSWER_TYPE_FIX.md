# P1.11 Visible Citation and Answer-Type Consistency Fix

The v6 diagnostic identified the deterministic verification failure:

`Answer contains claims but no inline citations`

The structured `claims` array already had 100% citation coverage, but the
model's free-form `answer` field sometimes omitted visible `[S#]` references.

This patch does not weaken verification.

Instead:

1. If the free-form answer contains no visible citations but structured claims
   are present, A.L.I.C.E. deterministically renders the visible answer from the
   already-cited structured claims.
2. If the model returns `contradictory_evidence` while the context package has
   zero actual contradiction groups, the normalized answer type becomes
   `grounded` when grounded claims exist, otherwise `insufficient_evidence`.
3. Model-invented contradiction notes are removed unless their labels exist in
   the context package.
4. The verifier explicitly rejects raw `contradictory_evidence` responses when
   no real context contradiction exists.
5. Private benchmark details now retain verifier error strings and visible
   citation counts for future debugging.

The evaluation checkpoint contract advances to v7.
