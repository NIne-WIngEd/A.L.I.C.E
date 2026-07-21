# P1.11 False Contradiction Label Fix

The v5 benchmark produced `contradictory_evidence` for 10 of 11 queries.

An earlier context metadata inspection showed a retrieved source whose
contradiction value was literally `No`. The grounded-context normalizer treated
only empty/none/null values as absence of a contradiction, so `No` became an
actual unresolved contradiction-group label.

That can turn metadata meaning "no contradiction" into a contradiction group
named `No`, which is then surfaced to the response model.

This patch treats false-like metadata values as no contradiction:

- empty
- none/null
- no
- false
- 0
- N/A / NA
- not applicable

Real labels such as `project-status` are preserved.

The response-evaluation checkpoint contract advances to v6 so prior v5
responses generated with false contradiction groups are not resumed.
