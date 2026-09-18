# Context Router v1

This file defines how Sol/ChatGPT should locate A.L.I.C.E. context without rereading the whole project.

## Retrieval order

For every substantial A.L.I.C.E. task:

1. Read `catalog/CATALOG_STATUS.json`.
2. Confirm its `source_commit` matches the live `alice-eipm-v1-build` head.
3. Use `catalog/BRANCH_CATALOG.json` to identify whether the task depends on a branch-specific knowledge surface.
4. Use one or more pointer catalogs:
   - `DOCUMENT_CATALOG.jsonl`
   - `FAILURE_LESSON_CATALOG.jsonl`
   - `FRONTIER_RESEARCH_CATALOG.jsonl`
   - `SUPERSESSION_CATALOG.jsonl`
5. For code topology, query Graphify.
6. Open the original source at the recorded branch + path before using a consequential claim.
7. Search private external sources only when the public/source catalogs say the required knowledge class is not sufficiently represented.

## Routing by question type

| Question | First route |
| --- | --- |
| Which implementation owns this behavior? | Graphify code graph |
| Why was this architecture chosen? | DOCUMENT + FRONTIER_RESEARCH catalogs |
| Did we already try/fail at this? | FAILURE_LESSON catalog |
| Is an older rule still active? | SUPERSESSION + branch catalog |
| What happened on Magnolia/Kaggle? | FAILURE_LESSON + runtime receipts |
| What is current N0 state? | source configs + runtime receipts + Graphify |
| What did a prior chat establish that Git does not contain? | private external-source router |
| What does the comic require from the destination system? | private external-source router |
| What did competitor/frontier research conclude? | FRONTIER_RESEARCH catalog, then branch-qualified original source |

## Truth rule

A catalog hit is a pointer, not evidence by itself.

The agent must preserve:
- branch;
- commit/blob;
- declared status;
- supersession;
- owner ratification;
- experimental versus canonical state.

Never flatten branch-specific context into one undifferentiated truth set.

## Miss rule

If the expected source class is absent, say retrieval is incomplete and route to the private/external layer. Do not substitute a nearby document merely because it sounds related.
