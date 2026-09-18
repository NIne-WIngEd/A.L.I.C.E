# Sol Graphify Retrieval Protocol

This protocol exists to stop repeated full-context reconstruction during A.L.I.C.E. work.

## Default behavior for a new A.L.I.C.E. session

When a current Graphify output is available:

1. Identify the active task and the repository ref that the user is actually working on.
2. Query the graph for the task, current gate, dependencies, and prior failure lessons.
3. Read the smallest relevant graph neighborhood or wiki pages.
4. Follow source pointers to original repository material for anything consequential.
5. Verify current branch/commit state before issuing execution commands.
6. Work from verified sources rather than rereading the entire repository.
7. If the graph misses required context, record the miss instead of compensating by permanently reverting to full-repo rereads.

## Trust ordering

For this research workflow:

1. Owner-ratified directives and canonical A.L.I.C.E. authority records.
2. Current original source files, receipts, evaluations, and commits.
3. Historical source files with valid temporal relevance.
4. Graphify EXTRACTED relationships as navigation hints.
5. Graphify INFERRED relationships as hypotheses only.
6. Generated summaries and graph reports as orientation only.

A lower layer cannot override a higher layer.

## Temporal rule

A relevant old fact is not automatically a current fact.

Before acting on a retrieved item, check:
- branch/ref;
- date or commit;
- whether a newer record supersedes it;
- whether the condition was temporary;
- whether the item was an experiment rather than a ratified decision.

## Failure rule

If a command or workaround already failed in prior work, retrieval must surface that failure before proposing a materially similar action.

If the graph does not surface it, that is a retrieval defect to log.

## Token discipline

Do not ingest broad folders merely to feel confident.

Escalate source reads only when:
- the graph leaves ambiguity;
- authority cannot be established;
- the current task crosses several subsystems;
- a high-risk execution step depends on exact operational history.

The target is **minimum sufficient verified context**, not minimum context at any cost.

## Current limitation

ChatGPT cannot directly attach itself to a local stdio MCP server from inside an ordinary chat. For now, Graphify outputs must be persisted somewhere accessible across chats, such as this research branch. The branch acts as the durable bridge until a direct supported integration is available.
