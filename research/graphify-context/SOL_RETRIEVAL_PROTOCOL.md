# Sol Graphify Retrieval Protocol

This protocol exists to stop repeated full-context reconstruction during A.L.I.C.E. work.

## Default behavior for a new A.L.I.C.E. session

When a current Graphify output is available:

1. Identify the active task and the repository ref that the user is actually working on.
2. Verify that the Graphify build receipt names the current `alice-eipm-v1-build` commit.
3. If the source SHA differs, request a source sync and rebuild before querying. Do not use a stale graph.
4. Query the graph for the task, current gate, dependencies, and prior failure lessons.
5. Prefer exact identifiers and typed relation contexts when known.
6. Read the smallest relevant graph neighborhood.
7. Follow source pointers to original repository material for anything consequential.
8. Verify current branch/commit state before issuing execution commands.
9. Work from verified sources rather than rereading the entire repository.
10. If the graph misses required context, record the miss instead of permanently reverting to full-repo rereads.

## GitHub bridge

Until ChatGPT can attach directly to Graphify's local MCP server, the research branch provides a durable bridge:

```text
Sol
  -> research/graphify-context/QUERY_REQUEST.json
  -> Graphify Context Query GitHub Action
  -> graphify-out/graph.json
  -> research/graphify-context/QUERY_RESULT.json
  -> Sol
  -> original-source verification
```

Source freshness uses a separate external path:

```text
alice-eipm-v1-build moves
  -> SOURCE_SYNC_REQUEST.json
  -> Graphify Source Sync Action
  -> merge source branch into research branch
  -> Graphify Context Substrate rebuild
  -> new LAST_GRAPH_BUILD.json
```

The research branch is a retrieval instrument. It is not part of A.L.I.C.E.'s trusted runtime.

## Freshness gate

`LAST_GRAPH_BUILD.json` must record:

- `source_work_branch`
- `source_work_commit`
- `research_branch`
- `research_commit`
- graph mode
- workflow run id
- authority = `navigation-only`

The query workflow fetches the live `alice-eipm-v1-build` head before every query. If the live SHA and recorded source SHA do not match, the query must fail closed.

A stale graph is worse than no graph because it can make an old implementation look current.

## Trust ordering

For this research workflow:

1. Owner-ratified directives and canonical A.L.I.C.E. authority records.
2. Current original source files, receipts, evaluations, and commits.
3. Historical source files with valid temporal relevance.
4. Graphify EXTRACTED relationships as navigation hints.
5. Graphify INFERRED relationships as hypotheses only.
6. Generated summaries and graph reports as orientation only.

A lower layer cannot override a higher layer.

## Retrieval rule

Use a two-stage process:

### Discovery

Use broad graph search to locate likely modules, scripts, tests, and symbols.

### Precision

Narrow with:
- exact node or file identifiers;
- relation contexts such as calls/imports;
- path/explain/affected-style traversal where useful;
- mission-local scope;
- temporal and authority checks outside the graph.

Do not solve noisy retrieval by simply increasing the token budget.

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

The first operational graph is code-only. It solves code-topology reconstruction but not the full continuity problem because architectural rationale, chat history, decisions, supersession, Magnolia lessons, and other semantic material live outside code topology.

The semantic/history layer must be added without publishing private ALICE material to the public repository. That layer remains the next substrate step.
