# Graphify / Context Substrate — Cross-Chat Usage Handoff

Date: 2026-09-17  
Purpose: let future A.L.I.C.E. chats recover the right context quickly without rereading the entire repository, every branch, old PDFs, and prior conversations.

## Read this first

**The context substrate is ready for normal A.L.I.C.E. development workflow.**

The important wording is **context substrate**, not Graphify alone.

Graphify is the structural **code-navigation layer**. It is intentionally not the source of truth, not A.L.I.C.E. memory, not an authority system, and not part of the A.L.I.C.E. trusted runtime.

The working context system is:

```text
live Git branches
        |
        +--> branch/freshness catalog
        +--> full-document lexical index
        +--> failure/research/supersession catalogs
        +--> active continuity + experiment-frontier state
        |
        +--> Graphify exact-branch code graph when code topology is needed
        |
private ChatGPT Library / saved artifacts
        |
        +--> comic, prior chats/PDFs, private handoffs, local/private evidence
        |
        v
task-specific source pointers
        |
        v
open and verify ORIGINAL sources
        |
        v
compact context capsule for Sol
        |
        v
actual A.L.I.C.E. work
```

Do **not** replace this with a full-repo/full-PDF reread at the beginning of every chat.

Do **not** treat a Graphify result, catalog result, or generated capsule as canonical truth by itself.

---

## Hard boundary

Graphify and the context-substrate research branch are external development scaffolding.

They must not silently become:

- A.L.I.C.E. runtime memory;
- A.L.I.C.E. identity authority;
- a canonical claim store;
- a replacement for source receipts;
- a replacement for Git history;
- a replacement for owner-ratified directives;
- a reason to import third-party Graphify architecture into A.L.I.C.E.

Research lessons from this substrate may later inform A.L.I.C.E.'s native continuous-context design. That is a separate architecture decision.

---

## Where the substrate lives

Research branch:

`research/graphify-context-substrate`

Primary files:

- `research/graphify-context/catalog/CATALOG_STATUS.json`
- `research/graphify-context/catalog/ACTIVE_MISSION_STATE.json`
- `research/graphify-context/catalog/BRANCH_CATALOG.json`
- `research/graphify-context/catalog/DOCUMENT_CATALOG.jsonl`
- `research/graphify-context/catalog/FAILURE_LESSON_CATALOG.jsonl`
- `research/graphify-context/catalog/FRONTIER_RESEARCH_CATALOG.jsonl`
- `research/graphify-context/catalog/SUPERSESSION_CATALOG.jsonl`
- `research/graphify-context/catalog/lexical/LEXICAL_STATUS.json`
- `research/graphify-context/QUERY_REQUEST.json`
- `research/graphify-context/QUERY_RESULT.json`
- `research/graphify-context/CONTEXT_CAPSULE.json`
- `research/graphify-context/CONTEXT_CAPSULE.md`
- `research/graphify-context/LAST_GRAPH_BUILD.json`

Supporting protocol files:

- `research/graphify-context/CONTEXT_ROUTER.md`
- `research/graphify-context/PRIVATE_EXTERNAL_SOURCE_ROUTING.md`
- `research/graphify-context/COVERAGE_GATES.md`
- `research/graphify-context/READINESS_AUDIT_2026-09-17.md`

Private Library routing manifest:

`/ALICE Context Substrate/ALICE_EXTERNAL_SOURCE_MANIFEST.json`

That private manifest is intentionally not copied into public Git.

---

## Readiness validation snapshot

At the final validation in this context session:

- all **30 / 30 pushed non-temporary Git branches** were classified as branch-qualified knowledge surfaces;
- `alice-telemetry` is included even though it has unrelated Git history;
- **2,115** branch-qualified text documents were indexed;
- **0 documents were skipped** by the lexical index;
- **867** failure/lesson pointers were available;
- **397** frontier/research pointers were available;
- **221** supersession pointers were available;
- the full-body lexical index contained **46,095 searchable terms** after common-term filtering;
- the lexical index used **no external model**;
- Graphify was pinned to `graphifyy==0.9.63`;
- Graphify extraction used an **exact source worktree**;
- a stale Graphify graph was deliberately tested against a newer experiment frontier;
- the query workflow detected the SHA mismatch and rebuilt the graph from the live frontier automatically;
- the resulting `LAST_GRAPH_BUILD.json` recorded the exact live branch and commit;
- the latest freshness test found the expected current implementation file rather than continuing to use the stale graph.

These counts are a validation snapshot, not permanent constants. The workflow is designed to refresh them as branches move.

---

## Branch coverage rule

The context catalog no longer guesses which A.L.I.C.E. branches are important.

Every pushed branch is a context source **except**:

- `research/graphify-context-substrate` itself;
- branches beginning with `tmp-`.

Each branch remains qualified by:

- branch name;
- head SHA;
- last commit time/message;
- relationship to the stable build;
- original file path/blob SHA.

Unrelated Git histories are kept explicitly unrelated. They are not forced into a fake ancestry.

A branch-specific file is never silently promoted to canonical truth.

### Local-only limitation

A branch or file that exists only on a local machine and has never been pushed is not visible to the GitHub context substrate.

If a task depends on unpushed/private local material, retrieve that material through the appropriate private/local source instead of pretending Git contains it.

---

## Freshness behavior

Before each query, the query workflow fetches all pushed context branches and checks their heads.

If a context-bearing branch:

- moves;
- appears;
- disappears;
- or no longer matches the recorded head;

the branch/document/lexical catalogs are rebuilt before the query continues.

This is important. A fresh stable build does not imply that experiment, telemetry, context, or research branches are fresh.

---

## Stable build vs active experiment frontier

The context compiler keeps these concepts separate:

1. **stable build base**;
2. **latest continuity handoff**;
3. **unmerged experiment frontier**;
4. **canonical/ratified authority**.

Do not collapse them.

For current EIPM work, the compiler finds ahead-only `alice-eipm-v1-*` branches and identifies maximal experiment heads by ancestry.

If exactly one maximal head exists, a code-topology query can automatically target it.

If multiple maximal heads exist, the workflow must fail closed rather than choose one arbitrarily. Inspect the competing branches and set `graph_source_branch` explicitly only after resolving which source is relevant.

An unmerged experiment head remains **non-canonical** even when it is the newest scientific state.

---

## Active handoff selection

The active N0 continuity handoff is selected by the file's **actual Git last-change timestamp**, not alphabetic filename order.

This was added after discovering that two same-day handoffs could otherwise select the wrong one.

The active handoff is a navigation/continuity overlay. Open the underlying source before executing consequential actions.

---

## Document/history retrieval

Do not use Graphify for everything.

The public Git document layer is deterministic and branch-qualified.

It includes:

- Markdown;
- JSON / JSONL;
- TXT;
- YAML;
- TOML;
- TSV;
- relevant receipts/config/state documents.

It has:

- a document catalog;
- a failure/lesson catalog;
- a frontier-research catalog;
- a supersession catalog;
- a full-document lexical index.

The lexical index searches document bodies, not just filenames/headings. It was specifically validated on a technical failure string buried inside a Magnolia receipt.

No embedding/LLM backend is required for this public-document index.

### Use this route for

- architectural rationale;
- research findings;
- competitor/frontier-model analysis;
- old failures;
- Magnolia/Kaggle lessons;
- receipts;
- supersession;
- current/old handoffs;
- policy/config history.

---

## Graphify code route

Use Graphify only when code topology is actually useful.

Examples:

- "where in code is this implemented?";
- "which function calls this?";
- "which module owns this behavior?";
- "what implementation path connects A to B?".

The query workflow uses `graphify_mode=auto` by default.

For document/history questions, Graphify is skipped. This saves time/tokens and prevents irrelevant code nodes from polluting context.

For code questions:

1. select the relevant live branch;
2. compare the cached graph branch+SHA to that branch;
3. rebuild if mismatched;
4. extract from an **exact detached worktree** for that SHA;
5. store the source branch+SHA in `LAST_GRAPH_BUILD.json`;
6. query that graph.

Graphify is pinned to:

`graphifyy==0.9.63`

Do not silently upgrade the package during routine work.

---

## Graphify trust rule

**Graphify nodes are navigation hints only.**

Natural-language graph seeding can still return generic or irrelevant nodes.

Known examples from qualification included generic nodes such as:

- `Path`;
- `_model()`;
- unrelated tests.

Therefore:

```text
Graphify result
    !=
verified project fact
```

A consequential code claim must be checked in the original file at the exact branch/commit returned by routing.

Prefer:

```text
task
 -> candidate entities
 -> exact branch/SHA
 -> typed code neighborhood
 -> original source file
 -> verified conclusion
```

Never:

```text
natural-language Graphify result
 -> assume truth
 -> execute
```

---

## Source authority rule

Catalogs and Graphify route to evidence. They do not replace evidence.

Use roughly this trust ordering:

1. owner-ratified directives / canonical authority;
2. current original source files, receipts, evaluations, and commits;
3. temporally valid historical sources;
4. branch-qualified working/experimental sources;
5. Graphify EXTRACTED/code relations as navigation;
6. inferred/generated summaries as hypotheses/orientation.

Preserve:

- branch;
- commit/blob;
- timestamp;
- status;
- ratified vs experimental;
- canonical vs working;
- superseded vs active;
- failed vs successful;
- infrastructure failure vs model evidence.

If two sources conflict, do not average them together. Resolve temporal/authority/supersession state.

---

## Private sources

Some of the most important A.L.I.C.E. context is intentionally outside public Git:

- the comic;
- prior ChatGPT exports/handoffs;
- private PDFs;
- private owner context;
- local execution transcripts that were never committed;
- other saved Library artifacts.

When `private_external_route_recommended=true`, search the user's private ChatGPT Library.

Use:

`/ALICE Context Substrate/ALICE_EXTERNAL_SOURCE_MANIFEST.json`

as the routing manifest.

The private search layer has already been validated against:

- the A.L.I.C.E. comic;
- old Magnolia/udocker lessons;
- PowerShell/Kaggle JSON lessons;
- old EIPM/frontier research handoffs.

Do not copy those private payloads into the public research branch just to make Graphify see them.

If the same claim was later promoted into Git, prefer the Git authority and use the private source as historical context.

---

## Operational-command safety

For Magnolia, Kaggle, PowerShell, SSH, Slurm, remote paths, or similar operational questions:

**never issue a command from one old Git document alone.**

The router intentionally recommends checking private operational history for these questions because newer hard-won fixes may exist outside an older source document.

Before generating a command:

1. retrieve current branch/current state;
2. retrieve known failure lessons;
3. verify the latest relevant original source;
4. check private operational history when recommended;
5. avoid any route already recorded as dead/obsolete;
6. never reinterpret infrastructure failure as model evidence.

This rule exists because old valid setup docs can become operationally stale.

---

## Query workflow for future chats

A future Sol chat should normally start here instead of reconstructing the project.

### 1. Read minimal substrate state

Read:

- `CATALOG_STATUS.json`;
- `ACTIVE_MISSION_STATE.json`.

Do not start with the whole repository.

### 2. Submit a task-specific query

Update:

`research/graphify-context/QUERY_REQUEST.json`

with a unique request ID and the actual question.

Recommended shape:

```json
{
  "request_id": "unique-task-id",
  "question": "the exact context question needed for the work",
  "budget": 1800,
  "traversal": "bfs",
  "contexts": ["call", "import"],
  "graphify_mode": "auto"
}
```

Do not add `graph_source_branch` unless branch ambiguity actually requires an explicit choice.

### 3. Wait for the Graphify Context Query workflow to succeed

The workflow may:

- refresh branch/doc catalogs;
- skip Graphify entirely;
- reuse a fresh graph;
- or rebuild Graphify from the exact current branch.

### 4. Read the result

Read:

- `QUERY_RESULT.json`;
- `CONTEXT_CAPSULE.json` or `CONTEXT_CAPSULE.md`.

### 5. Verify sources

Open the top original branch/path sources required for the actual decision.

The capsule is not enough by itself.

### 6. Follow private routing when required

If the result recommends private/external routing, search Library before concluding context is complete.

### 7. Do the actual work

Once the minimum authoritative context is verified, stop context gathering and work.

That is the entire point of this substrate.

---

## Do not reuse an old capsule blindly

`CONTEXT_CAPSULE.json` is task-specific.

A capsule generated for one question is not a universal project summary.

If the next task materially changes, issue a new query.

This avoids the exact failure mode where a compact summary becomes stale and is repeatedly treated as complete context.

---

## Token-saving rule

The default behavior is:

```text
query first
 -> retrieve a small candidate set
 -> verify originals
 -> work
```

Not:

```text
read repo
 -> read every branch
 -> read old PDFs
 -> reconstruct project
 -> run out of context
```

A full reread is allowed only when there is evidence that the context substrate missed something materially important.

When that happens:

1. identify the specific miss;
2. retrieve the missing source;
3. record the miss as a context-substrate research lesson;
4. improve routing/indexing if appropriate;
5. do not normalize full rereads as the default workflow.

---

## Fail-closed conditions

Stop and resolve context before consequential work if:

- branch freshness check fails;
- multiple maximal experiment heads make code-source selection ambiguous;
- Graphify source SHA does not equal the selected live branch SHA;
- a required original source cannot be opened;
- a private source class is required but not retrieved;
- current and historical sources conflict without a resolved supersession relationship;
- the query returns only generic/noisy Graphify entities;
- a command would rely on a previously failed/obsolete operational route;
- an unpushed/local-only source is required but unavailable.

Do not fill those gaps with inference.

---

## Race/concurrency protection

Earlier Graphify publication failed when the research branch advanced while a graph was building.

The current publishers preserve generated artifacts, fetch the newest research head, re-check source state, and refuse stale publication.

A publication/push failure does not automatically mean Graphify extraction failed. Check the workflow step that failed.

---

## What this substrate still does NOT guarantee

It does not mean "all context is permanently inside Graphify."

It does not automatically know:

- unpushed local work;
- every private conversation without Library retrieval;
- private local files that have never entered an accessible source;
- the authority meaning of a source without the surrounding A.L.I.C.E. governance rules.

It does not make natural-language graph retrieval perfectly precise.

It does not eliminate source verification.

Those are deliberate boundaries, not bugs.

---

## Readiness verdict

**READY FOR NORMAL CROSS-CHAT A.L.I.C.E. DEVELOPMENT USE.**

Use the context substrate as the first retrieval layer.

Use Graphify as code navigation.

Use the document/branch/lexical catalogs for public history and rationale.

Use private Library routing for private/external history.

Then open the actual authoritative sources and do the work.

The substrate should save context and tokens by preventing repeated mass ingestion. It should not save tokens by lowering evidence quality.

If retrieval becomes uncertain, fail closed and verify more source material rather than confidently using the wrong context.

---

## Boundary for the chat that created this handoff

The chat that produced this file was a **context-infrastructure chat**.

Its job ends with validating this substrate and documenting how other chats should use it.

Actual A.L.I.C.E. model development belongs in the normal A.L.I.C.E. development chats after they retrieve context through this workflow.
