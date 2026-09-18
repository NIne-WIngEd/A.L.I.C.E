# Graphify Context Substrate Readiness Audit — 2026-09-17

## Verdict

**NOT READY as the sole A.L.I.C.E. continuity/context substrate.**

The current Graphify deployment is ready and useful as a **current-branch code-topology navigator**. It is not yet sufficient to replace full-context reconstruction for A.L.I.C.E. development because major authority, history, rationale, external-material, and cross-branch surfaces are outside the graph.

This is a coverage finding, not a failure of Graphify as a tool. Graphify remains useful as one retrieval layer.

## Current deployed graph

Source:
- work branch: `alice-eipm-v1-build`
- source commit: `021c5021a98104b35f9c8e94b19e48d21f25f132`
- mode: `code-only`
- authority: `navigation-only`

Last observed extraction:
- 935 code-classified files
- 10,832 nodes
- 37,624 edges
- 322 communities
- 255 non-code files explicitly skipped by `--code-only`
- 49 files unclassified/skipped by Graphify

The source-tree inventory at that commit contains 1,238 blobs, including:
- 736 `.py`
- 232 `.md`
- 141 `.json`
- 57 `.sh`
- 34 `.sbatch`
- 11 `.jsonl`
- 11 `.txt`

The repository contains 233 files under `docs/`.

## Coverage audit

### 1. Current code topology — PARTIAL PASS

Graphify is useful for:
- imports/calls/classes/functions;
- locating N0 implementation components;
- identifying candidate source files;
- current-branch code relationships;
- code/config JSON relationships where supported by the structural parser.

Observed limitation:
- natural-language seeding can still resolve generic names such as `Path` and `_model()` to unrelated areas;
- typed traversal and exact entity resolution are required before trusting the returned neighborhood.

Disposition:
**Use now, with source verification.**

### 2. Markdown architecture, doctrine, receipts, failures, runtime reports — FAIL

The current workflow intentionally runs:

`graphify extract . --code-only`

Therefore the semantic document layer is absent.

Important skipped material includes, among many others:
- `README.md`
- `docs/CHAT_ARCHITECTURE_DECISION_LEDGER.md`
- `docs/RESEARCH_FRONTIERS.md`
- `docs/MEMORY_ARCHITECTURE_V4.md`
- `docs/MEMORY_EXTERNAL_SYSTEMS_REVIEW.md`
- `docs/eipm/EIPM_N0_FRONTIER_RESEARCH_AND_BOOTSTRAP_PLAN_2026-09-13.md`
- `docs/eipm/EIPM_CAPABILITY_LIMIT_AUDIT_2026-09-16.md`
- `docs/eipm/N0_ADAPTIVE_LATENT_POOL_V01_FAILURE_AND_V02_REPAIR_2026-09-16.md`
- Magnolia/Kaggle runtime receipts under `docs/eipm/n0/runtime-results/`.

Disposition:
**Must add a document/history retrieval layer before full-context admission.**

### 3. Git branches and branch-specific knowledge — FAIL

The repository currently exposes 34 branches.

The deployed graph is one checked-out snapshot. It is not a union of branch-specific knowledge and does not preserve branch lineage as first-class context.

Examples relative to `alice-eipm-v1-build`:
- `alice-context`: diverged; 183 commits ahead and 459 behind in the sampled comparison; large branch-specific context surface.
- `fable-builder-model`: diverged; 74 commits ahead and 446 behind.
- `alice-mc10b-live`: diverged; 3,097 commits ahead and 459 behind.
- `alice-mc10c-live`: diverged; 471 commits ahead and 459 behind.
- `alice-eipm-v1-causal-arbitration-binding`: 17 commits ahead.
- `alice-eipm-v1-downstream-arbitration`: 6 commits ahead.
- `alice-eipm-v1-post-arbitration-gate`: 7 commits ahead.

A key example is:
`alice-context/docs/chat-context/2026-09-10/sol/EIPM_NATIVE_FRONTIER_ARCHITECTURE_RESEARCH.md`

Commit:
`bf58df48b1519a92caa4b2f9471ad0153e3bf612`

That document contains frontier/competitor architecture research and supersession rationale that the current Graphify corpus does not contain.

Disposition:
**Need branch-aware manifest + lineage + selective branch retrieval. Do not naïvely merge all branches into one truth graph.**

### 4. Comic / visual destination specification — FAIL

The A.L.I.C.E. comic is not present in the current `alice-eipm-v1-build` source tree. The current repository tree has no corresponding comic HTML/media corpus.

The comic therefore cannot be retrieved from the deployed graph.

Disposition:
**Keep external assets private/outside public Git as required, but create an external-source manifest/pointer layer that can route the context compiler to them without copying private content into the public research branch.**

### 5. Prior chat handoffs and exported PDFs — FAIL

Important operational and architectural knowledge exists in prior chat exports and handoff PDFs outside the Git checkout.

Examples observed during this audit include:
- EIPM frontier-research discussion and architecture rationale;
- Magnolia whole-stage udocker/persistent-path lessons;
- PowerShell/Kaggle JSON ownership lessons;
- historical continuation failures and corrected workflows.

The current Graphify graph has no access to ChatGPT Library or old chat exports unless those sources are deliberately materialized into a separate corpus.

Disposition:
**Need a private external-source retrieval layer. Never copy private payloads into this public branch.**

### 6. Failure lessons — PARTIAL / FAIL for completeness

Some failures have been converted into code, tests, docs, or runtime receipts and therefore exist in Git.

Many other hard-won lessons exist only in:
- old chat PDFs;
- local execution transcripts;
- private context;
- branch-specific notes.

Code topology alone cannot reliably answer:
- what failed before;
- why it failed;
- whether the failure was infrastructure versus A.L.I.C.E. logic;
- whether a workaround is obsolete;
- whether a decision superseded another decision.

Disposition:
**Need explicit failure/lesson retrieval with provenance and supersession.**

### 7. Frontier research / competitor models — PARTIAL / FAIL for completeness

Some current-branch docs contain frontier research.

However, major research material is also stored on `alice-context` and in prior chat exports. The current code-only graph omits both classes.

Disposition:
**Not admitted until branch-aware and external-document retrieval are available.**

### 8. Temporal truth / supersession / authority — FAIL as a Graphify-only capability

Even a full Graphify semantic graph does not by itself establish A.L.I.C.E.'s authority semantics.

A.L.I.C.E. development requires:
- branch/ref awareness;
- commit/time validity;
- canonical versus working recommendation distinction;
- explicit supersession;
- ratified versus experimental distinction;
- temporary gate versus permanent constraint distinction;
- owner-ratified authority ordering;
- provenance preservation.

Graph relationships are navigation aids, not canonical truth.

Disposition:
**Must remain outside Graphify and be enforced by the context compiler / source-verification layer.**

## Admission standard before calling the substrate "ready"

Do not call the overall context substrate ready until all of the following exist:

1. **Code graph:** current Graphify structural graph, freshness-gated.
2. **Document index:** all relevant public repo docs/receipts indexed and searchable without forcing full rereads.
3. **Branch map:** branch heads, ancestry/divergence, branch-specific files, and selective branch retrieval.
4. **Temporal/authority layer:** canonical, working, experimental, superseded, ratified, failed, obsolete, and current states remain distinguishable.
5. **External-source catalog:** private pointers to ChatGPT Library/local/private sources without copying them into public Git.
6. **Failure ledger:** retrievable prior failures and proven fixes.
7. **Mission/frontier state:** current task, active stage, next valid action, blockers, and relevant prior decisions.
8. **Source verification:** material claims are opened in original authority before execution.
9. **Coverage tests:** benchmark queries covering code, docs, branches, failures, frontier research, comic requirements, and operational lessons.
10. **Miss detection:** the system can say that required context is missing rather than confidently acting on incomplete retrieval.

## Working rule until admission

Use:

`Graphify code graph -> locate implementation -> source verification`

Do **not** use:

`Graphify answer -> assume complete A.L.I.C.E. context`

For questions involving history, rationale, Magnolia/Kaggle lessons, frontier research, competitor work, comics, or superseded decisions, route to the relevant Git branch/document/private source in addition to Graphify.

## Research lesson for future A.L.I.C.E. continuous context

This audit validates a core design principle:

**A persistent graph is valuable, but continuity is not graph persistence.**

Continuous context requires a graph plus source authority, temporal state, supersession, mission relevance, external-source routing, failure learning, and context compilation.
