# Phase 4 P4.10b — Live Governed Research Execution

Status: additive implementation profile. P4.6a, P4.7a, and P4.7b remain frozen fixture compatibility profiles.

## Exact path

`explicit research mode → Brave live search → controlled public HTTPS fetch → injection inspection → temporal/freshness analysis → exact extractive grounding → Phase 3 projection → P3.6 pre-commit hook → P4.5b citation validation`

The executor performs one foreground PUBLIC research request. It permits one search call, at most five fetches, and at most two exact-extractive grounded sources. Search and fetch providers are selected through the P4.10a exact no-fallback registry.

## Deliberately inactive

P4.10b does not enable source-body persistence, Phase 5 storage, memory writes, external actions, recursive browsing, retries, provider fallback, authenticated page fetching, or background execution. These are profile boundaries, not permanent capability ceilings.

## Private runtime

The repository contains no API credential, private provider configuration, or model/runtime configuration. A private factory outside Git must export:

```python
build_phase4_live_research_runtime(repository_root: Path, evaluated_at: str)
```

The returned object supplies the exact `LiveInformationResearchExecutor`, `ConversationTurnCommand`, PUBLIC `InformationResearchRequest`, reference time, and grounding creation time. Run it with:

```powershell
py scripts\run_phase4_live_research.py `
  --repository-root C:\A.L.I.C.E-main `
  --runtime-factory C:\ALICE_Vault\config\phase4_live_research_runtime.py `
  --output C:\ALICE_Vault\reports\phase4-live-research-receipt.json `
  --evaluated-at 2026-07-30T00:00:00Z
```

The output is metadata-only. It excludes raw query text, source bodies, model prompts, API credentials, and provider response bodies.


## Bounded candidate rejection

A controlled rejection of one search result, initially
`http_status_rejected`, consumes one of the five fetch attempts and is bound
into the metadata-only live research receipt. The executor then advances to the
next already-returned Brave result. It never retries the rejected URL, changes
provider, rewrites the query, recursively browses, or exceeds the original
search/fetch budget. Unknown transport or policy failures remain fatal.
