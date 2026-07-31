# Phase 4 P4.10 Live Provider Configuration

P4.10a adds one exact live search provider (`brave-search-v1`) and one exact
credential-free page fetch provider (`controlled-live-http-v1`). It does not
activate the end-to-end conversation path; that occurs in P4.10b.

## Private files and environment

Create this file outside Git:

`C:\ALICE_Vault\config\phase4-live-provider.json`

```json
{
  "provider": "brave-search-v1",
  "country": "US",
  "search_lang": "en",
  "ui_lang": "en-US",
  "safesearch": "off"
}
```

Store the API credential only in the current process environment:

```powershell
$env:ALICE_BRAVE_SEARCH_API_KEY = "<private Brave Search API key>"
```

Never put the credential in JSON, Git, logs, receipts, exception text, command
history, test fixtures, or source-page requests.

## Private preflight

```powershell
py scripts\run_phase4_live_provider_preflight.py `
  --repository-root C:\A.L.I.C.E-main `
  --configuration C:\ALICE_Vault\config\phase4-live-provider.json `
  --output C:\ALICE_Vault\reports\phase4-live-provider-preflight.json
```

The preflight makes one foreground Brave search call. It performs no page
fetch, retry, fallback, persistence, memory write, action, recursive browse, or
background operation. The output is metadata-only and must remain outside Git.
