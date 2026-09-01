# A.L.I.C.E. MC10B Full E-INF Frontier v1.1.0

**Release class:** clean controller redesign after the v1.0.x preflight failures. This package supersedes v1.0.0, v1.0.1, and v1.0.2. Do not use those older packages.

## Scope

This stage generates the remaining **60 MC10A E-INF-eligible packets** after the already-audited five-packet MC10B1 pilot. The frozen canonical generation procedure remains **4 generation methods × 3 seeds = 12 GPT-OSS 20B proposals per packet**, for **720 raw proposals**. UNKNOWN remains a separate competitor for every packet.

The run has six telemetry checkpoints, one after each 10 packets / 120 proposals. Telemetry is diagnostic only. It has no identity or memory authority.

This package does **not** accept E-INF, generate A-SYN, train a model, close MC10B, open MC10C, close Stage G, activate Stage H, or replace Phase 2.

## Why v1.1.0 exists

The v1.0.1 launcher masked a failed CPU preflight by reading a success-only property under PowerShell StrictMode. v1.0.2 then exposed the actual deterministic transport mismatch: Kaggle was looking for the frozen transport contract name `mc10b1-private-input.bin`, while the full-frontier packaging path had used a different name. The GPU generation kernel was never submitted in either failed full-frontier attempt, so there is no generated frontier state to salvage.

v1.1.0 was rebuilt around one Python controller rather than further patching the old PowerShell orchestration. The controller derives the private blob name from the frozen transport module and performs local authority, dataset, CPU preflight, resume, telemetry, runtime, finalization, and cleanup validation as a single versioned contract.

## Deep qualification

Before release, the package is clean-extracted and requalified. The included qualifier verifies exact package membership, SHA-256 hashes, frozen bytes, Python syntax, JSON/JSONL parsing, the minimal PowerShell contract, 16 deep regression tests, and absence of generated cache artifacts.

The regression suite covers the known failure classes plus deterministic transport round-trip, exact Kaggle dataset file contracts, current Kaggle CLI file-list parsing, real MC10B1 pilot-audit binding, torn-final-record recovery, middle-record corruption rejection, all six telemetry blocks, telemetry tamper/ack rejection, a synthetic 60-packet/720-candidate final package, standalone and independent final validation, exact final file set enforcement, atomic publication/idempotency, deterministic-versus-transient CPU preflight retry policy, privacy guards, remote evidence preservation, and generation soft-deadline behavior.

No script can guarantee that Kaggle, GitHub, networking, GPU allocation, or model inference will never fail. v1.1.0's contract is that known deterministic defects are qualified out, external failures fail closed, private evidence is preserved when recoverable, and a failure does not silently advance authority.

## Run

Use Windows PowerShell from `C:\A.L.I.C.E-main`. Verify the outer ZIP SHA supplied with the release, clean-extract it, let Windows PowerShell parse the launcher, then execute it.

Normal invocation:

```powershell
& $Launcher[0].FullName -MaxGenerationMinutes 480
```

A normal time checkpoint prints `next_action=RERUN_SAME_PACKAGE_TO_RESUME`; rerun the same command. A telemetry pause prints a block number. Review that block before explicitly resuming with:

```powershell
& $Launcher[0].FullName -MaxGenerationMinutes 480 -AcknowledgeTelemetryBlock <BLOCK_NUMBER>
```

Do not pre-acknowledge future telemetry blocks.

## Pinned authority / compute

- Repository baseline: `0abaed85873c3f8de04765847eb7700b0e20433f`
- Live branch: `alice-mc10b-live`
- Compute: private Kaggle, NVIDIA Tesla T4, pinned Ollama runtime
- Canonical proposer: GPT-OSS 20B, proposal-only / zero acceptance authority
- Qwen reserve calls: zero
- Local machine: orchestration only, no model inference
