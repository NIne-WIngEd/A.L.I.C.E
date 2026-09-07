# Magnolia Runtime Route — Observed Result Summary

**Observed:** 2026-09-07  
**Source run:** `alice-qwen38-a3-750881d854fa`  
**Closed scheduler job:** `575182`  
**Trace ZIP SHA-256:** `11DBC2BD6B7ED7A3BE0CD333986EFB72DBB403D03639DCD8EDFE714202218DFF`

## Read-only observations

- Login host: `Linux 3.10.0-957.el7.x86_64`.
- GNU libc: `glibc 2.17`.
- The pinned prebuilt Ollama executable contains a `GLIBC_2.28` requirement.
- Bundled native libraries also contain requirements newer than the host baseline, including `GLIBC_2.27`.
- `apptainer`, `singularity`, `podman`, `bwrap`, and `proot` were not available in the observed login-node PATH.
- Modules expose CMake `3.24.2` and GCC `11.4.0`, `12.3.0`, and `13.2.0`.
- Observed Go modules are only `1.9.3`, `1.11.1`, and `1.13.4` (default). A modern Go toolchain was not established.
- 25 runtime ELF objects were hash-verified before inspection.
- The one nonzero command was the deliberate tool-presence query because the listed container tools were absent. Overall survey status was `COLLECTED`.

## Preservation proof

The survey made:
- model jobs submitted: **0**
- model downloads: **0**
- model services started: **0**
- inference requests: **0**
- runtime/system modifications: **0**
- package staging on Magnolia: **false**

Local a1/a2/a3 controller-state hashes were unchanged before/after. The selected closed a3 evidence files were unchanged.

## Scientific interpretation

`scientific_outcome = NOT_EVALUATED`.

This is an infrastructure/runtime compatibility result only. It does not qualify or reject Qwen. a1, a2, and a3 stay closed. Do not rerun V105.

## Next engineering question

Before another execution package, independently verify the exact source-build requirements of the pinned Ollama version and whether a reproducible Magnolia-compatible build can target glibc 2.17 while preserving the frozen Qwen model/profile/request contract.
