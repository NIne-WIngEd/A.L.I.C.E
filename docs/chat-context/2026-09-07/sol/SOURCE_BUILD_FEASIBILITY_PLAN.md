# Verified Source-Build Feasibility Boundary — 2026-09-07

This note records the upstream requirement check performed after the read-only Magnolia runtime-route survey. It does **not** report a successful source build. The remote build probe is still pending.

## Exact upstream facts

Pinned Ollama authority remains `v0.32.15`, commit:

`b7871fc0d1d82fe109536efa3e0e8e411c766c75`

The exact tag's `go.mod` declares:

`go 1.26.0`

The exact tag's development instructions require Go, CMake 3.24+, and a Linux GCC/Clang C/C++ compiler. Its root CMake file also declares CMake 3.24 and C++17. The normal non-macOS-arm64 local build is CPU-only unless GPU backends are explicitly selected.

The exact tag's `LLAMA_CPP_VERSION` is `b10488`. That tag resolves to full llama.cpp commit:

`9d77fa17254e1dee4b9e92504c91611a60b1359f`

Ollama's CMake superbuild uses `CGO_ENABLED=1` for the Go binary and builds the native llama-server payload.

Official Go 1.26.0 Linux amd64 archive:

`go1.26.0.linux-amd64.tar.gz`

SHA-256:

`aac1b08a0fb0c4e0a7c1555beb7b59180b05dfc5a3d62e40e9de90cd42f88235`

## What this means for Magnolia

The prior survey observed CMake 3.24.2 and GCC 11.4.0/12.3.0/13.2.0, so those parts of the source-build prerequisites are available. The observed Go modules stop at 1.13.4, which is not sufficient for this Ollama tag. Therefore the route must use a separately hash-verified Go 1.26.0 toolchain rather than the Magnolia Go module.

A source build is **plausible, not yet proven**. Because Ollama contains CGO/native code, the only acceptable next proof is a real Magnolia CPU build followed by ELF ABI inspection and an empty-service readiness check.

## Prepared bounded probe

Package:

`ALICE_MAGNOLIA_SOURCE_RUNTIME_FEASIBILITY_v1.0.0.zip`

SHA-256:

`411791925E9879F51E226ED7BCA7609901243D2F882907216B14B2253870B77B`

Launcher:

`Start-ALICEMagnoliaSourceRuntimeFeasibilityV100.ps1`

SHA-256:

`FCBC2C46D56B0684C8E46F93A8CFAFB3122B887040799FDF631301E01EE78F61`

The probe creates one bounded CPU infrastructure job only. It downloads no model and performs no inference. It does not reopen a1/a2/a3, does not rerun V105, does not touch MC8, does not mutate canonical main, and grants no downstream authority.

Pass requires exact Ollama/llama.cpp/Go lineage, successful CPU source build, max runtime GLIBC requirement no newer than 2.17, no unresolved shared libraries, and an empty Ollama service reporting version 0.32.15 through `/api/version`.

Do not create Qwen a4 or any other calibration execution until this result has been returned and reviewed.
