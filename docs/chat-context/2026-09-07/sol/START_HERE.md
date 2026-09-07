# START HERE — Sol continuation, 2026-09-07

Read in this order:

1. `STATE.json`
2. `SOL_CONTINUATION_HANDOFF.md`
3. `RUNTIME_ROUTE_RESULT_SUMMARY.md`
4. `SOURCE_MANIFEST.json`
5. Prior authority: `../../2026-09-06/astra/ASTRA_MASTER_CONTINUATION_HANDOFF.md`

Current observed boundary: Qwen is **not evaluated**. a1/a2/a3 are closed. V105 must not be rerun. The read-only Magnolia runtime survey proved a native ABI incompatibility between host glibc 2.17 and the pinned prebuilt Ollama runtime. No container route was observed. Modern CMake/GCC modules exist, but a modern Go toolchain was not established.

Do not create another wrapper/hotfix package yet. First verify the exact pinned Ollama source-build/toolchain requirements and then choose one consolidated provider/runtime route that preserves all frozen model, profile, request, provenance, and judge semantics.


## 2026-09-07 verified source-build update

Upstream requirements are now independently bound: Ollama v0.32.15 requires Go 1.26.0 and CMake 3.24+; its pinned llama.cpp tag resolves to the recorded full commit. A bounded Magnolia source-build/ABI/empty-service probe has been prepared but **not executed**.

Read `SOURCE_BUILD_FEASIBILITY_PLAN.md` before running anything. Do not create Qwen a4 until that probe result is reviewed.
