# Current continuation — a3 closed after native runtime failure

Read [STATE](STATE.json), the [actual a3 review](QWEN_V105_RESULT_REVIEW.json),
and the [master handoff](ASTRA_MASTER_CONTINUATION_HANDOFF.md). This checkpoint
supersedes older launcher recommendations.

V105 ran as job 575182 and exited 76. Its pinned Ollama binary could not load:
the host libc did not provide GLIBC_2.28. It never became ready; no CPU probe or
calibration task was attempted. Qwen remains NOT_EVALUATED. A1, a2 and a3 are closed.

Astra missed a known host constraint. Earlier context already recorded CentOS 7
with glibc 2.17. Kaggle judge success and Magnolia Python/infrastructure tests
did not establish Ollama compatibility. Model files were downloaded before
native executable startup was proven. Preserve those verified files for future
validated reuse. The tar zstd warning was recovered by the pinned decoder;
the terminal cause is the retained native loader error.

Telemetry now works: all five a3 publications returned zero, and all 25 GitHub
snapshot blobs match the returned publisher journals. The actual result pointer
records a3 at context commit a641981. Its rebuilt summary matches exactly.

Next: the [runtime route survey](MAGNOLIA_RUNTIME_ROUTE_RELEASE.json). Save its
ZIP and Start-ALICEMagnoliaRuntimeRouteV100.ps1 together in Downloads, run the
launcher, and return TRACE_ZIP and the transcript. Six bounded read commands
inspect host libc, the existing pinned CPU ELF objects, and available container
tools/modules. It uses no allocation, download, model service or inference.
Missing tools remain evidence; selected source and local controller bytes are
checked before and after. Actual Windows/Magnolia survey execution is pending.

Do not rerun V105, reset a closed run or construct a4 yet. No compatible container
route has been observed. Login-node availability will not by itself establish
compute-node compatibility. Any future runtime route must preserve scientific
pins and prove ABI plus empty-service readiness before model preparation, then
the unchanged CPU probe and worst-case budget gate before the 16 requests.

After actual passing calibration, verify the four exact family/profile/digest
receipts before successor binding and breadth prerequisite migration. Preserve
Gemma v182, Mistral/Granite, failed GLM, 287 candidates, 63 replacements and one
deferred slot. Private pointwise, acceptance, promotion, training and MC8 stay
gated. This operational work is not evidence for the identity/memory experiments.
The [Phase 2 roadmap](PHASE2_REPLACEMENT_ROADMAP.md) still requires Stage G, G+,
H, I and J with owner acceptance. Main remains
0abaed85873c3f8de04765847eb7700b0e20433f.
