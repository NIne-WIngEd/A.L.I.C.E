# Magnolia runtime route survey

A3 is closed. Its returned server log proves that the exact pinned Ollama binary
cannot load because the host does not provide GLIBC_2.28. Earlier project context
already identified Magnolia as CentOS 7 with glibc 2.17. No Qwen probe or task ran.

This survey answers the remaining operational question: which compatible runtime
or container route is already available on Magnolia? It reads the existing a3
runtime and six bounded command receipts: kernel, host libc, ELF interpreter and
library/version requirements, container command locations, available modules and
container tool versions. It verifies the selected closed source files and all
25 distinct top-level CPU runtime objects before inspecting them.

No allocation, download, container pull, model service, inference, package staging,
system-library replacement or execution-state repair is performed. The literal
program travels on SSH stdin through the previously checked Python bootstrap.
Raw output is returned even if a tool is absent. Existing Windows controller
states and selected remote source files are hashed before and after collection.
The observation is from the login node. Available software does not establish
compute-node startup or Qwen eligibility.

Save the ZIP and Start-ALICEMagnoliaRuntimeRouteV100.ps1 together in Downloads.
Run the launcher and return TRACE_ZIP plus the transcript. Keep a3 closed.

The next eligible runtime must prove ABI compatibility and empty-service
readiness before model preparation. Preserve the pinned model, profile and
request budget. Any compatible userland must have its own reproducible identity.
No host-wide libc upgrade or automatic GPU fallback is proposed.
