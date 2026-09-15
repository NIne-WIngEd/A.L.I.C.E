# N0 capability-first capacity rule — 2026-09-15

Owner correction: A.L.I.C.E. does not have a rigid parameter-count target or ceiling.

Rules:
- Exact parameter counts describe concrete checkpoints only.
- A prior experiment's parameter count is not a successor-architecture constraint.
- Personality purpose, fidelity, evidence reasoning, contradiction handling, temporal reasoning, and generalization take priority over minimizing parameter count.
- If controlled held-out evidence shows undercapacity, increase width, depth, heads, specialist modules, fusion capacity, or semantic-core capacity as needed.
- Efficiency remains important only after capability requirements are met. Between equally capable designs, prefer lower latency/memory/compute.
- Do not preserve a smaller model by accepting meaningful personality-capability loss.
- Finite per-run compute limits are allowed to prevent accidental waste; they are experiment controls, not architecture ceilings.

Current state:
- semantic base: N0 v0.2 targeted-repair step080
- structured-state current base: structured-state pilot step080 (1,656,064 parameters as a historical checkpoint fact)
- graph sidecar current implementation: ~0.94M parameters as a starting point, not a target
- graph curriculum/objective work has begun with no hard parameter ceiling
