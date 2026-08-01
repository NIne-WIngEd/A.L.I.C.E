# Phase 5 P5.0d — Attention Decisions and Workspace Projections

## Status

This milestone extends `cognitive_kernel` to version `0.3.0` with host-neutral attention-decision, host-window-override, adaptive-layout, and workspace-projection contracts.

It remains contract-only. It does not implement a persistent attention store, autonomous ranking execution, the Phase 6 Cognitive Workspace UI, speaker recognition, guest authority, or Friday product source.

## Delivered contracts

- `AttentionRankEntry` records an opaque subject reference, rank, score, interruption cost, priority class, explicit reason codes, host override, and selection or suppression explanation.
- `AttentionDecision` binds an ordered candidate snapshot, visibility limit, interruption preference, focus mode, layout-stability weight, provenance, and tamper-evident digest.
- `HostWorkspaceOverride` records host commands for pinning, foreground/background placement, visibility limits, layout lock, focus, interruption preference, security visibility, and automatic-layout restoration.
- `WorkspaceItemProjection` exposes only metadata digests and opaque references over canonical Mission Graph or Result Capsule state.
- `WorkspaceLayout` implements the ratified no-empty-slot composition table and records all omitted references by digest.
- `WorkspaceProjection` binds selected attention entries to one product/host scope, one deterministic layout, one audience, and explicit privacy/redaction boundaries.

## Safety and governance invariants

1. Product, host, schema, and encryption scope remain explicit.
2. Protected security interrupts cannot be suppressed or backgrounded.
3. Pinned and foregrounded entries must remain selected.
4. Commercial, advertising, sponsored, and engagement-maximization ranking reasons are rejected.
5. Attention decisions are explainable receipts; no learned ranker or UI executor is activated.
6. Canonical Mission Graph state remains in the kernel or product runtime, never in the frontend projection.
7. Workspace layouts contain no fixed empty placeholders.
8. Non-host projections cannot include restricted items and must redact sensitive items.
9. Cross-host attention entries, layouts, and projections are rejected.
10. Real private companion payloads and product-private data remain outside public package fixtures.

## Adaptive composition contract

| Visible work | Default layout mode |
|---|---|
| 0 | `empty` |
| 1 | `full_workspace` |
| 2 | `focus_support_split` |
| 3 | `primary_two_secondary` |
| 4 | `adaptive_grid` |
| 5–6 | `panels_live_cards` |
| 7–10 | `command_center` |
| More than 10 total candidates | `highest_value_set`; at most ten visible, remaining references recorded as omitted digests |

## Deferred work

- durable attention-decision and projection stores;
- learned or autonomous ranker execution;
- Phase 6 Mission Canvas and adaptive window UI;
- speaker context, guest sessions, and delegated guest grants;
- product-specific shell behavior and private host state;
- candidate learning, evaluation, and release promotion.
