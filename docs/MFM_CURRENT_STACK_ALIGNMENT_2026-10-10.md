# Current selected stack and MFM alignment

The updated owner-ratified Alice stack controls the MFM build. Older Fable JetStream-as-authority and JanusGraph wording is documentation lag, not a new backend choice or a reason to change MFM. The latest owner instruction explicitly reaffirms this. Current MFM proposal, caller and compiler interfaces align with the selected contracts and do not couple memory authority to a physical engine.

This clarifies the provisional physical-policy conflict in the earlier [roots audit](MFM_ROOTS_ALIGNMENT_AUDIT_2026-10-10.md). That sealed source snapshot stays intact. The chronological audit recovered the current owner-ratified policy and its exact historical selected-policy version; certain later historical document blobs were unavailable locally. Commit metadata alone is not presented as recovered patch evidence. The current policy and owner-directed documents agree, so documentation lag does not block neutral MFM construction or reopen stack selection.

## Current Alice/Fable selection and FloRA slice

| Role | Alice/Fable selected destination | FloRA implemented/configured slice |
| --- | --- | --- |
| Original evidence | Encrypted content-addressed objects behind an S3-compatible abstraction | Host-local encrypted opaque objects; S3 placement remains future |
| Canonical Experience/events | KurrentDB | KurrentDB |
| Device ingress/federation | NATS JetStream transport, governed reconciliation into canonical events | JetStream ingress; acknowledgment after Kurrent commit/replay |
| Accepted bitemporal Claims | XTDB v2 | XTDB v2 |
| Vector/multimodal candidates | Qdrant | Qdrant current-Claim/source-filtered candidates |
| Host-local graph | LadybugDB | LadybugDB derived evidence graph |
| Scale-out graph | NebulaGraph | Selected destination; no declared adapter/service in the inspected slice |
| Durable workflows | Temporal | Temporal with PostgreSQL history/visibility storage |
| Shared ephemeral state | Valkey with local L1 | Selected destination; no declared Valkey adapter/service in the inspected slice |

Neo4j remains historical/reference evidence. PostgreSQL's FloRA workflow storage does not replace XTDB Claim authority. Graph/vector/cache results remain derived candidates; only the governed authority path accepts knowledge. This table identifies selected architecture and executable/configured surfaces, not a live deployment or full product qualification.

FloRA main is885be57abb582510d27a98177facd2a358c11387. Its current active successor is e3b40b47603f83505297aa9deaab7fd3f2d39432; the latest capture-boundary candidate remains unadopted and the recorded connected experiment incomplete. The narrow experiment uses the chosen engines where exercised. Its old Phase2 implementation was retired. Personality and MFM remain supplied by their separate workstreams. The current README and compose were independently read from that exact revision;18 pinned source documents/components and all76 scoped Python sources support the bounded stack scan.

## Consequence for MFM construction

Continue the unchanged conditional-language foundation and the [admitted caller input](MFM_CALLER_MEANING_AND_JOINT_INPUT_2026-10-10.md). Build actual causal longitudinal prior state before separately authored targets, then the complete learned formation and governed consumer outcome path. No model/source, backend choice, training recipe, production cutover or FloRA implementation is changed by this audit. Full MFM capability and physical consumer integration remain unfinished; a documentary discrepancy is not a model failure.

The controlling machine policy is Alice main8ea804aa25c3d1c5f46e9be5810d072ed0fadb4b, `policies/stage_g_memory_fabric_candidate_qualification_policy.json`, owner_ratified decision `alice.stage-g.full-cognitive-memory-fabric.v3`. Its selected implementation semantics supersede stale prose. [The current FloRA README](https://github.com/NIne-WIngEd/FloRA/blob/e3b40b47603f83505297aa9deaab7fd3f2d39432/README.md) and [compose](https://github.com/NIne-WIngEd/FloRA/blob/e3b40b47603f83505297aa9deaab7fd3f2d39432/compose.integration.yml) show the exercised slice. [This receipt](receipts/MFM_CURRENT_STACK_ALIGNMENT_20261010_TRACE103_PUBLIC.json) binds the clarification without rewriting historical evidence.
