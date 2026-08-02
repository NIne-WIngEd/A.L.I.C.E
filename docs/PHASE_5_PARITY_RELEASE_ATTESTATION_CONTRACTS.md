# Phase 5.0f — Capability Parity Synchronization and Release Attestation Contracts

P5.0f synchronizes the product-family parity records with the kernel contracts
already merged through `cognitive_kernel` 0.4.0 and adds the host-neutral
`release_attestation.v1` verifier contract in 0.5.0.

## Status model

A kernel contract being implemented does not mean that A.L.I.C.E. has gained the
corresponding production capability. It also does not make Friday eligible to
implement or test that capability. Friday eligibility still requires A.L.I.C.E.
to implement, evaluate, approve, and gain the capability, or an exact-scope
owner override. Before Phase 6.5, Friday remains foundation-only.

The parity ledger therefore records these facts separately:

- kernel contract implementation;
- A.L.I.C.E. capability-gained status;
- Friday eligibility;
- the pre-Phase-6.5 foundation gate;
- evidence for the contract implementation.

## Release-attestation contract

The P5.0f contract binds one release candidate to:

- source commit;
- kernel version;
- dependency lock digest;
- exact artifact hashes and their canonical manifest digest;
- model-pack versions;
- schema versions;
- policy versions;
- migration manifest digest;
- evaluation bundle digest;
- deployment manifest digest;
- rollback manifest digest;
- release channel;
- A.L.I.C.E. audit determination;
- owner approval.

The verifier requires the nested A.L.I.C.E. audit and owner approval to match
the outer source commit, artifact manifest, evaluation bundle, and deployment
manifest exactly. It can represent rejected or incomplete decisions, but it
reports an authorized candidate only when both determinations are approving.

## Boundaries

P5.0f does not:

- generate an A.L.I.C.E. audit;
- generate owner approval;
- sign commits, artifacts, tags, or release records;
- deploy or promote a release;
- create Friday product source;
- bypass A.L.I.C.E.-first capability precedent;
- make any Friday capability eligible merely because a kernel contract exists;
- relax the pre-Phase-6.5 foundation-only gate;
- include private host or source-person payloads.
