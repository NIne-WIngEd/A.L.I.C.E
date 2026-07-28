# A.L.I.C.E.–Friday Separation Plan

**Decision:** A.L.I.C.E. remains the owner-specific research flagship. Friday becomes a separate distributable product. Both depend on a host-neutral Personal Cognitive Kernel.

## 1. Exact separation point

Separation starts in **A.L.I.C.E. Phase 5.0**, immediately after Phase 4 is released.

Phase 5 is the first new subsystem that must be designed host-neutrally: the Experience Ledger and evaluation substrate. Building it directly inside owner-specific namespaces would create avoidable extraction debt.

The formal repository split occurs at the **Phase 6.5 Product Separation Gate**, after the generic cognitive control plane and host identity interfaces exist and before Phase 7 integrations proliferate.

Therefore:

- **Phase 4.5:** Finish current web-information work. Do not interrupt it with a premature product fork.
- **Phase 5.0:** Begin internal kernel extraction in the existing A.L.I.C.E. repository.
- **Phase 6.5:** Create the Friday repository and a shared-kernel repository/package.
- **Phase 7.0 onward:** Develop A.L.I.C.E. and Friday as separate products against shared versioned interfaces.
- **Phase 8 release:** Friday becomes worthy of closed alpha because autonomous memory formation is the minimum credible product moat.

## 2. Target repository topology

### Before Phase 6.5

One repository, explicit internal boundaries:

```text
A.L.I.C.E/
  src/
    cognitive_kernel/
    products/
      alice/
      friday_incubator/
  policies/
    kernel/
    alice/
    friday/
```

### After Phase 6.5

```text
personal-cognitive-kernel/   # host-neutral libraries, schemas, evaluators
A.L.I.C.E/                   # Rayan-specific flagship and research system
Friday/                      # installable general product
friday-model-packs/          # distributable manifests, adapters, eval metadata
```

The kernel may initially remain private while interfaces stabilize. Open-source scope is a commercial and trust decision, not a requirement for the split.

## 3. Ownership boundaries

### Personal Cognitive Kernel

Owns:

- experience-event contracts;
- memory and belief schemas;
- learning curator interfaces;
- model/runtime abstraction;
- capability manifests;
- evaluation and champion/challenger framework;
- identity-capsule format;
- encrypted local storage interfaces;
- action and permission primitives;
- product-neutral migration tooling.

It must not contain:

- Rayan's personal memories;
- A.L.I.C.E.-specific constitutional identity;
- consumer-product branding, licensing, or commercial account logic;
- hard-coded user names, paths, credentials, or goals.

### A.L.I.C.E.

Owns:

- Rayan's Constitution and final authority;
- personal vault and learned identity;
- experimental autonomy profiles;
- private research integrations;
- frontier self-evolution experiments;
- owner-specific tools and data.

### Consumer distribution (internal codename Friday)

Owns:

- host enrollment, host-selected naming, and identity creation;
- Windows installer and desktop UX;
- hardware benchmarking and model selection;
- product privacy defaults;
- generic host Constitution templates;
- local data onboarding;
- update, licensing, and optional service integration;
- consumer-facing inspection and deletion tools;
- multi-user and household product policies when added.

## 4. Extraction criteria

A module enters the shared kernel only when:

1. all Rayan-specific data and assumptions are removed;
2. product behavior is configured by manifests rather than hard-coded identity;
3. interfaces support more than one host instance;
4. storage paths and encryption keys are instance-scoped;
5. tests run with synthetic hosts;
6. no developer service requires raw host data;
7. migration and rollback behavior is defined;
8. the module is useful to both A.L.I.C.E. and Friday.

## 5. Data separation

A.L.I.C.E. data must never seed Friday test users or product defaults. Friday fixtures use synthetic identities and generated non-sensitive corpora.

Each Friday installation receives a random instance identifier and locally generated encryption root. The vendor may know a license or update identifier, but that identifier must not be a decryption key or a direct index into personal content.

## 6. Versioning

- Kernel uses semantic versions and migration manifests.
- A.L.I.C.E. and Friday pin explicit kernel versions.
- Product-specific policies can evolve independently.
- Identity Capsules declare compatible kernel and schema versions.
- Experimental A.L.I.C.E. features enter the parity ledger. They ship downstream after generalization and productization; they may not be silently removed from the destination roadmap.

## 7. Phase 1–4 migration rule

Completed phases are released baselines, not untouchable architecture. Any Phase 1–4 document, schema, source file, validator, or test may be changed when required to:

- remove owner-specific coupling from the shared kernel;
- support multiple product identities;
- eliminate obsolete capability ceilings;
- add migration or versioning;
- support Friday's local privacy architecture;
- preserve compatibility through named profiles.

Changes require regression evidence and migrations; they do not require preserving an obsolete design solely because it shipped earlier.

## 8. Capability parity

A.L.I.C.E. and the consumer distribution have the same ultimate capability destination. A.L.I.C.E. may lead in experimental implementation, but every successful generalizable capability enters the shared-kernel catalog and downstream product backlog.

The products differ by personal state, release maturity, active hardware, host permissions, packaging, and private integrations—not by a permanent intelligence ceiling.

## 9. Host-selected identity

`Friday` is an internal codename only. The commercial brand is selected separately, and every host chooses the assistant's local name. Product identifiers and storage schemas remain neutral so renaming either the product or an assistant does not require a data migration.

## 10. Long-term organizational split

After the consumer product has a dedicated maintenance team capable of independent releases, hardware support, updates, incident response, and capability productization, Rayan may return primary development attention to A.L.I.C.E. The consumer team remains responsible for tracking the shared-kernel parity ledger and delivering the same destination capabilities to host installations.
