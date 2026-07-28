# Implementation Evolvability and Product-Neutrality Standard

**Version:** 1.1.0

## Design requirements

A.L.I.C.E. and shared-kernel implementations must prefer:

- registries over closed provider or tool lists;
- capability profiles over permanent boolean denials;
- mission budgets over universal resource caps;
- strategy interfaces over one fixed algorithm;
- versioned schemas with extension fields over exact-key rejection;
- champion/challenger promotion over one immutable implementation;
- adapters over vendor coupling;
- explicit data lineage over blanket data exclusion;
- reversible migrations over architectural immobility;
- explicit product and host identity over singleton-user assumptions;
- synthetic multi-host tests over owner-data fixtures;
- neutral shared-kernel namespaces over product-branded dependencies.

## Prohibited architectural anti-patterns

Outside the minimal authority kernel, active code must not impose a permanent project-wide rule such as:

- every capability must remain false;
- allowed tools must always be empty;
- only one provider may ever be used;
- live retrieval can never occur;
- private data can never participate in authorized learning;
- model weights may never contain personal adaptation;
- production self-modification always requires manual approval;
- background cognition is categorically forbidden;
- resource budgets have universal fixed maxima;
- a model or agent architecture may never replace itself;
- Phase 1–4 files may never change;
- A.L.I.C.E. identity may be embedded in reusable kernel code;
- multiple hosts may share unscoped personal caches, indexes, logs, or training data;
- Friday must depend on vendor-readable host content.

A compatibility profile may exhibit narrow behavior if the restriction is clearly scoped and broader profiles can coexist.

## Extension contract

Each subsystem should expose:

1. a capability description;
2. a product and host scope;
3. a profile or mission selector;
4. observed outcomes;
5. an evaluator;
6. a rollback or replacement path;
7. learning-event output;
8. migration and deletion lineage.

## Earlier-phase migration

Completed releases are evidence baselines. Their files may be refactored, replaced, or removed when they violate this standard. Compatibility is a deliberate product decision, not an automatic veto.
