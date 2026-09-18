# N0 Query–Relation Role Router — External Architecture Research v0.1

**Date:** 2026-09-18  
**Trigger:** Magnolia job 575798, `FAIL_RELATION_SEMANTIC_ROLE_RESIDUAL_DEV_STOP_NO_HELDOUT_EXPOSURE`  
**Scope:** architecture research before any further model repair

## Why the repair loop stops here

The previous two repair attempts establish two different failure modes:

1. **Shared-read semantic grounding v0.1** mutated the learned endpoint-read path. Some semantic behavior moved, but explicit endpoint-role capability degraded.
2. **Isolated additive role residual v0.2** froze the endpoint parent, but still consumed the parent's endpoint-specialized projected query and composed by additive endpoint bias. It preserved the parent only at early checkpoints and never produced a dev-eligible semantic candidate.

The second failure therefore does not justify more steps, more replay weight, or a larger additive residual. It motivates an architecture change.

## External work reviewed

### Progressive Neural Networks

Rusu et al., *Progressive Neural Networks* (2016), arXiv:1606.04671.

Key lesson: preserve learned columns and add new capacity with lateral access rather than rewriting old knowledge. This directly supports keeping the step-200 endpoint graph immutable.

### Side-Tuning

Zhang et al., *Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks* (ECCV 2020), arXiv:1912.13503.

Key lesson: a side network can adapt a frozen backbone without catastrophic forgetting.

### Ladder Side-Tuning

Sung, Cho, Bansal, *LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning* (NeurIPS 2022), arXiv:2206.06522.

Key lesson: the side path should receive intermediate / earlier frozen activations. A side module fed only the already task-specialized bottleneck can inherit that bottleneck's information loss.

This exposes a flaw in semantic-role residual v0.2: it was parameter-isolated but received `query` only after the endpoint parent's `pool_query + query_projection + normalize` bottleneck. The raw frozen semantic query vector was available but unused by the side residual.

### AdapterFusion

Pfeiffer et al., *AdapterFusion: Non-Destructive Task Composition for Transfer Learning* (EACL 2021), arXiv:2005.00247.

Key lesson: separate knowledge extraction from knowledge composition. Preserve specialist modules, then learn a separate composition mechanism rather than sequentially overwriting one specialist.

For N0, explicit endpoint-role behavior and natural relation-semantic interpretation should be treated as composable specialists, not one shared mutable skill.

### Neural Bellman-Ford Networks

Zhu et al., *Neural Bellman-Ford Networks: A General Graph Neural Network Framework for Link Prediction* (NeurIPS 2021), arXiv:2106.06935.

Key lesson: relational graph reasoning benefits from query-conditioned graph computation. The query should participate directly in relation reasoning rather than only influence a generic pooled readout.

### Heterogeneous Graph Transformer

Hu et al., *Heterogeneous Graph Transformer* (WWW 2020), arXiv:2003.01332.

Key lesson: edge / relation types deserve dedicated parameters in attention and aggregation. Relation semantics should not be collapsed into one generic endpoint bias.

### Hypernetwork Knowledge Graph Embeddings

Balažević et al., *Hypernetwork Knowledge Graph Embeddings* (ICANN 2018), arXiv:1808.07018.

Key lesson: relation-specific transformations can be generated or conditioned by relation representations, giving structured specialization without duplicating a full network per relation.

### Gradient-interference work

Yu et al., *Gradient Surgery for Multi-Task Learning* (NeurIPS 2020), arXiv:2001.06782.

Farajtabar et al., *Orthogonal Gradient Descent for Continual Learning* (AISTATS 2020).

Wang et al., *Orthogonal Subspace Learning for Language Model Continual Learning* (Findings EMNLP 2023).

Key lesson: when tasks share parameters, conflicting gradients can destroy prior capability, and orthogonal/subspace constraints can reduce interference.

For the current N0 defect, however, the stronger move is to avoid sharing the endpoint expert's parameters at all. Orthogonal-gradient methods remain relevant only if a future architecture truly requires shared trainable weights.

## Architecture decision

The next architecture is **not another additive residual**.

It is a frozen-parent + independent **Query–Relation Role Router**.

### Inputs

The router receives:

- raw frozen `query_semantic` directly from the semantic encoder cache;
- relation type for each graph edge.

It does **not** consume the endpoint parent's projected query as its only semantic input.

### Per-edge decision

For each active directed relation edge, the side router predicts:

- `SOURCE`
- `TARGET`
- `DEFER_TO_PARENT`

The role computation uses interaction features between a side projection of the raw query and an independent relation embedding.

### Composition

The parent endpoint graph remains frozen.

The semantic side expert produces an endpoint field distribution.

The final field distribution is a probability-level gated mixture:

`final = (1 - route_probability) * parent + route_probability * semantic_expert`

This avoids the v0.2 failure mode where an additive residual has to numerically overpower a strong parent logit.

`DEFER_TO_PARENT` is an explicit stability route rather than an emergent small residual.

### Supervision

The new semantic router should be trained with **direct role supervision**:

- natural semantic rows -> `SOURCE` or `TARGET`;
- preservation replay -> `DEFER_TO_PARENT`.

Field-selection loss remains a downstream consistency objective, not the only signal teaching the semantic abstraction.

### Data discipline

The next curriculum must be fresh and lexically split:

- no frozen challenge rows;
- no job-575795 localization rows;
- no relation-semantic v0.1 rows;
- no role-residual v0.2 rows;
- no endpoint-repair heldout rows;
- train/dev/test query paraphrase templates disjoint;
- subject/value pools split where practical;
- heldout remains unopened unless preservation and dev readiness pass.

## Capacity / ceiling decision

This redesign adds capacity because the previous architecture lacked an independent semantic information path and explicit composition mechanism.

That is **architectural capacity**, not arbitrary scaling.

There is still no hard parameter ceiling.

No evidence currently supports scaling the semantic backbone, graph message-passing depth, cross-context fusion, or latent pool.

## Anti-loop decision

After this architecture:

- one pre-GPU contract qualification;
- one P100 training run;
- no automatic hotfix;
- if dev fails again, stop model modification and perform a broader architecture-level audit before another gradient run;
- do not convert another failure into a sequence of local patches.

The purpose of this research checkpoint is to prevent MC10-style repair chaining.
