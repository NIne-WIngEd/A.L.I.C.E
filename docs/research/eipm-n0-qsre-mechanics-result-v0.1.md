# QSRE CPU/No-Gradient Reference Mechanics Result v0.1

**Date:** 2026-09-20  
**GitHub Actions run:** `35497790357`  
**Result:** SUCCESS

The full-envelope reference mechanics batch passed without learned parameters.

Receipt:
`PASS_QSRE_CPU_NO_GRADIENT_REFERENCE_MECHANICS`

## Coverage

18 fixtures covered:

- masked sparse support;
- adaptive support cardinalities 0, 1, 2, and 3;
- exact-zero exclusion of masked/high-score distractors;
- explicit SOURCE and TARGET incidence;
- support-local relational readout;
- exclusion of a globally high-scoring field outside relational support;
- RELATIONAL / FALLBACK / DEFER control states;
- ordered two-hop relation execution;
- permutation-stable path mechanics;
- dynamic shape with 257 fields and 513 edges.

The dynamic-shape case intentionally exceeded historical 64-field / 256-edge operating hints.

## Boundaries

The run used:

- no learned QSRE parameters;
- no optimizer;
- no gradient;
- no GPU;
- no causal TEST;
- no frozen challenge;
- no private identity gradient.

## Interpretation

The reference mechanics now validate two things together:

1. QSRE is bound to the full EIPM workload envelope rather than only the current relation benchmark.
2. The execution mechanics can represent adaptive support, explicit endpoint role, local relational readout, fallback, ordered paths, and dynamic sizes without reintroducing the old global-field competition pattern.

This is sufficient evidence to authorize a **tensorized CPU/no-gradient mechanics skeleton**.

It is not evidence that a learned QSRE will generalize.

The next layer may implement PyTorch-shaped contracts and deterministic tensor operations only. No learned projection, executor MLP, optimizer, or GPU training is authorized.
