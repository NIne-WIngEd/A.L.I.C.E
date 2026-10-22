# Checkpoint 10: formation failure and learning attribution

The complete native assessment finished successfully as an execution, but none
of the nine teaching cases produced a grounded formation decision. Eight outputs
failed complete-copy alignment and one failed parsing. All ended with EOS after
24–37 tokens, within the unchanged 8,192-token budget. The first departure is
query 1: gold token 2763 versus generated token 2394. The outputs retain the
earlier restaurant/value_text form; reference and copy prediction are downstream
and were not reached. This is a model-behavior failure, not evidence of a Kaggle
runtime failure.

The original outputs remain intact. The independent assessment audit checks 71
recovered originals, 34 execution-manifest members, 104 source interfaces and
the complete nine-case roster. Its 518 checks include the actual parser and
alignment failures, all nine authored structural controls, and the parent-output
bindings. Full target labels contain 4,921 tokens across 410,994 source positions.
Local checking verifies small original bytes and their relationships. Production
tensor loads, GPU execution and state preservation remain attestations from the
exact executed source; no production tensor was loaded on the PC.

## Ordered diagnosis

1. **Validate the evidence.** The original consumer rejects the complete observed
   failure, while authored complete decisions pass the same structural path.
   Short EOS-terminated generations rule out budget exhaustion. Do not alter raw
   output, supply a missing structure or count valid JSON as successful formation.
2. **Connect the claim to MFM.** Alice/Fable needs learned, source-grounded
   formation proposals and dispositions before independent acceptance. Lower
   teacher-forced losses do not demonstrate this capability. These nine cases
   are a teaching family, not independent gold or full-role qualification.
3. **Trace the architecture.** Task context is present in the rendered source:
   task, profile, context, claims, episodes and task snapshots reach the source
   representation. Frozen source states feed a trainable projection and the
   owned causal decoder. The decoder has tied input/output embeddings and shared
   states for reference and original-copy learning. No missing task input or
   accidentally frozen owned learner has been demonstrated.
4. **Locate the departure.** The first wrong choice occurs while the preceding
   prefix is still gold. Post-error exposure shift cannot explain that original
   choice. An unchanged winner also does not prove the correct token's likelihood
   stopped improving. Earlier checkpoint-2 measurements found only 3–10 correct
   teacher-forced top-one tokens per case; the learning problem extends beyond
   the opening schema token.
5. **Understand the objective.** Lexical loss averages all complete-wire token
   losses. Reference and copy losses separately average their requests and are
   added with weight one; all nine cases contribute their original masses.
   Query 1 has only 1/N of a case's lexical mass. That fact alone establishes
   neither harmful imbalance nor gradient interference. Saved Adam moments and
   parameter-specific update directions matter as well.
6. **Choose a coherent next operation.** Measure checkpoint 10's full-target loss,
   token correctness, early gold likelihood/rank/margin and matched-prefix parity.
   Retain parent-2 measurements for comparison. On the complete original roster,
   differentiate lexical, reference, copy, early-token cross-entropy, early margin
   and the actual total loss. Inspect aggregate and parameter-block geometry and
   the prospective AdamW direction using the saved moments, counters and recipe.
   This read-only operation changes no production weights and generates no new
   output. It resolves a construction decision; it is not a new evaluation gate.
7. **Consult research before correction.** The pinned
   [PyTorch loss contract](https://docs.pytorch.org/docs/2.11/generated/torch.nn.CrossEntropyLoss.html),
   [gradient API](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.grad.html)
   and [Adam implementation](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/optim/adam.py)
   ground the numerical measurements.
   [PCGrad](https://arxiv.org/abs/2001.06782) and
   [GradNorm](https://proceedings.mlr.press/v80/chen18a.html) motivate inspecting
   interactions and scale, but do not justify adopting an extra controller or
   gradient surgery from one negative cosine.
   [Tied-embedding research](https://aclanthology.org/E17-2025/) does not establish
   tying as a defect. The latest inspected frontier-watch commit
   `6f5ebc0fc3a155301b063627e840abf603af3f7c` keeps WeAct conditional on demonstrated
   interference; BetaMem's validity/support separation does not replace learned
   formation. No new architecture, constrained decoder or selective-token boost
   is adopted from these references.
8. **Verify and return to construction.** Real tiny-model controls compare the
   prospective direction against actual AdamW with nonzero moments, check complete
   case-mass accumulation, unused parameters, invalid state refusal and state/RNG
   preservation. Repeat those controls in the actual pinned production runtime
   before measurement. All original sources, target packets, checkpoints and
   interfaces remain bound. A numerical control pass is not learned behavior.

## Completed diagnostic and construction decision, October 22

The exact diagnostic completed with exit 0. Five actual-runtime numerical
controls passed. The independent audit verified 60 small originals, all 16
execution-manifest members, 42 embedded payloads, 104 physical source interfaces,
12 original assessment bindings, nine complete cases, 4,921 labels and 18
same-gold-prefix probes. All model, optimizer, RNG, mode and gradient preservation
checks passed. No production update or new generation was performed. Production
state remains bound remote evidence; local inspection did not load its tensors.

Checkpoint 2 to 10 improves complete lexical loss in every case, from a
case-mass-weighted 12.0091 to 9.3240. Reference loss falls from 3.3876 to 1.2616
and copy loss from 7.1688 to 1.1360. Query 1's gold likelihood and margin also
improve in all nine cases, but its probability remains only 3.10e-7–2.05e-6 and
the old greedy winner remains. This is incomplete learning with measured
progress, not a demonstrated stuck or disconnected training path.

The whole-wire comparison preserves regressions: 306 token positions have worse
likelihood. Gold EOS remains top-one in all nine teacher-forced cases but its
pooled NLL worsens from 0.0274 to 1.5357. Proposal, disposition and schema sections
improve in aggregate. These descriptive token-pooled means do not replace the
original equal-world/cut/variant training masses, and none qualifies native
formation behavior.

Raw gradient opposition does not establish a harmful optimizer update. The raw
joint-gradient dot early-token CE is -5.405, but the actual prospective AdamW
direction, using the saved moments and counters, predicts lower lexical,
reference, copy, early-token and total losses. Its early-CE directional derivative
is -0.5922 and early-margin derivative +0.6642. These are local first-order
measurements, not an executed finite step or a unique decomposition of adaptive
optimizer effects by head. They do not support detaching heads, changing their
weights, untied embeddings, gradient surgery or first-token emphasis.

The selected next construction is an unchanged continuation from checkpoint 10
through update 26: 16 complete-nine updates, retaining checkpoints 18 and 26,
original named optimizer moments/counters, history, objective, source/target
roster, numerical precision and all 104 interfaces. The actual trainer already
supports this v3-to-v3 successor. This is additional formation learning, not a
repeat of S5 or a changed data phase.

The previous eight updates took 11,778.94 seconds plus 10,977.19 seconds of other
execution overhead. Keeping that entire overhead predicts 9.59 hours at the
mean update rate or 10.20 hours at the slowest observed update rate. Those are
resource estimates, not timeouts or convergence promises. The unchanged
two-checkpoint footprint, current free allowance and provider session limit
govern dispatch; no custom timeout or automatic retry is introduced.

After completion, recover the exact terminal state and assess all nine complete
native decisions under the unchanged consumer. Compare the full wire, all heads,
early choice and retained regressions to checkpoint 10; do not substitute a
query-1 pass for complete formation. If finite progress contradicts this
diagnosis, return to the responsible learning/data path before another change.
When structure improves but grounding fails, inspect semantics and controlled
source interventions. The broader Rill source/target construction proceeds
independently, with its own explicit admission and eventual data-phase transition.

S5 stays complete at 20 epochs and 45,760 steps. The joint construction has 10
updates over this nine-case family. The full build still requires transferable
formation across time, uncertainty, episodes, corrections, relationships and
missions; broader authorized teaching and distinct development families; native
media; and accepted memory → judgment/action → observed outcome → governed
revision. Independent gates continue to own accepted writes. The full-system
stress draft remains parked until construction supports its intended claim.

Current dispatch and backup evidence is recorded in TRACE157. Earlier pending
snapshots in TRACE156 remain historical evidence.
