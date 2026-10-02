# Personality Gemma N0 feature interface v1

Contract IDs: `personality-gemma-n0-source-input-v1` and
`personality-gemma-n0-token-features-v1`.

This contract describes the implemented feature interface in
`src/alice_personality/gemma_n0/backbone.py`. It is not an ACFP/IDP codec,
a trained personality, or proof of semantic competence or neutrality.

The loader admits only a freshly verified, unchanged personality-role clone of
`google/gemma-4-12B@023679ed352de9bb66cc873c9009ce3482585c08`. Preparation
binds the clone, this contract, both implementation files and exact runtime
versions/dtype. Loading uses local safe tensors with remote code disabled.

Inputs are nonempty two-dimensional integer token IDs and an equally shaped
binary attention mask. Each row must contain an attended source position.
Processor-produced media tensors may accompany them; the initial public
diagnostic is text-only and does not qualify media support. Source context must
fit the publisher's declared context budget in full. Exceeding it fails; the
interface never silently truncates or summarizes evidence. The public probe
producer tokenizes raw source without an assistant chat template. The
tensor-only backbone cannot infer how supplied IDs were packaged; future
producers must implement and bind this input rule themselves.

The representation module is frozen in evaluation mode and executes without
gradients or cached decoding. Outputs are finite, detached floating-point hidden
states with shape `[batch, source positions, publisher hidden size]`, together
with the complete binary attention mask on the chosen device. Padding states
must be excluded by downstream consumers using that mask. The interface does
not average the document into a single vector.

Only the publisher representation module is retained. The public interface has
no generation, language logits, publisher-selected answer or stance. Labels,
decoder inputs, input embeddings, cache and generation controls are rejected.
Upstream tensor mutation and upstream private gradients are outside this v1
route. Later first-party personal layers must own source-supported identity,
candidate judgments, calibration and expressive intent.

Removing the generative path suppresses a direct source of inherited voice and
preference. Hidden states still contain pretrained priors. Receipt admission and
a successful forward cannot establish that those priors cannot steer a future
personal readout. Independent semantic tests and causal inherited-influence
tests are required before declaring N0 approved for personality learning.

Public diagnostic artifacts are always unqualified. They contain execution
metadata, not saved source representations or personality gold. Private
evidence requires its own authorization, source isolation and qualified
downstream learning path; this contract grants none of those automatically.
