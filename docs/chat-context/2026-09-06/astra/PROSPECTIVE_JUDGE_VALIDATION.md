# Prospective judge-validation design

This implements the accepted audit's scientific direction as a design artifact. It does not revise the current MC10D pass rule, open MC8, certify a judge or create a new execution prerequisite by itself. Choose the final numeric error budget and freeze the protocol before collecting prospective outcomes.

## What this must establish

Can the specified judge, model digest, runtime profile and rubric distinguish compatible personality novelty from arbitrary invention while enforcing source/target role, state, context, contradiction and history boundaries on new cases? Evaluate the exact downstream judging profile. A public calibration pass on reused cases cannot answer that generalization question.

## Separate the evidence streams

- Keep the existing 16 fictional tasks as development/calibration, including every historical prompt/profile/scoring revision and its outcomes. Never discard failures when an amendment is accepted.
- Author a fresh development set for debugging rubric ambiguities. Explicitly permit tuning only on that set.
- Have a separate evaluator author a prospective certification set from the frozen construct specification. Use fictional personas and histories, distinct surface forms and underlying source families. Do not copy the public tasks with renamed people or paraphrased answers.
- Keep the certification labels and packet identities unavailable to the agent/model-selection process until the run is frozen. A suite generated and inspected by the tuning agent is not described as sealed independent evidence.
- Continue to keep private MC8 sealed under its existing authority. Judge certification is not a reason to open it.

## Coverage and labels

Stratify cases by compatible unobserved behavior, arbitrary weak bridges, gratuitous specificity, core contradiction, source history transfer, fake lived memory, actor/role reversal, state direction, public/private context, target contrast, personality flattening and controlling versus noncontrolling behavior. Include difficult valid novelty as well as hard violations; a judge that simply rejects everything must fail the usefulness assessment.

Freeze labels before model responses are seen. Write a short evidence-based decision explanation and explicit applicability of each auxiliary field. Mark genuinely ambiguous cases during author review and resolve or exclude them under a preregistered rule; do not repair gold after seeing which model loses. Use independent label review on disputed and hard-boundary cases, with agreement and adjudication recorded.

Group train/development/certification splits by underlying persona, evidence family and generative template. A thousand paraphrases of one source are not a thousand independent observations. Blind generator/model origin when reviewing candidate outputs. Cross-model agreement is a diagnostic, not truth authority.

## Measurement and decision

Report separately: format completion, parsing/schema failure, ordinary verdict errors, false rejection of compatible novelty, false promotion of arbitrary material, critical decision errors, each hard-boundary category, and auxiliary-field diagnostics. Preserve denominators, uncertainty, runtime, attempts, token counts and cost. Record results per source family as well as pooled; distinguish correlated-family evidence from independent cases.

Select the practical error budget before execution. As a planning illustration only: with zero observed errors on n independent representative Bernoulli trials, the one-sided 95% exact upper error bound is `1 - 0.05^(1/n)`. To put that bound below 1% requires at least 299 such independent trials. This is not a new mandatory 299-task gate, and correlated templates or nonrepresentative cases do not justify that interpretation. Scope sample size and critical-case coverage to the actual failure cost and available resources.

Pin model digest, quantization, runtime version, profile, prompt, task/label hashes, schema, decoding parameters, retry rules and complete effective scoring-contract hash before the first prospective response. The run manifest must name the selected profile and frozen decision criteria. A runtime-only amendment still needs evidence that the deployed profile is the one evaluated.

## Cost and failure handling

1. Complete authoring, validation, corpus hashing, protocol-state tests and runtime preflight on local/Magnolia CPU where eligible.
2. Estimate cost from measured eligible throughput; use the actual scheduler/QOS and VRAM budget. Do not assume Magnolia's P100 fits a frozen 30B artifact or that Kaggle's old remaining-quota estimate is current.
3. Launch only the missing, validly authorized scope. Preserve exact successful outputs from other families when their bindings remain applicable.
4. Download and hash terminal output before resource cleanup. On a semantic failure, analyze the rows before changing the judge or criterion.
5. If a failed prospective outcome informs tuning, retire that suite into development and disclose the change. A fresh independent certification must then evaluate the revised system.

The immediate prerequisite remains inspection of the already downloaded v186 GLM rows. That evidence may change which model, profile or rubric issue deserves the next experiment. This document prevents the next experiment from repeating the same interpretation error while that recovery proceeds.
