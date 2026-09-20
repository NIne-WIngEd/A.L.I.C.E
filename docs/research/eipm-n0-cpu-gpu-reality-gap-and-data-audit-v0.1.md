# N0 CPU/GPU Reality Gap and Data Validity Audit v0.1

**Date:** 2026-09-20  
**Status:** architecture/data audit authority; all new gradient/GPU work frozen  
**Trigger:** T2 jobs 575933 and 575934 plus historical validation pattern

## Executive conclusion

The recurring pattern is real, but it is not simply “CPU good, GPU bad.”

There are three different things being mixed together:

1. deterministic/oracle CPU success — proves an interface can represent the desired computation;
2. learned-model GPU failure — valid evidence that a model/factorization did not learn the desired behavior;
3. runtime/harness failure — no model evidence at all.

The current workflow has repeatedly allowed category 1 to look more like a rehearsal of category 2 than it really is. Worse, several validators are authored from the same assumptions as the implementation, so a wrong assumption can be copied into both code and test.

Job 575934 is the clearest example: the malformed doubled-dollar shell expansion was present in the runner/sbatch and the CI expected the malformed spelling. The validator therefore confirmed the bug instead of challenging it.

That is the same class of epistemic mistake seen in early A.L.I.C.E. RAG work: local stage metrics and architectural sophistication were trusted until the actual downstream evidence was read directly.

No next GPU workflow is justified until the experimental substrate and data are audited as one coherent batch.

---

## 1. Early RAG analogy

The early Phase 1 retrieval stack used lexical + E5 retrieval and later added the generic MS MARCO cross-encoder for passage reranking.

The important result was not whether the reranker looked architecturally stronger. The mismatch-only support audit showed that grounded support fell from 0.421053 to 0.083333.

The reranker was therefore disabled for that role.

The lesson was:

> inspect the actual answer-bearing evidence and downstream support, not the apparent sophistication of an intermediate stage.

The current N0/QSRE work is far more advanced than the early RAG system, but the epistemic failure mode can still recur.

A stage can be internally correct while being tested against the wrong data abstraction or validated by a test that shares the same hidden assumption.

---

## 2. What the current CPU gates really prove

The clean-sheet QSRE work intentionally used increasingly concrete CPU/no-gradient layers:

1. deterministic oracle interface;
2. pure reference mechanics;
3. tensor mechanics;
4. static trainable source qualification;
5. prepared-cache lineage.

Those are useful. They prevented the old global-softmax-residual architecture from silently reappearing.

But they do not prove learnability.

The tensor-mechanics pass proved support-local readout, zero/one/many support, explicit role incidence, permutation stability, exact-zero sparse support, and isolation from outside-support global winners.

Those are representation/mechanics claims.

They do not prove that natural language will produce the correct operator state.

The workflow sometimes spoke of a “CPU pass” as though it were a miniature version of the GPU experiment. It is not. It answers a different question.

---

## 3. Historical failure taxonomy

The state history makes the distinction explicit.

### Non-model harness/runtime failures

- v0.7 — Python import layout mismatch in CPU harness. No diagnostic executed.
- v0.34 — dual-view qualifier double-counted reversal pairs. Bookkeeping bug.
- v0.38 — CPU localization was too slow for the actual bridge computation; moved unchanged inference to P100.
- v0.51 — uDocker did not expose repository Python paths correctly. Curriculum itself had already passed.
- job 575933 — evaluator was not total over provisional learned operators. Wrong intermediate PATH_FOLLOW prediction raised instead of being scored wrong.
- job 575934 — Slurm script contained malformed doubled-dollar shell expansion; Bash substituted the process id and failed before the run entered the container.

These are not architecture failures.

### Genuine learned-model failures

- v0.22 / job 575825 — relation-conditioned multilayer interface failed causally/preservation despite valid execution.
- v0.36 / job 575912 — dual-view specialist failed despite solved edge binding and good preservation.
- earlier relation-semantic grounding / role residual / query-relation-role router runs likewise produced valid learned failures.

### Genuine learned-model success

- QSRE T1 / job 575914 — exact P100 run reached the governed DEV contract at step 50.

Therefore the statement “all GPU stages fail” is not literally true. But the broader observation is correct:

> the first environment that combines learned outputs + exact runtime + exact evaluator has repeatedly exposed assumptions that CPU/static gates did not exercise.

---

## 4. Smoking gun: validator correlation in job 575934

The recovery sbatch and runner contain malformed parameter expansion with two dollar signs before a brace expression.

In Bash, two dollar signs expand to the current process id. That is why the path became a process-id prefix followed by the literal brace expression.

More important than the typo itself: the corresponding CI shell-boundary test expected the malformed form.

So the test and implementation were not independent.

This is a correlated-oracle problem:

> a validator generated from the same mistaken representation as the implementation can certify the mistake.

Future runtime tests must test behavior, not source spelling.

For shell/runtime contracts:
- execute the expansion under a controlled environment;
- assert the resolved value;
- do not merely grep for the expected literal string.

---

## 5. Direct read of the T1 data

The corrected T1 curriculum is intentionally structural.

Every field text is generated as:

`neutral evidence record <16-hex hash>`

The field semantics therefore contain no natural relation content.

This is appropriate for T1's causal question:

> if operator + support are oracle, can the relational executor perform role selection, relation filtering, aggregation, path traversal, reliability arbitration, temporal arbitration, plurality, fallback/defer, and distractor isolation?

T1 is deliberately not a natural-language realism test.

The T1 P100 PASS is therefore valid but narrow.

It proves the executor/readout mechanics can be learned over the governed structural workload.

It does not prove that actual A.L.I.C.E. evidence text will be bound or interpreted correctly.

---

## 6. Direct read of the T2 data

T2 does not currently use real user-style relation queries.

It generates query text from compact phrase banks keyed directly by the target operator.

Examples of highly available cues:

- aggregate/plural operations use phrases like “return all records”, “aggregate”, and “full plural answer”;
- reliability arbitration says “most reliable”, “highest-confidence”, or “most trustworthy”;
- temporal arbitration says “latest”, “most recent”, or “newest”;
- ordered paths explicitly say “follow ... then ...”, “first ... second”;
- fallback rows explicitly say “non-relational”, “ordinary evidence path”, or “fall back”;
- defer rows explicitly say “defer”, “do not guess”, or “relational intent is unresolved”.

Relation identity is also encoded through small synonym banks for SUPPORTS, CORRECTS, SUPERSEDES, DERIVED_FROM, CAUSES, and TEMPORAL_SUCCESSOR.

TRAIN and DEV phrase banks are disjoint, which is better than phrase reuse, but both splits are produced by the same small family-specific grammar.

That means T2 v0.1 is currently closer to:

> Can the frozen semantic model plus T2 head decode a generated operator language?

than:

> Can A.L.I.C.E. infer relational intent robustly from realistic public N0 language?

That difference matters.

The current T2 data is useful as a unit test of operator learnability. It is not yet a representative semantic benchmark.

---

## 7. Shortcut risk in the current T2 curriculum

The intended semantic factors and the easiest surface cues are too aligned.

A nonlinear classifier can potentially identify:
- operation from operation-specific keywords;
- control from fallback/defer vocabulary;
- role from active/passive directional phrasing;
- relation from a small synonym set.

Therefore perfect DEV performance could still be achieved without the robust compositional operator representation needed by the full EIPM workload.

This is the same kind of benchmark issue we already caught in T1 v0.1 when answer position and path structure leaked family information.

T2 needs the same level of skepticism.

Before another gradient run we need to know whether simple surface heuristics can solve most of the benchmark.

If they can, the benchmark is not a strong architecture gate even if the T2 model reaches 1.0.

---

## 8. Direct read of real public N0 data

The frozen semantic backbone was trained on much richer public language than the T2 template grammar.

Committed public N0 examples include:
- pragmatic implicature;
- presupposition and focus;
- indirect speech acts;
- domain-sensitive word sense;
- evidence/frame conflicts;
- counterfactual provenance;
- uncertainty calibration;
- causal equivalence;
- relationship-sensitive interpretation.

Examples include natural scenarios such as:
- a flight cancellation before an interview;
- a friend versus supervisor saying the same words;
- a structured shipping state conflicting with raw evidence;
- counterfactual bus statements;
- causal paraphrases such as “failed because” versus “caused ... to fail.”

This is closer to the semantic environment N0 is supposed to serve.

The next operator audit must connect QSRE to this kind of data rather than continuing to evaluate only generated relation instructions.

---

## 9. Current QSRE architecture: what remains supported

The present evidence does not justify abandoning QSRE.

Supported pieces:

### T1 executor/readout

T1 passed a real P100 learned-model gate.

### Query-conditioned operator factorization

Separating relation sequence, argument role, operation, applicability/control, and a continuous residual state still addresses the failure lineage better than the earlier global-field-residual designs.

### Full semantic hidden-state access

Prior audits showed token-level relation/binding information exists and can outperform pooled representations.

### Structural support-local execution

This directly fixes the old failure where correct edge binding merely nudged a global field softmax.

### Separate fallback

This remains architecturally justified.

What is not yet proven is whether the current T2 operator encoder and data can recover those factors robustly from realistic public language.

T2 has no valid model result yet.

---

## 10. Current T2 architectural tensions to audit

These are unresolved questions, not declared bugs.

### A. Discrete heads are easier than the intended continuous operator

The model predicts discrete relation/role/operation/control heads. The continuous state has an auxiliary supervised-contrastive signature objective but is not used by frozen T1.

That is acceptable for T2 isolation, but T2 success would primarily validate discrete operator extraction, not the final continuous operator semantics.

### B. Two relation slots are an experiment shape

The current two-slot relation sequence is explicitly not a product ceiling and must not become one accidentally.

### C. Downstream T1 is an evaluator, not the T2 training objective

The T2 loss is classification plus contrastive operator loss. T1 behavior is checked downstream.

That is good for causal isolation, but it means the operator benchmark itself must be especially resistant to shortcuts.

### D. Realistic ambiguity is underrepresented

Fallback/defer examples currently announce their control state unusually directly.

The full system will encounter ambiguous relation intent without words such as “defer” or “non-relational.”

---

## 11. What the next large workflow must be

Not another training run.

The next work should be one coherent reality audit, with all outputs produced together before a single architecture/training decision.

### Track 1 — Runtime parity

Exercise actual behavior rather than source strings:
- Bash parameter expansion under controlled environment;
- same uDocker entrypoint;
- exact repo/workdir/PYTHONPATH transfer;
- random/provisional operator outputs including invalid combinations;
- evaluator total over the model's full output domain.

### Track 2 — Data reality

Build a public-only relation/evidence inspection set from actual N0-style language:
- entity-rich natural text;
- relation paraphrases without relation names;
- role reversals;
- realistic ambiguity;
- reliability and temporal cases without announcing the operation label;
- multi-support and path cases phrased like normal questions;
- topic/lexicon/template split isolation.

No private identity data.

### Track 3 — Shortcut diagnostics

Measure how much of the T2 target can be recovered from shallow cues:
- bag-of-words / keyword heuristics;
- family-only priors;
- operation cue words;
- relation synonym matching;
- sentence-position or length features.

If a trivial baseline performs strongly, redesign the benchmark before judging the architecture.

### Track 4 — Frozen semantic sufficiency

On the realistic public set:
- no-gradient probes across semantic layers;
- relation information;
- role information;
- operation/control information;
- paraphrase invariance;
- counterfactual sensitivity.

This decides whether T2 should use the current frozen semantic backbone or whether N0 semantic representation itself must be reopened.

### Track 5 — Single architecture decision

Only after Tracks 1–4:
- keep current T2;
- change the T2 operator interface;
- or reopen semantic representation learning.

One decision. No chain of local fixes.

---

## 12. Immediate governance decision

The v0.54 recovery authorization is revoked.

Do not submit another GPU job from the current recovery scripts.

The current scripts are known-bad and are retained as evidence of the validation failure.

No T2 optimizer, gradient, GPU, T3 support learning, TEST opening, challenge opening, private identity gradient, or production promotion is authorized while this audit is active.
