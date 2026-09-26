# FBM received-data audit, 2026-09-26

**Scope:** `fable-builder-model@cf04d1ef`; `training/fbm/` and `docs/fable-builder/traces/`. This is a file and schema audit, not a model evaluation or a rights determination.

## Inventory and admission decision

| Existing material | Measured contents | Appropriate use now | Missing for a full-entity FBM |
| --- | --- | --- | --- |
| `training/fbm/bootstrap/sol_curriculum_seed_v0.1.jsonl` | 60 Sol-authored N0 `candidate_ranking` rows; 28 train, 32 dev; 58 have three candidates, two have two. All rows have a prompt, preferred indices, rationale, and competency. | Owner-authorized, provenance-bound N0 semantic teaching/diagnostic material under its origin manifest. Keep the current bytes and the historical split. | No builder input-to-decision cases, source/subject map, MFM or host/self/relationship assembly, observed outcome, independent label, source-family holdout, or authority decision. It cannot establish general FBM transfer. |
| `training/fbm/bootstrap/sol_curriculum_seed_v0.1.origin.json` | Identifies the 60-row blob `e926ad2c...`, source commit `610c8098...`, owner-asserted authorization and restrictions on private identity gradients. The recorded `training/eipm/n0/...` path exists at that source commit with the same Git blob. | Preserve the historical provenance. The FBM branch's copy lives at `training/fbm/bootstrap/...`; this is a mirror, not an invalid source. | A separate per-pack distribution/portability determination and downstream derivative ledger. The owner's project-use assertion does not prove redistribution rights. |
| `training/fbm/builder_events/*.json` | Two N0 operation narratives: failure-driven teaching and lineage-bounded compute. | Procedure seeds and engineering history. | No reproducible actual input state, choice/target pair, independent outcome, or cross-person transfer case. |
| `docs/fable-builder/traces/*.jsonl` | 107 files; 266 nonblank record lines after repairing two terminal literal `\\n` suffixes. Before repair, 264 lines parse as JSON. Of those 264, only 99 have `operation_type`, 84 have `input_refs`, 90 have `training_value`; 212 lack a `schema` field. Mixed legacy formats are present. | Audit trail, negative lessons, reusable methods, and candidates for later case reconstruction. A `training_value` tag is an author annotation, not proof of paired supervision. | Many rows have no source payload, checked target, outcome, permission map, and generator/person split lineage. Do not pool all rows as supervised examples. |

### Specific seed weaknesses

- 56 of 60 rows have `preferred_indices=[0]`, two have `[0,1]`, and two have `[2]`. A trivial policy selecting index 0 is in the preferred set for 58/60 rows; this makes accuracy on the historical rows a weak measure of semantic learning. Among the 32 dev rows, it is preferred in 30. Preserve the original data and split for reproducibility; create **new** position-balanced, independently checked challenge rows and assess a position-only baseline before citing progress.
- The only source family is `sol_authored`; there are no distinct-person or generator-held-out examples. Reusing this seed as FBM train and claiming its dev split independently validates builder transfer would be leakage of task and author, even though the prompts differ.
- The three relation-label candidates are intentionally identical in one train row and one dev row (`N0-SOL-0005`, `0006`), with different prompts and correct answers. That is ordinary label reuse, not duplicate-example contamination.

## Current admission rules

1. **N0 seed:** eligible only for its documented N0 curriculum purpose under the recorded owner authorization; not an FBM builder-target corpus or sealed FBM evaluation. Do not alter historic rows or retrospective N0 receipts.
2. **Builder events and traces:** process/provenance material only. A positive or evaluation tag does not admit a trace as a train/eval case. Require a linked versioned case with actual permitted input state, speaker/subject and rights map, alternatives including abstention where needed, externally checkable decision, independent outcome or explicitly unknown outcome, and split/generator lineage.
3. **Fictional contract fixtures:** can exercise parsers, authority routing, subject isolation, correction/deletion, and full-entity interface shape. They are authored examples with no independent outcome and must not be counted as train data, held-out evaluation, observed experience, or release evidence.
4. **Private personal evidence:** remains in authorized local custody. No raw private person material, adapted weights, or reconstructable derivatives enter a shared FBM seed. Source-person/host/relationship/assistant-self axes remain distinct.

## Required additions, in order

1. Freeze this inventory and an explicit manifest of corpus purposes; repair the two JSONL formatting defects without changing their JSON objects.
2. Exercise the linked-case shape on a tiny fictional multi-person fixture. Capture opened evidence, subject role, authority, candidate action, rejected alternative, correction dependency, and `outcome=unknown` where nothing actually occurred. Keep it out of train/eval metrics.
3. Collect multiple **independent** fictional source families with documented generation provenance plus genuinely authorized input-to-target builder operations. Include source attribution, outside quotes, unsupported inference/UNKNOWN, aspirations versus observed behavior, MFM memory use, host and self roles, relationships, native verdict and expression, correction/deletion, and later outcomes. Separate producer-generated labels from independently checked targets.
4. Freeze person/source-family and generator-family held-outs before builder optimization. Record independent authority and causal tests and compare position-only, rules, and learned baselines. Only then decide which examples are eligible for FBM training and whether the complete assembled entity qualifies.

N0's identity-neutral full-envelope workflow remains separate. Job `576166` exited before GPU measurement because its evidence root already existed. An earlier P43 attempt, job `576093` on `4270bfa2`, failed a half/float scatter during the two-rank fp16 forward and left the root empty. N0 code repair is now at `168c0315`; Magnolia CPU autocast job `576167` ran on its predecessor `657ba2a4`, with no result provided in this review. Preserve the existing roots and repeat source-bound CPU, mixture, and GPU gates only after runtime verification. Neither the revised code nor a failed preflight grants GPU memory or training authority.
