# MFM source receipt and formation objective

**State, 2026-09-29:** owner-authorized synthetic training curricula prepared.
No independently adjudicated FINAL, trained weights or GPU run exists. These
curricula can train a model; they do not certify its generalization. Simulator
truth and upstream QA are excluded from MFM inputs.

## Licensed source intake

The [Multi-Source Memory Benchmark](https://github.com/TianchengY/multisource-membench)
generator was cloned at `7a44de847315c244d675646b6e92910474493526`.
Its code is [Apache-2.0](https://github.com/TianchengY/multisource-membench/blob/main/LICENSE).
The upstream [`data/DATA_LICENSE`](https://github.com/TianchengY/multisource-membench/blob/main/data/DATA_LICENSE)
describes synthetic personas, generator outputs and renders as CC BY 4.0,
with attribution and a carveout for cached third-party model API outputs.
The [dataset card](https://huggingface.co/datasets/ytc1997/multisource-membench)
repeats the distinction. Preserve upstream notices and CITATION.cff for any
distribution. This local run used deterministic Python generation, not its
cached model outputs or published benchmark QA.

Reproduction from that pinned clone, with `PYTHONPATH=src`:

```bash
python -m survey2agent.data_generation.generate_personas --seed 20260929 --output-dir ../mfm-multisource-new-seed
python -m survey2agent.data_generation.generate_events --dataset-dir ../mfm-multisource-new-seed
python -m survey2agent.data_generation.generate_sources --dataset-dir ../mfm-multisource-new-seed
python -m survey2agent.data_generation.generate_ground_truth --dataset-dir ../mfm-multisource-new-seed
```

The final command produced upstream QA labels for source-chain inspection;
`inventory_multisource_sources.py` **excludes them** from MFM inputs. From the
MFM worktree, run:

```bash
python scripts/mfm/inventory_multisource_sources.py --dataset-dir ../mfm-multisource-new-seed --generator-repo ../mfm-generator-reference --seed 20260929 --output ../mfm-multisource-new-seed/mfm_source_inventory.json
```

The inventory SHA-256 is
`a5fcd53f84e049f84a2aaf0e1902da17a2d9d8fd4eec85deccb4bf8902c4d362`.
It records 480 distinct synthetic personas, 2,400 structural source files
(self report, planner, objective log, device log, profile), 30 days per
persona, exact source and ancestor event-table hashes, the generator commit,
seed, code hashes and license hashes. The whole output tree before this
inventory had 3,843 files, 75,359,948 bytes and canonical `(path,size,sha256)`
manifest digest `3930d31bb1b1a05ffcf64fc2f89b8ac3d95ce6e4b3f2387dc0ae2f758a594996`.
The inventory itself is outside the public branch and can be regenerated.
Its status is `source_candidate_only_no_formation_labels_or_review`.

## Owner-authorized diagnostic material

`generate_longitudinal_candidates.py --seed 20260929 --hosts 600` created
6,000 fictional diagnostic cases, SHA-256
`5de85384f0428fb168bc796fb53bd21d68996f383520143381a84e090cc6aab1`.
Its source templates and target labels were authored through this ChatGPT/Codex
session as owner-authorized teaching material for A.L.I.C.E., Fable and FBM.
Record that origin and authorization with the generator. The prior
ChatGPT-specific rights quarantine was an unsupported self-classification and
is withdrawn. These cases remain **unreviewed teaching examples**, not final gold:
the three renderers share the same ten scenario skeletons across nominal
train/development groups. Those groups cannot count as scenario-isolated
evaluation, and no independently custodied FINAL was created.

## Frozen training mixture

`assemble_formation_curriculum.py` extracts training targets only from the
fresh-seeded observed structural streams. It never supplies simulator truth or
upstream QA as model input. Across 480 hosts, it produced **43,819** cases:
14,400 multi-domain plan-only decisions, 14,400 multi-domain
plan-then-report decisions, 12,139 tracker reconciliation decisions, 1,920
seven-day bounded patterns, and 960 as-of day-15/day-30 bounded histories.
Every plan/report/tracker source state has one complete target across the six
supported domains. A prior 71,659-row draft had 28,800 identical source
inputs with conflicting single-domain answers and was replaced before GPU
training. The corrected curriculum SHA-256 is
`2595e5209c0cf9cc8077e330b9e288c08cf5752f6db46acde6a455ff781519a1`.
Source-grounded structured targets now carry field-level UTF-8 byte spans;
day-scoped proposals and weekly/monthly patterns have explicit end markers.
Date-only source fields are normalized to UTC-midnight **day markers** with
`temporal_granularity=day` in contract v1.5.0, not assertions about a precise
observation or ingestion time. A same-day `valid_from`/`valid_to` pair denotes
the inclusive calendar day, and a first-to-last pair denotes inclusive
calendar dates. Recorded clock time remains unknown. A tracker signal
agreement is a signal observation, and disagreement is uncertainty; neither
establishes the host's physical exercise outcome.
The code-generated targets are a source-grounded synthetic training tier.
They are not independently adjudicated semantic gold.

`convert_longitudinal_candidates.py` compiled all 6,000 owner-authorized
fictional teaching cases into the same structured training format, SHA-256
`2c92302390ebcf46ea222df8cf1c79b75710fdfb7a47a684c4897b9e0f6ae830`.
Their nominal development renderer rows are explicitly reassigned to training
only. The combined **49,819-case** manifest is
`mfm_training_mixture_v1.json`, SHA-256
`60fa44a5be62c8d2f0daf508e543dc1a41c34562dc0c4d399e26c9cabb772e21`.
It binds both component hashes, counts, generator families and the prior
`owner_authorized_service_teacher` authorization. The complete JSONL and
manifest are reproducible work artifacts outside this public code branch.

This mixture broadens training beyond the original ten template narratives.
The ten template families still use whole-utterance citations; the structural
curriculum supplies the narrower field-span supervision.
It is still synthetic, primarily structured/text, and does not cover every
audio, video, image, multilingual, deletion/revocation or lived-history case.
Those gaps are qualification and later curriculum work, not a legal hold on
starting a measured learned run.

## Qualification targets and learned objective

Independent annotators must work from the *observed* structural streams,
without upstream `event_table.json`, `ground_truth.json`, profile generator
knobs, published QA or model-generated target text. Freeze a source snapshot
first. For every candidate interval and source event, author a structured
formation target with:

1. Exact source ID/digest and narrow byte range or source-native locator;
   actor, speaker, subject, modality, observed/recorded times, duplicate and
   ancestor lineage, plus the interpretation's semantic text.
2. Type, domain, subject, epistemic status, validity interval, uncertainty,
   contradictory evidence and prior-claim revision. Record competing
   hypotheses and `UNKNOWN` rather than inventing a continuous biography.
3. A scoped disposition: propose, defer, retain raw, or abstain, with the
   particular claim/skill/outcome scope. A scheduled plan may be proposed
   while its unobserved completion is deferred. Correction, deletion and
   revocation requests need explicit *target source/derivative IDs* and must
   never be labeled as proof of completed erasure.
4. Source-supported outcome and rationale, changing preference and negotiated
   relationship norm, host/source-person/assistant-self separation, and
   permission/deletion influence. Add multilingual and multimodal material
   from separate rights-cleared families; this simulator alone covers neither.

For independently claimed development and FINAL results, two independently
authenticated reviewers must inspect exact source and target bytes blind to
the author's label, adjudicate disagreements, and attach verifiable receipts.
The steward confirms source permission and revocation state. This standard
does not require manual dual review of every synthetic training row. Hashes
and self-asserted JSON reviewer IDs alone only prove byte identity.

Freeze train, development and separately custodied FINAL by connected host,
source/duplicate, generator, scenario and ancestor families. The current
single Multi-Source generator cannot fill all three disjoint generator-family
splits. Additional independently generated or permissioned families are
required, and FINAL bytes must remain unopened to training/model selection.
An external custodian can verify FINAL after model and evaluator freeze.

The learned artifact must optimize evidence-span attribution, semantic claim
content, type/domain/subject, event and validity time, temporal update,
contradiction, scoped disposition, deletion target, uncertainty/abstention and
calibration jointly. Fast capture and slow consolidation are separately
measured. Compare on identical opened sources with strong retrieval,
long-context and extraction baselines. Count unsupported memories, omissions,
misbound subjects, deletion errors and downstream judgment outcomes separately;
no aggregate metric can conceal a critical failure. Checkpoint, optimizer,
tokenizer/encoders, data rights, sources and exact compute environment must be
versioned. The authority gate still decides whether a proposal changes memory.

**Native-lineage correction, 2026-09-29:** this mixture is formation-task
supervision, not a public foundation-pretraining corpus. The former pinned
Gemma processor and paid training handoff are superseded by
[`MFM_NATIVE_LINEAGE_DECISION_2026-09-29.md`](MFM_NATIVE_LINEAGE_DECISION_2026-09-29.md).
Build a separately sourced, fresh-initialized, identity-neutral foundation
from eligible public data and governed teacher material. Train the MFM
formation objective on top of that first-party lineage. New architecture,
tokenizer/media processor, data and runtime fingerprints require new CPU and
GPU receipts. The 49,819 cases remain available as training-only formation
material, with no independent FINAL or capability claim.
