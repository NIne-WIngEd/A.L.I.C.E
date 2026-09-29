# MFM source receipt and formation objective

**State, 2026-09-29:** source inventory and diagnostic candidates only. No
independently adjudicated formation corpus, sealed FINAL, trained weights or
GPU run exists. This receipt does not promote simulator truth, QA labels, or
assistant-authored targets into formation gold.

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

## Quarantined diagnostic material

`generate_longitudinal_candidates.py --seed 20260929 --hosts 600` created
6,000 fictional diagnostic cases, SHA-256
`4089c61e6bf9b0e309ef23fc09a521592ebdbf74b5f982e19d37d4a678ea55ac`.
Its source templates and target labels were authored through this ChatGPT/Codex
session. The [OpenAI Terms of Use](https://openai.com/policies/terms-of-use/)
restrict use of Output to develop competing models. Whether the intended
Fable model legally falls into that category requires a separate rights
determination. Ownership of Output alone does not settle that question.
Therefore these cases remain outside distributable-weight training absent
written clarification or independent permission. The three renderers share
the same ten scenario skeletons across nominal train/development groups; those
groups cannot count as scenario-isolated evaluation. No FINAL was created.

## Target and learned objective required before a GPU run

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

Two independently authenticated reviewers must inspect the exact source and
target bytes blind to the author's label, adjudicate disagreements, and attach
their signed or otherwise verifiable receipts. The steward must confirm live
consent/license rights for commercial distributable-weight training, no
revocation, and generator/provider ancestry. Hashes and self-asserted JSON
reviewer IDs alone only prove that a claim stayed byte-identical.

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

**Next executable handoff:** commission independent formation annotation and
rights verification on the frozen source inventory, add other licensed
longitudinal/generator/modal families, then freeze three nonleaking splits.
Only those admitted bytes can be handed to Magnolia or Kaggle for a measured
learned run. The present data cannot authorize that run.
