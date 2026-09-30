# MFM native training execution handoff

**V1 update:** This handoff is historical. The owner selected a new licensed
**pretrained/base** Gemma 4 route in
[`MFM_V1_LICENSED_BASE_DIRECTION_2026-09-30.md`](MFM_V1_LICENSED_BASE_DIRECTION_2026-09-30.md).
The old instruction-tuned Gemma snapshot, processor and hardware receipts
cannot be reused or purchased against that direction.

**State, 2026-09-29:** the former Gemma 4 12B staging, transfer, four-A100
probe and fit commands in this handoff are retired by the
[native lineage decision](MFM_NATIVE_LINEAGE_DECISION_2026-09-29.md). No native
foundation or specialized MFM has been trained. Do not rent MFM GPUs, stage
Gemma weights, or run the earlier Gemma-bound processor receipt under the
name of native MFM. This document records what must be prepared before a
new training command and hardware purchase can be made concrete.

## Two learned stages and their input closure

**Stage A: public-data foundation.** Select a language and sensory
architecture with fresh-initialized parameters. Train general semantic,
evidential, temporal, relational, code and image/audio/video competence on
actually available public or otherwise permitted source packs. As in N0,
owner-authorized assistant teachers may create examples, comparisons or hard
negatives with exact origin and rights records. Their outputs are training
candidates, not imported weights, independent gold or personal truth. Freeze
the tokenizer/processor, initialization, corpus and exact code in a lineage
receipt. There is no N0 weight dependency and no inherited external base
model.

**Stage B: formation specialist.** Initialize from the exact first-party
Stage A checkpoint, add any fresh formation/fusion/output parameters, and
train source attribution, uncertainty, subject separation, time, competing
interpretations, correction/deletion, episodes and typed evidence-citing
proposals. The specialist remains behind the deterministic memory gate.
Any scope-specific adaptation built by FBM for one user remains private and
versioned. A user-owned public pack can alter the authorized training set;
it does not make a fictional account evidence of that user's actual life.

FBM must recommend rights-cleared public base packs and let a user select or
add them when constructing an instance. Inventory exact source IDs and
revisions, rights for training and distributing resulting weights, duplicate
families, licenses and required notices, language, source systems, sensory
modalities and temporal/long-context coverage. Measure whether the selected
combination can support the intended claims. If coverage is lacking, report
the missing capability and request better sources or compute instead of
silently shrinking the goal or fabricating an identity. A verified first-party
shared checkpoint can save repeating all public pretraining for every
consumer, but the public-pack selection and per-instance construction must
still have explicit lineage.

## Existing frozen formation material

These three files remain outside the public branch and retain their exact
training-only meaning. They are suitable as one Stage B curriculum tier after
the Stage A representation is competent. They are **not** a broad Stage A
pretraining corpus, independent development, or a sealed FINAL.

| File | Cases | SHA-256 |
| --- | ---: | --- |
| `multisource_formation_train.jsonl` | 43,819 | `2595e5209c0cf9cc8077e330b9e288c08cf5752f6db46acde6a455ff781519a1` |
| `longitudinal_formation_train.jsonl` | 6,000 | `2c92302390ebcf46ea222df8cf1c79b75710fdfb7a47a684c4897b9e0f6ae830` |
| `mfm_training_mixture_v1.json` | 49,819 | `60fa44a5be62c8d2f0daf508e543dc1a41c34562dc0c4d399e26c9cabb772e21` |

The superseded 71,659/77,659-case draft gave identical inputs
contradictory single-domain targets. The corrected mixture has one combined
six-domain target for each source state and explicit day granularity; the
6,000 fictional cases remain unreviewed training-only material. See
[`MFM_CORPUS_SOURCE_RECEIPT_2026-09-29.md`](MFM_CORPUS_SOURCE_RECEIPT_2026-09-29.md)
for provenance and limitations. Neither this mixture nor a fresh-initialized
model on its own demonstrates general language or sensory ability.

## Before any paid training allocation

1. Assemble actual broad, permissioned Stage A text/code and image/audio/video
   source packs, plus the attributed teacher candidates. Check rights,
   duplicates, exact source bytes, contamination, intended modalities and
   coverage. Version user choice and FBM recommendations. Freeze independent
   evaluation families before optimizer access. **Do not submit a Stage A
   pretraining GPU job until these real packs and their admission receipts
   exist.**
2. Implement and CPU-verify the fresh-initialized Stage A architecture and
   processor. Its model-load path must reject Gemma and all third-party
   pretrained core weights. Bind parameter counts, tensor initialization,
   tokenizer/processor origin, code and complete input manifests. Define a
   training and checkpoint route that can resume the real objective without
   dropping a claimed modality or source family. A config-only constructor
   and a schema test are insufficient learning evidence.
3. Implement Stage B loading **only** the exact Stage A first-party artifact.
   CPU-process the complete formation curriculum and broader multilingual,
   media and long-history material. Freeze separately sourced development
   and custodied FINAL. Record critical false-memory, provenance, subject,
   temporal and deletion cases individually. A source anchor validates
   location; it does not prove semantic entailment.
4. Stage both admitted corpora, the code and first-party checkpoint on
   runtime-visible durable storage. Rehash locally and confirm dependencies,
   free space, planned checkpoint/export path and actual accelerator type.
   Then run a bounded **full forward/backward/optimizer/checkpoint** hardware
   step, measure memory and throughput, and price a full run using those
   measurements. Bind resume and export to the same exact data and source
   lineage. Hardware success does not qualify formation behavior.
5. Promote only after independent held-out tests and the end-to-end
   experience -> MFM -> gate -> projections -> retrieval -> native judgment
   -> outcome loop pass. Evaluate changes after correction, deletion,
   model replacement and distinct host histories. Do not relabel a training
   loss, a CPU preflight or a single GPU step as full MFM capability.

## Magnolia and rental notes retained from observed diagnostics

On 2026-09-29 the Magnolia `node` CPU partition was available, but
`node016.cluster` could not resolve `huggingface.co`; an earlier N0 Magnolia
attempt also failed on DNS. A future *permitted, selected* public source
pack or first-party checkpoint should be staged outside that compute node and
copied through the documented file gateway, then verified on a CPU node.
Keep MFM separate from the N0 container and run no large data pass on a login
node. The observed home volume had 3.1 TB free on a shared filesystem; it
was not a reserved quota. These facts do not authorize transferring the
retired Gemma snapshot.

The older Magnolia inventory listed P100/K80 GPUs and the Kaggle L4 request
once resolved to T4 hardware. Neither observation proves the capacity of a
new native two-stage model. The previous four-A100 suggestion and 750 GiB
storage gate belonged to the Gemma-specific 12B route and cannot be reused
as a native cost or hardware estimate. Select hardware only after the real
architecture, public corpus, checkpoint plan and CPU receipts exist. No paid
MFM GPU purchase is requested at this stage.
