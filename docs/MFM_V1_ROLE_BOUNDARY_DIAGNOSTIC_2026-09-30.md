# MFM V1 prepared-base and formation-role diagnostic

**Status:** CPU scorer and frozen public fixture only. No 12B forward pass,
prepared base, trained formation component or model-quality result exists yet.
The pristine `google/gemma-4-12B` checkpoint is an optional descriptive
reference; no source-only forward pass is reported here. Fable's
MFM operating base is a distinct, hash-linked MFM role clone of the publisher
source or an evidence-backed modified derivative, plus separately trained
formation weights. The older `-it` trainer is excluded. No tensor change is
required before specialist training.

`scripts/mfm/qualify_v1_role_boundary.py` compares two outputs from the actual
first-party specialist inference runner on the same seven public synthetic
cases. A third publisher output is optional descriptive context:

| Run | Model state | Purpose |
| --- | --- | --- |
| `trained` | Prepared base plus trained specialist weights | Inspect source-bound, non-authoritative memory proposals. |
| `seeded-untrained` | Same prepared base plus the saved decoder weights from before the first optimizer step | Measure whether specialist training improves the same role. |
| `untouched` (optional) | Pinned publisher pretrained checkpoint and its language head | Describe inherited defaults. It uses a different decoder and a 1024-token cap, so it is not a matched training control. |

The two specialist rows come from
`scripts/mfm/run_v1_formation_specialist.py --control trained` and
`--control seeded-untrained`. They record the exact input-file digest,
context, source and prepared-base lineage, selected specialist weight digest,
sealed receipt, control kind, deterministic decoder settings, token IDs, raw
completion, EOS and validation status. Their weight and receipt digests must
differ while the prompt, prepared-base receipt and decoding settings match.
The prepared-base weight digest may equal its publisher parent for an exact
role clone, while its separate clone receipt differs. The qualifier rehashes
the local base, component and seed files and checks their common sealed run
manifest before it can report a diagnostic pass. This does not independently
prove that a recorded JSONL completion was actually emitted by those files.

The scorer CLI rehashes the eight publisher files against pinned values and
verifies the corresponding source receipt before scoring. Its fixture is
`tests/fixtures/mfm/base_behavior_diagnostic_v1.json`; it opens no private
Elaina, Rayan or consumer data and makes no network requests. Generate the
unlabeled prompts with `evaluate_base_behavior.py emit`. The untouched runner
loads the exact source. The prepared-base runner verifies the MFM role clone
or derivative through `alice_foundation.gemma4_v1.verify_role_base`, checks a
loopback-only Linux network namespace when requested, uses local files only,
and applies the identical greedy decoding control. The default accepts active interfaces for this
**public synthetic diagnostic only** and records that network isolation was
not established. The prepared runner also pins the complete emitted prompt
file digest; a `public_synthetic` label on different text cannot admit it:

```bash
PYTHONPATH=src:/path/to/foundation/src \
  python scripts/mfm/run_gemma4_prepared_base_behavior.py \
  --snapshot /models/mfm-prepared-base \
  --receipt /receipts/mfm-prepared-base.json \
  --prompts /cases/public-synthetic.jsonl \
  --output /results/mfm-prepared-base.jsonl
```

Pair its output with the untouched result using:

```bash
PYTHONPATH=src python scripts/mfm/evaluate_base_behavior.py score \
  --untouched /results/untouched.jsonl \
  --edited /results/mfm-prepared-base.jsonl \
  --output /results/prepared-vs-untouched.json
```

For an unmodified role clone, paired source and clone runs are expected to
behave the same. This is a custody check, not an edit efficacy claim. Both
are public diagnostics only. `--require-network-isolation` makes the
runner fail if it observes a non-loopback interface. Even a loopback-only
interface check does not exclude a same-host Unix socket proxy. An externally
isolated process remains required before private evidence may be opened.
Run the real specialist inference command from the
[V1 training path](MFM_V1_SPECIALIST_TRAINING_PATH_2026-09-30.md) twice on the
same emitted JSONL. Set `--control trained` and `--control seeded-untrained`
with separate output paths and run IDs. Once the results exist, run:

```bash
PYTHONPATH=src python scripts/mfm/qualify_v1_role_boundary.py \
  --snapshot /models/gemma-4-12B-pretrained \
  --source-receipt /receipts/gemma-4-12B-pretrained.json \
  --component-dir /secure/run \
  --prepared-base-dir /secure/mfm-role-clone \
  --prepared-base-receipt /secure/mfm-role-clone.json \
  --preflight-receipt /secure/processor-preflight.json \
  --untouched /results/untouched.jsonl \
  --trained /results/trained-output.jsonl \
  --seeded-untrained /results/seeded-output.jsonl \
  --output /results/mfm-role-diagnostic.json
```

The output reports per-case critical errors, omissions and modality gaps.
`formation_contribution_cases` requires the seeded control to score worse on
at least one matched case. `role_boundary_diagnostic_pass` also requires a
rehash of both specialist artifacts, no trained-case errors and complete
trained/control fixture coverage. The optional publisher reference may leave
the image case unexercised; that appears separately under
`reference_coverage_gaps` and does not hide the specialist image result. This
remains a small public diagnostic, and `qualification_claim` is always false.
Running this scorer without local artifacts cannot yield a pass. An output
file could still be fabricated, so independently observed execution remains
unmeasured.

Before any capability promotion: verify actual prepared/component artifact
bytes and training lineage; run a process under an OS-enforced no-egress
boundary before it reads private data; freeze independent cross-person,
long-history and multimodal FINAL; compare stronger baselines and wrong-person
swaps; exercise MFM proposals through the Claim gate, projections, retrieval,
native judgment and observed outcome. The 49,819 formation cases are
training-only. A policy flag, fabricated output file or this scorer's unit
tests cannot substitute for these receipts.
