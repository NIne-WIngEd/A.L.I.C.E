# MFM V1 prepared-base and formation-role diagnostic

**Status:** CPU scorer and frozen public fixture only. No 12B forward pass,
prepared base, trained formation component or model-quality result exists yet.
The pristine `google/gemma-4-12B` checkpoint is a measured reference. Fable's
MFM operating base must be a distinct, hash-linked prepared derivative with
separately trained formation weights. The older `-it` trainer is excluded.

`scripts/mfm/qualify_v1_role_boundary.py` accepts three independently produced
JSONL output files on the same seven public synthetic cases:

| Run | Model state | Purpose |
| --- | --- | --- |
| `untouched` | Pinned publisher pretrained checkpoint only | Measure inherited defaults and compare general behavior. Never promote it directly. |
| `assembled` | Prepared base plus trained formation component | Inspect source-bound, non-authoritative memory proposals. |
| `ablated` | Same prepared base and recorded component with the component disabled | Measure whether learned MFM parameters actually affect the role. |

Each row uses the existing
`scripts/mfm/run_gemma4_base_behavior.py` output fields: `case_id`,
`context_digest`, `prompt_set_sha256`, `source_repository`, `source_revision`,
`generation`, `status`, `processed_modalities`, `output_text`,
`model_artifact_digest` and `model_receipt`. Preserve `raw_output_text` so
special tokens cannot disappear through display decoding. The latter two runs
also supply `base_source_sha256`, `prepared_base_parent_sha256`,
`prepared_base_sha256`, `prepared_base_receipt_sha256`,
`formation_component_sha256`, and `formation_component_active`. The first two
source ancestry fields are the pinned pretrained weight SHA-256. The prepared
base digest must differ. Both specialist runs must name the same prepared base
and formation component, and differ only in whether it is active. The runner
that eventually emits these rows must separately authenticate the prepared
base and component artifacts; metadata alone cannot establish their bytes or
prove the ablation truly occurred.

The scorer CLI rehashes the eight publisher files against pinned values and
verifies the corresponding source receipt before scoring. Its fixture is
`tests/fixtures/mfm/base_behavior_diagnostic_v1.json`; it opens no private
Elaina, Rayan or consumer data and makes no network requests. Generate the
unlabeled prompts with `evaluate_base_behavior.py emit`. The untouched runner
loads the exact source. The changed-base runner verifies the MFM derivative
through `alice_foundation.gemma4_v1.verify_derivative`, checks a loopback-only
Linux network namespace when requested, uses local files only and the identical
greedy decoding control. The default accepts active interfaces for this
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

Both are public diagnostics only. `--require-network-isolation` makes the
runner fail if it observes a non-loopback interface. Even a loopback-only
interface check does not exclude a same-host Unix socket proxy. An externally
isolated process remains required before private evidence may be opened.
The specialist and ablated output rows
need their own genuine inference runner. Once the three role outputs exist,
run:

```bash
PYTHONPATH=src python scripts/mfm/qualify_v1_role_boundary.py \
  --snapshot /models/gemma-4-12B-pretrained \
  --source-receipt /receipts/gemma-4-12B-pretrained.json \
  --untouched /results/untouched.jsonl \
  --assembled /results/mfm-assembled.jsonl \
  --ablated /results/mfm-ablated.jsonl \
  --output /results/mfm-role-diagnostic.json
```

The output reports per-case critical errors, omissions and modality gaps.
`formation_contribution_cases` requires a matched ablation to worsen at least
one observed case. `role_boundary_diagnostic_pass` also requires the assembled
run to have no scored errors and every fixture case to be exercised. Both
values describe this small diagnostic only; `qualification_claim` is always
false. An image case is currently unexercised by the text-only source runner,
so the diagnostic cannot pass until a genuine sensory execution exists.

Before any capability promotion: verify actual prepared/component artifact
bytes and training lineage; run a process under an OS-enforced no-egress
boundary before it reads private data; freeze independent cross-person,
long-history and multimodal FINAL; compare stronger baselines and wrong-person
swaps; exercise MFM proposals through the Claim gate, projections, retrieval,
native judgment and observed outcome. The 49,819 formation cases are
training-only. A policy flag, fabricated output file or this scorer's unit
tests cannot substitute for these receipts.
