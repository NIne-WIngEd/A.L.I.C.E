# Magnolia 2×P100 Real N0 Training — Job 575546

**Date:** 2026-09-14  
**Status:** completed successfully  
**Role:** first durable real N0 MLM learning segment on the imported public corpus/tokenizer lineage

## Scheduler result

- job: `575546`
- name: `rayan-n0-p100-pilot`
- state: `COMPLETED`
- exit code: `0:0`
- elapsed: `00:21:57`
- node: `gpu001`
- requested/observed route: 2× Tesla P100-PCIE-12GB

This run is not a runtime/DDP qualification. Those gates were already closed earlier. This is durable N0 learning state.

## Durable checkpoint evidence

The canonical workdir contains both:

- `checkpoints/step-00000100/receipt.json`
- `checkpoints/step-00000200/receipt.json`

The step-100 receipt was inspected directly and records:

- model id: `alice-n0-semantic-v0.1`
- step: `100`
- sequence length: `512`
- world size: `2`
- mixed precision: `fp16`
- gradient checkpointing: `true`
- parameters total/trainable: `352,184,960`
- tokens seen total at step 100: `1,638,400`
- corpus receipt SHA-256: `6cf717095e0eea4e790d0c852014a8047fd5ffc9f30550b2445c8a992a0bc293`
- corpus verified bytes: `422,476,606`
- corpus verified shards: `5`
- tokenizer SHA-256: `9ce759f063d92acfdfd65b24ee8f4cad180d7dd8a37fef1a5da8a67df0a8b88d`
- tokenizer origin git revision: `cb23ac483d0ec5bcc9107f895ada0a6a9c8c34ba`
- training code git revision: `79d4b2d0b4c3bff8932dbc5c5a68f79256d59b9e`
- private identity gradient: `false`
- resume parent: `null`

The step-100 model artifact hash includes:

- `model.safetensors`: `d9ed1faf1e777f2985ab8c8957f2faf0ef990ed85461c0632feddd7554df87a1`

The presence of the step-200 receipt after scheduler completion proves the run reached the requested terminal checkpoint. The exact step-200 receipt and loss trajectory must be inspected before choosing the next training horizon; do not infer or fabricate its values from step 100.

## Duplicate-run prevention

A second submission, job `575547`, was recognized as a duplicate attempt after job 575546 was already the active real training run. Do not treat the launcher as an instruction to restart from step zero. Durable checkpoint state is authoritative.

Before any future N0 MLM submission:

1. inspect the latest canonical `step-*/receipt.json`;
2. inspect the latest real learning loss trajectory;
3. choose the next target based on model evidence;
4. resume from the exact latest checkpoint;
5. never repeat a completed learning segment solely because the launcher is reusable.

## Next model-building boundary

Inspect job 575546 stdout/loss trend plus `step-00000200/receipt.json`. Then decide the next durable N0 MLM continuation horizon from learning evidence. Continue model development in parallel on the semantic/judgment curriculum and N1/N2 identity-learning stack. No additional Magnolia infrastructure qualification is warranted unless a concrete new failure appears.

No private E0/E-INF/A-SYN gradient is authorized by this result.
