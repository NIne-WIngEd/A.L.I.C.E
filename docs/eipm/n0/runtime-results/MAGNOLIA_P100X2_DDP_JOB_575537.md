# Magnolia 2×P100 N0 DDP Mechanics — Job 575537

**Date:** 2026-09-13  
**Execution branch:** `alice-eipm-v1-build`  
**Execution Git HEAD:** `fbee8a5b88a2be21cb1967a19c81abde0716af6d`  
**State:** COMPLETED  
**Exit code:** `0:0`  
**Elapsed:** `00:03:23`  
**Node:** `gpu001`

## Result

The non-promotable distributed mechanics job completed successfully on the hardened Magnolia udocker route.

Observed runtime:

- SLURM job `575537`;
- two visible Tesla P100-PCIE-12GB devices;
- udocker P2 mode;
- validated `rayan-n0-base` container;
- 352,184,960 trainable parameters;
- 2-process distributed training;
- fp16 inside the N0 trainer;
- step 0 -> 4 training leg PASS;
- step-4 cumulative tokens: 8,192;
- exact resume from the step-4 checkpoint;
- step 4 -> 8 continuation PASS;
- step-8 cumulative tokens: 16,384;
- final `ddp_mechanics_receipt.json` status: `PASS`;
- `world_size: 2`;
- `promotable_checkpoint: false`;
- `private_identity_data: false`;
- `private_identity_gradient: false`.

Receipt evidence:

- step-4 receipt SHA256: `ba7490c9b24260159bc78d5f6830ac74f1652a36d2e8ba0df91ca34fc4facfc0`;
- step-8 receipt SHA256: `654e7d7a21a92132eddf8a022790d0e37ec4d7f9554bf4b3415e3611beb1168a`;
- resume parent points exactly to `ddp-mechanics-checkpoints/step-00000004`.

## Warnings and interpretation

The run emitted three warning classes. None invalidated the mechanics result.

1. `accelerate launch` warned that `num_machines`, launcher-level `mixed_precision`, and `dynamo_backend` used defaults. The trainer itself was explicitly configured for fp16 and the generated checkpoint receipt recorded `mixed_precision: fp16`. The tracked launchers are nevertheless hardened after this run to pass these launcher settings explicitly and remove the ambiguity.
2. Accelerate warned that the Magnolia host kernel `3.10.0` is below its recommended `5.5.0`. The container cannot replace the host kernel. This is therefore retained as a long-run hang risk, not treated as a failed qualification. Real training remains checkpointed and bounded so an allocation can be resumed.
3. PyTorch warned that `destroy_process_group()` was not called before process exit. The distributed run and resume both completed successfully. `train_mlm.py` is hardened after this run to call `accelerator.end_training()` on normal completion and early already-complete exit.

The tied-weight save warning for `decoder.weight` was also observed. It was empirically non-blocking because the saved state reloaded successfully in leg 2 and completed the exact resume test.

## Gate decision

The DDP/checkpoint-resume mechanics gate is **CLOSED / PASS**.

Do not repeat this test merely to reconfirm the same mechanics. The next unresolved build task is meaningful data lineage rather than infrastructure qualification.

## Next stage

Build the first real bounded public N0 lineage under:

```text
$HOME/rayan-compute/rayan-n0/n0-v01
```

Smoke corpus/tokenizer artifacts remain qualification-only and must not be copied into this real lineage.

CPU preprocessing is preferred for the next two stages:

1. `magnolia_cpu_corpus_bootstrap.sbatch` — bounded `corpus-bootstrap` plus receipt verification;
2. `magnolia_cpu_tokenizer.sbatch` — verify the real corpus and train the real 48k tokenizer.

Only after those receipts exist should the prepared 2×P100 200-step MLM pilot consume the real corpus/tokenizer lineage.

## Evidence status

This record is based on owner-returned SLURM stdout/stderr and the returned DDP mechanics receipt. Raw private runtime artifacts remain outside public Git.
