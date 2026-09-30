# Exact-base Gemma 4 behavior run: compute placement

**Status:** no 12B forward pass or role-specific edit yet. The source is
historically verified in [the dissection receipt](MFM_GEMMA4_PRETRAINED_BASE_DISSECTION_2026-09-30.md),
but its temporary local copy was pruned. The current scratch holds incomplete
download fragments only. A new full byte verification is required at the
actual execution location.
The offline text diagnostic runner and frozen seven-case public synthetic
fixture are committed. This is a placement record, not a model-quality result.
The run probes the external base before constructing a separately trained
MFM. It cannot substitute for the later full formation objective or entity
qualification.

The first paired run needs a local copy of the exact pinned 23,919,549,408-byte
BF16 source, a compatible Transformers/PyTorch runtime, a BF16-capable GPU
allocation with enough combined VRAM for weights plus context/runtime, and
room for a separately hashed working derivative. Start with at least 48 GB
usable GPU memory and verify actual placement. The runner fails if weights
offload to CPU/disk. Its initial seven cases are diagnostic only; image input
is marked unexercised. A full multimodal, long-history and independent FINAL
suite remains to be built and run before promoting a role base.

| Route | Observed state | Decision for exact BF16 diagnostic |
| --- | --- | --- |
| This workspace | 9.7 GiB RAM, no GPU, 32 GB disk; source held outside synchronized scratch to prevent duplicate 24 GB transfers | Hash, inspect and prepare code here; no 12B inference or second physical copy. |
| USM Magnolia | Prior N0 observation was two 12 GB P100 GPUs. Current `gpu` nodes are uninspected. The CPU node could not resolve Hugging Face. This workspace has no reachable Magnolia SSH/DNS/auth route. | P100 is not native BF16 and lacks fit. Inspect current USM GPU GRES/VRAM first; if suitable, stage public source through authorized transfer, rehash on the new path, then run. Do not confuse USM Magnolia with Ole Miss/MCSR Magnolia documentation. |
| Kaggle | No local CLI/credential or connected app. Earlier L4 request yielded T4×2; standard T4 is not native BF16 and a single 16 GB T4 cannot hold the checkpoint. | No exact BF16 run available through the current route. A separately labeled FP16 conversion diagnostic would require authentication, new code and measured equivalence; it is not this receipt. |
| GitHub | Authenticated repository connector exists; local `gh` binary is absent. Existing Actions are CPU governance tests with no Magnolia/Kaggle bridge or credentials. | GitHub can share code and receipts after publication authorization, but it cannot supply this GPU route by itself. |
| Hugging Face | Model research access works. Jobs connector returns an unavailable-tool error; local `hf` CLI is logged out. Jobs require positive credits. | No job was created. A paid L40S/A100 route would need working authentication, pre-staged model and explicit cost decision. |

On the **USM Magnolia login node**, the read-only inventory command is:

```bash
sinfo -N -p gpu -o '%N %T %G %m %f'
```

If a suitable node exists, first stage the complete source and its small
receipt/scripts without opening private Elaina, Rayan or consumer evidence.
Run `verify_gemma4_pretrained.py` on the relocated source to make a new
absolute-path receipt. Then exercise `run_gemma4_base_behavior.py` on the
public emitted cases and score them with `evaluate_base_behavior.py`. Freeze
the untouched result before any candidate edit. Test edited-versus-untouched
competence and role authority; preserve a rollback and trace every result in
FBM. No paid GPU time is requested by this document.
