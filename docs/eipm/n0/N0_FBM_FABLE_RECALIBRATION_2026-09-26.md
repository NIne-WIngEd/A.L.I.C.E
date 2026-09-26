# N0 recalibration after FBM and Fable first-release review — 2026-09-26

**Branch:** `alice-eipm-v1-n0-full-envelope-foundation-build-v1`  
**Source reviewed:** `4270bfa2c856f9a7fbbbeab773b82ddae0608f31`  
**Status:** no new GPU or gradient authorization.

## Reviewed direction

Fable Sleight's first-release design assigns FBM the construction and evaluation of the connected personal foundation, including identity/personality, MFM, host, relationship, assistant-self, evidence/Experience, native judgment, and the local conversation handoff. The latter needs a versioned verdict containing evidence, uncertainty, subject perspective, disagreement intent, and expression obligations before an external language service drafts words. A conclusion-only personality score is insufficient.

The FBM data and seed program is recorded at [`docs/fable-builder/FBM_DATA_AND_SEED_PROGRAM_2026-09-26.md`](https://github.com/NIne-WIngEd/A.L.I.C.E/blob/fable-builder-model/docs/fable-builder/FBM_DATA_AND_SEED_PROGRAM_2026-09-26.md). Current FBM traces teach procedures but do not yet constitute a qualified cross-user builder corpus. N0 is one host-neutral foundation dependency; it does not train a consumer's whole entity.

Research watch verdicts retain the present N0 topology: Personalized RewardBench informs N1/N2 personal preference and downstream-proxy evaluation; LGM and RPMem are later context/parametric-memory challengers; MemCalib and Jev-Mem strengthen Stage G and later influence/control qualification. None requires changing the registered full-envelope N0 constructor, its public mixture, or its frozen gates on this evidence. Preserve the separate downstream arbitration owner and keep private identity gradients false. Later personality work must explicitly test characteristic expression, relevant host/relationship/self interventions, and outcome-driven judgment, including same-conclusion/wrong-voice candidates.

## Exact current gate

The continuity receipt records Magnolia P39PN CPU rematerialization job `576092` as PASS on exact head `4270bfa2`: eight optimizer-facing public lanes and ten macro families, 39,200/5,600 FewRel TRAIN/DEV, and 11,200 sealed FINAL rows. This is a CPU/data receipt, not GPU fit or training approval.

GPU memory job `576166` failed `92:0` after six seconds. Its stderr says `STOP: preserve existing GPU memory evidence root` at `/homes/01/mxrayan/rayan-compute/rayan-n0/n0-v02/full-envelope-gpu-memory-v1`. No stdout or memory result was shown. The script intentionally exits before creating a result when that root already exists. Do not interpret this as OOM, model failure, or a successful memory qualification. Do not delete or overwrite the root.

## Safe continuation on Magnolia

1. From the login host, inspect the existing root, its `result.json` (if any), timestamps, and running Slurm jobs. Verify the result's source revision, mixture hashes, two ranks, all three semantic cases, six full-fabric cases, 18 stress pairs, finite losses, and no-gradient/no-optimizer/no-FINAL flags before reusing a receipt. The existence of a directory alone is not a pass.
2. If the old root has no valid result and no active writer, preserve it and set `ALICE_N0_GPU_MEMORY_ROOT` to a **new** empty, versioned path. Keep all other exact-head paths and inputs unchanged. Submit the same no-gradient P43 script and capture numeric JOBID, sacct, stdout/stderr, and the JSON receipt.
3. A valid GPU receipt permits the next separately governed authorization check. GPU memory projection is diagnostic, not an authorization to create an optimizer, train weights, open FINAL, or start N1. If the memory cases fail, localize the actual failing axis and revisit placement/sharding without shrinking the registered capability.

No Magnolia filesystem inspection or rerun was performed in this documentation review. Its network and SSH route is unavailable from the current execution environment; the pasted `576166` accounting and stderr are the direct evidence for this failed attempt.

## P43 failure and full-path autocast correction (2026-09-26)

The original next-step advice above was overtaken by actual P43 job `576093` on `4270bfa2`: both fp16 ranks failed `scatter(): Expected self.dtype to be equal to src.dtype` in the graph after approximately 84 seconds. The root has no `result.json`. Job `576166` then exited 92 on the existing-root preservation guard. These are two separate failures and neither measures memory capacity.

The N0 branch now reaches `168c031556ee632061a2b9c2c3f151b865a79264`. The graph, its semantic node gate, and the active full-envelope Executor's message, path, activity and support accumulations have been audited for the fp16 projection/fp32 criterion mismatch; Binder's destination already follows its source dtype. A full-stack CPU bfloat16 no-gradient test includes graph and executor probability/finite checks. Python syntax and static diff checks pass locally; no Magnolia execution is yet verified for this head. User-submitted CPU test job `576167` was pinned to earlier `657ba2a4`; its result is not in the provided transcript and cannot qualify the revised Executor.

**Required order:** inspect `576167` sacct/stdout/stderr as diagnostic evidence; on a clean Magnolia checkout fast-forwarded to the current exact head, run the full-stack autocast test in udocker. After a pass, rerun P42 into a fresh versioned `ALICE_N0_CPU_RUNTIME_ROOT`; after that pass, rerun P39PN with that CPU root and a fresh `ALICE_N0_FULL_MIXTURE_ROOT`; only then submit the existing no-gradient two-rank P43 into a fresh `ALICE_N0_GPU_MEMORY_ROOT`. Keep all previous roots and logs. Validate all source, producer, rank and case receipts before interpreting P43. Do not create an optimizer, train, open FINAL, or claim a GPU fit on source inspection or CPU-only evidence.
