# N0 v0.2 Continuation Recovery and Teacher-Dev Gate

**Date:** 2026-09-15  
**Purpose:** durable continuation after the prior chat export ended before its last Git-backed updates.

## Recovered authoritative state

The exported chat ends before the repository's final N0 v0.2 updates. Git is authoritative for the missing tail.

The first permanent native v0.2 gradient tranche already completed successfully as Magnolia job `575583` on `gpu001` using 2 x Tesla P100-PCIE-12GB. It trained `alice-n0-semantic-v0.2` from random initialization for 500 optimizer steps. No private identity data or private identity gradient was used.

Current native model configuration is the practical P100 build, not the older pathfinder architecture:

- 16 encoder layers;
- hidden size 640;
- 10 attention heads;
- FFN 2560;
- exact parameters 136,594,435;
- governed 48k v0.2.1 tokenizer;
- 21-source frozen public corpus lineage;
- 1,020 governed public teacher rows across 51 competencies.

The 1,020 teacher rows are split exactly 765 train / 255 dev, with 15 train and 5 dev scenarios per competency. The first tranche consumed only `split=train`. The next gate consumes only `split=dev`.

## First-tranche result

Step 250 and step 500 both reached 0.9765625 fixed-core top1 and 0.976744 full fixed-core order invariance. All fixed voice cases passed by step 250 and remained passing. Held-out MLM NLL improved from 7.069161 at step 250 to 6.660788 at step 500. The sole surviving fixed-core failure is `N0V02-FIX-ALIGN-01`.

The fixed suite saturated too quickly. Score margins grew strongly from step 250 to 500 without another accuracy gain. Therefore neither checkpoint is promoted merely because it is later or because training loss improved.

## Exact next gate

Do not launch another gradient tranche yet.

The immediate runtime action is a one-P100 held-out teacher-dev comparison of step 250 and step 500:

`sbatch scripts/eipm/n0/magnolia_p100_n0_v02_teacher_dev_challenge.sbatch`

This gate evaluates all 255 untouched teacher-dev rows with deterministic candidate-order rotations. It measures top1, supported-set separation, order invariance, score margins, and score magnitude.

This teacher-dev gate is **not** the complete expanded N0 challenge. Even if one checkpoint wins clearly, it does not authorize another gradient update by itself. After inspecting the dev failures and score growth, the next evaluation layer must contain genuinely novel cross-competency, hard-negative, ambiguity/tie, multi-turn, and voice-expression cases. Repair teaching is generated only from observed failures.

## Hardening added before the gate

The build branch was advanced from the recovered head `c67fbbe526b29efdc940d7065476c349658cfa48` to `0c2fa5fdf0d126461b7ae38808d322dfa3c5c693` with four continuation commits:

1. `2c6c442785da2405c2ba79c1c081d6d3dfac2411` — adds fail-closed challenge preflight. It verifies config/tokenizer/teacher hashes, both checkpoint receipts, full-model/ranker/MLM artifact hashes, exact step lineage, and zero train/dev ID overlap.
2. `271340c8b3dcb4973564a2e37ea07c092167191a` — wires that preflight into the dev challenge and makes the comparison receipt explicit that execution `PASS` is not semantic promotion, the scope is teacher-dev only, the full expanded challenge is unsatisfied, and additional gradient is unauthorized.
3. `6abd43df8dad6fec4e8ab290d10bed21dacc9ed1` — preserves v0.2 workdir/checkpoint/evaluation overrides across the udocker boundary so host-side controls cannot silently disappear inside the container.
4. `0c2fa5fdf0d126461b7ae38808d322dfa3c5c693` — hardens Magnolia environment logging by creating the private runtime directory before `tee`.

## Non-regression rules recovered from the prior chat and branches

These are engineering knowledge, not experiments to repeat:

- **No A100 assumption.** The proven user-accessible Magnolia route is P100. Training uses 2 x P100 when required; the current dev evaluation intentionally requests 1 x P100.
- **Magnolia is container-first.** Do not rebuild the N0 Python stack on the CentOS 7 host. Use the validated Debian 12/Python 3.11/PyTorch 2.7.1+cu118 udocker environment.
- **Whole-stage udocker boundary only.** Do not wrap individual Python invocations with nested/per-Python udocker shims. Those previously caused recursive execution and MKL/libtorch failures. `magnolia_udocker_exec.sh` is the reusable stage boundary and refreshes NVIDIA binding on the allocated GPU node.
- **Never use container `$HOME` for persistent N0 paths.** Inside udocker it resolves to `/root`. v0.2 derives its persistent workdir from the mounted repository root instead.
- **Current Magnolia-facing namespace:** repo `$HOME/rayan-compute/rayan-eipm-main`; runtime `$HOME/rayan-compute/rayan-n0`; container `rayan-n0-base`; scheduler/output names `rayan-n0-*`.
- **Magnolia compute-node network is not a corpus-acquisition route.** Multiple CPU nodes failed at Hugging Face DNS resolution. Treat network capability and compute capability separately. Current frozen corpus/tokenizer artifacts are offline and must not be reacquired for this gate.
- **Do not reuse smoke lineage as real lineage.** Smoke corpus/tokenizer artifacts remain separate from permanent N0 artifacts.
- **Kaggle transport rule if it is needed later:** Windows PowerShell is only a thin launcher. Python owns JSON, state, and native Kaggle CLI calls. PowerShell 5.1 `Set-Content -Encoding UTF8` introduced a BOM and broke `kernel-metadata.json`; do not repeat it. Metadata must be UTF-8 without BOM. Kernel identity is deterministic. Push exactly once, wait for stabilization, poll the same identity, retrieve output only after terminal state, hash-verify output, and never blind-repush after an ambiguous/404 state.
- **No node roulette for known DNS failure.** Once the same outbound DNS failure repeated on a second CPU node, that route was closed rather than hotfixed repeatedly.
- **Do not rerun completed jobs blindly.** Inspect existing receipts/checkpoints first. Preserve step 250 and step 500.
- **Known non-blocking warnings:** the Magnolia 3.10 kernel warning did not prevent the completed run; the ModernBERT MLM-head `UNEXPECTED` warning is expected when loading the MLM checkpoint as a backbone for ranker evaluation.
- **Build Alice directly.** Validation stays compact and failure-driven. Do not recreate MC10-style model/judge/qualification bureaucracy.
- **Identity boundary remains closed.** N0 is still public semantic/pragmatic learning. N1 private Elaina gradients remain unauthorized until the owner explicitly opens that gate.

## Paired FBM rule

Continue the paired workflow: build A.L.I.C.E. -> preserve the reusable construction/failure lesson for FBM -> keep building A.L.I.C.E. FBM captures process knowledge and negative lessons without copying private Elaina payloads. It must not become a blocking side project.

## Continuation rule

At the next runtime boundary, pull `alice-eipm-v1-build`, run only the one-P100 teacher-dev challenge above, return its stdout/stderr plus `teacher_dev_challenge_comparison.json`, and inspect the result before authoring the novel challenge or any new training segment.
