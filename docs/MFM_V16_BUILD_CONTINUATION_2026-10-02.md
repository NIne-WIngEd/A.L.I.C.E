# MFM V1 continuation: recovered corpus and frozen-feature training

Status: recovered CPU admission and new training implementation; learned
capability remains unqualified. Continues `ae7d680286b84a5755caeb1d2e82e0d09c5176ae`.

## Destination and retained decisions

MFM learns to turn exact authorized experiences and current context into
grounded, uncertain, time-scoped memory proposals. The independent Claim gate
controls writes. Governed episodes, claims, relationships, missions, workspaces
and personal state must affect native judgment, and observed outcomes must
drive revision. Source person Elaina, host Rayan, and Alice's postactivation
self remain distinct. A Fable consumer begins their own host/self history and
needs no Elaina source; transferable methods must not transport another life.

The full comic v0.3 storyboard and story map were reviewed. Long-term continuity,
independent disagreement, changing priorities, decision rationale, negotiated
norms, learned skills, engine/device replacement and derivative deletion remain
requirements, not achieved model results. Full native modalities and the full
personal loop remain in scope for Alice/Fable V1. Flora work is separate.
See the ratified memory/identity architecture and the existing full-capability
build plan, including the main branch's MFM-1 through MFM-6 and Stage G gates.

V1 retains a verified **non-IT** Gemma 4 12B role clone supplying frozen source
representations, plus a separate fresh formation decoder. The Gemma language
head supplies neither answers nor labels. Old instruction-tuned commands and
fresh-foundation training are superseded for V1; a first-party foundation is
longer-term work. Contract fixtures and deterministic replay prove boundaries,
not learned skill. The earlier 49,819 cases are not full v1.6 admission.

## Recovered evidence

The previous chat ended with a local 441-case fictional corpus, without a
protected processor receipt or trained weights. The owner does not think they
transferred or ran that archive. The original expanded archive was unavailable.
The original 25-case v3 archive was recovered from existing owner Drive custody
and verified against SHA-256
`fd34cddc1c8b8958d00d5daf74f39b928e9fb2818ccaa4ee6294b928037f40a3`.
The pinned longitudinal issuer was replayed with fresh timestamped rights
records. This is a new issuance preserving case/source bytes, not the original
manifest under another name.

| Artifact | SHA-256 / binding |
| --- | --- |
| Recovered 432-train / nine-development manifest | `0268c134596326bb592e0afaf3fdab8626670a345e813b2148a11eb68dc93676` |
| Intake | `8549f8247dc32c0e055a15d775e7588718fb05aa4279178cb8ce8f6363757312` |
| Issuance receipt | `f013286f4aa642e1bdc05fab8240fb3d6047374a454f6450c952fbcea81cf79b` |
| Recovered archive, 1,358,927 bytes | `1a5155457a43af215ad295e67f039785b1aa8d5f059295ddf52189de48316586` |
| MFM role clone receipt | `54484dd523c396648fdf7069abe219c3f8f7c1fdcfec026cda1038810d4d7ac7` |
| Foundation commit | `3e1328410dac52ba18be243cb9f7144e7a3964d1` |
| Processor receipt, Magnolia 576606 | `f2a8d521a567d88833a485604dfd259e232efb20adda1334d4365c55f043230c` |
| Processor receipt file | `0551adb3a89cdcf5b73bfbd0c43dda6baacca30daea5acd26e1e090acf5b79e7` |

Job **576604** failed before opening corpus data: unsupported helper arguments.
The correction uses the helper's same-UID environment and `-- /bin/bash`
interface. Preserve failed logs. Replacement **576606 completed, exit 0**, in
4m46s as `mxrayan`. It verified loopback-only execution, exact input and role-base
bytes, 432 training/nine diagnostic development cases and Transformers 5.17.0
processing. Maximum source/target: 3,035/1,607 tokens. Modalities processed:
text, structured, code. No native media, base forward, backward, fit,
independent FINAL or downstream learning effect was proved by this job.
The earlier AMI/mixed upload rejection is not bypassed by this fictional route.

## Compute decision and implementation

The local Windows PC lacks CUDA and cannot hold the 23,919,549,408-byte base.
Available Magnolia P100/K80 devices do not individually hold that base plus the
specialist. The live trainer's two-device placement does not shard the base.
Kaggle has previously allocated T4 when L4 was requested; actual hardware and
backward/restart peaks must decide viability. Paid hours remain the last fallback.

A frozen representation does not change during specialist learning. Export it
once on allocated CPU in BF16, losslessly, then train the same decoder with
the same full targets and source mask. This follows the existing objective and
hardware constraint; it introduces no new representation architecture, adapter,
quantization or instruction-tuned replacement. Specialist **FP32** arithmetic
is separately versioned for P100 compatibility. CPU BF16 and FP32 training are
not claimed numerically identical to the old CUDA BF16/autocast path.

- `formation_feature_bank.py` binds exact source/context, case/split, base,
  processor and code lineage; verifies tensors and external bank digests;
  rejects incomplete, changed or unlisted payloads.
- `export_v16_frozen_features.py` re-admits the corpus, rehashes the role clone,
  verifies the processor receipt and runs only frozen `base.model` on CPU.
  Complete train/development source states are exported; targets never enter
  the backbone or bank tensors; FINAL is never opened.
- `train_v16_cached_specialist.py` trains fresh FP32 weights, preserves full
  targets and chunked vocabulary projection, uses train-only gradients,
  restores actual AdamW checkpoints and records seeded development CE controls.
- `magnolia_v16_frozen_feature_export.sbatch` requests 12 CPU threads, 64 GB,
  12 hours under the owner account; verifies isolation and external pins and
  installs job-local dependencies without altering the shared container.

Local tests prove identical live/serialized loss and every parameter gradient
in the same FP32 path, plus multi-step disk reload and identical next weights
and Adam moments. Full-size hardware evidence remains required. Resume verifies
committed cases and identical export lineage. Unknown interrupted staging
directories block finalization; preserve them outside the bank in the same
private custody before a reviewed resume. Never bless a partial artifact.
The cached artifact schema is distinct: original live inference/qualification
scripts cannot accept it merely by renaming files.

## Research grounding

Frontier-watch Oct 1–2 intake and primary sources were reviewed before choosing
this increment. [MemFit](https://arxiv.org/abs/2610.00872) motivates preserving
original turns and using summaries as retrieval aids.
[Memory LACE](https://arxiv.org/abs/2609.03201) supports lifecycle, contradiction
and supersession tests. [HaluMem](https://arxiv.org/abs/2511.03506) separates
extraction, update and downstream answering failures.
[Hindsight](https://arxiv.org/abs/2512.12818) and
[Zep](https://arxiv.org/abs/2501.13956) are competitor references for
belief/experience organization and temporal graph memory. Author-reported scores
do not qualify our model or prove identity formation. These reinforce existing
acceptance requirements; no architectural replacement follows from them.

## Remaining gates toward full capability

1. Complete and verify all source-only features; record actual runtime/memory.
   Submission/running state is not an export pass.
2. Probe full-size P100/T4 stress accumulation, first/later optimizer steps,
   actual checkpoint reload and subsequent weights/moments. Record actual
   accelerator and memory. Do not silently shorten cases to manufacture a pass.
3. Fit all admitted teacher training cases. Keep development gradient-free.
   Compare matched trained/seeded **generated formation** and gate diagnostics.
   Teacher-forced CE alone is not useful memory or role-boundary fidelity.
4. Build independently adjudicated source/host/time/generator-separated gold,
   native modality cases, corrections/revocations, calibrated abstention,
   changed priorities, scenes, outcomes/norms/skills and injection tests. The
   single-author fictional expansion is a learning diagnostic, not the scale
   destination or independent capability benchmark.
5. Exercise the governed memory-to-judgment loop with actual sequential feedback,
   consolidation order, memory ablations, deletion/replay and downstream-model
   swaps. Prove a second independent consumer without source-person leakage.
6. Qualify Alice/Fable V1 against sealed independent FINAL and Stage G. Capture
   FBM procedure traces and separately eligible input/target/authority/outcome
   cases. Procedure records alone are not supervised gold.

Magnolia is first; Kaggle and Hugging Face are retained as free-route candidates.
Kaggle CLI authentication and existing owner notebooks were verified. Hugging
Face is connected as `NineWinged`, currently non-PRO. Current official
[ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu) allows
eligible free personal accounts to host up to two Spaces with five daily GPU
minutes. Account age/email eligibility, remaining quota, Gradio integration,
runtime compatibility and isolated checkpointable training remain unverified
for this project. This may support short probes, not an assumed full training
allocation. [HF Jobs](https://huggingface.co/docs/hub/jobs-pricing) are billed
compute requiring positive credits; no free Jobs balance is claimed. The Jobs
skill's older paid-plan prerequisite is superseded by those current official
docs. No HF workload, credits or subscription were purchased.

Paid compute is the last fallback after concrete free-route failures. Every
unresolved gate remains visible; this increment is not full MFM completion.

## Continuation execution receipts

Magnolia **576608 completed exit 0**, 104 relevant tests in 16.711s on the
actual Torch 2.7.1 container, including Linux network/symlink and signed
adjudication checks. Twelve new frozen-feature/restart tests also passed on
local CPU Torch 2.14.1. The initial broader Windows run encountered existing
Linux-only tests and a missing test-only crypto dependency; these were checked
on Linux with job-local pinned dependencies, without weakening production guards.

The first export launcher, **576609**, failed before running the exporter:
Magnolia's older Bash treats an empty array as unbound under `set -u`. Use a
nonempty command array and an explicit resume flag. Preserve its logs and
job-local dependencies. A replacement export still requires its own actual
completion receipt; no base-forward success is inferred from this failed job.
