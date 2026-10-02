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

## Saved execution handoff: 2026-10-02

Execution code is pinned to `7f9621bddfe0b1136b776a9850e2f0e5c8f4842a`
for cached training, generated diagnostics and the precision benchmark. The
running BF16 exporter and its resumes remain pinned to
`eee3b30cf1791b92dbb509b7423b993d45e6596c`; do not change their bound modules
while the bank is being produced.

Magnolia **576615 completed exit 0**: five additional generated-diagnostic
tests passed on Linux, and **all 441 complete targets** tokenized identically
with the processor tokenizer and cache-only tokenizer. The parity receipt is
`5cb5c0bfdc0446f9574aacbf569fe03139aabf40ee65f79c3ae78785e2cabe19`.
This proves target tokenization, not semantic target accuracy.

Dedicated GPU runtime `rayan-mfm-cache-gpu-v2-576613` is ready. **576616
completed exit 0** on one actual Tesla P100-PCIE-12GB, Torch 2.7.1+cu118,
capability 6.0. A tiny FP32 specialist completed backward with BF16 source
features, without opening teacher data. Receipt:
`3a47b9e54245fcac392d9834f78f62c70e1712f699d23b3ec4b5d1b32ed32a00`.
**Full-size capacity is still unproved.** Failed container setup jobs 576611
and 576613 and their logs remain preserved. The repaired dedicated runtime
does not modify the shared N0/personality container's dependencies.

**576610** is running the source-only BF16 export on 12 CPU threads, 64 GB,
with a 12-hour allocation. The first measured cases took about 189–293 seconds
each. Export root:
`/homes/01/mxrayan/rayan-compute/rayan-mfm/features/rayan-mfm-features-441-eee3b30cf`.
The export record digest is
`ac8a80daff0dd22b33ca80e7ca71cca90e03ee3030cb1fd3e2383e97200b76ce`.
No completed bank digest or learned weights exists at this handoff.

**576614** is a separate CPU FP32 measurement on 12 threads, 96 GB. It measures
time and numerical differences against the first two committed BF16 cases.
Loading the FP32 base reached roughly 67.6 GiB peak resident memory. Timing and
numerical receipts are pending. It does not alter the BF16 bank, train the
decoder or justify switching the export precision before results are inspected.

Bounded continuation jobs **576619 → 576620 → 576621** follow 576610 through
Slurm `afterany` dependencies. Each requests 12 hours and resumes only an
upstream **TIMEOUT**, revalidating all committed cases with the identical
export invocation. Recognized interrupted atomic staging directories are
preserved separately in private custody. Unknown members, links, FAILED,
NODE_FAIL and CANCELLED stop the chain. A completed bank causes subsequent
resumes to skip duplicate work. This bounds the existing BF16 route to four
allocations; unfinished export after the last timeout requires another decision.

GPU pipeline **576622** waits for `afterok:576621`. It obtains the external
bank pin from the final complete producer receipt in the separately stored
stdout of **576610/576619/576620/576621**, then the trainer independently
verifies the bank, corpus, model, processor and code lineage. The stages are:

1. Full-size stress accumulation, three actual optimizer steps and disk reload,
   comparing subsequent weights and Adam moments. A failure stops the job.
2. A fresh two-epoch fit of all 432 train cases, with development gradient-free.
   Probe weights are not continued into this run.
3. Matched trained/seeded greedy FP32 generation on nine diagnostic cases, up
   to 2,048 new tokens, preserving raw failures and checking JSON grounding.

This job requests one P100, eight CPU threads, 64 GB and four hours under
`mxrayan`. Its completed receipts, exit status and full-size peak memory must
be inspected before claiming even a teacher fit. A timeout or partial
checkpoint is not a completed component. Generated JSON grounding does not
establish semantic accuracy, Claim-gate execution or useful later judgment.

Exact submitted scripts are retained as
`scripts/mfm/magnolia_v16_feature_resume_20261002.sbatch` and
`scripts/mfm/magnolia_v16_cached_pipeline_20261002.sbatch`. Remote launch copies
and submission receipt are under `rayan-mfm/jobs/` and
`rayan-mfm/receipts/rayan-mfm-pipeline-submission-20261002.txt`. Training outputs
are under `rayan-mfm/runs/rayan-mfm-{probe,fit,generated}-576622`; producer logs
are `rayan-mfm/slurm/rayan-mfm-features-<job>.{out,err}`, learning logs
`rayan-mfm/slurm/rayan-mfm-learning-576622.{out,err}`.

At the next continuation, inspect accounting and original receipt bytes first.
If a stage failed, diagnose it before any retry; use Kaggle/free HF options if
actual Magnolia capacity or runtime fails. No paid allocation has been made.
Independent gold, native modalities, accepted-memory feedback, sealed FINAL and
second-consumer qualification remain the subsequent full-capability gates.

## Completed-job audit and dedicated runtime recovery

This section supersedes the running/queued snapshot above. **576610 FAILED
exit 1 after 1h11m38s**, committing 17 cases. The traceback reports
`FileNotFoundError` while creating the bank's `cases` directory. The same
directory and its committed cases still exist in host custody with mode 700,
UID 1905. The other job using `rayan-n0-base`, 576600, completed roughly two
minutes before the failure. Concurrent shared-container mount visibility is
the leading explanation, not a proven causal result. No file loss, corrupted
case or numerical failure has been established from this traceback.

**576619/576620/576621** each stopped exit 3 on a failed upstream job, as their
TIMEOUT-only policy requires. Their logs are preserved. **576622** could never
satisfy its dependency and was cancelled. **576614 TIMEOUT** after one hour,
without completing even the first measured FP32 forward or writing a numerical
receipt. That run does not support a switch from BF16; it is not an accuracy
failure or a general proof that FP32 hardware execution is impossible.

Recovery execution code is **01ad19de001b47604593e15661f2d4cd4af963f1**.
Only the CPU batch launcher changed in the frozen-feature path: it now requires
a dedicated `rayan-mfm-source-cpu-*` container. `formation_feature_bank.py`,
`export_v16_frozen_features.py`, the production decoder and teacher corpus are
unchanged, allowing the identical export binding and 17 committed cases to be
reverified and resumed. All 17 existing feature/diagnostic tests passed locally
after this operational change. Batch syntax was checked on Magnolia.

Dedicated source job **576630** creates a fresh runtime from the already cached
Python image and copies only installed libraries, excluding Python caches and
unreadable kernel modules. It adds persistent empty bind-path ancestors, uses
P2, holds a parent-process runtime lock and enters same-UID loopback-only
isolation before inspecting private bank custody. It then repeats the original
complete admission, publisher-base rehash, processor and case verification. The
runtime is `rayan-mfm-source-cpu-20261002`; no shared N0 container setup or
dependency changes are made. The lock stays in the parent because the namespace
helper deliberately closes inherited nonstandard descriptors.

Bounded TIMEOUT-only continuations are now **576631 → 576632 → 576633**.
They share this dedicated runtime sequentially. The scheduler's
`--kill-on-invalid-dep=yes` behavior was tested before submitting the new jobs.
Recognized interrupted atomic staging directories remain preserved; unknown
members and other failure states still stop execution. This is a diagnosed
operational recovery, not a policy to retry arbitrary FAILED jobs.

In parallel, **576629** runs the **full configured decoder** (3840 source
width, 262144 vocabulary, specialist width 768, six layers/twelve heads) on
synthetic BF16 source tensors at **3035 source / 1607 target tokens**, with FP32
decoder arithmetic and 16-case gradient accumulation. It checks first/later
optimizer steps, actual checkpoint reload and the next weights/moments on the
dedicated P100 runtime. It opens only the externally pinned processor metadata,
not teacher payloads or the backbone. These combined maxima are a conservative
resource fixture, not a source-grounded teacher case. A completed result would
prove only synthetic-tensor capacity/restart; the admitted-corpus stress probe
remains mandatory. OOM, timeout or partial logs do not count as a pass.

Replacement learning pipeline **576634** depends on **afterok:576633:576629**.
It pins the complete bank from producer stdout for **576630/576631/576632/576633**
and still runs the separate empirical corpus stress/reload before any fresh
two-epoch fit or matched generated diagnostics. It uses execution code 01ad19de0
and a dedicated GPU-runtime lock. Failed prerequisite dependencies now cancel
the queued GPU job instead of leaving it pending forever.

New source, capacity and pipeline recipes are
`magnolia_v16_dedicated_export_20261002.sbatch`,
`magnolia_v16_synthetic_capacity_20261002.sbatch`,
`probe_v16_cached_capacity.py`, and
`magnolia_v16_cached_pipeline_recovery_20261002.sbatch`, under `scripts/mfm/`.
The remote submission receipt is
`rayan-mfm/receipts/rayan-mfm-dedicated-submission-20261002.txt`. Capacity logs
are `rayan-mfm/slurm/rayan-mfm-capacity-576629.{out,err}` and output custody
`rayan-mfm/runs/rayan-mfm-synthetic-capacity-576629`. Revised learning logs and
outputs use job ID 576634. At submission, these results remain pending and no
bank digest or trained model is available.

## Latest route: measured full capacity, protected CPU learning queued

**576629 failed exit 3 before reading private metadata**: the GPU node rejected
combined USER+NET namespace creation with EINVAL. **576634 was automatically
cancelled** when that prerequisite failed. The completed CPU isolation probe
does not establish GPU-node isolation.

The synthetic capacity CLI was then narrowed to **public code and random
tensors only**, with explicit source/target dimensions and an informational
shape-reference digest. It cannot receive a corpus, base or preflight-file
path. Actual private admission, training and diagnostic guards remain intact.
Execution code is **924ea46b3907297c912b34422890d142381f527b**.

**576636 COMPLETED exit 0**, 3m54s, on the P100 12 GB. The retained decoder has
**267,542,272 parameters**. It passed 16-case accumulation at 3035/1607 tokens,
first/later updates and actual checkpoint reload; next weights and Adam moments
matched with maximum absolute difference **0.0**. Peak CUDA allocation reached
**10,304,656,384 bytes (~9.60 GiB)**, reservation 10,961,813,504 bytes (~10.21
GiB). The sealed capacity receipt is
`6b27110a7fe87a0ef0da348f8e443a0f4e6dc56e6d2d511060e0969799c36bf1`,
bound to run
`e39d7ef57bf28890273ec0e53819553c362c4fa2c9a73b23c59fe1855316f429`.
This establishes **synthetic full-decoder capacity**, not empirical corpus
stress, semantic accuracy, useful formation or private GPU execution.

Two no-data versions of owner-private Kaggle notebook
[`mkrayanyan/rayan-mfm-nodata-gpu-20261002`](https://www.kaggle.com/code/mkrayanyan/rayan-mfm-nodata-gpu-20261002)
completed. Returned metadata confirms private=true, internet=false and no
attached dataset, model or notebook inputs. L4 was requested; the runtime
actually provided **two Tesla T4s**, 15,636,037,632 bytes each, Torch
**2.11.0+cu128**. Both nonroot USER+NET and root NET-only followed by UID drop
were denied with EPERM. No corpus, base checkpoint or private formation weight
was uploaded. Version two's sealed receipt is
`a80934a8d6689ec436cbdcb283284000638a4225816b13361fdaad9b9befb312`.
Internet=false metadata does not establish the required loopback-only boundary.
Notebook method follows the official
[Kaggle kernel metadata documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md);
actual allocation and no-data receipts determine feasibility.

The dedicated source export **576630** has reverified all **17** prior cases
under proven CPU isolation and resumed new source forwards. Its bounded
continuations remain **576631/576632/576633**. The free protected CPU route
already supports the same FP32 specialist and complete targets, so **576637**
is queued afterok:576633: twelve CPU threads, 64 GB, twelve hours. It uses a
dedicated source-runtime lock, the same loopback-only checks, externally pinned
complete producer bank and execution code 01ad19de0. It first measures actual
corpus stress/reload, then performs a fresh two-epoch fit and matched generated
controls. No architecture, vocabulary, target length or decoder width is
reduced to obtain a pass.

The fit saves every five optimizer steps and pauses at a completed optimizer
boundary after eight training hours if still incomplete. A missing completed
component stops the pipeline before generation; inspect the saved checkpoint
and resume only its identical run. A twelve-hour diagnostic timeout likewise
requires review, never a completion claim. CPU timing and learned behavior
remain pending. Hugging Face remains a candidate for an independently verified
free route; no paid credits or GPU allocation has been purchased.

Actual sealed receipts and method-only accounting are saved in
`docs/MFM_COMPUTE_RECOVERY_RECEIPTS_2026-10-02.json`. The no-data Kaggle source and
metadata and the selected protected CPU pipeline are under `scripts/mfm/`.
The CPU learning submission is
`rayan-mfm/receipts/rayan-mfm-cpu-learning-submission-20261002.txt`, with logs
`rayan-mfm/slurm/rayan-mfm-learning-576637.{out,err}` and eventual
`rayan-mfm/runs/rayan-mfm-{probe,fit,generated}-576637` custody. Source export
continues to use the original exact BF16 bank binding. Full-capability gates
listed above remain open.
