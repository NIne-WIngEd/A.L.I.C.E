# Approved Qwen public qualification v1.0.0

Rayan approved the exact Qwen fallback amendment with the message “approve”. This package executes that bounded public calibration. It does not execute private MC10D pointwise judging, breadth v104, candidate acceptance, promotion, training, or hidden MC8 evaluation.

## Run on the recorded Windows machine

Save the distribution ZIP and `Start-ALICEAstraQwenQualificationV100.ps1` in the same Downloads folder. Run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$HOME\Downloads\Start-ALICEAstraQwenQualificationV100.ps1"
```

The launcher verifies the ZIP hash, extracts into a fresh folder, verifies every package file, compiles Python, runs offline tests, and starts the controller. It writes a transcript in that fresh folder. The existing Anaconda Python is preferred. The known local repo is `C:\A.L.I.C.E-main`. The vault is `C:\ALICE_Vault`. The launcher checks the exact established R13 and R18–R25 provider receipts before compute.

The controller uses your existing SSH identity at `$HOME\.ssh\rayan_magnolia_ed25519` and connects to `mxrayan@magnolia.usm.edu`. It checks the existing host key. It does not copy or print the key. No SSH credentials are present in this package.

## Compute and runtime

| Item | Frozen execution choice |
|---|---|
| Model | `qwen3.8:27b-q4_K_M` |
| Full model digest | Resolve official registry manifest and freeze before model inference; compare captured bytes, registry header when supplied, installed manifest, and Ollama runtime |
| Runtime | Ollama `0.32.15`; exact archive and binary SHA-256 in `authority/runtime_policy.json` |
| Route | Magnolia CPU, partition `node`, QOS `normal` |
| Allocation | 20 CPUs, 48 GiB RAM, one exclusive node, six hours maximum, no requeue |
| Inference | Thinking on; temperature 1.0; top-p 0.95; top-k 20; min-p 0; presence penalty 0; repeat penalty 1 |
| Request | Context 8192; up to 6144 output tokens; seeds 9101–9116; `num_gpu=0`, `num_thread=20` |
| Local monitor | At most eight hours per launcher invocation, including queue time; rerun attaches |

The runtime archive is reused only after its exact hash verifies. If absent, only the previously pinned release is downloaded. If system tar lacks Zstandard support, the historical pinned `zstandard==0.25.0` fallback is installed into this run's own dependency directory. Its installation and file identities are retained. The Ollama archive and binary pins are unchanged. Runtime substitution is not allowed.

The model gets a dedicated cache at `/homes/01/mxrayan/rayan-compute/cache/models/qwen38-public-v1`. Only exact digest-addressed blobs may be reused from the two known existing model-cache locations. New blobs are fetched from the official registry and verified by size and full SHA-256. A tag change against an existing cached manifest stops the run. No model cache, previous judge result, or private source folder is deleted.

Before the 16 tasks, one separate public diagnostic request has a 128-token ceiling. It measures throughput and verifies the loaded model uses zero VRAM. It is not a calibration task. The worker projects the full 16 × 6144-token allowance using observed CPU throughput and prompt evaluation time, a 20% margin, and a five-minute cleanup reserve. If that does not fit the remaining allocation, it records a preflight stop. The estimate is conservative; it is not a guarantee of later throughput. There is no automatic Kaggle, GPU, smaller-model, or altered-profile fallback.

The full runtime identity, task hash, approved profile, exact pass rule and package identity are frozen in `effective-contract.json` before the diagnostic inference. Ollama stays on a private loopback port. Requests contain only the frozen fictional case and rubric; gold labels do not enter the model's messages.

## Checkpoints and telemetry

The stable approved run is `alice-qwen38-a1-8e7a384a745496f4`. Its local state is under `C:\ALICE_Vault\tools\alice-astra\qwen-fallback-a1`. Its remote run is under `/homes/01/mxrayan/rayan-compute/runs/` with that ID.

Rerun the same launcher after closing the terminal or losing connectivity. It attaches to the same scheduler job or reuses verified downloaded results. A durable submission intent, remote file lock, unique job name and Slurm reconciliation prevent blind duplicate submission after a lost acknowledgement. If an intent has no uniquely identifiable job, the controller stops for investigation. It never creates a second run to resolve that uncertainty. A terminated allocation is collected as terminal evidence; it is not automatically resubmitted.

Each task's exact request and intent are saved before HTTP dispatch. There is at most one request per frozen task. Every returned stream chunk is retained. Completed output is checkpointed before the next task. Missing terminal chunks, exhausted token budgets and malformed structured responses remain incomplete or failed. Rationales are not truncated into compliance. An unresolved in-flight attempt cannot be silently repeated. A complete captured response may survive a connection-close error only after runtime identity revalidation.

The existing Magnolia helper publishes operational telemetry to [Rayan-Compute-Ledger](https://github.com/NIne-WIngEd/Rayan-Compute-Ledger/tree/main/runs/alice-qwen38-a1-8e7a384a745496f4). It publishes at stage/task boundaries and every five minutes. Heartbeats and completed-task counts are separate. Raw model responses do not enter the public ledger log. A final telemetry failure preserves the result. Terminal collection retries the metadata publication without inference. The controller then independently rebuilds a typed public summary and appends it to `alice-context` using an isolated Git index. It preserves user checkout changes and uses no force push.

## Results and exit codes

The controller downloads and verifies an `ALICE_QWEN_PUBLIC_RESULT_<hash>.zip`. It copies that ZIP into Downloads and prints its full path and SHA-256. Return that ZIP and the terminal output. Full fictional responses, thinking output, token counts, durations, finish reasons, task statuses, runtime metadata and terminal logs remain available for review. The source results remain on Magnolia. No private candidate or hidden-evaluator data is mounted or collected.

| Exit | Meaning and next step |
|---|---|
| 0 | All public calibration gates passed; terminal telemetry and context publication completed. Four-family successor binding still requires receipt verification. |
| 74 | Transport or monitoring is pending. Run the same launcher to attach or collect. |
| 75 | Evidence is preserved; context or terminal telemetry publication is pending. The same launcher retries publication without repeating inference. |
| 76 | A prerequisite, format, runtime, budget, or semantic gate stopped the run. Return its ZIP if produced and the transcript. Do not delete state, switch profiles, or rerun old stages to work around it. |
| 86 in Slurm | Scientific output completed but the worker's final telemetry publication failed. Terminal collection attempts recovery and reports the final publication state separately. |

## Scientific boundaries

The pass rule is unchanged: all 16 unique tasks complete, at least 14 verdict matches, at least six of seven critical decisions correct, all five mandatory hard anchors correct, Q01 PASS, and Q03 HOLD. Secondary gold-field agreement is diagnostic. The 16 reused tasks are calibration, not independent certification.

Approval did not convert GLM v186 into a success. After a real Qwen pass, the next step is to verify exact Gemma, Qwen, Mistral and Granite receipts and their effective profiles and construct a new four-family lineage. The existing breadth-v104 prerequisite still requires an explicit migration to that eligible successor receipt and bundle. Preserve 63 replacements, one deferred item and the 287-candidate pool. Main remains frozen at `0abaed85873c3f8de04765847eb7700b0e20433f`.

## Sources and validation limits

The approved draft, owner acceptance, original public task bytes and source binding policy are included under `authority/`. The fictional prompt/schema were extracted unchanged from the historical worker with SHA-256 `931607aa75ac8fe16d7792acfe86be37aa0ba1e17d269170c677d0f8b132551d`. Official references used for the transport contract are [Ollama chat](https://docs.ollama.com/api/chat), [model identity listing](https://docs.ollama.com/api/tags), and [loaded-model metadata](https://docs.ollama.com/api/ps). The approved Qwen sampling profile came from the [Qwen model card](https://huggingface.co/Qwen/Qwen3.8-27B).

Offline tests exercise synthetic worker lifecycles, actual loopback HTTP transport, exact task/request validation, resource and completion gates, duplicate-submission recovery, verified collection, publication retries, and an isolated local Git repository. Test fixtures never leave temporary test directories. The build receipt distinguishes these tests from live Magnolia execution. This workspace has no PowerShell runtime or your SSH key; native Windows execution and the real Qwen result are not claimed by the build.
