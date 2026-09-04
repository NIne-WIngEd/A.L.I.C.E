# A.L.I.C.E. MC10D Continuation Handoff — 2026-09-04

## Purpose

This file is the continuation handoff for the dedicated MC10D working chat after completion of the new provider-neutral compute architecture.

The MC10D chat should treat this file as a reconstruction aid, **not as canonical scientific authority**. If anything here conflicts with a ratified repository file, frozen bundle, manifest, validator, checkpoint, or preserved execution receipt, the ratified/frozen evidence wins.

The immediate goal is to **resume MC10D from the exact preserved v1.7.2 boundary without restarting MC10D and without mutating frozen science**.

---

# 1. Current readiness: infrastructure is complete

The provider-neutral infrastructure roadmap **R0 through R25 is complete and live-qualified**.

Final infrastructure boundary:

- `R18_STORAGE_BUDGET_POLICY=PASS`
- `R19_MODEL_ARTIFACT_POLICY=PASS`
- `R20_SYNTHETIC_DATA_ARTIFACT_POLICY=PASS`
- `R21_CROSS_PROVIDER_SYNTHETIC_SMOKE=PASS`
- `R22_FORCED_INTERRUPTION_RESUME=PASS`
- `R23_FORCED_PROVIDER_SWITCH=PASS`
- `R24_DRIVE_RESTORE=PASS`
- `R25_GITHUB_RECONSTRUCTION=PASS`
- `MC10D_SCIENCE_MUTATED=false`
- `CANONICAL_MAIN_MUTATED=false`
- `R18-R25_COMPLETE=true`
- `INFRASTRUCTURE_R0_R25_COMPLETE=true`

Latest durable infrastructure receipt:

```text
local:
C:\ALICE_Vault\tools\rayan-compute\infrastructure\r18-r25-receipt-v1.0.4.json

local SHA256:
400113E93B855C4BA65E2D2B04B5798B55652CEC1D15817C45E98B5214760ABD

Drive:
rayan_gdrive:Rayan-Compute/manifests/infrastructure/r18-r25-receipt-v1.0.4.json

Drive round-trip hash verification:
PASS
```

The private compute ledger also contains the completed R22 resume, R23 provider switch, R24 restore, and R25 reconstruction telemetry.

Important recent ledger commits:

```text
R22 resume:
bc8c7edf1e9e011f116e72e7752aaa0f7f18f4e3

R23 provider switch:
656d0a159892111d7f5cb2734a369a3a836dc532

R24 Drive restore:
df6d4afebebf6a72f3c93501c0783b042ea2d049

R25 GitHub reconstruction:
ff6b6685407d2016b44f36a933de020390d8e819
```

### What R21-R25 proved

**R21 — cross-provider equivalence**

The same provider-neutral synthetic run was executed on Kaggle and Magnolia. Exact result-file SHA256 values matched. Semantic deterministic-output SHA256 values also matched.

**R22 — interruption/resume**

A Magnolia run completed three tasks. Its checkpoint was made durable on Drive. The run intentionally stopped. The resume job verified the first three results and did not recompute them. It executed only the remaining five tasks.

**R23 — forced provider switch**

Kaggle completed the first four tasks. Magnolia received the same checkpoint/results and executed only the unfinished tasks. Completed Kaggle results were preserved byte-for-byte.

**R24 — Drive restore**

A full run-state bundle was stored content-addressed on Drive. It was restored independently on Windows and Magnolia. Internal hashes were verified.

**R25 — GitHub reconstruction**

Run state was reconstructed from the private GitHub ledger plus the Drive artifact reference. The local preexisting run directory was not required.

Therefore the MC10D workflow may now use Kaggle and Magnolia as disposable execution providers while keeping canonical state outside either provider.

---

# 2. Canonical repository authority

Canonical local repo:

```text
C:\A.L.I.C.E-main
```

Canonical frozen `main` commit:

```text
0abaed85873c3f8de04765847eb7700b0e20433f
```

Do **not** advance or mutate canonical `main` merely to run MC10D compute.

Chat/context archives live on:

```text
alice-context
```

Current saved continuation checkpoint:

```text
docs/chat-context/2026-09-02/CURRENT_STATE.md
```

Saved MC10D boundary note:

```text
docs/chat-context/2026-09-02/MC10D_V1_7_EXHAUSTION_AWARE_BOUNDARY.md
```

If the new MC10D chat has GitHub access, it should read those files first and then compare them with the newer v1.7.2 evidence described below.

---

# 3. MC10D frozen science authority

Frozen pre-execution bundle:

```text
ALICE_MC10D_PREEXECUTION_FREEZE_BUNDLE_v1.1.zip
```

SHA256:

```text
22B0ADBCCF442B0B3654F964E35AB77A044AF7623891E2A60C72B06A94ECE9A3
```

Frozen high-level geometry:

```text
raw candidates                         288
deterministic hard-blocked originals    63
retained originals                     224
soft scope-review flags                 20
repair obligations                      64
UNKNOWN competitors                     24
judge families                           4
```

Judge families:

```text
Gemma
GLM
Mistral
Granite
```

MC8 remains sealed during repair and public-judge qualification.

A-SYN acceptance/promotion/training remain closed at this boundary.

---

# 4. Repair history and exhaustion doctrine

## v1.6 historical stop

MC10D v1.6 solved the Kaggle transport chain and reached the real frozen repair worker.

v1.6 package SHA256:

```text
A9CA1EB7DD05F57A4BF5F9F45CD54EFA975513257FE5F28D6D6C744ED839900E
```

It produced **62 valid replacements**, then exhausted all three allowed prompted repair attempts on repair obligation 63.

Slot 63:

```text
original:
ASYN-F694986B9D502671BCFC9D58

facet:
repair_initiation

target role:
peer

third attempted replacement:
ASYN-R1-4466155EC92CBAA563FCCE40

final deterministic blocker:
RAYAN_SUBSTITUTED_FOR_PEER
```

The worker stopped before slot 64.

## Frozen repair rule

The repair cap is **3 prompted attempts per slot**.

Hard rules:

- no fourth attempt;
- no weakening deterministic hygiene;
- a technical/model/JSON/runtime failure is **not** an abstention;
- a proven three-attempt deterministic-hygiene exhaustion may be deferred;
- a deferred/exhausted slot does not create evidence;
- a deferred/exhausted slot earns zero saturation credit;
- the hard-blocked original never re-enters the effective pool.

The v1.7 exhaustion-aware doctrine uses the packet's existing UNKNOWN/ABSTAIN competitor only after exact three-attempt deterministic exhaustion is proven.

---

# 5. Current MC10D boundary: v1.7.2

Do **not** restart MC10D from v1.6 or regenerate the 64 repair obligations.

Current durable repair boundary:

```text
MC10D v1.7.2
```

Package SHA256:

```text
4D9EE7FE7AACB510A1170B5C53B9E9F07D1182FFDC04EE55FA2A533097324533
```

v1.7.2 failure receipt SHA256:

```text
8C0066CAFA310A91845C8D7D7F746A17CCC37651BD25561B059C89121B7286B2
```

Current resolved repair geometry:

```text
valid replacement candidates     63
deferred exhausted slots           1
retained originals               224
effective candidate pool         287
UNKNOWN competitors               24
```

The one deferred slot is the verified slot-63 exhaustion described above.

Slot 64 is no longer an open generation obligation at the v1.7.2 boundary. **Do not regenerate slot 64.** Bind its resolved result from the preserved v1.7.2 evidence.

Current effective-pool equation:

```text
224 retained originals
+ 63 valid replacements
= 287 effective candidates
```

A-SYN state remains:

```text
accepted = 0
promoted = 0
training = false
```

The completion frontier remains open because the exhausted slot is deferred.

---

# 6. Scenario geometry after repair

For the 287-candidate effective pool, preserve the frozen scenario functions.

From the exhaustion-aware boundary:

```text
packet-shared scenarios                         1,152
candidate-specific scenarios for 287 candidates 4,592
candidate-probe upper bound                    55,104
```

The 1,152 packet-shared scenarios remain byte-identical to the frozen earlier boundary.

The retained-original candidate-specific scenarios also remain unchanged.

An exhausted/deferred slot has no candidate-specific scenarios because it is not a candidate.

Do not invent scenarios for the deferred slot.

---

# 7. Public judge state — exact current stop

The v1.7.2 CPU public-judge helper preflight passed.

The public fictional-judge qualification then stopped during Gemma qualification at:

```text
Q12_CONTEXTUAL_PRIVACY
```

Observed failure class:

```text
empty judge content
```

The pointwise scope-judge screen **never started**.

Therefore there are no valid pointwise-screen outputs to reuse or infer.

### Frozen public-judge bindings known here

Gemma:

```text
model:
gemma4:31b-it-q4_K_M

digest:
6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7
```

GLM:

```text
model:
glm-4.7-flash:q4_K_M

digest:
4475827791a269b02c8ec49b1c3bc1abb5846bacf3fae015b75d33986322d8f6
```

Qualification task count:

```text
16
```

Mistral and Granite were bound earlier. The continuation chat must retrieve their **exact model names/digests from the frozen v1.7.2 manifest/evidence**. Do not reconstruct or guess those bindings from chat memory.

### Critical runtime rule

The frozen MC10D public-judge worker uses the existing Ollama/llama.cpp-style judge runtime contract. It is **not** a Transformers rewrite opportunity.

Do not change:

- judge family;
- model digest;
- judge prompt;
- qualification task content;
- response parser semantics;
- frozen acceptance logic;

just because Gemma returned empty content.

An empty/technical response remains a technical failure. It cannot become UNKNOWN/ABSTAIN.

---

# 8. Immediate MC10D continuation action

The next MC10D chat should begin from the **v1.7.2 public-judge qualification stop**, not from repair generation.

Recommended execution sequence:

1. Re-read the frozen v1.7.2 package manifest, failure receipt, judge manifest, and helper checkpoint/evidence.
2. Reconcile all preserved remote/local identities before submitting anything new.
3. Confirm the exact allowed resume semantics from the frozen validator:
   - if the frozen qualification contract permits incomplete-only resume, execute only the unfinished qualification work;
   - if it requires requalification of the entire Gemma qualification set, do that;
   - **do not invent a partial-rerun rule**.
4. Keep Gemma's model/digest and frozen qualification science unchanged.
5. Route the execution through the provider-neutral architecture described below.
6. Require all four public judges to be successfully qualified and bound before opening the pointwise screen.
7. Once the four judges are valid and the current effective candidate/scenario registry is frozen, authorize **only** the blinded pointwise scope-judge screen.
8. Keep MC8 sealed during the pointwise screen.
9. Do not begin full simulation/falsification until the pointwise scope screen has frozen.
10. Do not claim final MC10 saturation or training authority while the completion frontier is still open.

The first package created by the MC10D chat should therefore be a **versioned successor to the v1.7.2 judge-qualification boundary**, not a repair package and not a new science package.

---

# 9. Provider-neutral architecture to use for MC10D

The infrastructure qualification synthetic workload is **not MC10D science**.

Do not reuse this infrastructure workload as an MC10D input:

```text
F07BBA479925E4CB3196795E134E5C83463D374C8DCBA92511B117F7D36D53B4
```

That SHA belongs only to the infrastructure qualification fixture.

For MC10D, wrap the exact frozen MC10D worker/input bytes in the neutral compute contract.

## Neutral run must bind

At minimum:

```text
schema
stage = MC10D
canonical repo SHA
exact frozen input/bundle SHA256
exact workload/package SHA256
entrypoint
arguments
ordered deterministic task IDs
task-set SHA256
model names/revisions/digests
judge bindings
environment/runtime lock
seeds
evaluation policy
capability requirements
execution/retry policy
qualification/resume nonce
```

Provider-specific names must stay outside the neutral science contract.

Do not place any of the following in the neutral scientific identity:

```text
Kaggle kernel slug
Magnolia host
Slurm job ID
gpu001
provider filesystem path
temporary provider run path
```

Those belong in provider observations, receipts, or adapter metadata.

## Deterministic task semantics

Task IDs must remain deterministic.

A checkpoint may contain:

```text
completed
active lease
failed
missing
```

A completed task is reusable only after its exact result SHA256 is verified.

Resume rule:

```text
verified completed task -> preserve, do not rerun
incomplete/retryable task -> requeue only if frozen policy allows
hard scientific stop -> never silently requeue
```

A duplicate task state or completed-result hash mismatch is a hard stop.

## Provider switch

Provider identity must not alter scientific task identity.

A valid provider switch is:

```text
same neutral run
same task IDs
same frozen workload/input SHA
same checkpoint
same completed-result hashes
different provider adapter
```

---

# 10. Durable-state architecture

## Google Drive — large artifact authority

Remote:

```text
rayan_gdrive:
```

Project root:

```text
rayan_gdrive:Rayan-Compute
```

Project hard cap:

```text
400,000,000,000 bytes
```

Operational write ceiling:

```text
380,000,000,000 bytes
```

Safety reserve:

```text
20,000,000,000 bytes
```

The budget gate must run before large writes.

### Content-addressed object layout

Models:

```text
Rayan-Compute/objects/models/<SHA256>/
```

Datasets:

```text
Rayan-Compute/objects/datasets/<SHA256>/
```

Checkpoints / restore bundles:

```text
Rayan-Compute/checkpoints/<SHA256>/
```

Infrastructure / policy manifests:

```text
Rayan-Compute/manifests/
```

Run-specific references may live under run/manifest structures, but the durable large object itself should be content-addressed.

### Durability rule

Never set:

```text
CHECKPOINT_DURABLE=true
```

until:

1. Drive upload completed;
2. the object was read back;
3. the read-back SHA256 matches the local/source SHA256.

A provider-local copy is not durable authority.

## Model artifact policy

Model artifacts must bind:

```text
artifact SHA256
size
format
precision
quantization
base model ref
base revision/digest
logical plane
provenance
```

Logical model planes may share a base artifact when scientifically valid.

Do not duplicate full 25–30B bases merely because logical planes differ.

## Synthetic-data artifact policy

Synthetic artifacts must explicitly carry provenance such as:

```text
E0
E-INF
A-SYN
A-EXP
A-SELF
```

Storage never implies acceptance.

For any not-yet-accepted synthetic artifact:

```text
acceptance_state = NOT_EVALUATED
canonical_memory_credit = false
```

Do not upload a broad raw personal corpus to compute providers. Send only frozen minimal payloads or derived artifacts required for the job.

---

# 11. Private GitHub compute ledger

Repository:

```text
NIne-WIngEd/Rayan-Compute-Ledger
```

Visibility:

```text
private
```

Magnolia clone:

```text
~/rayan-compute/telemetry/ledger
```

Magnolia GitHub SSH wrapper:

```text
~/rayan-compute/bin/rayan-github-ssh
```

Deploy key:

```text
~/rayan-compute/runtime/github/rayan_compute_ledger_ed25519
```

Telemetry publisher:

```text
~/rayan-compute/bin/rayan-telemetry-push.sh
```

The private ledger is for:

```text
run manifests
receipts
application logs
task/result hashes
provider metadata
checkpoint metadata
Drive artifact references
reconstruction anchors
```

Do **not** put large model/dataset/checkpoint blobs in GitHub.

The assistant can currently read this private repository through the connected GitHub integration. Future MC10D chats should use the GitHub connector to inspect it directly.

### Immutable run-ID rule

A telemetry run ID is immutable.

If a second execution tries to publish different bytes under an existing run ID, the ledger must hard-stop.

Example already proven:

```text
RUN_ID_COLLISION=true
exit 73
```

Do not "fix" that by deleting old evidence or overwriting the run.

Use a fresh execution/attempt identity when the frozen policy permits a retry.

---

# 12. Magnolia execution provider

User:

```text
mxrayan
```

Host:

```text
magnolia.usm.edu
```

Remote root:

```text
/homes/01/mxrayan/rayan-compute
```

Use Rayan-visible names:

```text
rayan-*
```

Canonical immutable payload internals may keep `A.L.I.C.E.` names when renaming would change hashes/provenance.

## Normal user interface

**Windows PowerShell is the only normal control interface.**

Do not instruct the owner to manually operate a Magnolia shell.

Magnolia work is controlled remotely from Windows through SSH/SCP/Slurm.

## CPU route

Qualified CPU partition:

```text
node
```

QOS:

```text
normal
```

## Qualified P100 GPU route

Node:

```text
gpu001
```

Partition:

```text
gpu
```

Hardware:

```text
2 x Tesla P100-PCIE-12GB
```

Qualified single-GPU route:

```text
--gres=gpu:1
```

P100 constraints:

```text
12 GB VRAM
compute capability 6.0
CUDA 11.8 qualified
no BF16
FP16 usable
FlashAttention2 should not be assumed
```

Do not route a declared >12 GiB GPU requirement to the P100.

## A100 warning

`gpu003` exposes A100 hardware but the user's known QOS does not authorize the required partition/QOS combination.

**Do not silently use the A100.**

If a task requires >12 GiB and no currently authorized provider satisfies it, hard-stop with an unsatisfied capability classification rather than weakening the task.

## Python runtime

Exact Python:

```text
/modules/pkgs/common/python/3.11.5/bin/python3.11
```

The Slurm job must resolve the Python shared library and export its directory in `LD_LIBRARY_PATH`.

Known library directory:

```text
/modules/pkgs/common/python/3.11.5/lib
```

A batch job should run `python --version` and an unresolved-library check before executing the scientific payload.

## rclone

Correct Magnolia binary:

```text
~/rayan-compute/bin/rclone
```

Config:

```text
~/rayan-compute/runtime/rclone/rclone.conf
```

Do not look for the binary under `runtime/rclone/`.

---

# 13. Kaggle provider

Known CLI:

```text
Kaggle CLI 2.2.4
```

Known account:

```text
mkrayanyan
```

Kaggle is a disposable execution provider. It is not canonical state authority.

Important transport lesson from R21:

Kaggle code-file execution should not assume arbitrary sibling files will be available exactly as local Windows staging laid them out.

The qualified infrastructure adapter used a self-contained `main.py` that materialized the exact neutral payload under Kaggle working storage and verified frozen hashes before execution.

For MC10D, preserve the exact frozen MC10D package/input bytes. Do not rewrite science to accommodate Kaggle transport.

### Kaggle retry rule

Do not blind-repush on:

```text
404
not found
ERROR
delayed discoverability
```

First reconcile the exact kernel identity and status.

A completed predecessor is evidence, not permission to resubmit the same immutable telemetry identity.

---

# 14. Windows / transport failure classes already solved

Do not reintroduce these bugs.

## PowerShell

Use:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$HOME\Downloads\<versioned-launcher>.ps1"
```

Do not permanently alter execution policy.

Do not embed Python in PowerShell here-strings.

Do not build a large state machine in PowerShell.

PowerShell should do only the thin launcher work:

```text
package SHA verification
clean extraction
Python resolution
py_compile
offline selftest
call plain Python controller
classify exit code
```

The Python controller owns state/reconciliation.

Windows PowerShell 5.1 may surface harmless native stderr as `NativeCommandError` under `$ErrorActionPreference='Stop'`. Judge native tools by exit code.

## SCP

Windows `scp.exe` does not reliably expand remote shell variables in the remote path.

Do not pass:

```text
$HOME/rayan-compute/...
~/rayan-compute/...
```

directly to `scp.exe`.

Normalize to absolute paths such as:

```text
/homes/01/mxrayan/rayan-compute/...
```

before invoking SCP.

Reject any unexpanded `$VARIABLE` in an SCP remote path.

## Magnolia Git

System Git is old:

```text
git version 1.8.3.1
```

Do not depend on global:

```text
git -C <dir> ...
```

Use:

```bash
cd <dir>
git ...
```

The dedicated GitHub SSH wrapper should remain the transport layer.

## Remote text files

Remote Bash/Slurm scripts must be LF-only raw bytes.

Do not stream multiline Bash through Windows text-mode stdin.

Write the script locally as LF-only bytes, transfer with SCP, then execute/submit it remotely.

---

# 15. Proven package-release pattern

Every new MC10D execution package should follow the known working release pattern:

1. Build a versioned ZIP.
2. Produce exact SHA256.
3. Verify deterministic/clean package contents.
4. Supply a tiny versioned PowerShell launcher.
5. Launcher verifies outer ZIP SHA.
6. Clean extraction to disposable directory.
7. `python -m py_compile`.
8. Full offline `--selftest`.
9. Live controller only after selftest passes.
10. On a deterministic stop, preserve everything and diagnose before producing a successor version.

Do not hotfix package internals after release.

Do not manually clean remote evidence to make a retry easier.

Do not blind-rerun the same failed package.

If a package bug is discovered, build a new version.

---

# 16. Provider-neutral MC10D release design

The next MC10D package should keep scientific bytes and provider mechanics separate.

Suggested package layers:

```text
MC10D frozen scientific payload
    |
    | exact SHA256
    v
provider-neutral run manifest
    |
    +-- deterministic tasks
    +-- capability requirements
    +-- retry/hard-stop policy
    +-- frozen judge/model bindings
    +-- checkpoint contract
    |
    +--> Kaggle adapter
    |
    +--> Magnolia adapter
```
Adapters may:

```text
stage files
verify hashes
select qualified provider route
submit
watch status
retrieve outputs
publish telemetry
restore checkpoints
```

Adapters may **not**:

```text
change prompts
change candidate sets
change judge models
change digests
change acceptance thresholds
turn technical failure into abstention
alter deterministic hygiene
alter task identity
alter MC8 visibility
```

---

# 17. Capability routing for MC10D judges

Do not force a large judge onto an under-capacity GPU.

The neutral run must declare actual requirements derived from the frozen model/runtime artifact.

Then:

```text
if requirement <= qualified P100 capability:
    Magnolia P100 may be eligible

if requirement > qualified P100 capability:
    Magnolia P100 adapter must reject

if another authorized provider satisfies it:
    use that provider without changing task identity

otherwise:
    deterministic UNSATISFIED_CAPABILITY stop
```

Do not route to Magnolia A100 without authorization.

Do not change quantization/model identity merely to make the job fit unless the frozen scientific authority explicitly permits that exact artifact.

---

# 18. OAuth durability caveat

The Google OAuth/rclone path is functionally qualified.

However, the OAuth app was still known to be in Testing, so refresh-token durability may be limited.

Before a long MC10D run:

```text
run the Drive/rclone preflight
```

If authentication is expired, repair authorization **before** expensive compute begins.

Never put OAuth client secrets, refresh tokens, private keys, or Kaggle secrets in:

```text
GitHub telemetry
application logs
chat messages
package manifests
```

---

# 19. MC10D authority gates after judge qualification

A successful public-judge qualification does **not** complete MC10D.

The next gates remain:

```text
all four public judges qualified and bound
        ↓
effective candidate/scenario registry frozen
        ↓
blinded pointwise scope-judge screen
        ↓
pointwise screen freezes
        ↓
full simulation/falsification
        ↓
MC10D scientific decision boundary
```

MC8 evaluator-only material remains hidden until the frozen boundary explicitly allows it.

The completion frontier remains open because of the verified repair exhaustion.

Therefore do not infer authority for:

```text
final saturation
personality-model training
Stage G closure
production promotion
Phase 2 retirement
```

from the repair/judge-screen milestones alone.

---

# 20. Broader Stage G sequence after MC10D

The current intended downstream order remains:

```text
MC10D
→ MC10E select / abstain
→ normalize / fuse / retest
→ shadow A-SYN
→ challengers + challenge debt
→ multi-turn tests
→ Identity Completion Graph
→ recursive MC10 / dual-frontier saturation
→ provenance-normalized personality training
→ PE1–PE22
→ blinded owner behavioral evaluation
→ canary / promotion
→ MC11 behavioral no-void
→ MC12 E0 fidelity / non-regression
→ model-plane qualification
→ full memory fabric / LifeSim
→ Stage G owner acceptance
→ G+ intelligent recollection
→ post-G+ roadmap reconciliation
→ H bounded canary
→ I full canonical transition candidate
→ J compatibility / rollback
→ owner final acceptance
```

Phase 2 remains canonical through Stage G and G+ until the explicit retirement boundary is reached.

Do not skip ahead merely because provider-neutral infrastructure is complete.

---

# 21. New MC10D chat startup checklist

The next chat should do this before generating a live package:

```text
[ ] Read this handoff.
[ ] Read alice-context CURRENT_STATE.md.
[ ] Read MC10D_V1_7_EXHAUSTION_AWARE_BOUNDARY.md.
[ ] Verify canonical main = 0abaed85873c3f8de04765847eb7700b0e20433f.
[ ] Verify freeze bundle SHA = 22B0ADB...ECE9A3.
[ ] Locate and validate MC10D v1.7.2 package/evidence.
[ ] Verify v1.7.2 package SHA.
[ ] Verify v1.7.2 failure receipt SHA.
[ ] Bind effective pool = 287.
[ ] Confirm slot 63 deferred and slot 64 resolved; no repair regeneration.
[ ] Confirm accepted=0, promoted=0, training=false.
[ ] Retrieve exact four public-judge model bindings/digests.
[ ] Reconcile the Gemma Q12_CONTEXTUAL_PRIVACY empty-content failure.
[ ] Read frozen qualification resume semantics from validator/checkpoint.
[ ] Verify R18-R25 durable receipt.
[ ] Preflight Drive/rclone.
[ ] Preflight chosen provider capability.
[ ] Build a new versioned provider-neutral MC10D continuation package.
[ ] py_compile + offline selftest.
[ ] Only then submit live work.
```

---

# 22. What the next chat must NOT do

Do not:

- restart MC10D;
- rerun MC10B or MC10C;
- regenerate all 64 repairs;
- regenerate slot 63;
- regenerate resolved slot 64;
- add a fourth repair attempt;
- weaken `RAYAN_SUBSTITUTED_FOR_PEER`;
- turn runtime/model/JSON failures into abstentions;
- accept or promote A-SYN early;
- start model training;
- expose MC8 during public judge qualification or pointwise screening;
- change judge model/digest because one runtime returned empty content;
- rewrite the frozen judge worker as Transformers;
- silently use Magnolia A100;
- force >12 GiB jobs onto the P100;
- make provider path/job ID part of scientific task identity;
- overwrite an existing GitHub telemetry run ID;
- delete failed telemetry to "clean up";
- store large blobs in GitHub;
- rely on provider-local cache as durable authority;
- claim `CHECKPOINT_DURABLE=true` before Drive round-trip SHA verification;
- hand-author remote state after a failure;
- blind-rerun a failed package;
- mutate canonical main for infrastructure bookkeeping.

---

# 23. Suggested first message/action for the MC10D chat

The MC10D chat should treat the situation as:

> Infrastructure R0–R25 is complete. Resume MC10D from the preserved v1.7.2 public-judge qualification stop. Do not restart repair generation. First reconcile the exact v1.7.2 package, failure receipt, judge manifest, and checkpoint. The last scientific execution stopped on Gemma qualification task `Q12_CONTEXTUAL_PRIVACY` with empty judge content. Preserve all frozen science. Use the provider-neutral run/task/checkpoint contract and content-addressed Drive + private GitHub ledger architecture. Determine the allowed resume set from the frozen validator, then build the next versioned ZIP + tiny PowerShell launcher.

That is the correct continuation boundary.

---

# 24. Key hashes / paths quick reference

```text
CANONICAL MAIN
0abaed85873c3f8de04765847eb7700b0e20433f

MC10D FREEZE
ALICE_MC10D_PREEXECUTION_FREEZE_BUNDLE_v1.1.zip
22B0ADBCCF442B0B3654F964E35AB77A044AF7623891E2A60C72B06A94ECE9A3

MC10D v1.7.2 PACKAGE
4D9EE7FE7AACB510A1170B5C53B9E9F07D1182FFDC04EE55FA2A533097324533

MC10D v1.7.2 FAILURE RECEIPT
8C0066CAFA310A91845C8D7D7F746A17CCC37651BD25561B059C89121B7286B2

GEMMA
gemma4:31b-it-q4_K_M
6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7

GLM
glm-4.7-flash:q4_K_M
4475827791a269b02c8ec49b1c3bc1abb5846bacf3fae015b75d33986322d8f6

LOCAL REPO
C:\A.L.I.C.E-main

LOCAL COMPUTE TOOL ROOT
C:\ALICE_Vault\tools\rayan-compute

MAGNOLIA ROOT
/homes/01/mxrayan/rayan-compute

MAGNOLIA PYTHON
/modules/pkgs/common/python/3.11.5/bin/python3.11

MAGNOLIA RCLONE
~/rayan-compute/bin/rclone

DRIVE ROOT
rayan_gdrive:Rayan-Compute

PRIVATE LEDGER
NIne-WIngEd/Rayan-Compute-Ledger

R13-R17 RECEIPT
C:\ALICE_Vault\tools\rayan-compute\provider-neutral\r13-r17-receipt.json

R18-R25 RECEIPT
C:\ALICE_Vault\tools\rayan-compute\infrastructure\r18-r25-receipt-v1.0.4.json

R18-R25 RECEIPT SHA256
400113E93B855C4BA65E2D2B04B5798B55652CEC1D15817C45E98B5214760ABD
```

---

# Final continuation statement

**Yes: the infrastructure layer is finished enough to return to MC10D.**

The next work is not more generic infrastructure. It is a provider-neutral continuation of the exact frozen MC10D v1.7.2 public-judge qualification boundary.

Preserve the science. Preserve the evidence. Reconcile before resubmission. Use external durable state. Requeue only work the frozen contract identifies as incomplete. Then continue MC10D.