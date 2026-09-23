# N0 qualification integrity review — 2026-09-23

## Scope and handoff

This is an independent, read-only source review of
`alice-eipm-v1-n0-full-envelope-foundation-build-v1@de37ff0c74cc9b767edc2ce6ece5d3684a156c0f`.
The review branch changes no N0 implementation, contract, test, evidence root,
threshold, or active execution branch. Its findings should be checked against the
current N0 head before applying any correction. The other chat owns the active
training path. This review owns only whether qualification receipts prove what
they claim.

Graphify's most recent stored code graph at the time of review was pinned to
`1eeff40ad07f8a60ded92113f1ecb3755e0c67a7`. I used it for navigation,
then inspected the exact branch source for the later commits. The old stable
build N0 selector is historical; it is not the current execution frontier.

## Finding 1 — CPU and GPU qualification do not share the registered constructor

**Observed source.** The topology contract at
`configs/eipm/n0/n0_v02_full_envelope_registered_topology_v1.json` declares a
single production-shape constructor for CPU qualification, GPU qualification,
training, DEV, and FINAL. The GPU qualifier calls
`load_registered_full_envelope_system(..., runtime_profile=None)` in
`scripts/eipm/n0/qualify_n0_v02_full_envelope_gpu_memory_v1.py`. In contrast,
`scripts/eipm/n0/qualify_n0_v02_full_envelope_cpu_runtime_v1.py` directly
constructs `AliceN0V02Model`, loads the checkpoint, and directly constructs
`N0FullEnvelopeTrainableSystemV1` from its separate CPU configuration. It does
not call the registered factory or include the topology-contract SHA in its
result. The CPU runtime config uses a 32-token window and 8-token overlap; the
registered production values are 4096 and 256.

**Why this matters.** The 32-token window is a legitimate operating-point
override for exercising cross-window behavior. It is not itself a product
ceiling or a defect. The unbound second constructor is the issue: a passing
P42 receipt could be produced by a model configuration that has drifted from
the registered production topology. The existing CPU result's
`registered_trainable_system` class name, parameter count, and 640-wide tensors
do not prove constructor or topology-contract identity. P42 is still useful
CPU evidence, but it should not be treated as exact registered-system proof
without this binding.

**Suggested acceptance check.** Use the registered factory in the CPU path
with an explicit declared operating-point-only runtime profile for its
32-token cross-window fixture. Include the topology SHA and effective runtime
profile in the CPU receipt. Verify learned topology dimensions and parameter
identity against a factory-created production-profile instance. Keep a
production-window resource check separate if the gate requires one. A static
assertion that only checks the class name or the JSON declaration would not
resolve the source/receipt gap.

## Finding 2 — GPU evidence root is created before artifact preflight

**Observed source.** The new
`scripts/eipm/n0/magnolia_p100x2_n0_v02_full_envelope_gpu_memory_v1.sbatch`
checks the Git head and tracked working tree. It then rejects an existing
`OUT_ROOT` and immediately runs `mkdir -p "$OUT_ROOT"`. Only afterward does
the launched Python qualifier read and validate the mixture manifest/audit,
tokenizer, public corpus, teacher registry, rows, bank, and checkpoint hash.
The script promises to preserve an existing evidence root and refuse a repeat.

**Why this matters.** A missing file, wrong hash, or unmaterialized mixture
can leave an empty or incomplete root. A corrected infrastructure input would
then hit the existing-root stop before any model inference ran. This is an
operational preflight issue, not negative model evidence. Do not delete an
already created run root to make a result look clean.

**Suggested acceptance check.** Perform read-only existence, schema/status,
lineage, and required hash checks before creating the new immutable GPU root.
Then create it once and preserve failures that occur after actual qualification
begins. A missing mixture row in a clean checkout should fail before the root
exists; a failure during GPU inference should leave its root intact. The
existing CPU launcher's prerequisite and semantic-checkpoint checks provide
a partial precedent, though they are not a substitute for a GPU-specific
preflight.

## Boundary: GPU memory projection is not training admission

The GPU script explicitly runs under `torch.inference_mode()` and computes a
training-memory projection from the no-gradient peak. Its configuration and
result correctly say `projection_is_authorization=false`. Keep that boundary.
Even a passing P43 receipt establishes a measured no-gradient route and a
projection, not observed backward/optimizer memory or permission to train.
This is a caution about interpretation, not a third defect claim.

## Reproduction and authority

These findings come from direct source inspection at the base SHA. No Magnolia
job or model experiment was launched. This environment lacks Torch, so there
is no claim of a passing or failing exact-head static suite. Graphify has
navigation authority only. The source on the active branch and later exact-head
receipts take precedence over this review branch if the other chat has already
addressed either finding.

The review does not change N0's goal: a full public, identity-neutral semantic
and judgment foundation with no fixed serving-axis or parameter ceiling. It
does not reopen frozen-semantic authority, alter the sealed FINAL corpus, or
authorize optimization, Magnolia execution, or N0 completion.
