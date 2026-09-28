# N0 5e29 paid-host transfer and full-route qualification handoff

**SUPERSEDED TRANSFER LIST — DO NOT TRANSFER FROM THE 76-FILE V1 MANIFEST.**
Independent review of the owner's uploaded manifest found two sealed FewRel
FINAL payloads. See the v2 correction below. No bytes had been transferred.

Status 2026-09-28: **prepared, no paid allocation, no transferred data, no GPU
authorization**. The source-side inventory from the owner's Magnolia terminal
completed at clean `5e29f7f69ba4a5d031c7036639b67bebbcdc0bd2`. The
two-rank P100 measured joint route failed its unchanged 85% policy by
1,098,265,396 bytes on both ranks at the sixth pair's first AdamW step. The
other 12 pairs, second step, checkpoint transport and same-stage resume remain
unmeasured. Preserve job `576237`, its old projection failure and rank JSONL.

## What transfers, and what does not

The observed inventory pins the public mixture manifest
`c5f18c6f7ebf8c1a198cd420740966b5e9c7ffe9143b1125c06edfb94de51b97`,
audit `171ba5d606e6c36739db07b0267a4df2886fc087a1f8e4890c2d8ea8163986bd`,
teacher registry `4c06e08cf7ca217fe803ec16a63da2d7381b1d400172af8ffd9f90de120f6de9`,
semantic checkpoint `6c2706984c0e05123c4d88ba456788fbd4c9e7f0fca41d43d9e575ac6e53bf43`,
and 21 public corpus shards. Seven teacher pairs are repo-relative; the v0.5
wave pair is absolute under
`/homes/01/mxrayan/rayan-compute/rayan-n0/n0-v02/teacher-bank-v0.5`.

The versioned `n0_5e29_paid_input_transfer_manifest_v1.py` runs the previous
read-only inventory again, enumerates each corpus shard and all eight teacher
pairs, includes tokenizer **and its receipt**, semantic checkpoint, exact-head
CPU proof originals and mixture copies, seven optimizer-facing row/bank files,
and mixture provenance files. It records each file's SHA-256 and byte length.
It excludes the sealed `final-v2` tree except the freeze receipt. It transfers
no bytes. Its verifier repeats the source authority checks and compares the
complete destination closure and every file hash, failing on missing, altered,
extra, escaping or FINAL paths. The earlier inventory did not individually
publish all shard hashes.

**Checksum correction after the owner's first retrieval:** The original
terminal handoff mistakenly used SHA-256 `a5652804...`, calculated before the
tokenizer-receipt addition. The published script in context commit
`5b155c0ae3590df9ac17167b105250bd926b5c18` has Git blob
`d2b9c32b43bbc5e3b2ec19664439c9248c6a2a6a` and actual SHA-256
`50bfac0720e8a4433cee6cf4dbf0d661e146dd22b9b535a53bbf19ff756e1a4b`.
The reported checksum failure occurred before the container call and created
no manifest. Verify the already downloaded file against both hashes, then run
the existing producer; do not fetch or replace it merely to match the stale
checksum.

**Source-side manifest observed, owner terminal (2026-09-28):** The corrected
Git blob and SHA-256 checks returned OK, and the producer completed in the
CPU udocker runtime. It reported
`CREATED_PUBLIC_INPUT_MANIFEST files=76 sha256=8eb21cf4cc4634e1a9c5ceeb7d6137760b6eb594027ded56ccdb26831be91c66`.
This establishes a locally retained, per-file source manifest of the public
input closure. The full manifest contents have not been supplied here for an
independent row-by-row review. No payload was transferred, no destination
hash or absolute-path binding was checked, and no paid host was allocated.
Keep the manifest at its original path and do not regenerate over it.

**Independent manifest audit and v2 successor (2026-09-28):** The owner
uploaded the 76-entry JSON. Its upload used CRLF line endings; normalizing
to LF reproduces the printed SHA-256
`8eb21cf4cc4634e1a9c5ceeb7d6137760b6eb594027ded56ccdb26831be91c66`.
All 14 repo-relative teacher files match size and SHA-256 in the pinned local
source checkout. The 62 work-root entries total with the repo entries
897,956,446 bytes. **V1 wrongly includes**
`fewrel/final_rows.jsonl` (18,144,173 bytes) and
`fewrel/final_bank.json` (28,449 bytes) under the mixture root. The script's
directory walk excluded only `final-v2`, so its `final_rows_staged=false`
field was false as a transfer claim. The 76-file manifest is retained as
failed procedure evidence; it must never be used for packaging or transfer.
The uploaded manifest contains paths, hashes and sizes, not those row bytes.

`n0_5e29_paid_input_transfer_manifest_v2.py` replaces the directory walk
with an explicit list of the 29 observed **non-FINAL** mixture files. It
includes all six source-bound CPU receipts needed by the later full training
authorizer: result, static proof, tokenizer stress, operator evidence token
alignment, semantic long-context token alignment, and long-context boundary
alignment. Four of these were absent from V1's transfer list. Each of those
four is checked against its mixture copy and exact-head PASS status. V2
rejects `final_*` payload names, preserves only the `final-v2` freeze receipt,
and checks the generated expected file set before opening entries supplied
by a destination manifest. Local syntax, source-list comparison and synthetic
sealed-file rejection passed. The expected successor has 78 entries if the
owner's source files still match; **no v2 Magnolia manifest has been
produced yet**. It requires a new evidence path and distinct hash. Do not
overwrite or relabel the failed V1 manifest. No new N0 scientific source or
training authorization is involved.

On Magnolia, after fetching this context revision and checking **both**
script hashes, run the manifest producer under the validated CPU container
with `--repo`, `--work`, `--mixture` and a **fresh** `--manifest` path outside
the source and mixture trees. Do not duplicate a large data tree on the nearly
full `/homes/01` disk. Retain the printed manifest SHA and manifest itself.
Transfer only its listed `work` files, plus a clean checkout of the pinned
source commit. Run `verify` inside the target runtime with the same logical
roots. The authoritative source repo must remain clean. The paid runtime must
expose the original v0.5 absolute path to the registry, for example through a
container bind of the staged work tree at that exact path. If it cannot,
**stop**: a versioned portable registry and new public-mixture proof chain are
required. Never edit the pinned registry in place or mark old hashes as PASS.

## Hardware and runtime decision before spending

Lambda's public catalogue lists a two-A6000 48 GB-per-GPU, 28-vCPU, 200-GiB
RAM, 1-TiB SSD shape at $1.09 per GPU-hour ($2.18 per pair-hour before tax),
and a two-A100 PCIe 40 GB-per-GPU shape at $1.99 per GPU-hour. This is a
candidate, **not** account availability or a guarantee the full route passes.
Its on-demand images include Docker and the NVIDIA container toolkit, but
default Python versions vary between 3.10 and 3.12; the Magnolia udocker
environment used Python 3.11. Do not install from the broad
`requirements-n0.txt` ranges and assume equivalence. Capture the Magnolia
container image identity and actual Python, PyTorch CUDA build, Accelerate,
Transformers, tokenizer/safetensors versions, then pin a runnable container
digest or wheel lock. Verify the offered host driver supports that CUDA build.
Prepaid credit or a catalogue page is not evidence of a live two-device
allocation. The user should inspect an account-specific two-GPU offer,
hourly charge, availability, disk persistence and termination controls before
authorizing launch. Lambda bills on-demand instances in one-minute increments
while running, including idle setup time. An explicit spend/time cap is needed.

## One bounded qualification after a real allocation

1. Bring the exact source and public files to the host; verify the manifest,
   original absolute teacher path, original CPU/mixture receipts and sealed
   FINAL freeze. No FINAL rows or private identity inputs are copied.
2. Record actual GPU names, `torch.cuda.mem_get_info`/total bytes per device,
   host driver, CUDA/Python/library builds, image digest and two-rank NCCL
   mapping. Mismatch stops the run; do not change source or thresholds on the
   rented host.
3. In a **fresh** evidence root, invoke
   `scripts/eipm/n0/run_n0_v02_measured_joint_route_v2.sh` with the same
   semantic checkpoint, corpus, teacher registry/audit and exact-head mixture.
   This first preserves the old no-gradient projection as its own result and
   then runs the measured two-rank complete J3 route. Both ranks must prove
   all ten families, 18 semantic × fabric pairs, eight microbatches, first and
   later AdamW steps, named gradients, 85% plus 1-GiB margin, checkpoint
   save/load and resumed optimizer step. A fail or incomplete result stops;
   retain rank JSONL and do not rerun with narrower shapes or gates.
4. Only the source-bound measured memory **and** complete joint-route PASS
   receipts can proceed into the existing training-authority chain. This
   diagnostic updates disposable weights but claims no production training,
   DEV model selection, private identity gradient or FINAL opening.

This route can clear the known P100 capacity failure with a larger device,
but its full numerical and resume result is an experiment, not something that
can be guaranteed before the actual hardware runs it. A provider mismatch is
also measurable before the expensive full route. Do not purchase on the
strength of a projected margin alone.
