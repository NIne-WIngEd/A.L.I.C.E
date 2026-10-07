# Source-feature fixture diagnosis and continuation

The first public source-feature export job, Magnolia **576999**, failed one of 86
runtime checks before loading Gemma. It produced no source export or learning
result. The original logs and [failure projection](receipts/MFM_CONDITIONAL_FEATURE_PILOT_576999_FAILURE_PUBLIC.json)
remain preserved. The unchanged production exporter is not implicated by that
test verdict.

## Ordered diagnosis

1. **Check the evidence.** The test expected wrong input IDs to reach tensor
   validation. Its actual exception came from the earlier receipt/directory
   guard. The fixture therefore did not establish the claim it asserted.
2. **Connect the claim to MFM.** Frozen source states must stay bound to exact
   processed inputs, masks, shape, dtype and source lineage before fresh MFM
   learning consumes them. This integrity test supports that prerequisite;
   it cannot prove Gemma source quality or useful MFM formation.
3. **Trace the responsible path.** `read_source` checks the sealed receipt,
   file digest and exact two-file source directory before loading tensors and
   validating IDs/masks/geometry. Diagnose fixture setup before changing this
   deliberately strict production sequence.
4. **Locate the first departure.** The bounded unchanged-runtime diagnostic
   **577003** independently pins its output to
   `8339705785757f25da7207997c08b2651dcbc5a16d47b2288e427ca5e1a7110d`.
   With the old tensors still mapped, same-path replacement leaves a hidden
   `.nfs` file with the old feature inode. Only the directory predicate fails;
   the other four predicates and the resealed digest match. Releasing the
   mapping removes that entry without changing the new feature bytes. The
   intended wrong-ID guard then runs and rejects the input.
5. **Understand the piece.** Safetensors 0.8.0 returns file-backed Torch
   storage and writes replacements through a temporary file and rename.
   NFS retains an open old inode under a temporary name. The fixture's retained
   mapping unintentionally changes a separate custody condition. Production
   exports publish new staged sources and never rewrite committed sources.
6. **Choose a coherent correction.** Give adversarial test tensors independent
   cloned storage and release all original mappings before replacement. Check
   exact directory/receipt setup, prove the intended tensor validator ran, and
   separately cover IDs, masks, geometry, dtype, nonfinite values and a valid
   rewritten control. Retain full-bank rejection as the connected consumer check.
   Keep production digest, directory, lineage and tensor guards unchanged.
7. **Consult primary mechanisms.** The pinned
   [Safetensors writer](https://github.com/huggingface/safetensors/blob/v0.8.0/safetensors/src/tensor.rs#L288),
   [Python storage implementation](https://github.com/huggingface/safetensors/blob/v0.8.0/bindings/python/src/lib.rs#L556)
   and [Linux NFS unlink implementation](https://github.com/torvalds/linux/blob/v3.10/fs/nfs/unlink.c#L418)
   explain the lifecycle. The actual three-control experiment establishes its
   relevance here; the older Linux example does not claim Magnolia's kernel
   runs that version. This correction adds no model technique or new dependency.
8. **Verify the claim and resume construction.** In 577003, the original live
   mapping fails before the tensor guard; released owned wrong-ID tensors reach
   that guard and fail; the valid owned control reaches it and passes. Actual
   Python 3.11.16, Torch 2.7.1+cu118 and Safetensors 0.8.0, original code hashes,
   complete controls and scope are recorded in the
   [diagnosis projection](receipts/MFM_CONDITIONAL_FEATURE_FIXTURE_DIAGNOSIS_577003_PUBLIC.json).
   The corrected file now passes 31 author checks, 31 independent checks and
   the root's 91 affected local checks. Magnolia577006 then passed all91 actual
   runtime checks in48.012s before real Gemma source export began. Test success
   remains a prerequisite, not a full MFM result.

The first diagnostic wrapper, **577002**, also remains preserved: it stopped
before the container because a checksum delimiter broke nested shell quoting.
The replacement verified the recipe and both decoded shell layers, retained all
isolation/ownership/hash checks and completed 577003. The correction follows
the [GNU Bash quoting rules](https://www.gnu.org/software/bash/manual/html_node/Single-Quotes.html)
and [parameter expansion rules](https://www.gnu.org/s/bash/manual/html_node/Shell-Parameter-Expansion.html).

## Storage and full-role boundary

The original material and completed census have a Drive archive whose complete
download matches SHA256
`857f472bd04e52208579ad3b195a6e317086cafa03cd9ae09c13416567ada512`.
The [scoped metadata gate](receipts/MFM_DRIVE_SCOPED_STORAGE_GATE_20261007_PUBLIC.json)
measures 79,290,797 logical bytes across the two distinct observed custody
trees. A provisional 3 GB next-public-bank archive allowance fits the recorded
380 GB operational policy. Reconcile actual archive bytes and retained copies
before a write; account-wide quota, trash and revisions remain unknown. The
earlier 24.36 MB archive preceded this gate, a preserved process gap.

Magnolia's saved direct Drive credential rejects its OAuth grant. The connected
Drive route works and has passed full archive restoration. Use it through the PC
when needed; avoid new base downloads and preserve live compute dependencies.
Retain public conditional archives separately from private teacher/owner custody.

Gemma remains the frozen base source encoder. Complete the entire4,862-source
bank through verified same-binding scheduling continuations, measuring throughput,
complete lengths, peak memory and storage during that work. The owner's October7
correction forbids the reduced development framing previously used for the first
30minute scheduling block. Do not rename a small-model experiment as construction:
preserve full roster, dimensions and contracts, and continue until the actual bank
is complete. Historical immutable filenames and failed receipts remain intact.
The continuation uses an eight-hour worker allocation within a twelve-hour
scheduler limit, successful-parent dependency and the parent's external progress
pin. These are execution details, not capability limits. MFM must still build its own semantic producer,
learned references and source spans, full-role formation, native modalities and
governed memory-to-consumer feedback. Foundation precedes personalization;
restaurant conditional teaching, fixture passes and exports do not qualify
Alice/Fable V1.
