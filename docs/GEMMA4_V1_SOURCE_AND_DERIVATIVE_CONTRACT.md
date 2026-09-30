# Fable V1 Gemma 4 source and derivative custody

This branch defines a shared **source**, not a qualified role model. The pinned
publisher artifact is the non-instruction-tuned, already pretrained
`google/gemma-4-12B` revision `023679ed352de9bb66cc873c9009ce3482585c08`.
Its `model.safetensors` file is 23,919,549,408 bytes and has SHA-256
`fe054ae05ff7f44318fd8ae90d58992531455c7ed31356704088f0f2d8c8009a`.
That roughly 24 GB file is the 12B model's weight file, not a separate neutral
checkpoint. Keep the publisher attribution and Apache-2.0 terms with every
derivative. Copying or changing weights preserves source ancestry.

The custody functions in `src/alice_foundation/gemma4_v1.py` run offline:

1. `verify_source(source_dir)` hashes all eight pinned publisher files and
   checks the Gemma 4 Unified configuration. `seal_source(source_dir,
   source_receipt)` writes an append-only JSON receipt outside the source.
2. `clone_verified_source(source_dir, immutable_clone_dir, clone_receipt,
   role="mfm")` makes a separately rehashed copy. Never change this clone
   after its receipt is written. It is still pristine upstream weights.
3. Independently specify and justify one same-shape replacement tensor payload.
   The custody tool chooses neither the tensor nor its replacement values.
   Supply an operation manifest that names the exact existing tensor and the
   old and new payload digests:

   ```json
   {
     "schema": "alice-gemma4-v1-transform-operation-v1",
     "parent_receipt_sha256": "<digest of clone or preceding derivative receipt>",
     "transform_id": "mfm_formation_v1",
     "implementation_ref": "<immutable code revision and operation reference>",
     "parameters": {
       "tensor_name": "<exact existing safetensors tensor name>",
       "expected_tensor_sha256": "<64-character digest of parent tensor payload>",
       "replacement_sha256": "<64-character digest of replacement payload>",
       "replacement_size": 0
     },
     "intent": "<reason and predicted effect>",
     "evaluation_receipt": null
   }
   ```

   Set `replacement_size` to the tensor's actual byte length. Then
   `stage_tensor_replacement(parent_receipt, replacement_payload,
   candidate_dir, operation_manifest, stage_receipt)` streams a new candidate.
   It checks both payload hashes and the whole parent checkpoint, preserves
   every other source byte and writes a hash-linked stage receipt outside the
   candidate. A staged candidate is still unqualified.

4. `materialize_transform(parent_receipt, candidate_dir, derivative_dir,
   operation_manifest, derivative_receipt, stage_receipt_path=stage_receipt)`
   rehashes the unchanged parent and candidate, checks the stage receipt, and
   independently compares both checkpoints. It requires exactly the declared
   tensor payload to change. The header, all bytes outside that tensor, every
   nonweight file and the file set must match the parent. Both tensor payload
   digests must match the manifest. It then copies and rehashes the derivative
   and writes `qualification: unqualified`. Changing another tensor or only
   rewriting a stage receipt cannot pass this comparison.
5. Before any role trainer uses the prepared base, call
   `verify_derivative(derivative_receipt, snapshot=derivative_dir,
   expected_role="mfm")`. This checks the receipt and files again, publisher
   notices, structural integrity and changed weight bytes. The trainer must
   reject the pristine source, the pristine clone, and a missing or mismatched
   derivative receipt.

The CLI equivalents are `python -m src.alice_foundation.gemma4_v1` with
`verify-source`, `clone`, `stage-tensor-replacement`,
`materialize-transform` and `verify-derivative`.

```bash
python -m src.alice_foundation.gemma4_v1 verify-source /abs/source /abs/source.json
python -m src.alice_foundation.gemma4_v1 clone /abs/source /abs/clone /abs/clone.json --role mfm
python -m src.alice_foundation.gemma4_v1 stage-tensor-replacement /abs/clone.json /abs/replacement.bin /abs/candidate /abs/operation.json /abs/stage.json
python -m src.alice_foundation.gemma4_v1 materialize-transform /abs/clone.json /abs/candidate /abs/derivative /abs/operation.json /abs/derivative.json --stage-receipt /abs/stage.json
python -m src.alice_foundation.gemma4_v1 verify-derivative /abs/derivative.json --snapshot /abs/derivative --role mfm
```
Receipts use canonical UTF-8 JSON (sorted keys, compact separators), with
`receipt_sha256` over the object before that field is added. The digest is
integrity evidence within trusted local custody, not a third-party signature.
Use private owner-controlled directories without concurrent writers. Reverify
the derivative immediately before a safe loader opens it; the file checks do
not defend against a malicious process with write access racing those opens.

This custody chain verifies **which bytes** are being used. It does not train
MFM, prove a tensor edit suppresses inherited behavior, create independent
gold, confer exclusive ownership over Google-origin weights or qualify a
personal model. Role evaluation must compare the full specialist against the
unmodified control and independently test source selection, attribution,
uncertainty, revision, deletions, subject separation and inherited influence.
No full Gemma checkpoint has been edited in this workspace.
