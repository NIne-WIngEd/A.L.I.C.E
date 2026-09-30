# Gemma 4 pretrained source dissection for MFM

**Status:** exact source and structural inspection verified; inherited behavior
and MFM specialization unmeasured. This is the shared Fable V1 foundation
source for five personal roles, not an MFM checkpoint. It does not change the
parallel N0 personality work.

**Custody update (2026-09-30):** The complete bytes were verified at the
absolute path recorded in the historical receipt, but that temporary workspace
was subsequently pruned. The current scratch directory contains only incomplete
download fragments, not a usable eight-file snapshot. Reacquire on persistent
storage and rehash all eight files before loading, cloning, or transforming.
The structural findings below describe the previously verified checkpoint;
they do not imply that the source is presently available at that path.

## Historical verified source and current availability

The complete eight-file `google/gemma-4-12B` **pretrained** repository at
revision `023679ed352de9bb66cc873c9009ce3482585c08` was downloaded
outside Git and rehashed locally. The 23,919,549,408-byte weight file matches
the publisher SHA-256
`fe054ae05ff7f44318fd8ae90d58992531455c7ed31356704088f0f2d8c8009a`.
All seven other file sizes and hashes match their pinned publisher objects.
The local source was moved to `/workspace/gemma4-12b-base-snapshot` after
background scratch synchronization exhausted available disk, then rehashed
at that path before workspace pruning. See the [historical source receipt](receipts/GEMMA4_12B_PRETRAINED_SOURCE_20260930.json)
and [tensor inventory](receipts/GEMMA4_12B_PRETRAINED_TENSOR_INVENTORY_20260930.json).
The receipt's content digest is
`b80271bf8ff30023fac84aa247f7e21d8eb25fe2dbc4cd19709c9e7a24dde6e5`;
the inventory manifest digest is
`29d3f925c3be6cc72266973285fa87e198bf9b504a7beb61f48cb6cbaf18d578`.
The old MFM `-it` staging and fit receipts are unrelated and remain research
history. Google's Apache-2.0 origin and notices remain attached to derived
weights.

## Structural pieces

| Piece | Observed tensors / bytes | Function and editing risk |
| --- | ---: | --- |
| Text token embedding, tied to output | 1 / 2,013,265,920 | Shared vocabulary/output interface; token deletion or zeroing can break output and multimodal delimiters. |
| 48 decoder layers | 40 local attention, 8 global attention; 1,024-token local window | Long evidence is integrated through global layers, including the last layer; blanket layer removal risks temporal and source reasoning. |
| MLP projections | 16,986,931,200 bytes, 71.02% of tensor bytes | General representations and learned priors are distributed here; size does not identify a behavioral trait. |
| Attention projections | 4,813,021,184 bytes, 20.12% | Context routing is shared by text and sensory inputs; pruning needs causal evidence and regression checks. |
| Vision patch and projection path | 10 named tensors / 99,844,608 bytes | The 12B Unified checkpoint has no separate vision encoder to strip without consequence. |
| Audio projection | 1 named tensor / 4,915,200 bytes | Raw audio enters the shared decoder; removing this loses audio input, with negligible size benefit. |

The header accounts for **677 BF16 tensors and 11,959,730,224 elements** with
contiguous, valid offsets and an exact expected file end. The config specifies
3,840 hidden width, 15,360 MLP width, 262,144 vocabulary and maximum text
positions, and tied embeddings. Twenty-four bounded tensor samples covered
4,417 values with no nonfinite value observed; this does **not** assert that
every weight is finite or that numeric outliers are behavioral causes.

The processor and tokenizer contain chat/turn, tool, channel and thinking
markers. The pretrained repository does not supply a chat template in its
tokenizer configuration. The default generation configuration samples tokens
(`do_sample=true`, `top_k=64`, `top_p=0.95`) and suppresses only two modality
end tokens. These are **candidate output surfaces** for foreign chat/tool
behavior, not proof that the pretrained weights will spontaneously exhibit it.
The publisher README includes an `-it` example, so every load must resolve
the pinned local no-`-it` source by digest, never that example identifier.

## Behavioral influence and interventions

Weight names, magnitudes, a tokenizer inventory and the publisher's
instruction-tuned benchmark table cannot locate or quantify inherited model
behavior. The possible foreign influences relevant to MFM are generic
assistant self-reference, unsourced confident facts, fabricated autobiography,
tool calls, chat/thinking delimiters in proposals, wrong-person attribution,
source-instruction obedience, false memory consolidation and deletion echoes.
They are **hypotheses for measurement**, not observed failures of this source.

The [public diagnostic fixture](../tests/fixtures/mfm/base_behavior_diagnostic_v1.json)
and offline runner compare untouched source output to future edits with fixed
processor/decoding and synthetic inputs. The diagnostic scores source/actor,
time, authority, injection and foreign output markers. It is not independent
FINAL and its cases must not become optimization targets. For MFM, candidate
suppression is a role-specific learned formation component plus constrained
structured proposals and the existing independent Claim gate; direct memory
writes remain forbidden. A role-local output-token constraint or tensor edit
is eligible only after a matched before/after test reduces a demonstrated
failure without harming long-history, multimodal or general competence.
Private evidence remains outside this base and is never used for untrusted
base probing.

**No weight was deleted, rewritten or promoted.** The inspection workspace had 9.7 GiB
RAM and no GPU; it cannot execute the 12B BF16 model to locate a causal
behavior direction. Its 32 GB filesystem cannot hold a second full 23.9 GB
working copy alongside the immutable source, and copy-on-write reflinks are
unsupported here. A full local clone and measured five-role derivations need
adequate storage and a BF16 GPU host. Unmeasured deletion would be an
unqualified regression risk, not suppression evidence.

The destination MFM is a separately learned formation system on top of this
licensed representation source. The model must learn source selection,
subject and time attribution, uncertainty, consolidation and governed
revision through its own trained weights and full memory architecture.
A fine-tuned Gemma chatbot with personal data is only a control, never the
finished MFM. Qualification needs component removal/swaps to show that the
formation specialist, evidence fabric and Claim gate actually determine
behavior.

Reproduce the source inspection with
`scripts/mfm/verify_gemma4_pretrained.py` and
`scripts/mfm/inspect_gemma4_checkpoint.py`; use
`scripts/mfm/run_gemma4_base_behavior.py` for an exact-base offline behavior
run on sufficient hardware. Preserve each role's source hash, transformation,
before/after failures and rollback in FBM. The five roles are
identity/personality, MFM, host, relationship and assistant self; their
distinct specialization does not require exactly five standalone weight files.
