# Explicit CPU source execution

[TRACE144](receipts/MFM_EXPLICIT_CPU_SOURCE_EXECUTION_20261015_TRACE144_PUBLIC.json)
records the reviewed optional execution profile in the existing source exporter.
The dedicated Magnolia production export remains pending. No full-width source
bank, production memory result, new joint update or learned formation is claimed.

`export_prepared_bank(..., execution_profile=None)` preserves the default route.
Explicit `cpu-sdpa-flash-v1` selects CPU/PyTorch2.7.1, loads SDPA and permits only
the native FLASH_ATTENTION backend. Each complete forward must actually observe
one CPU flash call per configured text layer and zero math calls. The existing
Gemma4 masks, positions and KV semantics remain intact, as do all three native
source fields, frozen BF16 width3840 and whole safetensor save/reload checks.

The existing sealed bank/receipt format gains optional execution metadata.
Policy, case ID, source alignment and length bind the measured operator events,
wall/CPU time and Linux process-lifetime high-water RSS snapshots. The RSS
measurement includes loading; it is not an isolated per-forward peak. Readers
check stored producer evidence before feature tensors, while permitting a
different consumer runtime. Historical banks retain their exact exporter FILE
pins and require their archived reader source.

Eleven focused controls passed in 113.413 seconds, including actual small BF16
CPU flash against causal/sliding/additive-mask math references, eager/math/
partial refusal, foreign case/policy refusal, different-consumer handling and
existing native-field, prepared-clone and path controls. The width4 fixture's
resource operands are mocked and production consumption refuses that fixture.
The retained Kineto GPU-enumeration diagnostic supplies no production result.
Root independently reviewed and rehashed 23 packet artifacts.

Review packet008 FILE: `e7b3326ecc6f37a03b08d66bc3fa066c62ebd58717bfe34dc49ecbb329991f27`.
Exporter FILE: `40d0507629780454e67b59a4d6f798cd5092ca5cd41b947ae3a5f7c702014174`.
Focused test FILE: `6c1a9060738efaf51b988617722dfabdf98ac35785213695a65c9816cfc8f66d`.

The compatible104-interface closure and inherited complete state remain
unchanged. Next, execute the separate longest-first 59,303-position source in
the finite nine-case production bank and retain its actual output/resources
before continuing the remaining sources and owned joint learning. Fresh
source/rights/author admission, independent behavior and accepted-memory
authority remain separate requirements. The full-system stress draft stays parked.

The choice follows the pinned [PyTorch CPU selector](https://raw.githubusercontent.com/pytorch/pytorch/v2.7.1/aten/src/ATen/native/transformers/sdp_utils_cpp.cpp), [CPU flash kernel](https://raw.githubusercontent.com/pytorch/pytorch/v2.7.1/aten/src/ATen/native/cpu/FlashAttentionKernel.cpp), [Transformers SDPA integration](https://raw.githubusercontent.com/huggingface/transformers/v5.17.0/src/transformers/integrations/sdpa_attention.py) and [Gemma4 text call path](https://raw.githubusercontent.com/huggingface/transformers/v5.17.0/src/transformers/models/gemma4_unified/modeling_gemma4_unified.py). Internal attention blocking avoids a materialized dense score matrix while retaining the native whole-source forward. These implementation facts and small controls do not establish the actual long-source resource result or byte equality with a different attention backend.
