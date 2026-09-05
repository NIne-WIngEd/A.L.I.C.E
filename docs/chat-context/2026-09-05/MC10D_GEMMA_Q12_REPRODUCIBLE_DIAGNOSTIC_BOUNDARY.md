# MC10D Gemma Q12 Reproducible Diagnostic Boundary — 2026-09-05

## Status

The clean MC10D v1.8.0 rebaseline successfully revalidated canonical/frozen authority, provider-neutral infrastructure, the 63 replacements / 1 deferred / 287 effective candidate boundary, and the exact v1.7.2 public-judge science.

It then reproduced the same Gemma qualification failure at Q12_CONTEXTUAL_PRIVACY:

- family: gemma
- failure: empty judge content
- private data used: false
- v1.7.2 failure receipt SHA256: 8C0066CAFA310A91845C8D7D7F746A17CCC37651BD25561B059C89121B7286B2
- pointwise screen: not started
- A-SYN accepted: 0
- A-SYN promoted: 0
- training: false

This confirms the remaining blocker is inside the frozen Gemma qualification/runtime interaction rather than the v1.7.3-v1.7.6 provider-adapter chain.

## Current hypothesis

The leading hypothesis is that Gemma thinking consumes the frozen num_predict budget and leaves message.content empty before a final answer is emitted. This is plausible but is not yet accepted as fact.

The frozen worker does not persist done_reason, eval_count, or thinking/content lengths, so the exact cause cannot be proven from the current failure receipt.

## Diagnostic release

Package:
ALICE_MC10D_GEMMA_Q12_DIAGNOSTIC_v1.0.0.zip

SHA256:
CB2AE3FC05AA3E90C6554EC3BE100D0980794FD717DA0033282506C1F1D31659

Launcher:
Start-ALICEMC10DGemmaQ12DiagnosticV100.ps1

SHA256:
823072BACE0669E318177127B4777780531A08A9B021EA37CEC2B9BD3231E6E7

This diagnostic:
- binds the exact v1.7.2 parent;
- binds the exact frozen Q12 task, prompt/schema authority, model/digest/profile, seed, temperature, num_ctx, and 2048 final-attempt budget;
- persists only safe response metadata;
- never persists model thinking text;
- grants no qualification, acceptance, promotion, training, pointwise, or Stage G authority;
- if and only if the exact 2048 attempt proves prediction-length exhaustion, runs non-authoritative capacity probes at 3072, 4096, and 6144 and stops at the first valid structured final answer.

## Next action

Run the diagnostic only. Review its classification before designing or ratifying any qualification-parameter amendment.

Do not rerun v1.8.0 qualification until the diagnostic result is reviewed.
