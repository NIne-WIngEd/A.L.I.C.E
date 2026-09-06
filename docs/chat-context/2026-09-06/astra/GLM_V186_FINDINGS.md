# v186 GLM findings and next decision

Status: the actual Windows forensic bundle has been received and independently reverified. Its public summary was already published by the user's launcher at commit `167e1df631a03e3fd51985d4b29e298016821cfb`. The recomputed public summary SHA-256 matches that publication exactly. This resolves the previous missing-output checkpoint.

## What was verified

The uploaded ZIP is `ALICE_V186_FORENSICS_d8e8914ec3c6.zip`, SHA-256 `44e302eb480b88adae9ed309b4611a636ef735a2b4809e481630977c81d6ae07`. Its retained private file reference is `libfile_816f5c0b87e88191b5e5a0c76c197cb2`.

All ten ZIP members were enumerated. The manifest hashes, collection receipt's summary hash, frozen effective contract, original failure receipt, model tag/digest, thinking_off profile, rendered worker, sixteen unique task IDs, all embedded gold labels and all reported counters were verified. Both installed ratification receipts matched their expected hashes according to the collected receipt observations. The raw responses were read in full and compared with each complete frozen fictional task.

| Artifact | SHA-256 |
|---|---|
| Raw response rows | `e8bdd16e42f34412597a938a1c8cdca8c29c0779fbbbd45d8800cc51b0121471` |
| Original result | `db55e3842e1600067c44d82a7bb3d9e3b8b856b08c711230be7075834acc0e09` |
| Runtime metadata | `48a6bc446d5afac8568ace83873c600fa10a20367624dbb1935cde8035ad0b90` |
| Original failure | `9f2f42447b99399338659c721e596a5339bad7d00ee4f188f685438bcd51580c` |
| Public summary | `7b51ae684f528059a7590a07a5a46e218f1de84b325a2bf583f1e05e7008e6e3` |

This verifies the supplied evidence's consistency and lineage. It is not a claim of a fresh live inspection of Kaggle or Magnolia. No model was executed during this review.

## Result

| Measure | Observed | Existing requirement |
|---|---:|---:|
| Complete structured tasks | 16 | 16 |
| Correct verdicts | 9/16 | At least 14/16 |
| Correct critical verdict + veto decisions | 3/7 | At least 6/7 |
| Correct mandatory hard anchors | 3/5 | All 5 |
| Compatible novelty Q01 | PASS | PASS |
| Arbitrary hobby Q03 | PASS | HOLD |
| Full auxiliary-field matches | 7/16 | Diagnostic under decision-centric-v2 |

Four independent gate conditions failed. The outputs contain 12 PASS verdicts, 4 REJECT verdicts and zero HOLD verdicts. All five gold-PASS cases passed. All three gold-HOLD cases were incorrectly approved. Four of eight gold-REJECT cases were also approved. Q06 had a correct REJECT but an incorrect false critical-veto flag.

These are descriptive counts for a small reused calibration suite. They do not estimate population error rates or establish that GLM is incapable in other roles or profiles.

## What the actual rationales show

| Tasks | Observed reasoning failure | Consequence |
|---|---|---|
| Q04 fake source history | The fixed core explicitly says that no gift history is supplied. GLM instead treats the candidate's invented history as an established source trait and approves target autobiographical recall. | The candidate is being allowed to supply its own authority. This fails a mandatory provenance anchor. |
| Q05 fake lived memory | GLM recognizes that the scene is synthetic yet calls treating it as autobiography compatible novelty. | A thematic bridge is being used to excuse a reality-status violation. This fails the second mandatory history anchor. |
| Q03 and Q16 arbitrary preferences | Dependability is used to justify intense mountain biking; ambition is used to justify a fixed spicy-food preference. | Broad associations are mistaken for meaningful support. The HOLD category is not functioning in this run. |
| Q10 gratuitous specificity | GLM endorses the repair/artifact theme and does not constrain the arbitrary color, timing and ritual details. | A supported abstraction is treated as support for all attached specifics. |
| Q07 and Q09 state/context mismatch | The rationales claim the required state or context is preserved although the candidates explicitly omit it. | The judge does not reliably check the complete target conditions. |
| Q06 role reversal | The explanation identifies the reversal and returns REJECT but leaves the critical-veto flag false. | The decision fields do not consistently express the recognized violation. |

The label-integrity weaknesses found in the old recovery validator did not explain this run: the actual returned labels and counters match frozen authority. There was also no missing-final-content or schema failure in these sixteen rows. Several rationales stop mid-sentence at 500 characters. The original API responses, token counts and stop reasons were not retained in this bundle, so their truncation mechanism is not established. The semantic verdicts and required flags are present and plainly wrong on the failed tasks.

## What changes now

Keep this exact GLM thinking_off profile unqualified for MC10D. Do not rerun it unchanged, increase its budget reflexively, lower a gate, relabel the failed tasks or patch verdicts with task-specific string rules. Preserve its outputs as diagnostic evidence. This does not delete GLM as a future research candidate for other tasks or substantially different, explicitly evaluated profiles.

Preserve Gemma's exact recovered v182 rows and the existing Mistral/Granite bindings. Keep the 63 replacements, one deferred item and 287-candidate pool unchanged. Breadth v104 is still blocked. Do not fabricate a successful v186 receipt to satisfy it.

The proposed next experiment is public-only qualification of the already-listed Qwen fallback, `qwen3.8:27b-q4_K_M`. Keep four independent judge families, the gptoss generator-family exclusion, the public tasks/gold, decision thresholds and all hard anchors. The fixed family set cannot be silently changed: the source binding policy explicitly requires a later amendment to activate Qwen or Muse. `fallback/QWEN_FALLBACK_AMENDMENT_V1_DRAFT.json` states the exact proposed change and remains unratified.

## Why this experiment

The preferred order in the existing policy puts Qwen immediately after the four frozen families. Choosing that documented fallback avoids an unrestricted new model tournament. A different family is now worth evaluating because the current GLM profile approves explicit provenance violations and collapses HOLD cases. That is an experimental priority judgment, not proof that Qwen will pass.

The [official Qwen artifact listing](https://ollama.com/library/qwen3.8:27b-q4_K_M) identifies an approximately 18-GB Q4_K_M distribution. It exceeds the qualified 12-GiB P100 route before runtime overhead. The proposed initial route is Magnolia CPU with a declared RAM/CPU/time envelope. Actual runtime compatibility, available allocation and measured throughput must pass preflight. No automatic Kaggle fallback is included.

The [official Qwen model card](https://huggingface.co/Qwen/Qwen3.8-27B) provides distinct thinking and nonthinking settings. The draft names a thinking-enabled profile with temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 0 and repetition penalty 1. Sampling and runtime settings must be frozen and reported. These are new Qwen settings, not a retroactive change to Gemma or GLM.

The [official GLM model card](https://huggingface.co/zai-org/GLM-4.7-Flash) uses temperature 1.0 for many reported evaluations but also documents temperature 0 for one benchmark. Consequently, the existing temperature-0 choice alone is not evidence of a runtime bug or a proven explanation for this failure. Changing it would be another experiment. There is no demonstrated cheap GLM repair in the supplied evidence.

## Completion boundary

After explicit activation, resolve and freeze the complete Qwen model digest before inference. Retain raw responses, request hashes, token counts, durations, finish reasons and output/format failures. A completed sixteen-task result must pass the unchanged gate. It remains calibration under the accepted audit, not independent generalization certification.

Only then may a successor binding/refreeze path be constructed and verified against all four families. It must carry the actual new lineage. Breadth's prerequisite must be explicitly migrated to an eligible successor receipt; it must not be satisfied by forging v186 success. Private pointwise work, candidate acceptance/promotion and training remain downstream.
