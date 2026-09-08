# START HERE — Sol continuation, 2026-09-07

Read in this order:

1. `STATE.json`
2. `KAGGLE_PROVIDER_SWITCH.md`
3. `SOL_CONTINUATION_HANDOFF.md`
4. Prior authority: `../../2026-09-06/astra/ASTRA_MASTER_CONTINUATION_HANDOFF.md`

Current boundary: Qwen is still **NOT_EVALUATED**. a1/a2/a3 remain closed. V105 must not be rerun.

The Magnolia source-runtime feasibility job was executed as job `575231` and reached a terminal `FAILED` scheduler state. The local collector then hit a Windows/POSIX path-comparison bug and refused the result path. That collector defect is not a Qwen scientific result. The owner has now directed the project to stop spending time on Magnolia infrastructure and use Kaggle so the architecture can advance.

**Magnolia is paused. Do not rerun job 575231 and do not create another Magnolia repair package.**

The next action is one bounded Kaggle public-calibration execution using the already validated private `alice-kaggle v0.2` route with two Tesla T4 GPUs. The model, quantization, prompt, public 16 tasks, decoding settings, gold labels and pass rule remain unchanged. The compute profile is newly frozen as `qwen38_thinking_on_public_calibration_kaggle_t4x2_v1` before any Qwen response is observed.

No private pointwise candidates. No MC8. No A-SYN acceptance/promotion. No training. Canonical main remains frozen.

## 2026-09-07 Kaggle v1.0.2 transport correction

Two local-only launch failures produced zero remote Qwen execution: V100 referenced a stale missing `alice-kaggle.ps1` wrapper, and V101 recreated the already-solved Windows PowerShell 5.1 UTF-8 BOM bug in `kernel-metadata.json`. Do not rerun either launcher.

The replacement is `ALICE_KAGGLE_QWEN_PUBLIC_v1.0.2.zip` with a tiny PowerShell launcher and a Python controller. The controller owns all JSON/state, writes UTF-8 without BOM, stages exactly `kernel-metadata.json` plus one self-contained `script.py`, persists one deterministic Kaggle identity before push, pushes at most once, reconciles 404/not-found without repushing, resumes the same identity after local interruption, preserves raw CLI exit/stdout/stderr receipts, retrieves output before cleanup, and preserves failed remote kernels for diagnosis.

The exact deterministic kernel ref is `mkrayanyan/alice-qwen-k1-v102-55b8e7acb4c1`. The previously validated shared runtime dataset `mkrayanyan/alice-tournament-runtime-50539c5fe9bf` supplies the pinned Ollama v0.32.15 archive so the job does not waste Kaggle ephemeral disk on another 1.42 GB runtime download. The worker still verifies the archive and binary hashes, requires two T4s, proves material model use on both GPUs, performs a pre-model disk gate, then runs the frozen throughput probe and 16 public tasks only.

No private pointwise data. No MC8. No A-SYN acceptance/promotion. No training. Canonical main remains frozen.

## 2026-09-07 Qwen result verified

Qwen public calibration on the frozen Kaggle 2x-T4 profile completed successfully. Returned result ZIP SHA-256: `d5f9f75d33ca92d7455677ba4f04ca8fa090ccba03b3946732671a0403669d53`.

Independent evidence audit verified all 84 manifest-listed files, all task request/response hashes, 16/16 unique complete tasks, exactly one attempt per task, frozen model/profile identity, and the predefined score. Qwen passed with 14/16 verdict matches, 7/7 critical decisions, and 5/5 mandatory hard anchors. This is public calibration only, not independent certification.

Known Qwen public-suite errors are Q09 (expected REJECT, observed HOLD) and Q12 (expected PASS, observed REJECT). Preserve them as challenge debt. Do not tune Qwen on the same 16 tasks and rerun.

Read `MC10D_QWEN_V102_VERIFIED_AND_SYNTHESIS_FIDELITY_GUARDRAILS.md` before any next MC10D package. It locks the bigger-picture E0/E-INF/A-SYN fidelity doctrine and the exact MC10D objective.

Immediate next action: verify preserved Gemma/Mistral/Granite receipts and effective profiles, construct a new truthful Gemma+Qwen+Mistral+Granite successor binding, refreeze the unchanged 287-candidate pool as pointwise-ready, then explicitly migrate the breadth prerequisite. Do not start private pointwise before those gates.

## MC10D v1.9.0 consolidated pre-pointwise step

The owner requested that all MC10D work that can safely be combined should be combined. The resulting v1.9.0 package batches every deterministic/local step through the private-pointwise authorization boundary: Qwen evidence revalidation, preserved Gemma recovery, frozen Mistral/Granite verification, Qwen successor adapter, truthful four-family binding, unchanged 287-candidate refreeze, and controlled-synthesis breadth-prerequisite migration.

Package SHA-256: `5A83B08399A8AB8A06FAE0DCA7345E87D2355544FC593E7207FDAF17AC97FDD0`.

Launcher SHA-256: `8ABF9EEB5BF022DEA701AA4F9A007FAFE61E19FCA5A16DEA2267D1A86374E169`.

Read `MC10D_QWEN_SUCCESSOR_V190_PREEXECUTION_BOUNDARY.md` before continuing.

Do not bundle private pointwise or full simulation into this step. Pointwise must first bind the exact v1.9.0 output hashes. Full simulation must then bind the exact frozen pointwise result. Current next action is `RUN_MC10D_QWEN_SUCCESSOR_REFREEZE_BREADTH_V190`.

## MC10D v1.9.1 current execution boundary

v1.9.0 reached the correct 287-candidate Gemma+Qwen+Mistral+Granite refreeze but stopped because the copied historical v1.7 pointwise-ready validator still hard-coded GLM. This was a deterministic validator migration bug, not a scientific failure.

Current package: `ALICE_MC10D_QWEN_SUCCESSOR_REFREEZE_BREADTH_v1.9.1.zip` SHA-256 `397EE77874D9D2AF7AE197AF5EC3FEB478D222930D46F22AE52C953B31023D72`.

Current launcher: `Start-ALICEMC10DQwenSuccessorRefreezeBreadthV191.ps1` SHA-256 `E03304DED31E0A9492DE111428B27A0044AEACD91BD7D245CC9D37C708154E02`.

Read `MC10D_V190_VALIDATOR_STOP_AND_V191_TECHNICAL_SUCCESSOR.md`. Next action: `RUN_MC10D_QWEN_SUCCESSOR_REFREEZE_BREADTH_V191`.

## MC10D v1.9.2 current execution boundary

v1.9.1 reached the correct 287-candidate Gemma+Qwen+Mistral+Granite refreeze and scenario freeze, then exposed a second independent GLM hard-code in the compact historical pointwise-ready validator. v1.9.2 audits and migrates the entire legacy family-set surface instead of fixing one assertion at a time.

Package SHA-256: `5BA15F6807A66FEA8601333F809F3C798E0621F95B4C9A9383BFEBD940873A39`.

Launcher SHA-256: `C87BF4ABCAA59B7970C6CDB7CF9FA562929DADD71CFF215EA9AA0F04726D017E`.

Read `MC10D_V191_SECOND_VALIDATOR_STOP_AND_V192_WHOLE_VALIDATOR_FIX.md`.

Next action: `RUN_MC10D_QWEN_SUCCESSOR_REFREEZE_BREADTH_V192`.

## MC10D v1.9.3 current boundary

v1.9.2 completed the actual 287-candidate successor refreeze, passed the internal pointwise-ready validator, atomically published the canonical ready tree, and created pointwise-ready ZIP SHA-256 `5CC3C64EF4E64D75BD41E860F34CCD8FE8C882EEDBB6F6C6D0F68CB13218EF30`.

Its exit 76 occurred afterward because the outer validator assumed a single wrapper directory around the historically flat v1.7 ZIP. **Do not rerun the historical refreeze.**

Current package: `ALICE_MC10D_QWEN_SUCCESSOR_POSTREFREEZE_RECOVERY_v1.9.3.zip` SHA-256 `8A8AC8C9FAA1C999E6F1236F80BC04FBA625F10CA9B3DF4D9F9FF94D2EBC8D9D`.

Current launcher: `Start-ALICEMC10DPostRefreezeRecoveryV193.ps1` SHA-256 `28C90285AAAFA1EE8C3E57A38EA67A18ED74C5D8EC29FD3B59A1E03C70567D53`.

Read `MC10D_V192_POST_REFREEZE_STOP_AND_V193_RECOVERY.md`.

Next action: `RUN_MC10D_POST_REFREEZE_RECOVERY_V193`.

## MC10D v1.9.4 current boundary

v1.9.3 verified the exact v1.9.2 pointwise-ready bundle, the flat historical archive layout, the 287-candidate Gemma+Qwen+Mistral+Granite binding, the 55,104-probe scenario geometry, and equality with the 83-file canonical atomically published Vault tree. It stopped only because the recovery validator incorrectly required legacy Gemma result JSON identity fields that may be absent.

Do not rerun the historical refreeze.

Current package: `ALICE_MC10D_QWEN_SUCCESSOR_POSTREFREEZE_RECOVERY_v1.9.4.zip` SHA-256 `215C671664B560B8DD13EAD150B7D8BEDFE6A9A8D38DD8E4CCB9BD98F7D3906D`.

Current launcher: `Start-ALICEMC10DPostRefreezeRecoveryV194.ps1` SHA-256 `DCD9AD4BD2267BB1141927097E485B1B66AD3A19DBF6E86D21130980B0B6ADEA`.

Read `MC10D_V193_GEMMA_SCHEMA_STOP_AND_V194_RECOVERY.md`.

Next action: `RUN_MC10D_POST_REFREEZE_RECOVERY_V194`.
