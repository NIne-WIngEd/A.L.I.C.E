# Run instructions

Required existing local state:

- `C:\A.L.I.C.E-main` at frozen main commit `0abaed85873c3f8de04765847eb7700b0e20433f`.
- `C:\ALICE_Vault` containing canonical MC10C and the preserved v1.6 workroot/checkpoint.
- `ALICE_MC10D_PREEXECUTION_FREEZE_BUNDLE_v1.1.zip` in Downloads with SHA-256 `22B0ADBCCF442B0B3654F964E35AB77A044AF7623891E2A60C72B06A94ECE9A3`.
- Kaggle CLI authenticated as the existing project account.

Do not manually delete or edit the v1.6 checkpoint, failure receipt, workroot, or remote resources before running v1.7.

Use the supplied PowerShell launcher. It verifies package/freeze hashes, clean-extracts the package, compiles Python, runs the full local self-test, and starts the Python controller.

Exit 75 is a safe remote-reconciliation pause. Exit 76 is a deterministic transport/preflight stop. For either result, preserve the terminal output and remote state; do not blind-retry or hotfix.

A successful run ends at `MC10D_POINTWISE_SCOPE_JUDGE_SCREEN_CURRENT_EFFECTIVE_POOL_WITH_OPEN_COMPLETION_FRONTIER`. It does not start full MC10D simulation/falsification or personality-model training.
