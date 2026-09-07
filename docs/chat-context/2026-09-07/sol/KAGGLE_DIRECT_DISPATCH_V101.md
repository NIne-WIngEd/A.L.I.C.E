# Kaggle direct-dispatch correction — 2026-09-07

The first Kaggle launcher did not submit a remote job. It stopped locally because it depended on a stale handoff path:

`C:\A.L.I.C.E-main\tools\alice-kaggle.ps1`

The current local repository does not contain that wrapper. The worker SHA was verified successfully before this stop:

`8BF89028BB9798FA60FEFA23F0CE1FAD3DA1E380CE6C7D6CCE33593D58011CB8`

Scientific state is unchanged. No Qwen Kaggle inference occurred. Qwen a1/a2/a3 remain closed. Magnolia remains paused. Private pointwise execution, MC8, A-SYN acceptance/promotion, and training remain blocked.

## Correction

Do not spend another cycle restoring the old wrapper. Use the official Kaggle CLI directly. Kaggle CLI 2.2.4 supports private script-kernel push, kernel status, logs, output download, and deletion.

Prepared launcher:

`Start-ALICEKaggleQwenQualificationDirectV101.ps1`

SHA-256:

`6B153EDEDCF51FF09D17B8CD48DB3619450BAA1F6F99A1D327C3B4D66DE46AAF`

It uses the already-frozen worker:

`ALICE_Kaggle_Qwen_Public_Calibration_v1_0_0.py`

SHA-256:

`8BF89028BB9798FA60FEFA23F0CE1FAD3DA1E380CE6C7D6CCE33593D58011CB8`

The launcher creates a private Kaggle script kernel with internet and `NvidiaTeslaT4`, requires the worker itself to observe two T4 GPUs, waits for terminal state, captures logs and outputs, and deletes the remote kernel only after terminal success plus evidence download are both proven. Failed or unresolved remote runs are preserved instead of blindly rerun.

This correction changes dispatch infrastructure only. The public Qwen calibration task/model/rubric/profile frozen in the worker is unchanged.
