# N0 Real Training 575546 Complete — 2026-09-14

The first real durable N0 MLM learning segment completed successfully on Magnolia.

Scheduler evidence:

- job `575546`
- name `rayan-n0-p100-pilot`
- `COMPLETED 0:0`
- elapsed `00:21:57`
- node `gpu001`

Canonical checkpoints now exist at step 100 and step 200 under:

`$HOME/rayan-compute/rayan-n0/n0-v01/checkpoints`

The inspected step-100 receipt binds the run to the verified imported public corpus, real tokenizer, 352,184,960-parameter native N0 model, 2×P100 world size, fp16, sequence length 512, and `private_identity_gradient=false`. It records 1,638,400 total tokens by step 100 and training-code revision `79d4b2d0b4c3bff8932dbc5c5a68f79256d59b9e`.

A second submission (`575547`) was a duplicate attempt and must not be used as a fresh step-zero run. Durable checkpoint state is authoritative. Future N0 learning continues by inspecting the latest receipt and resuming from that exact checkpoint.

Immediate next action is model evidence, not infrastructure: inspect the step-200 receipt and the 575546 loss trajectory. Use that evidence to select the next MLM continuation horizon. In parallel continue building the semantic/judgment curriculum and N1/N2 identity-learning path in Git.

Do not reopen runtime, DDP, corpus acquisition, Kaggle transport, or Magnolia import qualification unless a genuinely new failure occurs.

No private E0/E-INF/A-SYN gradient is authorized yet.
