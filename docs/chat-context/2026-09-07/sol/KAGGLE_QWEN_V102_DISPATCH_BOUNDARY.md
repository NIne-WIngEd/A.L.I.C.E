# Kaggle Qwen public calibration v1.0.2 dispatch boundary

Prepared after auditing the prior A.L.I.C.E. Kaggle v0.2 workflow, the solved Windows UTF-8 BOM failure class, the adaptive tournament 404/reconciliation behavior, MC10D single-code-file transport, and the Q12 retrieval-only recovery path.

## Frozen package

- package: `ALICE_KAGGLE_QWEN_PUBLIC_v1.0.2.zip`
- package SHA-256: `7352FE1A32713D2E85A400E70FFFD656AE05C760F134AE6AE83D87B9D4053006`
- launcher: `Start-ALICEKaggleQwenQualificationV102.ps1`
- launcher SHA-256: `89BA0E1121C829D84A03C05D1B7217CA89AFF5E36E64DDBB2CB7C525DBECD738`
- worker SHA-256: `35389B3D7B83FA43AC1A358C9610670183544378EFEDE3D095F616D8AA1B4F4A`
- controller SHA-256: `4DB3219A1AA48994F13D09D52DA7FE73751F2472246A3AEE4F0A96074CCD3F2E`
- selftest SHA-256: `147D8B25F85304B41E0A25C66E876C0ADC2AA72732B4788837C17792B5B8C700`

Deterministic request SHA-256:

`55b8e7acb4c16284773f371196d0cb9d6e080eef83f46b41476d45302d6593a3`

Deterministic private Kaggle kernel:

`mkrayanyan/alice-qwen-k1-v102-55b8e7acb4c1`

## Recovered transport rules now enforced

1. Machine-readable JSON is written by Python as UTF-8 without BOM. PowerShell never serializes Kaggle metadata.
2. Kernel staging contains exactly two files: `kernel-metadata.json` and one self-contained `script.py`.
3. The kernel identity is content-derived and persisted before the first push.
4. The controller can invoke `kernels push` only once for this identity. After a push attempt, 404/not-found triggers exact-ref reconciliation only, never a repush.
5. Local interruption exits to a reconciliation pause; rerunning the same launcher resumes the same Vault state and same kernel identity.
6. Every Kaggle CLI call stores argv, exit code, stdout and stderr as raw receipts before interpretation.
7. Kaggle output is retrieved and hash-validated before a successful kernel is deleted. Failure kernels remain preserved.
8. The shared runtime dataset `mkrayanyan/alice-tournament-runtime-50539c5fe9bf` is mounted. The worker verifies the exact 1,422,416,084-byte Ollama archive SHA `50539c5f...e42bb` and exact binary SHA `eb99a47a...99c8` before use.
9. The worker gates ephemeral disk before the Qwen model pull using the historical one-GiB reserve rule, verifies two Tesla T4s, and proves material memory use on both after the frozen public throughput probe.

## Local gates passed

`metadata_utf8_no_bom=true`
`metadata_json_parse=true`
`title_slug_binding=true`
`single_code_file_transport=true`
`deterministic_kernel_identity=true`
`push_once_contract=true`
`404_never_causes_repush=true`
`native_exit_receipts_preserved=true`
`resume_same_identity=true`
`terminal_output_retrieval=true`
`retrieval_failure_never_causes_inference_rerun=true`
`kaggle_scratch_disk_preflight=true`
`dual_t4_runtime_proof=true`
`frozen_16_task_contract=true`
`private_candidate_files=0`
`hidden_MC8_files=0`
`A_SYN_acceptance=false`
`A_SYN_promotion=false`
`model_training=false`
`canonical_main_mutation=false`

## Scientific boundary

Qwen a1/a2/a3 remain closed. V100 and V101 are local launcher failures only and created no Qwen scientific evidence. V102 is prepared but not yet executed. MC10D remains 287 effective candidates, 63 valid replacements, one defer, slot 63 no attempt 4, slot 64 done. Private pointwise screening has not started and MC8 remains sealed.
