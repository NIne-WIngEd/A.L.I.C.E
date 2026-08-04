# Memory Repository Inventory

**Baseline commit:** `d5d311ec49f1e4a3e5a7cf688062c7dc2f46d4ec`<br>
**Generated:** `2026-08-04T05:07:30Z`<br>
**Scope:** Public repository metadata and static architecture surfaces

## Purpose

This inventory records the current public memory, evidence, storage,
retrieval, lifecycle, and governance surface before Memory Architecture v4
runtime renovation begins. It does not claim that every listed file is a
complete implementation.

## Summary

- Memory-related tracked files: **165**
- Aggregate size: **1.80 MiB**
- Runtime behavior changed by M0: **No**
- Private content inspected: **No**

## Subsystem totals

| Subsystem | Files | Bytes |
|---|---:|---:|
| `cognitive_kernel_memory_storage` | 20 | 427.15 KiB |
| `memory_benchmark` | 1 | 9.65 KiB |
| `memory_documentation` | 18 | 163.22 KiB |
| `memory_governance_tooling` | 5 | 47.67 KiB |
| `memory_policy` | 7 | 144.77 KiB |
| `phase2_memory_core` | 33 | 496.04 KiB |
| `phase2_tests` | 37 | 449.29 KiB |
| `phase5_tests` | 44 | 100.58 KiB |

## Tracked file manifest

| Public path | Subsystem | Lines | Bytes | SHA-256 |
|---|---|---:|---:|---|
| `src/cognitive_kernel/attention.py` | `cognitive_kernel_memory_storage` | 827 | 31812 | `478048075AC9B1CB5737C714262BF3825F733771C9E022D7FF3F3FC598D1D620` |
| `src/cognitive_kernel/attention_policy.py` | `cognitive_kernel_memory_storage` | 411 | 16090 | `16C6B8FDB8EED5B1DE2CE281ADE378BF096D5322C0E8E5A923FFD65D1310B287` |
| `src/cognitive_kernel/canonical.py` | `cognitive_kernel_memory_storage` | 137 | 4837 | `40455537A16D55C5CDC26BB5A393BE1B8D4FB9CA7883C8DE45674666C335C03F` |
| `src/cognitive_kernel/contracts.py` | `cognitive_kernel_memory_storage` | 425 | 14898 | `97A51EC61A6122E843669B5D497BA5B139BE2EA20CFDA4E622207A47BB609E6E` |
| `src/cognitive_kernel/experience.py` | `cognitive_kernel_memory_storage` | 406 | 14658 | `2EFEDF5602991A3383E35C4B6B35F21198EC109CE971D05A06755349E50337CF` |
| `src/cognitive_kernel/ledger.py` | `cognitive_kernel_memory_storage` | 426 | 15697 | `D0CE1599F3F9BACEC979134D1D58878C1F0ED51684CAF7031DFF1FF3666FA47A` |
| `src/cognitive_kernel/ledger_policy.py` | `cognitive_kernel_memory_storage` | 334 | 12526 | `6922FD34474FEF9FDBE8470DC09F85BEBBC6A322AF910408857142FC20F803A9` |
| `src/cognitive_kernel/ledger_store.py` | `cognitive_kernel_memory_storage` | 599 | 23836 | `45A6056A196D619D080706ECFA3F176E321EC8500991844319B44CF3E16DB0CA` |
| `src/cognitive_kernel/lifecycle.py` | `cognitive_kernel_memory_storage` | 1305 | 48974 | `129EFEFFFF4E45C33AB0A505CFFC05B3FA17F8456E88E6C9A216FB2F5EC1DA4D` |
| `src/cognitive_kernel/lifecycle_policy.py` | `cognitive_kernel_memory_storage` | 542 | 20057 | `E07514CB6CDFF4F95EC195F163B4AFDAD9EFBA6087FDB6A6BF6953403B4007B0` |
| `src/cognitive_kernel/lifecycle_store.py` | `cognitive_kernel_memory_storage` | 824 | 33388 | `417688A17C8832027A1BC0C8C878B8301C5E50E7C5C02345ADCDEA7D0D248B30` |
| `src/cognitive_kernel/mission.py` | `cognitive_kernel_memory_storage` | 658 | 27144 | `4F66800485F75181E28FD2A82776408154C5A3632664C5F54360214D3FA82C5C` |
| `src/cognitive_kernel/mission_policy.py` | `cognitive_kernel_memory_storage` | 190 | 10094 | `AFE2ED88FBCC58D304B8AE56916248EFA714FCF5D83821A570D99E3CA477B1B5` |
| `src/cognitive_kernel/payload_store.py` | `cognitive_kernel_memory_storage` | 565 | 21085 | `1525848BFEF458D809C33CB711115CEA9FC9FBAE4A637CCFD526985B7AA77AFB` |
| `src/cognitive_kernel/raw_buffer.py` | `cognitive_kernel_memory_storage` | 345 | 13127 | `D3863C74D3483C575E3AE0BFAC40840E0D305B95873702C0A78EE3B6CB989378` |
| `src/cognitive_kernel/raw_buffer_policy.py` | `cognitive_kernel_memory_storage` | 209 | 8091 | `91231D3B69323F758129867512F44223F243027F0458691524981A71A40FAEA6` |
| `src/cognitive_kernel/tier_transition.py` | `cognitive_kernel_memory_storage` | 792 | 32264 | `D04B33AAF58A865755FE56677657515404F1C5BE7FF44999BC3831268C118B74` |
| `src/cognitive_kernel/tier_transition_policy.py` | `cognitive_kernel_memory_storage` | 348 | 12451 | `C0EAACD5587D6BE07622E2B27CF13B85666C138B95F9CE32D7A629BFE90CA6EF` |
| `src/cognitive_kernel/tier_transition_store.py` | `cognitive_kernel_memory_storage` | 1166 | 46676 | `D3CA9065A207C7FAC4002F5B080CDEE7C8C8E58CD9B7A8B96F107D4F2D677A8A` |
| `src/cognitive_kernel/workspace.py` | `cognitive_kernel_memory_storage` | 792 | 29698 | `D18E47D9C27EAEB5429825D68BEFA7B57E3F89A63075E388AE0626B500106430` |
| `benchmarks/phase2/memory_core_evaluation_v1.json` | `memory_benchmark` | 334 | 9886 | `0B430B2021E5E445DE15179A451C9E2303262CDF34B360525EDAFF34C6094C42` |
| `docs/FRIDAY_COGNITIVE_WORKSPACE_AND_PRODUCTION_GOVERNANCE_PLAN.md` | `memory_documentation` | 1727 | 44576 | `16805DAD98739BF2DD311B06B8D6E83B973AD03CBA2182E5DCF180A8B78474DE` |
| `docs/MEMORY_ARCHITECTURE_HOLD.md` | `memory_documentation` | 137 | 6348 | `5A74077F223ED3B8909850F898A700A71D8F19AEC9BC28965D69E6A0BF0B7A82` |
| `docs/MEMORY_ARCHITECTURE_V4.md` | `memory_documentation` | 659 | 19043 | `6B60CDC6E0DCAF65A5C1F47303FE943445B48ED9D21F7AB407F7F66D4A8E6F0E` |
| `docs/MEMORY_CLAIM_COVERAGE_MATRIX.md` | `memory_documentation` | 87 | 8123 | `F89996F6E3C06EA910104C50ECF7EF025746CC893388FA7DB7D50C72ACF9EA95` |
| `docs/MEMORY_EXTERNAL_SYSTEMS_REVIEW.md` | `memory_documentation` | 447 | 17583 | `470DB739F1E63DBD50C3AD34BAB13CC6FF09D55CDFD38BF6B2A28DFD0A149D0F` |
| `docs/MEMORY_PERFORMANCE_AND_RELIABILITY_STANDARD.md` | `memory_documentation` | 191 | 5625 | `8E2A744107C89B7A4E084CB949FF068AAA168A16E2A0F3DDAF4C49F7E0DBB953` |
| `docs/MEMORY_POLICY.md` | `memory_documentation` | 63 | 4834 | `9476AAEEEE857906ECAA5930825E30C67A0B40A41C3173952B38910B523B03D1` |
| `docs/MEMORY_RECORD_AND_PROVENANCE_STANDARD.md` | `memory_documentation` | 190 | 5073 | `53A36F40320882FFE5236B7F14FD5A55CB199E827FCF5B95F7C713311125331E` |
| `docs/MEMORY_RENOVATION_PLAN.md` | `memory_documentation` | 191 | 4681 | `43CEDB3FAE298A1C8104C72814BC79EF6EF9B580846A7614418BD1A60062127F` |
| `docs/PHASE2_TO_KERNEL_MEMORY_MIGRATION_PLAN.md` | `memory_documentation` | 200 | 5495 | `ABAEAA7D0E391A03D75D7273008127068F3374584F74185D37DEC364B8913FFC` |
| `docs/PHASE_2_MEMORY_CORE_ARCHITECTURE.md` | `memory_documentation` | 581 | 20563 | `02522A38D86739BBB5531F6EDB2D117C899E6F2A01DF6FEA25633E725CC04088` |
| `docs/PHASE_5_COMPACT_EXPERIENCE_LEDGER.md` | `memory_documentation` | 54 | 2239 | `4B47AAEB6BAF39631FF2008E37851CFB79BCDF403698F76BC7DC3C937DB605AD` |
| `docs/PHASE_5_GOVERNED_TIER_TRANSITIONS.md` | `memory_documentation` | 27 | 2215 | `9A8F7992CE7BDFA60C402A55482A374ADEDC409D81662E1A113A21FEDCBF1A95` |
| `docs/PHASE_5_MISSION_GRAPH_CONTRACTS.md` | `memory_documentation` | 32 | 2024 | `9A90521238141FC44A61D53CC17755E5AC310263B5310840A82B53B2E9B6B2FB` |
| `docs/PHASE_5_RAW_BUFFER_CONTENT_STORE.md` | `memory_documentation` | 19 | 1671 | `9675DBD20E56D2E5E8EE29BB868D4356E89DB5B034868E31F6E8135D54117EFA` |
| `docs/PHASE_5_RETENTION_LIFECYCLE_JOURNAL.md` | `memory_documentation` | 25 | 2188 | `90DA2DD06716FBFF6071084176EAB7835648C01CC385E5344724AE5FFD4A9D06` |
| `docs/STORAGE_LIFECYCLE_AND_RETENTION_POLICY.md` | `memory_documentation` | 295 | 11826 | `9B7B3241576811408184D521E1861DC578538A096352DFDA809FAE472C3153D7` |
| `docs/decisions/ADR-008-aggressive-capture-selective-retention.md` | `memory_documentation` | 59 | 3032 | `7683B587B536AE1720E006336297338E5C6386E6749F86F434F6342AAA541849` |
| `scripts/audit_capability_barriers.py` | `memory_governance_tooling` | 530 | 20408 | `F109886036D114A8880638FE5DF0BE64630600B7A4F44F50525D0EBADD96FDB5` |
| `scripts/audit_repository_phase_boundary.py` | `memory_governance_tooling` | 250 | 8496 | `879F821B458219B96216963460DD589C7FCD59C47F0ABB7B4F59A06FD48DEB0F` |
| `scripts/migrate_capability_barriers.py` | `memory_governance_tooling` | 147 | 5944 | `C9A92256C30EC939DBAF5D0936614DCAF4B2F844D65327F1BAEA67D1FFFD38BA` |
| `scripts/run_phase2_memory_release_audit.py` | `memory_governance_tooling` | 95 | 3510 | `720C457C7405E85CCCFC3ABF05BF11BCBC64C68A9BF0A23D2361E7AD5D1D3B3B` |
| `scripts/validate_phase5_parity_release.py` | `memory_governance_tooling` | 241 | 10453 | `7FBB2D6612207DB01395436F5E3BCC2954A115B30A987E269CEF59FEE1C2EC01` |
| `policies/capability_profiles.json` | `memory_policy` | 456 | 18014 | `1F6EA249159DF64D7B1E9D6F3C7E0E065ABB9D91E806D2D8E8D1D474E3D234BC` |
| `policies/cognitive_kernel_lifecycle_policy.json` | `memory_policy` | 129 | 3303 | `963062A05D972602C4598F9B0A7DF9BC3F0DA00AD10C6AE92DF826B7BE6534E5` |
| `policies/cognitive_kernel_raw_buffer_policy.json` | `memory_policy` | 45 | 1604 | `13678440971A9CB8E2C3EACC669AD748AE71D01D0BF29348121BC3F78ACA9A1E` |
| `policies/cognitive_kernel_tier_transition_policy.json` | `memory_policy` | 81 | 2488 | `3C404BAC2CB3745A42033697E16D948CAFBEE31436562D8A6F22C83C8C0E98E7` |
| `policies/memory_evaluation_policy.json` | `memory_policy` | 104 | 2517 | `0B9622B61603081C14C8EFF679B7A19B95A62DC36C7E9E3F7BAB3AB36C93FA6D` |
| `policies/phase_scope_registry.json` | `memory_policy` | 1983 | 115993 | `2CC86E70D704E0C6A31B53FC74D9FEC659351D6DC06890E6B257471F0670C55B` |
| `policies/storage_lifecycle_policy.json` | `memory_policy` | 146 | 4326 | `C98CDF0E6AF148B18F294CA0C9C0F5218492835E9D6564F54837C620593ACF03` |
| `src/alice_memory/__init__.py` | `phase2_memory_core` | 3 | 92 | `4B80F4F4A31F975A775B2E3365DFA5C781CEDF8027CFE9D04FA588315C9C1D45` |
| `src/alice_memory/candidate_assessment.py` | `phase2_memory_core` | 464 | 14682 | `7C28F1893C0DDBDEE3D9B837CEA1ED3BDB8C908861043B6587E47996838132D0` |
| `src/alice_memory/citation_evaluation.py` | `phase2_memory_core` | 507 | 16974 | `5EA71F19A20B171BF6E226C116F50BB8DB15009E691D793C1CAAA1E7C0D89AEF` |
| `src/alice_memory/cited_answer.py` | `phase2_memory_core` | 446 | 14345 | `C08ACBFDA750B0758E28C1DE213054E2230E15650F6847BEE3B14F9B131977B4` |
| `src/alice_memory/deletion.py` | `phase2_memory_core` | 1010 | 34307 | `636FAFE90098083985F072E5F69A521720F2C58193ECC40BE66E195398CC95B7` |
| `src/alice_memory/deletion_evaluation.py` | `phase2_memory_core` | 243 | 8976 | `22C7AD078761CF5F2DE8CC8EA8FC4BFCE184F8ABE02C658EF67C44383F118100` |
| `src/alice_memory/deletion_indexes.py` | `phase2_memory_core` | 395 | 12196 | `B3A5ED87E2C8CE7B712F147834A8650AD09130BBF3A4943ADE55CFD0927C3BEC` |
| `src/alice_memory/deletion_integrity.py` | `phase2_memory_core` | 469 | 16173 | `99A7F05F5304210C83593B64E79AF14F59FC46BFAE411ED8FD9A489122A21330` |
| `src/alice_memory/evaluation_contract.py` | `phase2_memory_core` | 619 | 21264 | `9E3C2C446A89E1E731B3CB564FEE92CB887D580B357EA05C7C0AAFB7095BEBC3` |
| `src/alice_memory/evaluation_fixtures.py` | `phase2_memory_core` | 714 | 22333 | `318E874D3D19CAC3056577DA256B98043FD3D7731AC4E14B601EF18CABF02DB3` |
| `src/alice_memory/final_evaluation.py` | `phase2_memory_core` | 456 | 17359 | `5E5081839A2667A97B053F83426A15336FF946DCD0DF2DFD93281927F68D4F4F` |
| `src/alice_memory/formation.py` | `phase2_memory_core` | 572 | 17878 | `1B7DBCFB3AD117E81E845B5F20AF89D86554C7721AD17DC1C214EA21F6B9FD34` |
| `src/alice_memory/hybrid_retrieval.py` | `phase2_memory_core` | 374 | 10014 | `1BD2BB4878B9135F8E32F37DAB962A9D1EDD85DB77E874B386CA0BD7AF34250F` |
| `src/alice_memory/inspection.py` | `phase2_memory_core` | 283 | 7817 | `58B1C7CFC7597B6F41E4539C17E2606DE48CEB3B4A135DC48C73CE178B7DFB7C` |
| `src/alice_memory/lexical_index.py` | `phase2_memory_core` | 510 | 14728 | `BC98C05FB6F78834EAF7902C8178B48EBC02C3E647F9E10A39A3D3017C66E139` |
| `src/alice_memory/migrations.py` | `phase2_memory_core` | 211 | 6519 | `770EEBD09D42CCA165F6E528949F439007EAC1F3EEC7CA31FEFB2135CB90571F` |
| `src/alice_memory/phase1_bridge.py` | `phase2_memory_core` | 400 | 13009 | `EC7528EFBDD50260D8BB2B8AF7839B4E18CCF266F951ED7A7E2EDAB31154D828` |
| `src/alice_memory/promotion.py` | `phase2_memory_core` | 706 | 23627 | `6127270F3279B38F65FBFEA45412C79FCFC016C173F173911C8109B08764C6C6` |
| `src/alice_memory/provenance.py` | `phase2_memory_core` | 200 | 5851 | `9D6661301719DF0F71845A1F0C4427CEECD97DDECF58B538D4EC1C9BAB6BA198` |
| `src/alice_memory/release_audit.py` | `phase2_memory_core` | 350 | 13320 | `91B48E44D008185D7B09D63600BD68684D04142E0D02C46AD94F84C76508556E` |
| `src/alice_memory/retrieval.py` | `phase2_memory_core` | 373 | 9838 | `58E6D87C2125946AB21631CEBCE28486CBEAFCF220834971E16FE69DA36FD4C2` |
| `src/alice_memory/retrieval_models.py` | `phase2_memory_core` | 82 | 2132 | `DD1DA3C1EB2A8AF16C27D0C6BAF8948722106FC790B7E679CD81ED3607A9B5DB` |
| `src/alice_memory/schema.py` | `phase2_memory_core` | 478 | 15611 | `E96EF07BA536AD1F98DD0D916EC334A3F815C60DB4179395D3BD7F16B7B122A0` |
| `src/alice_memory/semantic_index.py` | `phase2_memory_core` | 705 | 19931 | `5D64861A354E5F1B80BAEE3938BBA387E17E3D7B00D81DC093D6B2B49E434DC5` |
| `src/alice_memory/sensitive_access.py` | `phase2_memory_core` | 644 | 20983 | `B4084C5BEFE1748D9E3FB23546B1D89206F3839B532059831D37430D8763A8B0` |
| `src/alice_memory/sensitive_crypto.py` | `phase2_memory_core` | 378 | 12227 | `3DDCF91BE455A8AEC7B1ABE641C92EAD2C3C2E036B6991DA23B719BE282B0AA0` |
| `src/alice_memory/sensitive_deletion.py` | `phase2_memory_core` | 910 | 32285 | `2654690C5CF6CD3D6861466890A62A1E6B45CE11B478477DE9CB36D712EC8A78` |
| `src/alice_memory/sensitive_storage.py` | `phase2_memory_core` | 272 | 8394 | `ED000063867DE86AB75E8269B762C23E3B7AD4AC1DB3BEA52A4B93010064F003` |
| `src/alice_memory/service.py` | `phase2_memory_core` | 654 | 19248 | `909C94340B4452313AB99054D62C7932111BF40AC7D669DC6BC8BBF8EE349A62` |
| `src/alice_memory/sources.py` | `phase2_memory_core` | 307 | 9208 | `763B0FA655CD5CED9C3601236FD394EE1F64A2FA82E6A67AA09FDADDA4DA3AF8` |
| `src/alice_memory/store.py` | `phase2_memory_core` | 159 | 4757 | `8D236175E10E58176888EEAC82DF0FE8A9D3797E3383CE74D8EF5D9B9236C836` |
| `src/alice_memory/temporal.py` | `phase2_memory_core` | 872 | 23281 | `02A47EF021D15C011ED0E22E85E1BEFEA4BE9044571600EC95D77C6857CFDC75` |
| `src/alice_memory/transition_promotion.py` | `phase2_memory_core` | 1155 | 38582 | `7DEFF0C85E71061D093D359C261C6C9B403CD894A46313BB3CD79A3CBC8EC924` |
| `tests/phase2/test_memory_candidate_assessment.py` | `phase2_tests` | 445 | 14615 | `06074C6BE90A9C22AA705A4C4546CB53F53CDD8C2196BFD979498BC94A883443` |
| `tests/phase2/test_memory_candidate_promotion.py` | `phase2_tests` | 602 | 18976 | `5E8E4CB3816E64B1D3C7280839B8180A3B7C257A148368A9362469DE551B0384` |
| `tests/phase2/test_memory_candidate_security_gates.py` | `phase2_tests` | 797 | 25709 | `DF5B94A538E77B572B41B7DE65BCA5DCBCFCC7CD608BBF6D3F514B0CEF07CCC0` |
| `tests/phase2/test_memory_candidate_transition_promotion.py` | `phase2_tests` | 642 | 21799 | `FEC31BDBD61F4A465A3CDC4B12F3868F785DFCF56CCC2A3A9E5DAFE7023BAEA1` |
| `tests/phase2/test_memory_candidates.py` | `phase2_tests` | 363 | 11633 | `42D38DBDA2AC42A4B57A0CBDDF24D4E0D9DDED5538E2C2BB961EC1AC61CDED0B` |
| `tests/phase2/test_memory_citation_evaluation.py` | `phase2_tests` | 417 | 13936 | `AC76CFD9385AA0CE689652712FE89F3C21A3A1817648B0C048330873A3C2C59E` |
| `tests/phase2/test_memory_cited_answer.py` | `phase2_tests` | 371 | 12228 | `263ED01AE89485E82BFA5DEEACA8EC1847B7C9ABEAE51DF9B4FA3C1813B5FDDC` |
| `tests/phase2/test_memory_deletion.py` | `phase2_tests` | 678 | 23945 | `883E4D2C21C6C1DEBC253C945F640A85F1425127C34BC79B6D4C7F045199AD08` |
| `tests/phase2/test_memory_deletion_indexes.py` | `phase2_tests` | 635 | 21297 | `982A8C5EB1F35700492DC88FFB22CF0211BFED309549FD97DD415857719D8B39` |
| `tests/phase2/test_memory_deletion_integrity.py` | `phase2_tests` | 711 | 27157 | `42E221BF20D9349B2516D562F93ADD01D03B1BA91330677DCD135C066E257246` |
| `tests/phase2/test_memory_deletion_security_gates.py` | `phase2_tests` | 528 | 20089 | `A8DDCC8671D2709F2B2C897317AB5ED4F449D06D5384534EBE47A5C5F294D837` |
| `tests/phase2/test_memory_evaluation_contract.py` | `phase2_tests` | 305 | 9091 | `7BECCD0872386B60BB52C8714A4D90DF8CCFD83C9A187A1AFCA9E747A575ED71` |
| `tests/phase2/test_memory_evaluation_fixtures.py` | `phase2_tests` | 448 | 14020 | `2FB76579DE08485A1D3EECC39F44D7DB4E37C427724C6DB8DAFEEE165FC71B5B` |
| `tests/phase2/test_memory_final_evaluation.py` | `phase2_tests` | 284 | 11952 | `8500CF6F19D00FBF62FF0820EDF2D523EF23F59665498C3067226A980DB8BF8A` |
| `tests/phase2/test_memory_hybrid_retrieval.py` | `phase2_tests` | 376 | 9773 | `DBC90EEA16374EC46F7E6BE5668E7C95E2113E3BD5CE90C18D842E7BAD2DE823` |
| `tests/phase2/test_memory_inspection.py` | `phase2_tests` | 270 | 7430 | `F661E6C3A25C6E7F93E609AF42CA7A803433C283E81DD3194406F09D9DEF0193` |
| `tests/phase2/test_memory_lexical_index.py` | `phase2_tests` | 238 | 6403 | `963A1E4EC933056CA14936F436DC23BAB113BEC86E504A1632EA68A97E0A0EFA` |
| `tests/phase2/test_memory_migrations_v2.py` | `phase2_tests` | 115 | 3296 | `0FE9E7C3764747AE01EA2E2C77148CB3617E1C223E82F791B6FC4E1AE6690B15` |
| `tests/phase2/test_memory_migrations_v3.py` | `phase2_tests` | 140 | 3865 | `66E6DF0CC24AD7EB6A4844A05167EBA54976902D16FAF226F5DEEED6117134E3` |
| `tests/phase2/test_memory_provenance.py` | `phase2_tests` | 412 | 11854 | `06FA27D17D080EB8B13B293AC63C5A8C08B3174A710BCB5654962F2CF04710F4` |
| `tests/phase2/test_memory_provenance_invariants.py` | `phase2_tests` | 268 | 7497 | `2EDF0604F1CE898CA093BAAB7B381476B5AB0C949EE129B4D2844A1A85D173ED` |
| `tests/phase2/test_memory_release_audit.py` | `phase2_tests` | 257 | 10146 | `5BB496D4D49C9F6D57250C70E89A3574A11A1960E5EE2291B224E5AEF9A18C81` |
| `tests/phase2/test_memory_retrieval.py` | `phase2_tests` | 404 | 11042 | `F725D55AE94C61CD7FDA2EDA2AAD0F1CAE27BB48D20ECFEDF6F1B1C78480357E` |
| `tests/phase2/test_memory_retrieval_security.py` | `phase2_tests` | 239 | 6784 | `131ABD642EBDDF1C6062E74CFC0ACB4772D5AA4C1009C30C6B4E8769E5E681D5` |
| `tests/phase2/test_memory_schema.py` | `phase2_tests` | 237 | 6432 | `250C86C578334F1D3B50BCF8D04D7DB2FCD9309AA536A331CAAA274E3A4E5DE1` |
| `tests/phase2/test_memory_semantic_index.py` | `phase2_tests` | 197 | 5283 | `12AA915AF24EEFDE2DDF6A05E94D5F9E667F7B093A822582D85EC6306316262B` |
| `tests/phase2/test_memory_sensitive_access.py` | `phase2_tests` | 554 | 18854 | `16F25FB028DD636208F6E3610E00BCEFB5880B372ED5913B2FB7584CAD56E3E5` |
| `tests/phase2/test_memory_sensitive_crypto.py` | `phase2_tests` | 143 | 4074 | `9EFEEA045AC6C245368D9E019C52B3A959466EAD5F632C92E0CE5B50DD8EB420` |
| `tests/phase2/test_memory_sensitive_deletion.py` | `phase2_tests` | 625 | 23067 | `9A1A684358BD4FA8182A95C34A6B92934EFA24FFDD9FBA10507F2A8FEE4D7244` |
| `tests/phase2/test_memory_sensitive_security_gates.py` | `phase2_tests` | 284 | 9508 | `C97E6ACF3BF88330E7E185926F4AE160A690F5F0510BC9D8C181765FDDB79126` |
| `tests/phase2/test_memory_sensitive_storage.py` | `phase2_tests` | 312 | 10138 | `B0E221068E043B6E3CBFFFC88FFE5FF5F87B1B7594683833F268A6F19F331152` |
| `tests/phase2/test_memory_service.py` | `phase2_tests` | 373 | 10381 | `EE16E7B191DFC64802A982DA30B7A665CB9D420BB1792848EFC18EBF624F8125` |
| `tests/phase2/test_memory_store.py` | `phase2_tests` | 369 | 9431 | `06DDE8EC3073285D561D965862DFB73363DBA3AE72995B5A53F1CAAB1F1F4101` |
| `tests/phase2/test_memory_temporal.py` | `phase2_tests` | 341 | 9280 | `8091356C560EF97786A07071F9C7E2665BC9AAB97AA4BA6A500AADBCB1082A2D` |
| `tests/phase2/test_memory_time_validation.py` | `phase2_tests` | 117 | 3383 | `8DE0FBC20A8EE1A389AEACA1F5CD85B52BDB47E8EC963A2DD0C7497152026F84` |
| `tests/phase2/test_memory_transitions.py` | `phase2_tests` | 520 | 15766 | `3AAE60ECF7C277DB6119931631F51600D32ACFBCE1FCCBAC57F80FE67DF95B9A` |
| `tests/phase2/test_phase1_bridge.py` | `phase2_tests` | 326 | 9937 | `09B1A50D0D281657D97E4E66092E6C70B193A7FAFDFC800192C42E54BF92FA07` |
| `tests/phase5/attention_workspace_helpers.py` | `phase5_tests` | 140 | 4369 | `FE4EFFABBB937FB6DC3D7BB22AEBE32BDF1CA3509900BA321DD268BC6A98EEF0` |
| `tests/phase5/experience_ledger_helpers.py` | `phase5_tests` | 71 | 1991 | `4CF8D872E8A90E97522CFCC116CDF2BF65AB9877FA99B1ACE19277362172238D` |
| `tests/phase5/lifecycle_helpers.py` | `phase5_tests` | 120 | 3968 | `90828E6FB9D8E8BF8F47173F5CC71135B9696D2209516F130E564C0CCAF21F09` |
| `tests/phase5/mission_graph_helpers.py` | `phase5_tests` | 10 | 568 | `A22D54F054C62FC158B35516D6CEFE3C08B943BA912ED2E84A82B21587BA475B` |
| `tests/phase5/raw_buffer_helpers.py` | `phase5_tests` | 39 | 1170 | `B3C8A8A101511B0440E58710E4F9A9F18968043AC86F914B6CB35B93790E7E08` |
| `tests/phase5/test_attention_decision.py` | `phase5_tests` | 119 | 4064 | `5871BF4383959B302654F1C58D2A00A7C4FAD5782A556C3BE677AE2B3AE89440` |
| `tests/phase5/test_attention_workspace_isolation.py` | `phase5_tests` | 34 | 1303 | `547007453910C8994BFECE37AAD6E976224F591AEF5A9DBF242A87D18E6969F3` |
| `tests/phase5/test_attention_workspace_policy.py` | `phase5_tests` | 44 | 1631 | `348345217E78E53ABD08756F2219D06B433AC84502FEAE851E8FE623D313D413` |
| `tests/phase5/test_cognitive_kernel_experience.py` | `phase5_tests` | 105 | 3462 | `FD41FE046E5FE89F73FAF0449948B5D2B8D4DDB81E56CF185AD8993EC74E0B5B` |
| `tests/phase5/test_experience_ledger_append.py` | `phase5_tests` | 89 | 2743 | `3101EB26A900C1638DC81E9284EA134C673ED73EE877D6E702AFEF33C472F5F4` |
| `tests/phase5/test_experience_ledger_inspection.py` | `phase5_tests` | 37 | 1178 | `DDF5E2E4EFB6AEC34DF8C029268AC1B9B7F80E27B9E4BBADC0A100973497E983` |
| `tests/phase5/test_experience_ledger_integrity.py` | `phase5_tests` | 79 | 2361 | `DFCDED61A146D545EF91FDF23D5AA04C618F12D6AFA8DC3B32B7AE1E70F55DF5` |
| `tests/phase5/test_experience_ledger_isolation.py` | `phase5_tests` | 71 | 2018 | `5DAFD216E68372E9F565D5B3DC1ACACF0C90A1F0F1F203AC6D53166E59D5EF4C` |
| `tests/phase5/test_experience_ledger_path_safety.py` | `phase5_tests` | 31 | 899 | `4A567FC2AEFAA633F59466AA6110BA85CE123A30F178503306CB70D9CAE97BF1` |
| `tests/phase5/test_experience_ledger_policy.py` | `phase5_tests` | 46 | 1498 | `7C34C069586265E0A6B97CA882EE6A4E8A1482FE3C465608B2A1DB4BB4D21BA6` |
| `tests/phase5/test_experience_ledger_transactions.py` | `phase5_tests` | 83 | 2495 | `D664A520A366473F21AD3A04DDCDBECA1535E5A2FB7405C0EE7959372945CBBD` |
| `tests/phase5/test_host_workspace_override.py` | `phase5_tests` | 68 | 2335 | `C3EF4F4EBC34780FB9886B297D42343F155D935E94DAAC3F42916FCC585026FB` |
| `tests/phase5/test_lifecycle_blockers.py` | `phase5_tests` | 74 | 2249 | `5AB59CE6B273DB64BC8994A7CA860E3FC8FDF12CB34DC603624C7B5C2A67BFC5` |
| `tests/phase5/test_lifecycle_decisions.py` | `phase5_tests` | 51 | 1693 | `61614995A36FE7FF484871DC924C9AB913F9198681FB7F8CE0ABC3CA4FD4B903` |
| `tests/phase5/test_lifecycle_integrity.py` | `phase5_tests` | 118 | 3757 | `1334AC852B0E7B71619322EFC3940047F746E3C939BCB2B7FC02CE7DF1715807` |
| `tests/phase5/test_lifecycle_isolation.py` | `phase5_tests` | 48 | 1342 | `0B9A2897EDCB2643E9FE318DC4774141747CC696CC81FA5EF2026CB1A62EDDFF` |
| `tests/phase5/test_lifecycle_transitions.py` | `phase5_tests` | 90 | 3011 | `66524E8CF8357680BB85D240E54E1BACCFC5BEBB84FB3E3A2682FF142CD5116C` |
| `tests/phase5/test_mission_graph_contracts.py` | `phase5_tests` | 28 | 1901 | `7CAA8A34B4DAE37AB87286C798173512CF7D2B8CCDF10D27E291FF2CBDC4E081` |
| `tests/phase5/test_mission_graph_policy.py` | `phase5_tests` | 21 | 1169 | `E516E7C35B705E36C2CC231DD2035382DADDD7B5DE39940C06B10CCC82BF1A82` |
| `tests/phase5/test_mission_graph_snapshot.py` | `phase5_tests` | 30 | 2839 | `97FDCBA5B9E1D4425F89AE517A69A754F5EA7CD1E63ED794871C87CD9873E14F` |
| `tests/phase5/test_payload_store_accounting.py` | `phase5_tests` | 14 | 619 | `1146EAE42981483C6EA0F27F5A0506ABF96510AD60C7077892B384EC796259DE` |
| `tests/phase5/test_payload_store_deduplication.py` | `phase5_tests` | 31 | 1531 | `D1C0EAAE2F7F15E7EB337820050167923BB68C817DBA5525576B829B3CC4E1A9` |
| `tests/phase5/test_payload_store_integrity.py` | `phase5_tests` | 37 | 1514 | `B8F24B067E20F4186D6BFF4740B87DEC3176F88D0A66F180807969AB16274486` |
| `tests/phase5/test_payload_store_isolation.py` | `phase5_tests` | 41 | 1511 | `BB1D493EC4FDBFB7DC1E41CC53E10885A95BAFE4B62DA2CDF2CACE09DEF3B95A` |
| `tests/phase5/test_payload_store_path_safety.py` | `phase5_tests` | 24 | 754 | `B0057AF4F420BB080A39AE13774DB7EA42F615BC4F7675A556F96E2D83D794C0` |
| `tests/phase5/test_raw_buffer_capture.py` | `phase5_tests` | 31 | 1230 | `9F1A0EF93126C5DC51FC996B208A091600B990C18368327C90F7EDFE634184C3` |
| `tests/phase5/test_raw_buffer_policy.py` | `phase5_tests` | 26 | 929 | `6CF052051FCD4DAF2F86AA32E928388ED442BF6D401B63F165FC739D8341CBFE` |
| `tests/phase5/test_raw_buffer_reopen.py` | `phase5_tests` | 24 | 929 | `DA662A76D24A18D9C3CC3236D12505696325F7BB7B745F3BD6141B9DDFDAF9A8` |
| `tests/phase5/test_tier_transition_blockers.py` | `phase5_tests` | 174 | 6873 | `D0CB1ADE67E3191B905A8FA16ABE31DFA7430D729239BF360F1377CF589C6D57` |
| `tests/phase5/test_tier_transition_execution.py` | `phase5_tests` | 145 | 6231 | `55999F5CD5553F688FE03DE146734F83EF4321CE41ECBB5E0FCABB1CC8243C09` |
| `tests/phase5/test_tier_transition_integrity.py` | `phase5_tests` | 98 | 3464 | `139E3A7776B59A87757D82A434A3BD99CBDC9BAA4851C56618F60692E82B4D10` |
| `tests/phase5/test_tier_transition_isolation.py` | `phase5_tests` | 70 | 2622 | `89228EFED70ABD817D850BB2E499CFD86D97CB9E51AD145EE9A679BEFAA51A9B` |
| `tests/phase5/test_tier_transition_path_safety.py` | `phase5_tests` | 21 | 623 | `BFB20563E559DE3F16B74B745CBF8B88F19798005A5F3F75F621C441EDB1B4B7` |
| `tests/phase5/test_tier_transition_policy.py` | `phase5_tests` | 18 | 702 | `A4EA08349414AB8290387C6AB5DED703CD659BDA215602013A4F25B1D0AEE917` |
| `tests/phase5/test_tier_transition_recovery.py` | `phase5_tests` | 124 | 4696 | `CD1A9AD439D58063708FE94C4B066DBEBEC962DFA7DE53E20E532B1553EF0CAE` |
| `tests/phase5/test_tier_transition_reopen.py` | `phase5_tests` | 47 | 1537 | `65EE2330C6E14E4EDDEABB88FAE31CF2D5116F2E1DA316F800E64B0C076784AD` |
| `tests/phase5/test_workspace_layout.py` | `phase5_tests` | 67 | 2173 | `A5A3973D244E2B49EFB069B26BF596F139414FBD35C1D5D8429D12B8D968FEB2` |
| `tests/phase5/test_workspace_projection.py` | `phase5_tests` | 113 | 3842 | `F481BABD91A8D37865D4A6D2EDEF56CE05A370E75C3EED89632D67AA1FB39E56` |
| `tests/phase5/tier_transition_helpers.py` | `phase5_tests` | 179 | 5197 | `40247D25142281875BF4EADB14A238B2D7D719A53D4A30F3DD384C37F41CF643` |

## Static architecture observations

| Severity | Public path | Observation |
|---|---|---|
| `major` | `src/alice_memory/retrieval.py` | Ordinary lexical retrieval verifies the complete derived index before use. |
| `major` | `src/alice_memory/retrieval.py` | Each lexical query builds a global set of corrected targets. |
| `major` | `src/alice_memory/retrieval.py` | Candidate hydration is performed through individual record loads. |
| `moderate` | `src/alice_memory/retrieval.py` | The lexical candidate pool may expand to 800 candidates. |
| `major` | `src/alice_memory/hybrid_retrieval.py` | Semantic and hybrid retrieval repeat per-candidate authoritative hydration. |
| `moderate` | `src/alice_memory/store.py` | Phase 2 writes serialize through an immediate SQLite write transaction. |
| `moderate` | `src/alice_memory/store.py` | Phase 2 uses WAL with synchronous=NORMAL and requires explicit durability review. |
| `moderate` | `src/cognitive_kernel/ledger_store.py` | Experience Ledger writes serialize through an immediate SQLite transaction. |
| `moderate` | `src/cognitive_kernel/ledger_store.py` | Full ledger integrity verification scans the complete ledger. |
| `major` | `src/alice_memory/semantic_index.py` | Semantic-index verification hashes complete index artifacts. |

## Interpretation boundary

This inventory describes the code and documentation surface. It does not
measure private-memory correctness, owner-model fidelity, or future Claim
Store behavior.
