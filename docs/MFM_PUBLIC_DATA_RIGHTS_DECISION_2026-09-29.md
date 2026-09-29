# Public memory datasets: MFM admission decision

**Date:** 2026-09-29. **Status:** owner-authorized synthetic training corpus
assembled and validated; no GPU training or independent qualification yet.
The destination is a distributable, host-neutral Fable builder and MFM. Public
availability of a benchmark or code repository does not grant permission to
put its examples in shipped model weights.

| Primary source | Observed use and terms | MFM decision |
| --- | --- | --- |
| [LoCoMo official repository](https://github.com/snap-research/locomo) and [license](https://github.com/snap-research/locomo/blob/main/LICENSE.txt) | Ten long conversations, annotated QA/event summaries, generated observations. Repository license is **CC BY-NC 4.0**. The released images are URLs/captions, not image bytes. | **Exclude from distributable MFM training** under current rights. It may inform research methodology or a separately permitted noncommercial evaluation. Do not import its gold into shared weights or mistake generated observations for human historical truth. |
| [LongMemEval-V2 official repository](https://github.com/xiaowu0162/LongMemEval-V2) and [dataset card](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2) | Apache-2.0 dataset card; 451 curated questions with long web/enterprise trajectories, including workflow knowledge and environment failures. It is an evaluation benchmark, not a general formation target corpus. | Reserve for an independently frozen evaluation route. Do not use benchmark questions/answers for MFM training or iterative model selection that claims clean evaluation. Verify original trajectory and screenshot lineage before a qualification run. |
| [Community Alignment official dataset](https://github.com/facebookresearch/community-alignment-dataset) | CC BY 4.0; multilingual response-preference comparisons with repeated annotators. The maintainers flag possible personal information and removal requests. | Candidate **supplementary preference/feedback** source after rights, PII, consent, subject, attribution and removal review. It does not supply direct formation proposal gold or longitudinal personal history by itself. |
| [A-MEM official research code](https://github.com/WujiangXu/A-mem) | MIT code for dynamic memory organization; examples run against LoCoMo. | Architectural comparison for linking/revision. The code license does **not** change the LoCoMo data license or certify MFM formation behavior. |
| [CareCall official repository](https://github.com/naver-ai/carecall-memory) | Dataset terms permit only noncommercial AI R&D and restrict modification and redistribution. | Exclude from a shipped Fable MFM training pack under current rights. |
| [Multi-Source Memory Benchmark code](https://github.com/TianchengY/multisource-membench) and [data license](https://github.com/TianchengY/multisource-membench/blob/main/data/DATA_LICENSE) | Apache-2.0 deterministic generator; CC BY 4.0 synthetic personas, source projections and renders with attribution. Cached third-party API outputs are carved out. | Fresh-seeded structural sources and MFM-specific, source-grounded targets supply 43,819 synthetic **training** cases with recorded commit/seed/license hashes. Exclude cached model outputs, upstream QA and published benchmark rows. Independent qualification and disjoint held-out generator families remain missing. |

The fresh Multi-Source generation and receipt are described in
[`MFM_CORPUS_SOURCE_RECEIPT_2026-09-29.md`](MFM_CORPUS_SOURCE_RECEIPT_2026-09-29.md).
The ChatGPT/Codex-authored diagnostic generator carries the owner's prior
authorization for A.L.I.C.E., Fable and FBM teaching use. Its provenance is
recorded in the [source receipt](MFM_CORPUS_SOURCE_RECEIPT_2026-09-29.md).
The earlier ChatGPT-specific rights hold has been withdrawn. The two
owner-authorized components form **49,819 train-only cases**, bound by the
frozen mixture manifest described in the source receipt. Independent target
review, nonleaking development/FINAL splits and further generator families
are needed for a generalization claim, not for starting learned training.
A licensed retrieval/QA benchmark cannot substitute for MFM's source-role,
temporal, subject, correction and uncertainty labels. Preserve benchmark and
generator-family isolation during qualification.
