# Public memory datasets: MFM admission decision

**Date:** 2026-09-29. **Status:** source review, not a data download or training run.
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

No dataset was downloaded or admitted to training in this review. A real
training corpus needs permissioned, independently sourced histories with
formation targets and traceable rights. A licensed retrieval/QA benchmark
cannot substitute for MFM's source-role, temporal, subject, correction and
uncertainty labels. Retain benchmark and generator-family isolation when that
corpus is built.
