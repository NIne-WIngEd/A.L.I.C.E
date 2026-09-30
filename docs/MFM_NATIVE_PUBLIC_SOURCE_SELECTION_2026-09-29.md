# Native MFM public foundation: source candidates

**Status:** candidate selection, not admitted training data. No foundation pack,
multimodal pack, rights audit, tokenizer, native checkpoint, optimizer step or
capability result is claimed here. This selection implements the two-stage
lineage in [MFM native lineage decision](MFM_NATIVE_LINEAGE_DECISION_2026-09-29.md).

## Data belongs to different stages

| Use | Input and learning role | Boundary |
| --- | --- | --- |
| Stage A: first-party foundation | Broad, licensed public text, code, structured records and original image/audio/video; language, sensory, temporal, evidence and source representations learned from newly initialized weights. | Public content builds general competence. It never asserts that a Fable owner lived a public event. No third-party pretrained backbone, encoder, embedding or covert runtime dependence. |
| Stage B: formation specialist | Exact first-party Stage A checkpoint plus source-bound cases for subject, speaker, event/valid time, evidence span, correction, uncertainty, abstention, pattern and scoped disposition. New fusion/output weights may be initialized here. | The frozen 49,819 train-only cases in [the corpus receipt](MFM_CORPUS_SOURCE_RECEIPT_2026-09-29.md) are **one supervised tier**, not enough to pretrain Stage A or qualify Stage B. |
| Personal construction by FBM | Authorized owner data and governed instance adaptation after shared, identity-neutral competence. | Rayan and Elaina evidence stays in its private instance. A selected public pack does not become personal history or a canonical Claim. |

An owner may select or add rights-cleared public packs for a versioned rebuild.
The recommended default list below is a **candidate menu**. It cannot be
advertised as an adequate full-scale mixture until real byte counts, diversity,
languages, source systems, modality coverage, training curves and unseen-case
performance establish adequacy. If a selected mixture cannot support the
required capability, FBM must report the concrete data deficit and stop the
affected build until adequate permitted sources are supplied. It must never
label an undertrained result full capability.

## Recommended default candidate families

| Family and primary source | Intended signal | Admission work |
| --- | --- | --- |
| [Common Pile filtered collections](https://huggingface.co/common-pile), initially the 21 N0 exact-revision sources in `configs/mfm/native_foundation_source_candidates_v1.json` | Broad text, discourse, code, technical writing and long documents. Preserve genre tags. Stack V2 Edu belongs to **code**, not educational explanation. A YouTube transcript is text, not raw audio/video. | Recheck each exact dataset revision and per-row license, source URL/provenance, ID, removals and rights for MFM distribution. The N0 activation at `alice-n0-review/configs/eipm/n0/public_corpus_v0.2.1.activated.json` is evidence of an earlier audit, not automatic MFM admission. Do not widen the allowlist to fill a quota: N0 removed OER Commons when its admitted rows underfilled. The [Common Pile authors](https://huggingface.co/blog/stellaathena/common-pile) warn that license metadata can be wrong. |
| [Wikidata entity dumps](https://www.wikidata.org/wiki/Wikidata:Database_download) | Structured entities, claims, qualifiers and dates. [Wikidata's structured data is CC0](https://www.wikidata.org/wiki/Wikidata:Licensing). | Pin a dump date/files and checksums; inspect namespace/content and privacy. A graph record is world data, never a person's authorized statement. Generate as-of and contradiction tasks only with separately attributed labels. |
| [Wikimedia full revision dumps](https://meta.wikimedia.org/wiki/Data_dumps/Dump_format), restricted to vetted article namespaces | Time, revisions, correction and supersession from real document histories. Full history differs from a current-only snapshot. | Per-wiki/page license and attribution; revision/page lineage; suppressions, removals, copied material and living-person PII. Exclude User and Talk namespaces by default. Do not equate an edit with ground-truth correction or infer a real person's private history. [Wikimedia reuse rules](https://foundation.wikimedia.org/wiki/Terms_of_Use) vary with content. |
| [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html) | General visual grounding on original pixels with metadata and separate labels. | Individual image `License`, `Author`, `OriginalLandingURL` and bytes; filter to compatible per-image grants such as CC0/CC BY after checking originals, faces/PII, removals and duplicates. The annotation license does not license every image. |
| [LibriSpeech](https://www.openslr.org/12/) and [MUSAN](https://www.openslr.org/17/) | Speech/text alignment and speech/noise/music distinction. Both pages list CC BY 4.0. | Pin archive versions and checksums; preserve attribution, speakers, transcripts, recording splits and overlapping source ancestry. Speech corpora do not by themselves teach personal consent or memory authority. |
| [NSynth](https://magenta.tensorflow.org/datasets/nsynth) and [Expanded Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/e-gmd) | Non-speech sound structure, timbre and temporal audio events. The project pages list CC BY 4.0. | Audit exact recording and annotation rights, versions, artists, splits and duplicates. Do not treat MIDI or timbre labels as conversational audio. |
| [Wikimedia Commons media](https://commons.wikimedia.org/wiki/Commons:Licensing), [USGS multimedia](https://www.usgs.gov/products/multimedia-gallery) and [NASA SVS media](https://svs.gsfc.nasa.gov/help/) | Original image, sound and video with timestamped frames and possible accompanying descriptions. Public-sector material broadens video beyond one camera/domain. | File-level rights and provenance. Prefer verified CC0/CC BY or public-domain works; review CC BY-SA obligations separately. Commons [license metadata is often incomplete](https://commons.wikimedia.org/wiki/Commons:Machine-readable_data); USGS/NASA exceptions, third-party footage, soundtrack/music, endorsements and people must be checked file by file. These families alone may not cover full real-world video diversity. |

The Common Pile is a **per-source/per-document licensed** collection, not one
blanket MIT-licensed dataset. Its public source code license does not replace
the document licenses. The 21 N0 revisions and observed allowlist strings are
copied into the machine-readable candidate inventory solely to make a fresh
MFM audit concrete. N0's initial 120M-character tokenizer materialization is
not a broad foundation-pretraining volume or an adequacy receipt. Scale actual
unique admitted tokens and media exposure with capability evidence, not a
fixed small bootstrap target.

## Conditional owner-selected candidates

- [Mozilla Common Voice](https://mozilladatacollective.com/datasets/cmu5wui2w00e8nq07idithpm6): CC0 speech clips, but the Mozilla dataset terms prohibit rehosting/redistribution and speaker identification. Check exact version and direct-download requirements before using in a build; do not bundle raw clips in Fable.
- [FSD50K](https://zenodo.org/records/4060432): environmental sounds carry per-clip terms, including noncommercial/Sampling+ entries. Authors request contact for commercial use. Exclude until every admitted recording has an applicable grant and distribution route.
- [V3C1](https://catalog.data.gov/dataset/vimeo-creative-commons-collection-v3c1): research access requires the NIST agreement and individual Vimeo video licenses differ. A catalog-level open-data marker does not grant commercial training rights to every clip. Keep out of a default shipped pack pending access, license, withdrawal and distribution review.
- New owner-selected Common Pile, Wikimedia or other public families require the same admission process. Open web wrappers and code-repository licenses alone do not establish underlying work or personal-data rights. No public benchmark answer file should be used to train away the benchmark.

These are candidate sources, not a claim that the listed set already supplies
full sensory competence. Broader multilingual and varied video/audio sources
will be required if held-out performance exposes gaps; each new family must
pass the same rights and provenance gate.

## Admission and evaluation rule

1. Freeze each upstream URL, dataset commit or dump edition, files and
   checksums. Retain the original attribution/license text, download terms,
   row/file IDs, author, source URL, ancestry and removal state. The JSON
   inventory marks every family `candidate_not_admitted`; it must not drive a
   trainer until a separate validated `admitted` manifest names actual files.
2. Apply source-specific and **row/file-level** commercial training and model
   distribution review. Fail closed on unknown license, provenance, PII,
   conflicting terms or withdrawal. Keep an attribution and deletion ledger.
   Audit personal faces, names and voices in public media. Never train shared
   weights on private Rayan/Elaina histories or inferred consumer consent.
3. Preserve native item boundaries and temporal order. Verify raw media bytes
   and timestamp/frame/audio spans. Avoid treating captions, transcripts or
   simulator hidden state as the media itself. Deduplicate within and across
   source families; keep duplicate and derived-source links.
4. Freeze benchmark and independent FINAL exclusions before materialization.
   [LongMemEval-V2](https://github.com/xiaowu0162/LongMemEval-V2) is reserved
   for evaluation. [LoCoMo](https://github.com/snap-research/locomo) and
   [CareCall](https://github.com/naver-ai/carecall-memory) remain outside
   distributable training under their noncommercial terms. Keep all related
   questions, trajectories, paraphrases and source ancestry out of Stage A/B
   train and model selection where independent evaluation is claimed.
5. Teacher-authored data may provide explained examples, hard negatives,
   source comparisons and formation targets. Log the exact teacher, version
   when exposed, prompt/input-source digests, output digest, date, author,
   authorization and split. Teacher targets are training candidates, not
   independent gold or historical owner testimony. Keep teacher generation
   disjoint from independently reviewed FINAL.
6. Bind a Stage A receipt to fresh initialization, tokenizer/processor origin,
   exact data manifest and gradient data paths. Stage B may load **only** a
   verified first-party Stage A checkpoint. Measure full backward and export
   on the actual proposed compute topology before buying a long fit. Report
   modality, language, evidence, subject and temporal errors separately.

The public default and each owner's selected pack must remain versioned and
auditable. Training on a source establishes representation learning, not
permission to make a canonical memory from it. The deterministic memory gate
continues to decide writes after MFM emits source-citing proposals.
