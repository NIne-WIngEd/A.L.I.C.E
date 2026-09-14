# N0 v0.2 Public Source Selection Review — 2026-09-14

Status: source mixture refined; exact source activation still pending runtime probe.

## Why the initial 19-source plan was changed

The first v0.2 sketch placed `common-pile/stackv2_edu_filtered` under educational Q&A/explanation because of its name. Dataset-card review shows that Stack V2 Edu is still a **code** corpus drawn from openly licensed repositories. It therefore belongs in the software/code slice, not the instructional-explanation slice.

For actual educational/explanatory language, the refined plan adds:

- `common-pile/libretexts_filtered` — open-access textbook sections;
- `common-pile/pressbooks_filtered` — open-access books/textbooks;
- `common-pile/oercommons_filtered` — instructional material such as lessons, syllabi, problems, and worksheets.

`common-pile/stackexchange_filtered` remains useful as question/answer and explanatory discourse.

This correction matters because N0 is intended to learn semantic, pragmatic, epistemic, social, causal and structured reasoning representations. A code-heavy source mislabeled as education would make the mixture look more explanatory than it actually is.

## Refined mixture

The active *plan* now assigns:

- 14% reference / encyclopedic;
- 16% science / research;
- 22% educational Q&A / explanation;
- 15% conversational / spoken pragmatics;
- 12% books / long-form prose;
- 6% software / code discussion;
- 6% government / legal deliberation;
- 9% news / expository prose.

There are 22 candidate sources. No single source exceeds 10%, below the 12.5% source-share ceiling.

## Rights/provenance observations from dataset-card review

The Common Pile filtered collection exposes per-document license information in `metadata.license` for the reviewed variable-license corpora. Examples include:

- Wikimedia: CC BY-SA;
- ArXiv abstracts: CC0 metadata;
- ArXiv papers: filtered to CC BY, CC BY-SA and CC0;
- PubMed: filtered to CC BY, CC BY-SA and CC0;
- YouTube: curated CC BY channels;
- LibreTexts: public domain, CC BY, CC BY-SA or GNU FDL;
- OERCommons and PressBooks: public domain, CC BY or CC BY-SA;
- DOAB: CC BY and CC BY-SA;
- Project Gutenberg / pre-1929 books / Ubuntu IRC / US federal sources: public-domain-oriented sources;
- UK Hansard: Open Parliament Licence;
- GitHub Archive and Stack V2 Edu: repositories filtered through approved open-source license policies.

Dataset cards also explicitly warn that license metadata can contain mistakes. Therefore **dataset-card claims are not sufficient to activate a source**. Alice keeps the stricter rule: exact revision + row-level license + row-level provenance must be observed and frozen before materialization.

## Runtime activation procedure

`probe_n0_v02_sources.py` resolves each planned Hugging Face dataset to an exact immutable revision and samples its row schema. The probe records:

- exact revision SHA;
- top-level and metadata fields;
- presence of text / id / license / provenance;
- observed exact license strings;
- source target share;
- any retrieval/schema error.

The probe deliberately leaves `candidate_allowed_license_values` empty and marks manual license review required. It cannot authorize tokenizer or model training.

After the probe passes, the next build step is to review exact observed license strings conservatively and write a frozen activated manifest. Only that manifest may feed the v0.2 tokenizer/corpus builder.

## Storage constraint

Magnolia home storage is not treated as a place to keep the entire low-billions-token corpus resident. The v0.2 plan now requires deterministic source-balanced tranches that can be rebuilt from exact revisions. Active tranches may be rotated after their checkpoint and lineage receipt are durable.

This allows broad unique-token exposure without turning storage into the bottleneck or silently repeating a tiny local corpus.
