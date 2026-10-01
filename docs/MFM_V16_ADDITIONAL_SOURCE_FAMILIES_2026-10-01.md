# MFM v1.6 additional public source families: acquisition note

**State:** source research only. These are three independent upstream corpora,
not admitted MFM cases, reviewed targets, source-rights receipts, independent
splits or model results. No payload was downloaded for this note. The stated
licenses identify upstream releases; a steward still has to authenticate the
exact acquired bytes, applicable terms, attribution, consent and permitted
training/distribution scope before any use in the owner-teacher or signed-fit
lane.

| Source family and exact acquisition pin | Original evidence for v1.6 roles | Upstream rights and boundary |
| --- | --- | --- |
| **AMI Meeting Corpus**, [official corpus](https://groups.inf.ed.ac.uk/ami/corpus/) and [download page](https://groups.inf.ed.ac.uk/ami/download/). Select **manual annotations v1.6.2, 10 April 2017** and named meeting signals by ID; the page says those annotations were unchanged since 16 June 2014 and relicensed in 2017. The signal chooser provides mixed/individual WAV, AVI, slides, shared documents and whiteboard files. A scenario session has `a`–`d` meetings, e.g. `ES2008a` through `ES2008d`, under [the published ID convention](https://groups.inf.ed.ac.uk/ami/corpus/meetingids.shtml). | The same four people act as project manager, marketing expert, interface designer and industrial designer through kickoff, functional, conceptual and detailed design. Original timed speech, audio/video and contemporaneous artifacts can show who proposed a requirement, what another person corrected, how a decision or task changed in later meetings, and what a slide or whiteboard actually displayed. [Scenario description](https://groups.inf.ed.ac.uk/ami/corpus/scenariomeetings.shtml). This is one-day simulated work; familiarity or a lasting real-world relationship must not be invented from the roles. | The official corpus and download pages state **CC BY 4.0** for all signals and transcription and *some* annotations. Start with the named manual release and original signals; inspect each included annotation's terms. Avoid automatically derived or contributed labels as original observations, and do not inherit older mirror terms (for example, [an older OpenSLR AMI entry](https://openslr.org/16/) advertises a noncommercial license). |
| **ICSI Meeting Corpus**, [official corpus](https://groups.inf.ed.ac.uk/ami/icsi/) and [download page](https://groups.inf.ed.ac.uk/ami/icsi/download/). Select **core annotations v1.0, 22 July 2016** (orthographic transcript and dialogue acts) and matching original mixed WAV or individual SPH audio by meeting ID. Keep the separate contributed-annotation package out until independently reviewed. | The original collection has 75 naturally occurring meetings from 2000–2002, generally regular weekly meetings of Berkeley research teams ([LDC's original corpus description](https://catalog.ldc.upenn.edu/LDC2004T04)). Chronological meetings and recurring speakers can ground ongoing team tasks, corrections to earlier statements, status changes and interpersonal interactions. Audio and manual words are source evidence; topic summaries and future meeting contents must not be substituted into an earlier window. | The [official ICSI license page](https://groups.inf.ed.ac.uk/ami/icsi/license.shtml) explicitly states **CC BY 4.0** for the corpus and its annotations; the download page scopes its public release to all signals/transcription and some annotations. Pin this Edinburgh distribution and each selected file, since the separately distributed LDC release has its own access terms. |
| **Wikimedia revision history**, [MediaWiki Content File Exports](https://wikitech.wikimedia.org/wiki/MediaWiki_Content_File_Exports). A bounded English pilot is the **Simple English Wikipedia 2026-09-01** [history export directory](https://dumps.wikimedia.org/other/mediawiki_content_history/simplewiki/2026-09-01/xml/bzip2/): seven `simplewiki-2026-09-01-p*.xml.bz2` shards, `_SUCCESS` and `SHA256SUMS`. The new export is monthly, per wiki, and contains raw content of every past and present revision. | Retain Talk/User talk discussions alongside article revisions, edit metadata and timestamps; [Wikimedia's dump guide](https://meta.wikimedia.org/wiki/Data_dumps/Dumps_sizes_and_growth) distinguishes all-namespace history from article-only/current files. A signed comment, an edit summary, the subsequent diff and a later reply can jointly ground who corrected which statement, whether another editor acknowledged it, and how a shared writing task evolved. Repeated editor IDs can connect episodes, but the dump does not supply personal relationship labels or MFM answers. Thread and person resolution would require source-bound review. | The [Wikimedia Terms of Use](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use) license contributed text under **CC BY-SA 4.0** (with a GFDL alternative and exceptions) and explicitly permit compliant commercial reuse. Attribution and share-alike obligations apply to distributed adaptations. Non-text media have item-specific licenses, and imported text may carry additional attribution; restrict this pilot to examined text and preserve source page/history links. |

AMI and ICSI are distinct original collections, despite being distributed from
the same Edinburgh site. AMI supplies controlled, cross-meeting role and
multimodal evidence; ICSI supplies real, recurring research-team meetings;
Wikimedia supplies asynchronous discussions tied to actual document revisions.
None contains a verified memory disposition, owner consent or an MFM target.
An annotator would have to establish speaker/subject, event versus record time,
scoped relationship and workspace links, epistemic status, sensitivity,
contradiction, correction and abstention from the selected source window.

For acquisition, freeze the upstream release/dump URL and terms page, enumerate
each selected meeting ID or XML shard, compute local SHA-256 for each original
file, verify Wikimedia's published `SHA256SUMS`, and keep byte/timestamp/speaker
or revision-ID pointers in separate custody. Extract only an as-of slice for a
teacher; a complete meeting series or full-history shard would leak later
evidence. Hold together all meetings of one AMI team, all connected ICSI
speakers/teams, and all connected wiki pages/editors/threads when proposing
partitions. Cross-corpus duplication and benchmark overlap need their own
checks. These pins are release/date-level acquisition instructions, **not**
file-hash receipts or a claim that any partition is independent.

The case/rights requirements and independent authority boundary remain those
in [the v1.6 authoring and admission route](MFM_V16_CORPUS_AUTHORING_AND_ADMISSION_2026-09-30.md)
and [owner-teacher lane](MFM_V16_OWNER_TEACHER_TRAINING_LANE_2026-10-01.md).
