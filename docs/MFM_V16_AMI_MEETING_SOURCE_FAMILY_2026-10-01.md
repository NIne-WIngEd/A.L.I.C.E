# AMI meeting histories: exact source-only intake

**State:** pinned public source candidate; 24 meetings from six four-meeting
scenarios and 72 bounded transcript windows. These are not formation labels,
reviewer judgments, independently split development, sealed FINAL, or a
trained MFM. All six sessions have **one AMI annotation/source family**. Do
not count different AMI sessions as independent generator families.

## Original release and rights evidence

| Item | Pin |
| --- | --- |
| Official annotation release | [`AMI manual annotations v1.6.2`](https://groups.inf.ed.ac.uk/ami/download/), 10 April 2017 |
| Original ZIP | [`ami_public_manual_1.6.2.zip`](https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip), 22,887,865 bytes, SHA-256 `b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d` |
| ZIP's `LICENCE.txt` | SHA-256 `d52fd9c7c19eec9f0bec3107554c10937203e261259d73b01637a61501fec7f7`; its preamble identifies AMI corpus and annotations as CC BY 4.0 |
| ZIP's `corpusResources/meetings.xml` | SHA-256 `8ab6cdcf03ed863e839418f31c7380392397a2eed892d73b9bc1b438c7ec520a` |

The official [download page](https://groups.inf.ed.ac.uk/ami/download/)
specifically says this release changed its license to CC BY 4.0. This is a
distinct release from OpenSLR's older `ami_manual_1.6.1.tar.gz` mirror, whose
page describes different terms. Do not mix the archives or infer the license
of other AMI material from these selected manual-word members. Retain AMI
project credit, the release link, the CC BY 4.0 link, and an indication of
adaptations when distributing a derivative. The copyright license alone is
not an independent privacy, publicity, or participant-consent review.

## Source shape and bounded export

`scripts/mfm/inventory_ami_meeting_sources.py` checks the whole ZIP hash and
the embedded license and metadata members. It reads the original `words/*.xml`
members for the complete four-meeting sessions `ES2003`, `ES2005`, `IS1000`,
`IS1001`, `TS3003`, and `TS3004`. No abstract, extractive summary, decision
annotation, or dialogue-act label enters the export. Each source inventory
member has its exact size and SHA-256, meeting stage, speaker ID, role, and
original AMI participant identifier. The release contains role-playing design
meetings and some real speakers; these are meeting sources, not memories of
Fable's owner.

For each meeting the exporter chooses three retrospective word-quantile
centers and emits up to 120 seconds of timed words around each. A short or
partial annotation may end before the metadata duration. Thus the selection
uses observed word timing, not a fictional dialogue-length assumption.
`as_of.window_end_seconds` is the upper bound: no word ending after that bound
is included. The `a`/`b`/`c`/`d` order is a scenario stage order, while the
metadata's original date and start-time strings are kept verbatim. No missing
wall-clock value is invented.

Every word has an original XML member SHA-256, original member byte interval,
raw XML span SHA-256, NITE word ID, speaker/participant/role, and timed text.
The additional `derived_turns` and `derived_display` group consecutive words
within a speaker before sorting turns by start time. They prevent a naïve
time-sort from merging simultaneous speakers into one utterance. The displayed
text and turn boundaries are *derived*, can clip utterances at the window, and
are separately hashed. The raw word list and original archive remain the
authority; overlapping speakers and transcription uncertainty must be
reviewed before labels are made.

An actual source-only acquisition produced 72 windows (26,556 selected word
occurrences), with these outputs outside Git:

| Output | SHA-256 |
| --- | --- |
| `ami-v16-source-inventory.json` | `c459b15898fa9f1fbaf51154ef15fbbf05e8a1ec736be4cc4aa0eda2d2c3eb3d` |
| `ami-v16-word-windows.jsonl` with attributed turn view | `f414869926088e241f694db00f20862609e6992933e8713fda4a1733f659311d` |

Reproduce in owner-controlled external source custody:

```bash
umask 077
CUSTODY="$HOME/rayan-compute/mfm/ami-manual-v162"
mkdir -p "$CUSTODY"
curl -fL --retry 3 -o "$CUSTODY/ami_public_manual_1.6.2.zip" \
  https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip
python scripts/mfm/inventory_ami_meeting_sources.py \
  --archive "$CUSTODY/ami_public_manual_1.6.2.zip" \
  --output-dir "$CUSTODY/turns-v1"
```

The importer refuses any changed source ZIP, embedded license, or meeting
metadata. A downstream reviewer still needs to freeze source rights and target
provenance, inspect ambiguous speaker turns, adjudicate MFM's formation roles,
and establish independent splits before these windows can enter a teacher-fit
manifest. This source export alone does not unblock the full-corpus CPU pass.
