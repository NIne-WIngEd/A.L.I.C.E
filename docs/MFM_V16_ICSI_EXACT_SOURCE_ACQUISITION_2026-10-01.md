# ICSI v1.6 source acquisition

**State:** original source material acquired and byte-anchored; source-only and
unadmitted. It supplies no memory-formation targets, teacher review, split,
owner consent, or authenticated `mfm-source-rights-v1` receipt.

The [Edinburgh ICSI download page](https://groups.inf.ed.ac.uk/ami/icsi/download/)
publishes **ICSI core annotations v1.0, 22 July 2016** (orthographic words and
dialogue acts) separately from contributed annotations. The downloaded file is
[`ICSI_core_NXT.zip`](https://groups.inf.ed.ac.uk/ami/ICSICorpusAnnotations/ICSI_core_NXT.zip),
19,488,722 bytes, SHA-256
`cf4860245b9ca9c9ed11a66e5a74cfea2a99a686a52f7580aa0a6bc2225895a9`.
Its packaged `LICENCE.txt` and the [official ICSI licence page](https://groups.inf.ed.ac.uk/ami/icsi/license.shtml)
state CC BY 4.0. Attribution, notice of modifications and a link to the
licence must follow reused material. The licence does not resolve every
personality/privacy right or substitute for the MFM steward's source-use and
distribution attestation. The separately distributed LDC release is outside
this acquisition.

`scripts/mfm/acquire_v16_icsi_source_windows.py` verifies the original ZIP
digest and licence, inventories its 75 declared meeting IDs, and extracts
only timed orthographic `Words/*.words.xml` entries. Each observed word carries
the original ZIP member SHA-256, original member byte start/end, exact element
SHA-256, NXT word ID, speaker channel and participant ID where a matching core
`DialogueActs` channel establishes one unambiguous ID. It does not infer a
participant for the 29 channels without dialogue acts. It records excluded
untimed words and the single unnamed, untimed `Buw001..words.xml` file in the
receipt. No contributed annotations or audio have been acquired here.

Each 600-second JSONL row is a delta of words ending by its `as_of_end_seconds`
**within one meeting**; `previous_window_id` links earlier deltas for that
meeting. The extraction establishes order within meetings only. Numeric meeting
IDs must not be treated as dates, and a teacher must not open later deltas.
As-observed text and speaker IDs are evidence; memory disposition, relationship
meaning, consent and corrections are not supplied by ICSI. Connected speakers
and meeting series should stay together when constructing splits.

Reproduce in private custody outside Git:

```bash
curl -fL --retry 3 -o "$PRIVATE/ICSI_core_NXT.zip" \
  https://groups.inf.ed.ac.uk/ami/ICSICorpusAnnotations/ICSI_core_NXT.zip
sha256sum "$PRIVATE/ICSI_core_NXT.zip"
python scripts/mfm/acquire_v16_icsi_source_windows.py \
  --zip "$PRIVATE/ICSI_core_NXT.zip" \
  --output-dir "$PRIVATE/icsi-windows-v1"
```

The acquired run produced 458 as-of windows and 870,650 timed words; 133,318
untimed words were omitted because they cannot be placed safely in an as-of
slice. The `source_windows.jsonl` digest is
`de15370f4a747f41818af568764630c8bf2f69e39ab9e1b2a8ba421a51fc0ebb`.
`acquisition_receipt.json` contains the ZIP digest, every included original
member digest and size, source window digest, counts and exclusions. These
large source files are retained in owner custody and are not committed to Git.

A small readable source packet can be made with
`scripts/mfm/prepare_v16_icsi_source_excerpt.py`. It verifies the windows
digest, selects at most two minutes from one as-of window, and groups adjacent
words by known speaker channel with a bounded gap. Its groups are **derived
presentation spans**, not official dialogue-act/turn labels. All original
word IDs and byte anchors remain in the packet. One acquired example selects
`Bmr005` seconds 300–420 as of second 600 (12 spans, 232 words, packet SHA-256
`fcf91008a52bc7394ab3b5eeada4b2fddfdb965c220a8c680fcad1d7bd6b2381`).

```bash
python scripts/mfm/prepare_v16_icsi_source_excerpt.py \
  --windows "$PRIVATE/icsi-windows-v1/source_windows.jsonl" \
  --receipt "$PRIVATE/icsi-windows-v1/acquisition_receipt.json" \
  --meeting-id Bmr005 --as-of-end-seconds 600 \
  --clip-start-seconds 300 --clip-end-seconds 420 \
  --output "$PRIVATE/Bmr005-300-420-source-excerpt.json"
```

The excerpt may be selected for teacher authoring after rights and speaker
review. It does not automatically satisfy MFM v1.6 positive role coverage,
separate train/development lineage, or independent FINAL qualification.
