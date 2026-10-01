# Public-history teacher conversion for MFM v1.6

`scripts/mfm/prepare_v16_observed_history_teacher_case.py` converts a bounded
source-only AMI/ICSI style history window into one unadmitted v1.6 candidate.
It never supplies rights receipts, authenticates speakers, judges semantic
entailment, makes a split independent, or trains a model. The source extractor
must establish exact original XML byte/member/archive links and select a
source-only as-of window before using this converter. Preserve that original
window JSON as a separate SHA-pinned file; do not add future meetings or
reference model answers to it.

The private envelope (`mfm-v16-observed-history-envelope-v1`) contains
`case_id`, `split`, `authorization_id`, `host_id`, `self_id`, `authority_id`,
`source_family`, `as_of`, `source_window`, `sources`, `targets`,
`training_admitted:false`, and `rights_status:"unverified"`. `source_window`
is a relative path and SHA-256 to the exact source-only extraction row. Every
`sources[]` item has the v1.6 evidence fields `ref_id`, `role`, `modality`,
`subject_ref`, `speaker_ref`, `source_item_ref`, `observed_at`, `recorded_at`,
`temporal_granularity`, `minimum_sensitivity`, plus `path`, `sha256` and
`relative_time`. Its `path` is an exact UTF-8 source payload under the private
envelope directory. For public meeting data, `role` is `outside_source`;
owner and assistant-self events are disallowed. `targets[]` contains only
registered entity/scene/mission/workspace references supported by listed
evidence; it is often empty.

For this AMI/ICSI adapter, `as_of` is
`{"timeline_ref":"ami-...", "cutoff_seconds":291}`; every item has
`observed_at:null`, `temporal_granularity:"instant"`, and
`relative_time:{"timeline_ref":"ami-...","end_seconds":...}` bounded by the
cutoff. This does not fabricate a date from an ambiguous meeting date label.
Calendar-time sources require their own adapter and replay checks.
The extractor's original-window receipt remains necessary to prove its source
selection and the transcription relationship to the XML archive.

Create source files and the envelope outside Git in a private directory. Pin
the exact envelope hash externally, then prepare the source view and teacher
prompt:

```bash
python -m scripts.mfm.prepare_v16_observed_history_teacher_case prepare \
  --envelope /private/case/envelope.json --envelope-sha256 SHA256 \
  --view /private/case/view.json --prompt /private/case/prompt.txt
```

The teacher response must be retained byte-for-byte as a separate JSON file
with schema `mfm-v16-observed-history-teacher-response-v1`. It binds the case,
exact source-view SHA, exact rendered-prompt SHA, `proposals`, `dispositions`
and all ten scoped `adjudications`. Proposal anchors are exact, unique text
quotes. `convert` rejects a changed source, prompt, response or unsupported
v1.6 output:

```bash
python -m scripts.mfm.prepare_v16_observed_history_teacher_case convert \
  --envelope /private/case/envelope.json --envelope-sha256 SHA256 \
  --view /private/case/view.json --view-sha256 SHA256 \
  --prompt /private/case/prompt.txt --prompt-sha256 SHA256 \
  --response /private/case/teacher-response.json --response-sha256 SHA256 \
  --case /private/case/teacher-candidate.json \
  --provenance /private/case/teacher-provenance.json
```

The output case has `mfm-full-role-curriculum-case-v1.6`, the pinned train or
development split and owner authorization ID, and exact source base64. The
provenance records exact source, prompt, raw teacher, converter and case hashes.
The separate corpus assembler may consume this case, exact source files,
prompt, raw answer and converter bytes after source rights and connected
lineage are resolved by their actual custodians. A single AMI extraction family
and a single ICSI extraction family do not provide six independent role types
in both splits. The training guard still requires ten positive dimensions and
six critical constructs in each opened split. Do not fill absent assistant
self, deletion, revocation or procedural skill targets from mere meeting talk.
