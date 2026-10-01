# AMI cross-stage MFM teacher history (2026-10-01)

`scripts/mfm/prepare_v16_ami_longitudinal_teacher_case.py` prepares a bounded
same-team history from multiple original AMI meeting windows. The current
IS1001 example uses stages b, c and d and only selected speaker turns within
each pinned as-of window. Its public transcript sources remain unadmitted;
rights, independent semantic review and holdout qualification are separate.

The source envelope pins each original window, its transcript word/XML byte
ancestry, team, stage, and within-meeting cutoff. The converter rejects mixed
teams, reordered stages, later turns and modified source bytes. Each registered
source payload has a **derived** header containing the meeting, stage, current
as-of stage, attributed speaker, relative seconds and exact original turn hash,
followed by the unaltered turn text. This header is model-visible. The original
window and teacher prompt/response remain in private custody outside Git.

The example traces a voice-recognition time concern in stage b, a speaker's
report about a manufacturing email and a deferred decision in stage c, and a
project manager's “No” to whether voice was included in stage d. The last
answer is scoped to the meeting; it does not establish a shipped product or
feature outcome. The teacher response therefore labels `source_person`
and `abstention` present, `outcome` and `correction` negative. It is one
unreviewed teaching example, not a measure of longitudinal MFM capability.

The earlier single-window converter and its eight public-source cases retain
their exact implementation hash. Multi-window cases use the new converter hash
and record the earlier converter hash as `base_converter_sha256`; an assembler
must preserve both implementation bytes as well as source, rendered prompt,
raw answer, converted target and provenance. All case construction rejects
future or reordered turns before teacher rendering and replays the checks at
conversion time.
