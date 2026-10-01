# AMI v1.6 public teacher expansion

**Status:** two source-grounded, single-teacher, unadmitted cases. They are
training candidates, not independent development, FINAL, an independently
reviewed gold set, or an MFM capability result. The private case payloads and
rights receipts are outside Git. The public configuration pins six case
artifacts for each case without publishing source text.

| Case | AMI scenario stages | Training target |
| --- | --- | --- |
| `ami-es2005bcd-design-tradeoffs` | ES2005 b → c → d | Distinguish a usability/manual argument, preference for a reference design, a suggested hinged cover, and a later report about scroll buttons. Abstain on selected final design and manufacture. |
| `ami-ts3003bcd-remote-tradeoffs` | TS3003 b → c → d | Distinguish questions about a docking station, a participant's reported market research and speech-recognition advice, an LCD rationale, and a later button-cost suggestion. Abstain on verified research, final features, and realized cost. |

The source turn payloads contain derived stage, meeting, speaker, time and
original-turn hash headers followed by exact AMI transcript text. The
cross-stage converter rejects foreign or reordered sessions, future turns and
changed bytes. Each teacher answer records an exact quote anchor, attributed
speaker and epistemic status. It does not turn meeting speakers into the owner
or Alice, infer a shipped product from meeting discussion, or promote a reported
survey to verified evidence. All ten dimensions receive explicit scoped labels.
The teacher answer and its conversion are retained as separate SHA-pinned
private bytes. The case digests are `ce22062da67efe4820c2c971148f55b11d91f82a4f91235b54539fc6e84c2368`
and `93d2a8bd0f3329d35974fee2d1687dfb42ebf17ade4a91fc54a488eea0a6a76d`.

## Release and permission evidence

The [publisher's AMI manual annotations v1.6.2 release](https://groups.inf.ed.ac.uk/ami/download/)
identifies this 2017 release as CC BY 4.0. The original ZIP is pinned by
SHA-256 `b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d`;
its embedded `LICENCE.txt` is pinned by
`d52fd9c7c19eec9f0bec3107554c10937203e261259d73b01637a61501fec7f7`.
The exact source-only window JSONL is pinned by
`f414869926088e241f694db00f20862609e6992933e8713fda4a1733f659311d`.
The [CC BY 4.0 legal text](https://creativecommons.org/licenses/by/4.0/legalcode.en)
licenses reproduction, sharing and adaptation under its conditions, including
attribution, a license link and notice of modifications upon sharing. It does
not license others' privacy or publicity rights. Our exact-source receipts
record the project-side interpretation for **this released transcript text**:
`formation_training`, `formation_evaluation` and `model_distribution` are
`true` in the copyright-license scope, subject to CC BY conditions. The issuer
is `alice-agent-under-owner-direction`, not the AMI publisher; no participant
consent, noncopyright clearance, publisher endorsement or independent label
review is asserted. The original text remains attributed and CC BY licensed;
the receipt does not turn that text into proprietary Alice data. A person's
name or sensitive detail that appears in a public transcript still needs
appropriate handling in the product.

`issue_v16_ami_public_license_receipts.py` replays the pinned case/source
files against the ZIP and window inventory, then writes 22 exact-source
`mfm-source-rights-v1` records and an issuance manifest to a new private
folder. The current private manifest SHA-256 is
`c732b56b95b2888a9427d24c1cd108c9e731fe16339b90d05cb14636f5c58475`.
The receipt records copyright permission and unassessed categories; it is not
an independent legal opinion. The cases remain unadmitted because their original authorization ID differs
from the issued synthetic training/development corpus, their exact sources and
teacher provenance have not been staged into a combined assembly intake, and
no combined CPU processor pass has run. Preserve both original authorizations;
a case-scoped admission rule or explicit umbrella authorization with each
original evidence record is needed before a combined manifest can pass the
current single-authorization assembler and admission verifier. The owner-authorized teacher
training lane permits single-teacher labels without independent gold or FINAL;
those reviews are needed for later independent capability qualification. The
public license does not by itself establish target correctness.

The [ICSI publisher license](https://groups.inf.ed.ac.uk/ami/icsi/license.shtml)
also identifies its corpus and annotations as CC BY 4.0. This AMI-specific
issuer does not silently issue ICSI receipts. ICSI has a different original
ZIP, license member and source extraction lineage and needs exact ICSI source
bindings. The two source families may both be used as attributed public-text
teaching candidates after their own byte/rights checks, but they do not alone
supply every owner/self role in MFM.

## Lineage and replay

Both cases are in `train`, with distinct ES2005 and TS3003 scenario histories
but **one** AMI manual annotation/source family and **one** service teacher.
All AMI-derived cases remain in the training side of any proposed assembly;
two AMI sessions are not independent generator families. The same service
teacher cannot write diagnostic development targets and be counted as an
independent check: `formation_dataset_admission` connects its producer ID
across the split. Keep later AMI stages and duplicate transcripts with the
same scenario, and register speaker/scenario ancestry during actual corpus
assembly. The separately generated fictional development histories are
contract diagnostics, not independent public-history gold. Existing ICSI
teacher drafts were also produced by this teacher and currently say `train`;
they cannot simply be relabeled as independent development.

With the private bundle rooted at `$PRIVATE` and the already pinned original
sources, replay without opening a model or GPU:

```bash
PYTHONPATH=src:. python scripts/mfm/audit_v16_ami_teacher_expansion.py \
  --bundle-root "$PRIVATE/mfm-public-teacher-expansion" \
  --archive "$PRIVATE/ami_public_manual_1.6.2.zip" \
  --windows "$PRIVATE/ami-v16-word-windows.jsonl"
```

The audit checks the ZIP and embedded license hashes, entire window export
hash, each original selected meeting window, the model-visible source turn,
rendered prompt, raw response, replayed v1.6 target and provenance. It fails
on altered source, prompt, response, candidate, source chronology or rights
status. The config is
`configs/mfm/v16_ami_cross_stage_teacher_expansion_20261002.json` with SHA-256
`5801669ddc24bcef6eed2458dc6dba5fe51c5760f7d0bab9c06550ada52a1781`.
The run reported two cases and the ES2005/TS3003 scenario sessions; it issued
no fit, independent review, FINAL split or paid GPU receipt.

The exact private archive of the two candidates and 22 rights receipts has
SHA-256 `f8d4d9d279da0ebf38e017a1b7eae4ba01ec53750ce2239e49295524fd546913`.
It was not published to Drive because the upload was blocked by automatic
approval review. That local archive is the current custody copy; the code
branch intentionally contains no original transcript or teacher answer bytes.
