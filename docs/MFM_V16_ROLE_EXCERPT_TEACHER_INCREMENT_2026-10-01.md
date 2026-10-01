# Bounded v1.6 role-excerpt teacher candidates

**State:** six fictional, source-grounded, single-teacher candidates. None is
admitted for training, independently adjudicated, rights authenticated, an
independent generator family, or a model capability result. The source,
teacher-visible prompt, answer, converted target and provenance are in private
custody outside Git under `mfm-review-repro/role-excerpt-candidates/v7`.
The earlier private v1–v6 revisions are exploratory and must not be loaded.
The six-case private manifest SHA-256 is
`42da926754900edf7610951b0a9c7c17446d44b35c57767bb9502522eddef3ba`.
It binds the converter SHA-256
`b56c698e493e9350db85eefea1ea99cb9dfa47b1b7c92a441b5b19f927ad33fd`
and prompt template SHA-256
`c3cb9b3cf20b8a4b57b6abd7609efe19e057897c8d9d88928cab989cfda0c251`.
The source, answer, converted target and provenance for all six cases reproduced
byte for byte through the pinned CLI path.

`scripts/mfm/prepare_v16_role_excerpt_teacher_cases.py` prepares only the
selected source packet. Conversion binds a pinned source/view/prompt/response,
turns unique verbatim quotations into UTF-8 byte anchors, builds the v1.6
context and target, and runs `learning_example_v16_from_record` plus
`supervised_output_record_v16`. A matching quote does not establish semantic
entailment. A synthetic registration and an owner-directed teaching session
do not authenticate source consent, independent review, or write authority.

| Case | Bounded distinction | Positive v1.6 labels | Converted case SHA-256 |
| --- | --- | --- | --- |
| `role-scene-01` | Two time-bounded scenes from one note recorded after the second scene, with host and other participant registered per scene; the first event's ending cites the move | episode | `94b785c7b53f0a74c69dc1fbb9f180130da06bd285f9687e494bde73a4689477` |
| `role-self-01` | Assistant's own check remains assistant-self state; the host does not acquire it | sensitivity | `f0c10674bfacb13faa97656404e003d538557304990f9ae8cc07c98c8dbb34f2` |
| `role-sensitive-01` | Fictional health note proposes a stricter handling hint than the synthetic private source floor | sensitivity | `a7af57cfaabd6d2cd4249853f13607a2c99010ebb1b82735c29fa2adcd4cf274` |
| `role-conflict-01` | Two dated notes disagree about a day; actual attendance remains unresolved | contradiction, abstention | `eff7b33f0090176dfb8d946a64745526131e27ba8a45bc2cf5ba9fb84fc6873f` |
| `role-procedure-01` | One code snippet and one local test support only a narrow candidate after the test; general competence is withheld | outcome, abstention | `84cc35cf883df8ce5090d2a7f08146e2972b9a8f282597480fd956741d0db0ff` |
| `role-audio-unheard-01` | Uninterpreted synthetic WAV bytes and an upload note lead to abstention on spoken content | abstention | `a38d7eaf0303279c3fed96b8e60657f3e63580c9c45677a3d778e09dbfc77859` |

The positive labels are only for the selected evidence scope. The audio case
tests **refusal to fabricate perception**, not audio understanding. The
converter refuses a non-text proposal anchor because it lacks a media-native
verifier. This case is a diagnostic only until the full processor handles
real audio bytes and a qualified target can be reviewed. The code/test case
emits `procedural_skill` as a proposal kind, but `FULL_ROLE_DIMENSIONS` has no
skill dimension. Its positive labels cannot be reported as scored skill
coverage. A future versioned schema needs an independently judged skill target
and outcomes across new tasks before claiming this ability.

Every source and target in these six cases was authored in the same fictional
teacher family. They cannot form a sealed train/development/FINAL partition by
splitting case IDs. Source registration metadata and synthetic event logs are
assumptions for contract testing. No real owner health record, real A.L.I.C.E.
continuity event, or verified audio observation is included. Next corpus work
needs independent source families, authenticated rights, separate adjudicators,
exact source-only review, and later gate/judgment interventions. Re-run the
exact full-corpus CPU processor only after the *actual* full-role corpus is
frozen; the old 21-row processor receipt is historical.
