# MFM v1.6 source-bound role teacher candidates

**State:** six fictional, single-teacher, unreviewed training candidates.
The public method is `scripts/mfm/build_v16_role_teacher_candidates.py`.
The generated JSONL is kept outside Git in a private 0700 directory as a
write-once 0600 file. Its current SHA-256 is
`2096cb9a148678e9a2a4a2fea9a3c09d44e9f45db448713998667f3f4219d08e`.
These fictional literals are public method examples. There is no real owner
data in them. Source rights, genuine source authentication, independent blind
review, independent development and FINAL, gradient step, trained MFM, and
downstream memory behavior remain unverified.

| Connected fictional family | Candidates | Scoped teaching contrast |
| --- | ---: | --- |
| Jo sharing | 2 | Same angry outburst, later negotiated rule versus no agreement. Only later source content and its distinct source ID change. Both variants stay in train. |
| April review | 1 | Earlier completed-review claim versus a later explicit owner correction. A correction request cites the old claim ID; no overwrite is asserted. |
| Mira archive permission | 2 | One host identity persists across both times. Before withdrawal, attributed outside-source evidence belongs to Mira while the host diary belongs to the host. After withdrawal, only a registry tombstone is exposed in place of the archive payload. The teacher requests revocation of the source and deletion of derived claims, and abstains from making a new Mira memory or claiming erasure is complete. |
| Birch migration | 1 | Goal, tool-observed 40% copy and aborted deployment, and revised workspace plan retain their separate epistemic and time states. |

The six rows have **one assistant-authored generator family** and four fictional
scenario/host families. This is not four independent source generators. Every
source byte buffer is SHA-256 bound to its context evidence. Every proposal
uses a unique exact UTF-8 byte span in each cited source. The serializer and
v1.6 grounding checks run for every case. All ten dimension labels are
explicitly chosen for the listed source items and checked against the target;
they do not describe a host's whole history. Only the withdrawal/deletion
requests propose a `highly_sensitive` hint above their private source floor.
The model does not set the policy floor.

The original archive payload is deliberately absent in the after-withdrawal
case. A teacher may see a tombstone and the separately authorized owner's
statements. The owner's withdrawal explicitly identifies the registry item
`mira-note-2023`, so that proposal's exact target ID is grounded in compatible
owner evidence; the tombstone does not become owner speech. It must not reopen
withdrawn bytes to make a negative example.
An actual gate must still authenticate revocation rights, identify all
derivatives and duplicates, execute deletion, and rebuild projections. The
candidate records only request those actions. Likewise, the target references
in the April and Mira cases are examples for a later gate to resolve; this
method does not certify their registration.

To reproduce in a private directory outside the repository:

```bash
mkdir -m 700 -p "$HOME/mfm-role-custody"
chmod 700 "$HOME/mfm-role-custody"
PYTHONPATH=src:. python scripts/mfm/build_v16_role_teacher_candidates.py \
  --output "$HOME/mfm-role-custody/role-v16-candidates.jsonl"
PYTHONPATH=src:. python scripts/mfm/build_v16_role_teacher_candidates.py \
  --check --output "$HOME/mfm-role-custody/role-v16-candidates.jsonl"
PYTHONPATH=src:. python -m unittest \
  tests.governance.test_mfm_v16_role_teacher_candidates
```

The write is exclusive: an existing file is never replaced. These six cases
are too small and share one author to justify a full v1.6 fit or a paid GPU
probe. Before any trainer admission, actual source rights must be authenticated
and bound to exact files; teacher prompt/response/conversion bytes need a
provenance manifest; a distinct source family and held-out assessment must be
built. The CPU processor pass must run on the exact eventual admitted corpus.
