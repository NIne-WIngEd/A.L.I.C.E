# Deterministic v1.6 diagnostic development family

This is a public **fictional, single-author** source and target family. It is
separate from the 21-case authoring seed and its generator. Its two invented
histories have nine bounded source windows. The chronological ledger, exact
event text, target choices, and ten explicit positive/negative judgments per
case live in `scripts/mfm/v16_development_history_spec.py`. The separate
`scripts/mfm/build_v16_development_history.py` converter uses exact unique
UTF-8 quote spans and validates every result through the v1.6 formation codec.
It cannot certify that the labels are semantically correct.

The generated source-only windows, candidate targets, and receipt are frozen
in `benchmarks/mfm/v16_deterministic_development_{sources.jsonl,targets.jsonl,receipt.json}`.
They are **unreviewed, rights-unverified, unadmitted diagnostic candidates**.
No real owner or assistant history is represented. The two story settings and
the code were authored with the same AI assistance as other MFM synthetic
material, so a different code path does not create independent semantic gold.
Do not use these cases for sealed FINAL or report their score as generalization.

| History | As-of windows | Supervised distinctions |
| --- | --- | --- |
| Studio repair | June 3, 9, 14, 25; June 9 has an unsettled-rule counterfactual | Source-person attribution versus owner goal, two-scene episode, agreed versus unsettled sharing boundary, tested relay scope, failed high-temperature outcome and decision reason, repeated behavior, assistant self-check, correction, deletion and revocation requests |
| Garden survey | July 3, 7, 14, 24 | Separate mission/workspace, attribution, episode and relationship, shallow versus deep gauge evidence, rationale, three observed manual checks, assistant self-check, correction and explicit withdrawal |

Across development, all ten v1.6 adjudication dimensions have a positive
example, and proposals include assistant self, procedural skill, behavior
pattern, decision rationale, deletion and revocation. The negative labels are
explicit in the authored spec for the **selected visible evidence** of each
case. `sensitivity=negative` means no explicit sensitivity hint was proposed
in that target; it does not mean the source lacks confidentiality. The
registered source minimum remains in the context and the gate must enforce it.
The model does not see every earlier event at each later window. This
keeps an unrelated earlier observation from becoming a false negative in a
later focused adjudication. The source-only record carries `as_of`, parent
lineage and original source-item identities; no future event enters the
selected packet. A withdrawn external item's text disappears in the later
packet, which contains its registry tombstone instead. The counterfactual
preserves the nondecisive visit bytes but case/source IDs necessarily differ
for independent admission paths, so it is not a perfectly identical-input
causal intervention.

The aggregate SHA-256 values are:

| Artifact | SHA-256 |
| --- | --- |
| Source windows JSONL | `4f963c19c4aebabcc04c12222322832425c9245f2d568c47071c42e8ac7e09be` |
| Candidate target JSONL | `75a63c9df0a6380646c53ff911cd3833c282666b4a304fd1d40a1f9491bba54d` |
| Generation receipt | `c4bcd0b68b434dc815bcf42b083228d48349c216d451ac09e4b17e97a075c1d8` |

To materialize separate files for `assemble_v16_owner_teacher_corpus.py`,
select a new private output directory outside Git:

```bash
PYTHONPATH=src:. python3 scripts/mfm/build_v16_development_history.py \
  --authorization-id owner-directed-mfm-v16-synthetic \
  --output-dir /owner/mfm-v16-development-candidate
PYTHONPATH=src:. python3 scripts/mfm/build_v16_development_history.py \
  --authorization-id owner-directed-mfm-v16-synthetic \
  --output-dir /owner/mfm-v16-development-candidate --check
```

The resulting `bundle-index.json` lists per-case candidate, ordered source
bytes, canonical target output, original generator input, converter bytes and
provenance hashes. Its target output is the deterministic generator's exact
output, rather than a fabricated service-teacher response. It intentionally
contains **no issued source-rights receipt**. The per-source draft binds the
exact ID, hash and host but leaves issuer, authority, permissions and revocation
`null` under a distinct unissued schema that the assembler cannot admit. A
steward has to authenticate source
and model distribution permission, owner authorization, generator identity,
and truthful lineage before an assembler may admit these cases. An assembler
intake and the receipt must use the same authorization ID; pass a different
ID at generation time if the owner authorizes a different combined corpus.
Never alter candidate JSON after generation to make that ID match.
Pass only a selected per-case `source_packet.json` to an annotator; the
aggregate JSONL contains later windows from both fictional histories.

Even after an authenticated synthetic teacher fit and a complete CPU pass,
these data do not establish Alice-level formation capability. That requires
independently adjudicated longitudinal histories, sealed FINAL, governed
memory acceptance, later retrieval and controlled changes in downstream
judgment.
