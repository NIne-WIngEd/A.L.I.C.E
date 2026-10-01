# V1.6 owner-teacher corpus assembly

`scripts/mfm/assemble_v16_owner_teacher_corpus.py` freezes already authored
source-grounded v1.6 candidates into the private owner-teacher manifest. It
does not make teacher answers, source permissions, or independent reviews. Run
it on an owner-controlled filesystem outside Git after the source steward has
recorded exact file hashes and authenticated the source permission assertions.

## Input custody

The private bundle root must already hold each candidate JSON, its separate
source payloads, source rights receipts, the rendered teacher prompt or
deterministic generator input, raw output, converter implementation, and a
converter provenance JSON. All intake paths are relative to this root and
carry externally recorded SHA-256 hashes. The intake JSON itself is externally
pinned by `--intake-sha256`; no path may escape the bundle root.

The intake schema is `mfm-v16-owner-teacher-assembly-intake-v1`, with
`status=owner-authorized-teacher-training-only-unqualified`, `corpus_id`,
`authorization_id`, and nonempty `cases`. Each case has:

| Field | Required meaning |
| --- | --- |
| `case_id`, `split` | Candidate ID and exact original `train` or `development` assignment |
| `author_id`, `producer_version`, `target_origin` | Authenticated teacher/generator identity and version; origin is `owner-authorized-service-teacher` or `licensed-deterministic-generator` |
| `host_family`, `source_family`, `generator_family`, `scenario_family`, `duplicate_group`, `parent_case_ids` | Truthful connected lineage. `generator_family` covers the target generation family as well as source generation; do not rename one teacher to force a split |
| `candidate` | `{path, sha256}` to an exact `mfm-full-role-curriculum-case-v1.6` case with matching ID, split and authorization |
| `sources` | Ordered rows with `source_id`, `path`, `sha256`, `rights_path`, `rights_sha256`, `parent_source_ids`; exact source bytes match the candidate evidence bytes |
| `prompt`, `response`, `converter`, `converter_provenance` | Existing `{path, sha256}` records for complete input, raw output, implementation and converter provenance |

The converter provenance JSON must bind `case_id`, `case_sha256`,
`rendered_prompt_sha256`, `converter_sha256`, plus
`teacher_response_sha256` for service teachers or `generator_output_sha256`
for deterministic generators. Those are exact bytes, not an assertion of
semantic correctness. The candidate must pass the v1.6 codec and grounded
proposal validator, with all ten adjudications explicit and no `unknown`
supervised label. The source rights receipts must exist and match source IDs,
hashes, host and permitted split use under `mfm-source-rights-v1`.

For a service teacher, the assembler writes a canonical target under
`targets/CASE_ID.json`, preserving its raw response separately. For a licensed
deterministic generator, `response` is the already existing exact canonical
target bytes; it becomes both the target and the provenance output. The
assembler never substitutes a generated answer for it. The intake and rights
files are never rewritten. Existing output paths cause a failure.

```bash
python3 scripts/mfm/assemble_v16_owner_teacher_corpus.py \
  --bundle-root "$PRIVATE_CORPUS_ROOT" \
  --intake "$PRIVATE_CORPUS_ROOT/intake.json" \
  --intake-sha256 "$EXTERNALLY_RECORDED_INTAKE_SHA256"
```

The receipt prints the manifest path, frozen manifest SHA, train/development
counts and `qualified_for_product=false`. The default command checks source
rights, exact provenance, connected lineage, v1.6 source and target grounding,
ten positive role dimensions and the six critical proposal constructs in
**both** opened splits. Only source/target paths enter optimizer handoff. It
rejects one service teacher producer appearing in both train and development
even if someone gives the rows different `generator_family` labels. The
owner/steward must independently authenticate the declared producer identity,
license, rights receipt issuer and connected lineages; these checks cannot
establish those facts from hashes.

Training development is diagnostic and is never sealed FINAL or independent
qualification. The old 49,819 v1/v1.5 rows, 21 public v1.6 CPU diagnostics,
and unadmitted fictional role candidates cannot be admitted by relabeling
them. Once a real manifest passes, record its SHA outside the job and use
`magnolia_v16_teacher_cpu_preflight.sbatch` under P2 isolation for a full
processor pass. No paid GPU time is justified by assembly alone.
