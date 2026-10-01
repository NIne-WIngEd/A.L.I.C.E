# V1.6 owner-teacher formation training lane

The owner-authorized synthetic curriculum can teach formation behavior. Human
dual review is a qualification standard for independent development and FINAL,
not a prerequisite for every training target. This lane permits a measured
training-only fit without presenting teacher output as independently reviewed
gold. It is separate from the signed `--full-fit` corpus and never substitutes
for independent qualification, downstream judgment or the memory authority gate.

## Admission contract

`--teacher-fit --teacher-training-manifest` selects schema
`mfm-v16-owner-teacher-corpus-v1` with exact `--input-sha256` and
`--owner-authorization-ref`. Its `status` is
`owner-authorized-teacher-training-only-unqualified`; `case_count` must equal
the number of listed cases. There is no arbitrary row-count threshold. Every
case has an owner authorization ID, explicit train or diagnostic-development
split, author, host/source/generator/scenario/duplicate families and parent
lineage, exact source and v1.6 target payload hashes, and **no asserted
independent reviews**. Each source has an exact-byte `mfm-source-rights-v1`
receipt bound to its source ID, hash and host, with unrevoked formation
training/distribution rights for train or evaluation rights for development.
The owner/steward must authenticate those assertions outside this hash check.

Each case declares `target_origin` as
`owner-authorized-service-teacher` or `licensed-deterministic-generator` and
pins `target_provenance`: `producer_id` (equal to `author_id`),
`producer_version`, and `input`, `output`, `conversion` objects, each with a
relative `path` and SHA-256. The exact prompt/response or generator input/output
and conversion code bytes must exist at admission and inference. Raw teacher
response bytes are kept separately from the converted v1.6 target. These
prompt bytes are also separate from the opened source payloads. For a
deterministic generator, the declared output path and hash must be the exact
target payload. The prompt/generator input and converter must occupy distinct
canonical paths from each other, opened sources and the target; a service
teacher's raw response also has its own path. Provenance files are verified
and retained for audit; only
source and target payloads enter the optimizer. These fields record origin;
matching hashes alone do not prove that the converted
label is semantically correct or that an external service's identity and rights
were authenticated.

The target file must be a complete `mfm-formation-target-v1.6` record with all
ten explicit adjudications, grounded proposal spans, scoped dispositions and
the registered v1.6 context. `unknown` cannot become a negative supervised
label. Both opened splits must contain a positive example for each full-role
dimension. Source, generator and connected parent families cannot cross
train/development; development is **diagnostic**, not independent or FINAL.
Only train enters gradients. The source and target processor must pass over
the entire pinned corpus and write a fresh preflight receipt before a GPU
probe. The run, checkpoint and component bind `teacher_fit=true`, the corpus
manifest SHA and processor preflight, with no fake roster or review receipt.
The research inference runner requires that same pinned teacher manifest and
owner authorization; outputs are marked `teacher_fit_generated` and
proposal-only. The signed
qualifier continues to reject teacher fits.
Teacher-fit artifacts remain unqualified: a separate comparison against an
independently signed, sealed-FINAL corpus must be implemented before any
capability claim.

This is the research runner's local artifact check. A shipped Fable runtime
must use a separately qualified release receipt that binds model lineage and
rights without requiring the user's full raw training corpus at every
inference. This lane does not define or claim that release qualification.

The old 49,819-case mixture is v1/v1.5 target material and cannot be read as
v1.6. The current 21-case v1.6 JSONL is a CPU diagnostic seed with rights still
pending authentication, shared generator lineage and no source-rights
receipts. It cannot be admitted as-is. A repackaged manifest cannot itself
prove that altered lineage or rights claims are true; owner/steward
authentication remains external. The next data
work is to author or convert a complete source-grounded **v1.6** training
corpus from rights-cleared observed sources, retaining exact teacher prompt,
response and conversion bytes, then audit it before any paid GPU minute.

The trainer flags for that future corpus are:

```text
data-preflight --teacher-fit --teacher-training-manifest MANIFEST \
  --input-sha256 MANIFEST_SHA256 --owner-authorization-ref OWNER_REF
processor-preflight --teacher-fit --teacher-training-manifest MANIFEST \
  --input-sha256 MANIFEST_SHA256 --owner-authorization-ref OWNER_REF \
  --prepared-base-dir BASE --prepared-base-receipt BASE_RECEIPT \
  --preflight-receipt NEW_CPU_PREFLIGHT
train --teacher-fit --teacher-training-manifest MANIFEST \
  --input-sha256 MANIFEST_SHA256 --owner-authorization-ref OWNER_REF \
  --prepared-base-dir BASE --prepared-base-receipt BASE_RECEIPT \
  --preflight-receipt NEW_CPU_PREFLIGHT --probe-only --output-dir NEW_PROBE \
  --max-cross-attention-pairs VERIFIED_PEAK
```

Only after that bounded probe fits the selected hardware should a full
`train --teacher-fit` run use the same pinned corpus, a new output directory
and appropriate training settings. The signed independent `--full-fit` lane
remains separate.
