# Case-scoped MFM teacher authorization

The v1.6 teacher admission v2 schema accepts a corpus containing separately
authorized cases. It preserves the v1 manifest and intake behavior. The top
`authorization_id` is the owner's authorization to run this combined teacher
training job; it does not replace a case's original `authorization_id`.

Each v2 intake case has its original `authorization_id`, which must equal the
exact candidate bytes. `authorizations` registers every used
`(authorization_id, source_family)` with its source rights `authority_ref` and
`issuer_id`. The verifier checks each pinned rights receipt against the
registered pair, source ID/hash and host, use permissions, revocation flag,
target provenance and split lineage. The corpus and every source/rights byte
remain SHA-pinned. No receipt alone proves an issuer's identity, participant
consent, semantic correctness or independent review. AMI's CC BY receipt is a
project-side copyright permission record for the released transcript text,
with attribution conditions; it makes no broader claim.

## Private 443-case assembly

The first reproducible composite uses the separately issued 441-case
fictional corpus (432 train, 9 diagnostic development) and two source-grounded
AMI cases (train). No candidate, teacher output, original authorization ID,
source payload or rights receipt is rewritten. Source paths are namespaced
under `synthetic/` and `ami/` in a new private directory. The issuer registry
has 25 authorization/source-family pairs. The two AMI cases retain
`owner-directed-mfm-v16-public-teacher` and source authority
`ami-manual-v162-ccby4-publisher-release`; fictional cases retain
`owner-directed-mfm-v16-synthetic` and their original issued receipts.

Run from a checkout containing the AMI expansion config, with its two private
custody directories available. The output directory must not yet exist:

```bash
PYTHONPATH=src:. python scripts/mfm/compose_v16_mixed_teacher_corpus.py \
  --synthetic-root "$SYNTH" \
  --synthetic-intake-sha256 57d968b97475b052e873ce84b1484b64dedde52f976b941fe8a8121234a1acdc \
  --synthetic-manifest-sha256 f54f03ace36b7bb8afde531faf5a283b1cb894ff40389ddaeb96d2f5caa4c48a \
  --ami-root "$AMI" \
  --ami-config configs/mfm/v16_ami_cross_stage_teacher_expansion_20261002.json \
  --ami-config-sha256 5801669ddc24bcef6eed2458dc6dba5fe51c5760f7d0bab9c06550ada52a1781 \
  --ami-issuance-sha256 c732b56b95b2888a9427d24c1cd108c9e731fe16339b90d05cb14636f5c58475 \
  --output-root "$OUT" \
  --owner-authorization-ref owner-directed-mfm-v16-mixed-teacher-20261002
```

The assembly and admission checks passed locally with 434 train and 9
diagnostic development cases. Exact output: intake SHA-256
`3eb8e3395c963db5cb41cf09d1c9e5407a3dfdb005406ea794bda9002b37f374`,
manifest SHA-256
`3ba6e4f40451c51246ed7601e1e7ae95aec473d33e8a4be9425d6ce50de6c655`.
An independent second replay produced both same digests. The private bundle
stays outside Git with a private directory and intake permissions. The full
processor CPU pass under Magnolia's proven P2 loopback namespace is still
pending; this is not a fit, independently reviewed gold, downstream behavior
result or product qualification. It does not authorize paid GPU training.

The local private bundle was also archived deterministically with GNU tar
(`--sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner
--mode='u+rwX,go-rwx'`) and `gzip -n -9`. The resulting private
`mfm-v16-mixed-443-private.tar.gz` has SHA-256
`8eb58481f59658cc3c1b0c3c2f5d616b654c28c14dd4eff15b3c9ac4c928a9e2`,
is mode `0600`, and contains 4,205 entries. `gzip -t`, extracting the manifest
through `tar -xOf`, and a separate deterministic compression replay verified
the archive and the frozen manifest digest. It remains local, outside Git.
