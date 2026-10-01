# Synthea clinical chronology: source-only family

**State:** public synthetic source candidate, not admitted training data. No
formation labels, teacher responses, reviewer decisions, signed rights,
independent development, sealed FINAL, trained MFM or capability score result
from this inventory. The original ZIP and metadata inventory remain outside
Git in source custody.

## Exact source and lineage

| Item | Pin |
| --- | --- |
| Official sample repo | `https://github.com/synthetichealth/synthea-sample-data` at `9959d9178ea28f4ec10f17ee238b6fabe6eb0de5` |
| November 2021 FHIR R4 ZIP | `downloads/synthea_sample_data_fhir_r4_nov2021.zip`; SHA-256 `6d3c5433bcae4791bc5c30469d1445b430fb4894d5c13bda15fee0584bbd7778` |
| Inspected source code | `https://github.com/synthetichealth/synthea` at `d9d07a6eef91ee5144293b42ab64224d84d124f8`; Apache 2.0 `LICENSE` SHA-256 `b40930bbcf80744c86c46a12bc9da056641d722716c378f5659b9e555ef833e1`; `NOTICE` SHA-256 `aefac5c5d632a0cf595688712eff8a45da0bdadc92c07718aef9876f2f346b97` |
| Source inventory | Private `synthea-nov2021-source-inventory.json`, SHA-256 `18934bb58539650c4cd42c556c83e4a6c011fa1e54ae7386c04689aaf1df8a55` |

The inspected 2026 code commit is **not** asserted to be the generator
revision of the archived 2021 ZIP. That revision remains unverified. All 555
patient bundles belong to one Synthea November 2021 generator component. Do
not split its patients into supposedly independent train/development/FINAL
families. The archive also has two practitioner/hospital information bundles;
the inventory excludes them from patient histories.

The archive has 555 patient JSON bundles and 1,134,819,627 uncompressed
patient-byte content. Among the resource types are 27,812 Encounters,
131,703 Observations, 17,253 Conditions and 1,831 CarePlans. The aggregate
date markers span 1912-09-25 to 2021-11-19. These are simulator output counts,
not independent histories, clinical truth or MFM target labels. The inventory
records each original ZIP member's SHA-256, byte size, host family, resource
counts and source date bounds; it contains no source payload.

`scripts/mfm/inventory_synthea_sample_sources.py` verifies the exact ZIP,
both clean upstream commits, and source license/NOTICE digests before making
the inventory. Its `isolated_resource` helper rechecks the externally pinned
inventory and archive, opens one inventoried patient's original FHIR member,
and returns the **verbatim byte span** of a single Encounter or Observation
entry. The receipt binds the archive member SHA, entry ordinal, original byte
offsets, narrow source SHA and an as-of date. It refuses an undated entry,
Patient profile or entry containing a date-looking value after the requested
cutoff. A downstream teacher should only see separately reviewed, selected
as-of slices, never the entire patient Bundle (which contains future events).
This helper is a byte-level custody aid, not a semantic or rights validator.

The first source-only slice is entry 1 of the inventory's first patient. It
binds member SHA-256
`4ea2517eb45340fc6cbcf5e492db34a6eac8c4f280db08d07c68c00873a6191c`,
original byte range `[5345, 7566)`, source SHA-256
`45bb0913cfa0d96243c8bf06b3ee4d27056b7c5313420335275ac66748e0411d`,
and an observed date of `1954-04-03` before the explicit cutoff. It is not a
formation target.

Synthea's [official download page](https://synthetichealth.github.io/downloads.html)
describes synthetic data usable for secondary research and industry uses.
Its [maintainer's licensing answer](https://github.com/synthetichealth/synthea/discussions/1167)
also identifies separate licenses for FHIR and embedded medical terminology.
The Apache license on the simulator code is not a standalone rights receipt
for every FHIR record. An actual steward must resolve those terms and sign
source rights against exact bytes before the records enter the teacher-fit
manifest. This source is structured text JSON. Clinical notes inside JSON
are text, and imaging references do **not** provide actual image pixels.

## Role coverage and partition limit

Clinical encounters and observations can challenge subject attribution,
event versus recording time, episode boundaries, uncertain observations,
source precedence, sensitivity, and outcome follow-up. None is automatically
a memory of Fable's owner. Synthetic medical encounters do not supply the
owner's negotiated relationship norms, mission and workspace evolution,
correction/deletion consent, or actual image/audio/video grounding. Any such
target needs a genuinely relevant observed source and explicit adjudication.
Multi-Source and Synthea can form distinct *source* lineages, but two families
still cannot fill isolated train, independent DEV and sealed FINAL under the
current connected-family rule. Additional genuinely separate generators and
an independent reviewer/custodian are required. Do not count artificial
patient partitions as independent generator families.

Reproduce the inventory into a new external custody path after obtaining the
exact sample and source repos:

```bash
python scripts/mfm/inventory_synthea_sample_sources.py \
  --sample-repo ../mfm-synthea-sample-data \
  --sample-commit 9959d9178ea28f4ec10f17ee238b6fabe6eb0de5 \
  --generator-repo ../mfm-synthea-source \
  --generator-commit d9d07a6eef91ee5144293b42ab64224d84d124f8 \
  --archive-sha256 6d3c5433bcae4791bc5c30469d1445b430fb4894d5c13bda15fee0584bbd7778 \
  --output ../private-source-custody/synthea-nov2021-source-inventory.json
```

Test the source-byte contract with
`PYTHONPATH=src python -m unittest tests/governance/test_mfm_synthea_sample_source_inventory.py -v`.
