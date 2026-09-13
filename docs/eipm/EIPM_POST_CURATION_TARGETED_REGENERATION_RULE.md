# EIPM Post-Curation Targeted Regeneration Rule

**Status:** owner-directed working rule

Broad synthetic generation is not a one-way gate. After curation, the cleaned coverage map must be recomputed. If curation removes, factorizes, or weakens material enough to expose meaningful personality holes, targeted E-INF/A-SYN generation is allowed and expected.

## Regeneration policy

- Do not restart a giant blind enrichment campaign merely because rows were removed or factorized.
- Generate against explicit post-curation gaps.
- A-SYN generation may remain relatively aggressive because later curation can filter competing branches.
- E-INF still requires direct canonical E0 support.
- Preserve historical UNKNOWN when E0 does not justify a historical inference.
- Use A-SYN to avoid runtime behavioral blanks where a compatible starting policy can responsibly be created.
- Prefer multiple distinct candidate branches for genuinely ambiguous personality regions.
- New generated material returns to curation before corpus freeze.

## Saturation

Saturation is evaluated after curation, not from raw generator row count. A raw frontier can appear saturated while curation reveals unsupported or mechanically duplicated regions.

The intended loop is:

raw expansion -> curation/repair -> recompute coverage -> targeted regeneration if needed -> curation -> corpus freeze.

## Weight gate

Targeted regeneration does not authorize training. The owner final pre-weight review remains mandatory before the first private EIPM gradient update.
