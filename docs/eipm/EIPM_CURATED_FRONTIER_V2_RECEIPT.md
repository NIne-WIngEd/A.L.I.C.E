# EIPM Curated Frontier v2 Receipt

**Status:** working candidate-frontier successor; no training authority  
**Branch:** `alice-eipm-v1-build`

## Source chain

- Curated Frontier v1 SHA-256: `E95B5903F955B7D1661B91DA21668159B0F05A0F8D660FA0FDB5042B3FBDD116`
- Post-curation targeted Sol handoff SHA-256: `DBB2C15DC50B8924B660DA3524C1CB22DBF6693A486D9D6D7A6515F1FB5535D2`
- Targeted Sol output SHA-256: `7B567176019148B7FD665D76BB2960E53307A11C5A5AEE7FDE01A23F3BBBD369`
- Curated Frontier v2 SHA-256: `3867FF04D1E326086B9086B2F106B9156B3A3EC8D637D3161E7BF01616183EE9`

## Targeted pass result

The 67-item post-curation queue was closed without restarting broad generation.

The targeted generator produced:

- 0 new E-INF proposals;
- 119 new A-SYN proposals;
- 15 clean regenerated context variants.

Independent curation retained all 119 A-SYN proposals provisionally because they passed structural/provenance review and were useful under the owner-approved non-overconservative curation posture.

They are lane-separated rather than flattened:

- 37 clean E0-informed synthetic identity-prior candidates;
- 15 context-informed synthetic-prior candidates;
- 67 runtime meta-policy candidates for historically unsupported regions;
- 15 regenerated contextual-transform candidates.

Runtime meta-policies and contextual transforms are not independent Elaina identity evidence.

## Coverage

Direct core-E0 A-SYN anchoring increased from 158/187 to 167/187.

The remaining direct-anchor units are not treated as unresolved automatically. The targeted queue dispositioned them as already semantically covered, historical/context-only, or represented through context-informed synthetic behavior.

The explicit post-curation queue now has zero unresolved items.

## Forward rule

Do not resume broad E-INF/A-SYN generation now. Targeted generation may resume later if corpus construction, evaluation, or owner review exposes a genuine behavioral hole.

Curated Frontier v2 remains a candidate substrate, not accepted final training data. E0 is immutable. No weights may be created before the owner final pre-weight review and explicit approval.

No private Elaina payload is contained in this receipt.
