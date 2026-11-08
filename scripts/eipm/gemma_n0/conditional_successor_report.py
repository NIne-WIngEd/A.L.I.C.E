"""Descriptive fit of reviewed binary regions; not qualification or selection.

This report checks correctness, separately from invariance or movement. It
does not generate targets, change a packet, or replace the governing consumer.
"""
from __future__ import annotations

import math


def reviewed_region_fit(packet: dict, targets: list[dict]) -> dict:
    """Compare one native packet with explicitly available 0/1 facts.

    The 0.5 division is the binary logit-sign region, not a calibrated release
    threshold. Fractional probabilities and other output families are outside
    this report and remain listed as unmeasured.
    """
    fields = {
        'co_valid_probabilities': ('candidate_ids', packet.get('co_valid_probabilities')),
        'contraindications': ('candidate_ids', packet.get('contraindications')),
        'concept_applicability': ('concept_ids', packet.get('conditional_concepts', {}).get('concept_applicability')),
    }
    result = {'all_measured_regions_correct': False, 'unmeasured_fields': []}
    compared = 0
    errors = 0
    seen_fields = set()
    for target in targets:
        field = target['field']
        if field in seen_fields:
            raise ValueError('duplicate reviewed target field')
        seen_fields.add(field)
        if field not in fields:
            result['unmeasured_fields'].append(field)
            continue
        key, values = fields[field]
        ids = packet.get(key)
        # The native codec preserves tuple pointer rows; a JSON round trip
        # yields lists. Both describe the same ordered, source-bound pointers.
        if (not isinstance(ids, (list, tuple)) or len(ids) != 1
                or not isinstance(ids[0], (list, tuple))
                or any(not isinstance(pointer, str) or not pointer for pointer in ids[0])
                or len(set(ids[0])) != len(ids[0])):
            raise ValueError('report requires one row with unique pointer IDs')
        if not isinstance(values, list) or len(values) != 1 or len(values[0]) != len(ids[0]):
            raise ValueError('report requires a complete native output row')
        lookup = dict(zip(ids[0], values[0]))
        measured, failed, absolute_error = 0, 0, 0.0
        seen_pointers = set()
        for fact in target['facts']:
            if type(fact['available']) is not bool:
                raise ValueError('availability must be explicit Boolean')
            if not fact['available']:
                continue
            pointers = fact['pointers']
            if len(pointers) != 1 or pointers[0] not in lookup or pointers[0] in seen_pointers:
                raise ValueError('reviewed binary fact requires one unique native pointer')
            seen_pointers.add(pointers[0])
            desired = fact['value']
            predicted = lookup[pointers[0]]
            if isinstance(desired, bool) or desired not in (0, 1):
                raise ValueError('this report only measures reviewed binary regions')
            if isinstance(predicted, bool) or not isinstance(predicted, (int, float)) or not math.isfinite(predicted) or not 0 <= predicted <= 1:
                raise ValueError('native probability must be finite and bounded')
            failed += int((predicted >= .5) != bool(desired))
            absolute_error += abs(predicted - desired)
            measured += 1
        result[field] = {'comparisons': measured, 'errors': failed,
            'mean_absolute_error': absolute_error / measured if measured else None}
        compared += measured
        errors += failed
    result['all_measured_regions_correct'] = compared > 0 and errors == 0
    result['qualification'] = False
    return result
