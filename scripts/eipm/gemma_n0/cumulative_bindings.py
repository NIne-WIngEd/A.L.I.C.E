"""Pure cumulative allocation and original optimized-batch scope bindings.

Inputs have already passed their external pins and native source/checkpoint
verification. These helpers preserve those declarations; they confer no review,
gradient, teacher, N0 or personality authority. Calibration events cannot become
governing N1/N2 teaching. A case hash is never an optimized batch fingerprint.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from src.alice_personality.teaching.cases import ReviewedTeachingCase
from src.alice_personality.teaching.contracts import TeachingBatch, TeachingError
from src.alice_personality.teaching.lineage import ALLOCATION_SCHEMA, validate_allocation


def _require(condition, message):
    if not condition:
        raise TeachingError(message)


def merge_allocations(allocations: list[dict]) -> dict:
    """Union pinned native v2 allocations, allowing only identical duplicates.

    Each input must remain independently valid, including ancestor closure.
    Reference identities were checked against the source by the caller; this
    helper preserves their union rather than silently replacing the bank.
    """
    _require(type(allocations) is list and bool(allocations),
        'cumulative allocation needs a nonempty list of native allocations')
    merged = {'schema': ALLOCATION_SCHEMA,
        'qualification_scope': 'DEVELOPMENT_CONDITIONAL_CASES',
        'family_splits': {}, 'reference_family_ids': [],
        'supervision_family_splits': {}, 'supervision_parents': {}}
    references = set()
    for allocation in allocations:
        validate_allocation(allocation)
        for name in ('family_splits', 'supervision_family_splits', 'supervision_parents'):
            for key, value in allocation[name].items():
                _require(key not in merged[name] or merged[name][key] == value,
                    f'cumulative {name} has conflicting duplicate declarations')
                merged[name][key] = deepcopy(value)
        references.update(allocation['reference_family_ids'])
    merged['reference_family_ids'] = sorted(references)
    validate_allocation(merged)
    return merged


def bind_parent_scope(selection: dict, original_cases: Mapping[str, ReviewedTeachingCase],
                      batch_by_case: Mapping[str, TeachingBatch], manifests: list[dict]) -> dict:
    """Resolve explicit original-case choices to exact optimized native batches.

    ``batch_by_case`` may lazily replay/materialize native batches. Only one
    value is read per case, and only its digest is retained. Manifests must be
    the caller's already native-verified parents; this is scope reconciliation,
    not a replacement for checkpoint/admission verification. Extra controller
    selection metadata and replacement-packet validation belong to the caller.
    """
    _require(isinstance(selection, Mapping)
        and isinstance(original_cases, Mapping) and bool(original_cases)
        and isinstance(batch_by_case, Mapping)
        and set(batch_by_case) == set(original_cases),
        'parent scope needs the exact original case and native batch mappings')
    _require(type(manifests) is list and bool(manifests),
        'parent scope needs native-verified parent manifests')

    case_hashes, case_phases, case_batches = {}, {}, {}
    for key, case in original_cases.items():
        _require(type(case) is ReviewedTeachingCase and key == case.case_id,
            'original case map must bind each exact native case ID')
        case.validate()
        _require(case.phase in ('n1', 'n2') and case.split == 'TRAIN',
            'original governing cases must be native N1/N2 TRAIN cases')
        case_hashes[key], case_phases[key] = case.sha256, case.phase
        batch = batch_by_case[key]
        _require(type(batch) is TeachingBatch and batch.case_ids == (key,)
            and batch.phase == case.phase and batch.split == 'TRAIN'
            and batch.source_class == case.source_class
            and isinstance(batch.materialization_binding, Mapping)
            and batch.materialization_binding.get('case_sha256') == case_hashes[key],
            'original batch case ID, phase, TRAIN or materialized case hash differs')
        case_batches[key] = batch.fingerprint()
        # Drop the full native frame before requesting the next lazy batch.
        del batch
    _require(len(set(case_batches.values())) == len(case_batches),
        'each original case must resolve a unique native batch fingerprint')

    expected = {case_batches[key]: case_phases[key] for key in case_batches}
    optimized = set()
    for manifest in manifests:
        _require(isinstance(manifest, Mapping)
            and manifest.get('phase') in ('n1', 'n2', 'calibration'),
            'parent governing manifest phase differs')
        if manifest['phase'] == 'calibration':
            continue
        events = manifest.get('events')
        _require(type(events) is list and bool(events),
            'parent governing manifest has no actual optimized teaching events')
        for event in events:
            _require(isinstance(event, Mapping), 'parent teaching event is malformed')
            digest = event.get('teaching_batch_sha256')
            _require(isinstance(digest, str) and digest in expected
                and expected[digest] == manifest['phase']
                and event.get('phase') == manifest['phase'] and event.get('split') == 'TRAIN',
                'parent optimized event phase, TRAIN or exact native batch digest differs')
            optimized.add(digest)
    _require(optimized == set(expected),
        'parent optimized events must account for every exact original native batch')

    retained, superseded, accounted = [], {}, set()
    for group in ('retained_original_cases', 'superseded_original_annotations'):
        rows = selection.get(group)
        _require(type(rows) is list, 'original parent selection requires explicit row lists')
        for row in rows:
            _require(isinstance(row, Mapping), 'original parent selection row is malformed')
            key = row.get('case_id')
            _require(isinstance(key, str) and key in case_batches and key not in accounted
                and row.get('packet') == 'original'
                and row.get('case_sha256') == case_hashes[key]
                and row.get('phase') == case_phases[key],
                'original selection case ID, packet, case hash, phase or uniqueness differs')
            reason = row.get('reason')
            _require(isinstance(reason, str) and bool(reason.strip()),
                'retained and superseded original annotations require explicit reasons')
            accounted.add(key)
            if group == 'retained_original_cases':
                retained.append(case_batches[key])
            else:
                superseded[case_batches[key]] = reason
    _require(accounted == set(original_cases),
        'retained/superseded selection must account for every original case exactly once')
    return {'retained_parent_batches': tuple(retained),
        'superseded_parent_batches': superseded, 'case_batch_sha256': case_batches}
