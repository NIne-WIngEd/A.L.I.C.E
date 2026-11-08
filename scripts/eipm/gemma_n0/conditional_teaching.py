"""Pinned same-context TRAIN corrections through one actual native forward.

This is additional admission, never source/review/gradient authorization. The
owner must independently admit both original batches and pin each record in its
plan. Records contain original annotations and belong in protected storage.
There is no cached N2 replay, new target, confidence interpolation or optimizer.
"""
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from hashlib import sha256
import json
import re

import torch

from src.alice_personality.identity.codec import canonical, fingerprint
from src.alice_personality.identity.contracts import IdentityRepresentation
from src.alice_personality.identity.model import IdentityModel
from src.alice_personality.teaching.cases import ReviewedTeachingCase
from src.alice_personality.teaching.contracts import ReviewedTarget, TeachingBatch, TeachingError
from src.alice_personality.teaching.layout import target_layouts
from src.alice_personality.teaching.materialize import _document_frame


SCHEMA = 'alice-personality-conditional-teaching-v1'
CONDITIONAL_FIELDS = ('concept_applicability', 'concept_uncertainty',
                      'concept_source_support', 'concept_source_disconfirmation')
_KEYS = {'schema', 'n1_case', 'n1_case_sha256', 'n2_case_id', 'n2_case_sha256',
         'document_sha256', 'frame_sha256', 'frame_binding', 'n1_batch_sha256',
         'n2_batch_sha256', 'n1_materialization_binding', 'n1_review_record'}


def _require(condition, message):
    if not condition:
        raise TeachingError(message)


def record_sha256(record):
    """Canonical JSON digest; the caller supplies an independently pinned digest."""
    try:
        return sha256(canonical(record)).hexdigest()
    except (TypeError, ValueError) as exc:
        raise TeachingError('correction record must contain finite JSON values') from exc


def _seal(value, key):
    _require(isinstance(value, Mapping) and value.get(key) == record_sha256(
        {name: item for name, item in value.items() if name != key}),
        'correction materialization or frame binding seal differs')


def _targets(case, frame):
    """Check the exact original named facts against native permission layouts."""
    case.validate()
    _document_frame(case.document, frame)
    layouts, targets = target_layouts(frame, case.phase), {}
    for annotation in case.targets:
        _require(annotation.field in layouts, 'correction case names an unknown native target field')
        layout = layouts[annotation.field]
        _require(layout.distribution == (annotation.mode != 'independent'), 'correction target mode differs')
        values = torch.full(layout.allowed.shape, float('nan'), device=layout.allowed.device, dtype=torch.float32)
        uncertainty, available = values.clone(), torch.zeros_like(layout.allowed)
        for fact in annotation.facts:
            _require(len(fact.pointers) == len(layout.axes), 'correction pointer geometry differs')
            try:
                index = (0, *(axis.index(pointer) for axis, pointer in zip(layout.axes, fact.pointers)))
            except ValueError as exc:
                raise TeachingError('correction names an unknown frame pointer') from exc
            if fact.available:
                _require(bool(layout.allowed[index]), 'correction cannot grant forbidden authority')
                values[index], uncertainty[index], available[index] = fact.value, fact.uncertainty, True
        target = ReviewedTarget(values, available, uncertainty)
        target.validate(tuple(layout.allowed.shape), layout.allowed, probability=layout.probability)
        targets[annotation.field] = target
    return targets


def _receipt(receipt, *, case_sha, document_sha, frame_sha, targets_sha,
             source_families, lineage, phase, source_class, binding, session_sha):
    _seal(receipt, 'receipt_sha256')
    expected = {'schema': 'alice-personality-teaching-case-materialization-v2',
        'case_sha256': case_sha, 'document_sha256': document_sha, 'frame_sha256': frame_sha,
        'target_sha256': targets_sha, 'source_families_sha256': fingerprint(source_families),
        'construction_lineage_sha256': fingerprint(lineage), 'phase': phase, 'split': 'TRAIN',
        'source_class': source_class, 'feature_binding_sha256': binding['binding_sha256'],
        'closed_session_sha256': session_sha, 'qualification': 'UNQUALIFIED',
        'review_authority_granted': False, 'source_acceptance_authority': False,
        'gradient_authority_granted': False}
    _require(all(receipt.get(key) == value for key, value in expected.items()),
             'correction materialization differs from its case/frame/targets/lineage')


def _review(review, case, lineage):
    _require(isinstance(review, Mapping) and review.get('split') == 'TRAIN'
             and review.get('case_ids') == [case.case_id]
             and review.get('source_family_ids') == [list(case.source_family_ids)],
             'correction needs the exact original reviewed TRAIN N1 batch')
    if lineage is not None or 'construction_lineage_sha256' in review:
        _require(review.get('construction_lineage_sha256') == fingerprint(lineage),
                 'correction reviewed construction lineage differs')


def _batch_case(case, batch, phase):
    _require(type(case) is ReviewedTeachingCase and type(batch) is TeachingBatch,
             'correction requires native reviewed cases and materialized batches')
    case.validate()
    _require(case.phase == batch.phase == phase and case.split == batch.split == 'TRAIN'
             and batch.case_ids == (case.case_id,) and batch.source_family_ids == (case.source_family_ids,)
             and case.source_class == batch.source_class and not batch.contrasts
             and batch.lineage == ((case.lineage,) if case.lineage is not None else None),
             'correction case/batch phase, TRAIN split, source or lineage differs')
    _require(fingerprint(_targets(case, batch.frame)) == fingerprint(batch.targets),
             'correction case facts differ from the complete materialized targets')
    _require(isinstance(batch.production_binding, Mapping) and isinstance(batch.production_session, Mapping),
             'correction requires produced-frame and closed-session bindings')
    _receipt(batch.materialization_binding, case_sha=case.sha256, document_sha=case.document.sha256,
             frame_sha=fingerprint(batch.frame), targets_sha=fingerprint(batch.targets),
             source_families=case.source_family_ids, lineage=batch.lineage, phase=phase,
             source_class=case.source_class, binding=batch.production_binding,
             session_sha=batch.production_session.get('receipt_sha256'))
    _require(batch.materialization_binding.get('artifact_sha256') == dict(case.artifact_sha256),
             'correction annotation artifact provenance differs')


def build_correction_record(n1_case, n2_case, n1_batch, n2_batch, *, frame_binding, reviewed_n1_batches):
    """Bind original admitted N1 annotations to the same document's N2 objective.

    ``reviewed_n1_batches`` is the externally admitted manifest's
    ``reviewed_batches`` mapping, not a newly generated diagnostic manifest.
    The owner still verifies its original case/artifact/allocation admissions.
    """
    _batch_case(n1_case, n1_batch, 'n1')
    _batch_case(n2_case, n2_batch, 'n2')
    _require(n1_case.document.sha256 == n2_case.document.sha256
             and n1_case.lineage == n2_case.lineage and n1_case.source_class == n2_case.source_class
             and n1_case.source_family_ids == n2_case.source_family_ids,
             'correction must use the same document, source class and lineage; no partner swaps')
    _require(frame_binding == n1_batch.production_binding == n2_batch.production_binding
             and n1_batch.production_session == n2_batch.production_session
             and fingerprint(n1_batch.frame) == fingerprint(n2_batch.frame),
             'correction must use the same produced frame and closed session')
    _require(isinstance(reviewed_n1_batches, Mapping), 'original N1 review mapping is required')
    n1_sha = n1_batch.fingerprint()
    review = reviewed_n1_batches.get(n1_sha)
    _review(review, n1_case, n1_batch.lineage)
    record = {'schema': SCHEMA, 'n1_case': n1_case.to_mapping(), 'n1_case_sha256': n1_case.sha256,
        'n2_case_id': n2_case.case_id, 'n2_case_sha256': n2_case.sha256,
        'document_sha256': n1_case.document.sha256, 'frame_sha256': fingerprint(n2_batch.frame),
        'frame_binding': dict(frame_binding), 'n1_batch_sha256': n1_sha,
        'n2_batch_sha256': n2_batch.fingerprint(), 'n1_materialization_binding': dict(n1_batch.materialization_binding),
        'n1_review_record': dict(review)}
    record = json.loads(canonical(record))
    validate_correction_record(record, n2_batch, expected_sha256=record_sha256(record))
    return record


def validate_correction_record(record, n2_batch, *, expected_sha256):
    """Revalidate a pinned record and return eligible exact field/pointer values.

    Availability and annotation uncertainty are retained independently. A fact
    with uncertainty one is never a correction; other uncertainty is not used
    to interpolate the fact value. Unknown and forbidden coordinates fail.
    """
    _require(type(record) is dict and set(record) == _KEYS and record.get('schema') == SCHEMA,
             'correction record schema differs')
    _require(type(expected_sha256) is str and re.fullmatch('[0-9a-f]{64}', expected_sha256)
             and record_sha256(record) == expected_sha256, 'correction record differs from its external pin')
    for key in ('n1_case_sha256', 'n2_case_sha256', 'document_sha256', 'frame_sha256',
                'n1_batch_sha256', 'n2_batch_sha256'):
        _require(type(record[key]) is str and re.fullmatch('[0-9a-f]{64}', record[key]),
                 'correction requires exact case/batch/document/frame digests')
    _require(type(n2_batch) is TeachingBatch and n2_batch.phase == 'n2' and n2_batch.split == 'TRAIN'
             and n2_batch.case_ids == (record['n2_case_id'],)
             and n2_batch.fingerprint() == record['n2_batch_sha256'], 'correction N2 batch differs')
    case = ReviewedTeachingCase.from_mapping(record['n1_case'])
    lineage = (case.lineage,) if case.lineage is not None else None
    _require(case.phase == 'n1' and case.split == 'TRAIN' and case.sha256 == record['n1_case_sha256']
             and case.document.sha256 == record['document_sha256'] and lineage == n2_batch.lineage
             and case.source_class == n2_batch.source_class
             and n2_batch.source_family_ids == (case.source_family_ids,),
             'correction original N1 case/document/source/lineage differs')
    binding, frame = record['frame_binding'], n2_batch.frame
    _seal(binding, 'binding_sha256')
    _require(binding == n2_batch.production_binding and binding.get('frame_sha256') == record['frame_sha256']
             and fingerprint(frame) == record['frame_sha256']
             and binding.get('document_sha256') == record['document_sha256'], 'correction produced frame differs')
    _require(isinstance(n2_batch.production_session, Mapping), 'correction closed session is missing')
    session_sha = n2_batch.production_session.get('receipt_sha256')
    _require(type(session_sha) is str and re.fullmatch('[0-9a-f]{64}', session_sha),
             'correction closed session digest is missing')
    n1_targets = _targets(case, frame)
    for receipt, case_sha, targets, phase in (
            (record['n1_materialization_binding'], case.sha256, n1_targets, 'n1'),
            (n2_batch.materialization_binding, record['n2_case_sha256'], n2_batch.targets, 'n2')):
        _receipt(receipt, case_sha=case_sha, document_sha=case.document.sha256,
                 frame_sha=record['frame_sha256'], targets_sha=fingerprint(targets),
                 source_families=case.source_family_ids, lineage=lineage, phase=phase,
                 source_class=case.source_class, binding=binding, session_sha=session_sha)
    _require(record['n1_materialization_binding'].get('artifact_sha256') == dict(case.artifact_sha256),
             'correction original annotation artifact pins differ')
    original_n1 = TeachingBatch(frame, n1_targets, (case.case_id,), (case.source_family_ids,),
        'n1', 'TRAIN', case.source_class, production_binding=binding,
        production_session=n2_batch.production_session,
        materialization_binding=record['n1_materialization_binding'], lineage=lineage)
    _require(original_n1.fingerprint() == record['n1_batch_sha256'],
             'correction original N1 batch digest differs from its complete materialization')
    _review(record['n1_review_record'], case, lineage)
    return {target.field: {fact.pointers: fact.value for fact in target.facts
                          if fact.available and fact.uncertainty < 1}
            for target in case.targets if target.field in CONDITIONAL_FIELDS}


def _detached(value):
    if isinstance(value, torch.Tensor):
        _require(not value.requires_grad and value.grad_fn is None,
                 'correction must not introduce provider or N1 gradients')
    elif is_dataclass(value):
        for field in fields(value):
            _detached(getattr(value, field.name))
    elif isinstance(value, Mapping):
        for item in value.values():
            _detached(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _detached(item)


def _patch(representation, frame, values):
    _require(type(representation) is IdentityRepresentation, 'correction requires native N1 output')
    _detached(representation)
    layouts, updates = target_layouts(frame, 'n1'), {}
    for name, facts in values.items():
        layout = layouts[name]
        mask = representation.concept_source_mask if 'source_' in name else representation.concept_mask
        original, original_logits = getattr(representation, name), getattr(representation, name + '_logits')
        _require(torch.equal(mask, layout.allowed) and original.shape == original_logits.shape == mask.shape,
                 'correction N1 authority mask or field geometry differs')
        changed_values, changed_logits = original.clone(), original_logits.clone()
        changed = False
        for pointers, target in facts.items():
            index = (0, *(axis.index(pointer) for axis, pointer in zip(layout.axes, pointers)))
            _require(bool(mask[index]), 'correction cannot change a forbidden coordinate')
            value = original.new_tensor(target)
            if bool(value != original[index]):
                changed = True
                changed_values[index] = value
                # N2 receives the exact reviewed probability. Only the paired
                # audit logit uses finite dtype endpoints; sham logits survive.
                epsilon = torch.finfo(original.dtype).eps
                probability = value.clamp(epsilon, 1 - epsilon) if bool(value == 0 or value == 1) else value
                changed_logits[index] = torch.logit(probability)
        if changed:
            updates[name], updates[name + '_logits'] = changed_values, changed_logits
    return replace(representation, **updates) if updates else representation


def forward_with_correction(model, batch, record, *, expected_record_sha256,
                            apply_correction=True, subset='full'):
    """One actual forward, with a temporary hook on frozen N1's output only.

    Requires exclusive use of the model during this call. The caller performs
    natural and replay half-loss backwards separately and owns optimization.
    ``first_half``/``second_half`` select complete eligible concept groups by
    sorted SHA256 of exact semantic pointer IDs, independent of scores/targets.
    Corrected packets are supplied-correction results, never natural inference.
    """
    _require(type(apply_correction) is bool and subset in ('full', 'first_half', 'second_half'),
             'correction switch or concept subset is invalid')
    values = validate_correction_record(record, batch, expected_sha256=expected_record_sha256)
    _require(isinstance(model, IdentityModel) and model.phase in ('n2', 'inference'),
             'correction forward requires native N2-only training or inference')
    for module in (model.n1, model.n3):
        _require(all(not p.requires_grad and p.grad is None for p in module.parameters())
                 and not any(m.training for m in module.modules()), 'correction requires frozen N1 and N3')
    _require(all(p.requires_grad == (model.phase == 'n2') for p in model.n2.parameters()),
             'correction N2 trainability differs from the selected phase')
    batch.validate(model.config)
    _detached(batch.frame)
    if subset != 'full':
        concepts = sorted({pointers[0] for facts in values.values() for pointers in facts},
                          key=lambda pointer: (sha256(pointer.encode('utf-8')).hexdigest(), pointer))
        middle = (len(concepts) + 1) // 2
        selected = set(concepts[:middle] if subset == 'first_half' else concepts[middle:])
        values = {name: {p: value for p, value in facts.items() if p[0] in selected}
                  for name, facts in values.items()}
    calls = 0

    def correct(_module, inputs, output):
        nonlocal calls
        _require(len(inputs) == 1 and inputs[0] is batch.frame, 'correction hook received a different frame')
        calls += 1
        representation, query = output
        _detached(query)
        return (_patch(representation, batch.frame, values), query) if apply_correction else output

    handle = model.n1.register_forward_hook(correct)
    try:
        packet = model(batch.frame)
        _require(calls == 1, 'correction must consume exactly one native N1 forward')
        packet.validate()
        return packet
    finally:
        handle.remove()
