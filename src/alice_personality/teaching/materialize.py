"""Exact reviewed cases -> actual sparse teaching tensors, without authority.

Requires the actual complete feature frame and closed producer session. The
existing trainer still independently checks all source, N0, review, split and
current authorization pins before any governed gradient.
"""
from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping

import torch

from ..identity.codec import fingerprint
from ..identity.feature_producer import FrozenFrameProducer, ProducedFrame
from ..identity.contracts import IdentityModelConfig, SOURCE_KINDS
from .cases import ReviewedTeachingCase, validate_artifact_pins
from .contracts import ReviewedTarget, TeachingBatch, TeachingError, canonical
from .layout import target_layouts


@dataclass(frozen=True)
class MaterializedCase:
    batch: TeachingBatch
    receipt: dict


def verify_materialized_batch(batch, artifacts, reviewed):
    """Couple the complete materialization to the actual training admission."""
    receipt = batch.materialization_binding
    if not isinstance(receipt, Mapping):
        raise TeachingError('materialized batch requires its exact materialization receipt')
    body = {key: value for key, value in receipt.items() if key != 'receipt_sha256'}
    if (receipt.get('schema') != 'alice-personality-teaching-case-materialization-v1'
            or receipt.get('receipt_sha256') != sha256(canonical(body)).hexdigest()
            or receipt.get('frame_sha256') != fingerprint(batch.frame)
            or receipt.get('target_sha256') != fingerprint(batch.targets)
            or len(batch.case_ids) != 1 or len(batch.source_family_ids) != 1
            or receipt.get('source_families_sha256') != fingerprint(batch.source_family_ids[0])
            or receipt.get('phase') != batch.phase or receipt.get('split') != batch.split
            or receipt.get('source_class') != batch.source_class):
        raise TeachingError('materialization schema, bytes, targets, lineage or phase differs')
    for name in ('review_authority_granted', 'source_acceptance_authority', 'gradient_authority_granted'):
        if receipt.get(name) is not False:
            raise TeachingError('materialization cannot grant source/review/gradient authority')
    if receipt.get('qualification') != 'UNQUALIFIED':
        raise TeachingError('materialization cannot grant personality qualification')
    pins = receipt.get('artifact_sha256')
    validate_artifact_pins(pins, batch.source_class, label='materialization')
    declarations = reviewed.get('reviewed_case_artifact_sha256')
    if not isinstance(pins, Mapping) or not isinstance(declarations, Mapping) \
            or declarations.get(receipt.get('case_sha256')) != pins:
        raise TeachingError('materialized annotation/protocol/review pins lack explicit externally bound review')
    roles = ('source_package', 'compiled_receipt') if batch.source_class == 'public_mechanical_fixture' \
            else ('source_package', 'compiled_receipt', 'prepared_receipt')
    if any(role not in artifacts or pins.get(role) != artifacts[role].sha256 for role in roles):
        raise TeachingError('materialized source/compiler/preparation pins differ from actual admitted artifacts')
    if (not isinstance(batch.production_binding, Mapping) or not isinstance(batch.production_session, Mapping)
            or receipt.get('document_sha256') != batch.production_binding.get('document_sha256')
            or receipt.get('feature_binding_sha256') != batch.production_binding.get('binding_sha256')
            or receipt.get('closed_session_sha256') != batch.production_session.get('receipt_sha256')):
        raise TeachingError('materialized document/feature/session bindings differ')


def _document_frame(document, frame):
    """A resealed feature record cannot substitute pointers or permissions."""
    groups = [(frame.query, tuple(f.field_id for f in document.fields)),
              (frame.sources, tuple(e.record.record_id for e in document.sources)),
              (frame.concepts, tuple(e.record.concept_id for e in document.concepts)),
              (frame.candidates, tuple(e.candidate_id for e in document.candidates)),
              (frame.relations, tuple(e.entry_id for e in document.relations))]
    groups.extend((frame.labels[family], tuple(s.label_id for s in labels))
                  for family, labels in document.labels.items())
    for bank, ids in groups:
        if bank.entry_ids != (ids or (None,),) or bank.entry_mask.tolist() != [[True] * len(ids) or [False]]:
            raise TeachingError('produced frame pointers/availability differ from the target-free document')
    if (frame.source_records != (tuple(e.record for e in document.sources) or (None,),)
            or frame.concept_records != (tuple(e.record for e in document.concepts) or (None,),)
            or frame.label_specs != {family: (labels,) for family, labels in document.labels.items()}):
        raise TeachingError('produced frame source/concept/label authority differs from the document')
    for name, tensor in (('authority_allowed', frame.candidate_authority_mask), ('context_allowed', frame.candidate_context_mask)):
        if tensor.tolist() != [[getattr(e, name) for e in document.candidates] or [False]]:
            raise TeachingError('produced candidate authority differs from the document')
    nodes = [e.record.record_id for e in document.sources] or [None]
    nodes += [e.record.concept_id for e in document.concepts]
    relations = [e.entry_id for e in document.relations]
    if (frame.graph.senders.tolist() != [[nodes.index(e.sender_id) for e in document.edges]]
            or frame.graph.receivers.tolist() != [[nodes.index(e.receiver_id) for e in document.edges]]
            or frame.graph.relation_indices.tolist() != [[relations.index(e.relation_id) for e in document.edges]]
            or frame.graph.roles != (tuple(e.role for e in document.edges),)
            or frame.graph.edge_mask.tolist() != [[True] * len(document.edges)]):
        raise TeachingError('produced support graph differs from the document')


def _production(case, produced, session):
    if type(produced) is not ProducedFrame or not isinstance(session, Mapping):
        raise TeachingError('materialization requires an actual frame and closed feature session')
    binding = produced.binding
    for value, key in ((binding, 'binding_sha256'), (session, 'receipt_sha256')):
        if not isinstance(value, Mapping) or sha256(canonical({k: v for k, v in value.items() if k != key})).hexdigest() != value.get(key):
            raise TeachingError('feature binding or closed session bytes changed')
    if (binding.get('document_sha256') != case.document.sha256
            or binding.get('frame_sha256') != fingerprint(produced.frame)
            or binding.get('schema') != 'alice-personality-produced-frame-binding-v1'
            or session.get('schema') != 'alice-personality-feature-session-v1'
            or binding.get('source_geometry') != [produced.frame.query.states.shape[2], produced.frame.query.states.shape[-1]]
            or binding.get('binding_sha256') not in session.get('frame_binding_sha256', [])
            or binding.get('implementation_sha256') != FrozenFrameProducer._code_hashes()
            or session.get('implementation_sha256') != FrozenFrameProducer._code_hashes()):
        raise TeachingError('document/frame/closed session membership or producer implementation differs')
    for value in (binding, session):
        if (value.get('qualification') != 'UNQUALIFIED' or value.get('source_acceptance_authority') is not False
                or value.get('persistent_feature_values_retained') is not False):
            raise TeachingError('feature materialization cannot grant source or personality acceptance')
    if binding.get('session_source_recheck_pending') is not True \
            or any(binding.get(name) is not False for name in ('generation', 'chat_template', 'truncation')):
        raise TeachingError('complete source-only feature production is required')
    if case.source_class == 'public_mechanical_fixture':
        if (binding.get('source_class') != 'public_mechanical_fixture'
                or session.get('source_class') != 'public_mechanical_fixture'
                or session.get('status') != 'PUBLIC_MECHANICS_ONLY'
                or session.get('mechanical_fixture_only') is not True
                or session.get('session_source_recheck_passed') is not False
                or binding.get('prepared_receipt_file_sha256') is not None
                or session.get('prepared_receipt_file_sha256') is not None):
            raise TeachingError('public mechanics must retain actual fixture provenance')
    else:
        from .trainer import _production as verify_production
        verify_production(produced.frame, binding, session,
            prepared_file_sha256=case.artifact_sha256['prepared_receipt'],
            prepared_receipt_sha256=session.get('prepared_receipt_sha256'),
            implementation=FrozenFrameProducer._code_hashes())


def materialize_case(case: ReviewedTeachingCase, produced: ProducedFrame, closed_session: Mapping,
                     config: IdentityModelConfig, *, expected_case_sha256: str,
                     expected_artifact_sha256: Mapping, family_splits: Mapping) -> MaterializedCase:
    """No inference of targets, split bridge, reviews or authorization."""
    if type(case) is not ReviewedTeachingCase:
        raise TeachingError('source compiler rows are not reviewed teaching cases')
    case.validate()
    if case.sha256 != expected_case_sha256 or dict(case.artifact_sha256) != expected_artifact_sha256:
        raise TeachingError('case or source/compiler/annotation/review differs from its external binding')
    if not isinstance(family_splits, Mapping) or any(family_splits.get(family) != case.split for family in case.source_family_ids):
        raise TeachingError('source family crosses explicit teaching/calibration split allocation')
    _production(case, produced, closed_session)
    frame = produced.frame
    frame.validate(config)
    _document_frame(case.document, frame)
    layouts = target_layouts(frame, case.phase)
    targets, coverage = {}, {}
    for annotation in case.targets:
        if annotation.field not in layouts:
            raise TeachingError('target field is not implemented in the selected teaching phase')
        layout = layouts[annotation.field]
        if layout.distribution != (annotation.mode != 'independent'):
            raise TeachingError('target mode differs from the output distribution/independent semantics')
        shape = tuple(layout.allowed.shape)
        values = torch.full(shape, float('nan'), device=layout.allowed.device, dtype=torch.float32)
        uncertainty = values.clone()
        available = torch.zeros(shape, device=layout.allowed.device, dtype=torch.bool)
        for fact in annotation.facts:
            if len(fact.pointers) != len(layout.axes):
                raise TeachingError('target pointers differ from the exact output axes')
            try:
                position = (0, *(axis.index(pointer) for axis, pointer in zip(layout.axes, fact.pointers)))
            except ValueError as exc:
                raise TeachingError('target names an unknown output pointer') from exc
            if fact.available:
                if not bool(layout.allowed[position]):
                    raise TeachingError('target addresses unavailable or unauthorized output positions')
                values[position], uncertainty[position], available[position] = fact.value, fact.uncertainty, True
        target = ReviewedTarget(values, available, uncertainty)
        target.validate(shape, layout.allowed, probability=layout.probability)
        active = available.any(-1) if layout.distribution else None
        if layout.distribution:
            known = torch.where(available, values, torch.zeros_like(values))
            if bool((torch.abs(known.sum(-1)[active] - 1) > 1e-5).any()):
                raise TeachingError('reviewed distribution must sum to one on its explicitly known choices')
            if annotation.mode == 'closed_distribution' and not torch.equal(available[active], layout.allowed[active]):
                raise TeachingError('closed comparison must explicitly review every eligible alternative in its active rows')
        if annotation.field in {'candidate_pair_probabilities', 'value_tradeoff_probabilities'}:
            both = available & available.transpose(-1, -2)
            if bool((torch.abs((values + values.transpose(-1, -2))[both] - 1) > 1e-5).any()):
                raise TeachingError('reviewed pair directions must be complementary when both are known')
        if annotation.field == 'provenance':
            for col, record in enumerate(frame.source_records[0]):
                if record is not None and bool(available[0, col].any()) \
                        and float(values[0, col, SOURCE_KINDS.index(record.source_kind)]) != 1:
                    raise TeachingError('reviewed provenance contradicts immutable source-kind authority')
        if annotation.field == 'voice_control_values':
            for col, spec in enumerate(frame.label_specs['voice'][0]):
                if spec is not None and bool((((values[0, :, col] < spec.minimum) |
                    (values[0, :, col] > spec.maximum)) & available[0, :, col]).any()):
                    raise TeachingError('reviewed voice controls exceed portable-unit ranges')
        targets[annotation.field] = target
        coverage[annotation.field] = {'mode': annotation.mode, 'available_facts': int(available.sum()),
            'certain_facts': int((available & (uncertainty < 1)).sum()),
            'discriminating_rows': int(((available.sum(-1) > 1) &
                (torch.where(available, 1 - uncertainty, 0).sum(-1) > 0)).sum()) if layout.distribution else None}
    receipt = {'schema': 'alice-personality-teaching-case-materialization-v1', 'case_sha256': case.sha256,
        'document_sha256': case.document.sha256, 'frame_sha256': fingerprint(frame),
        'target_sha256': fingerprint(targets), 'feature_binding_sha256': produced.binding['binding_sha256'],
        'closed_session_sha256': closed_session['receipt_sha256'], 'artifact_sha256': dict(case.artifact_sha256),
        'source_families_sha256': fingerprint(case.source_family_ids), 'phase': case.phase, 'split': case.split,
        'source_class': case.source_class, 'coverage': coverage, 'qualification': 'UNQUALIFIED',
        'review_authority_granted': False, 'source_acceptance_authority': False, 'gradient_authority_granted': False}
    receipt['receipt_sha256'] = sha256(canonical(receipt)).hexdigest()
    if case.sha256 != expected_case_sha256 or fingerprint(frame) != produced.binding['frame_sha256']:
        raise TeachingError('reviewed case or feature frame changed during materialization')
    batch = TeachingBatch(frame, targets, (case.case_id,), (case.source_family_ids,), case.phase, case.split,
        case.source_class, production_binding=produced.binding, production_session=dict(closed_session),
        materialization_binding=receipt)
    batch.validate(config)
    return MaterializedCase(batch, receipt)
