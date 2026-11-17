"""Complete native personal judgment to replaceable response realization.

This helper preserves the original scientific checkpoint implementation. It
neither grants action authority nor silently repairs a wrong personal ranking.
"""
from copy import deepcopy
from hashlib import sha256

import torch

from src.alice_personality.identity.codec import (canonical, decision_to_mapping, fingerprint,
                                                label_semantics_fingerprint)
from src.alice_personality.identity.consumer import DecisionConsumer, RenderedResponse
from src.alice_personality.identity.contracts import IdentityError


def respond_with_governance(consumer, document, produced, closed_session, *, renderer=None):
    """Bind the full IDP and actual teaching scope to its response construction.

    The existing .5 co-valid region distinguishes acceptable alternatives from
    preference. It is not a measured risk bound or independent qualification.
    Untaught native outputs remain diagnostic values with explicit scope.
    """
    if not isinstance(consumer, DecisionConsumer):
        raise IdentityError('governing response requires the actual native consumer')
    original = consumer.respond(document, produced, closed_session)
    evidence = consumer._evidence()
    with torch.no_grad():
        packet = consumer.model(produced.frame)
    if fingerprint(packet) != original['packet_sha256']:
        raise IdentityError('governing packet differs from the actual selected response')
    decision = decision_to_mapping(packet)
    n1_taught = set()
    taught = set(evidence['reviewed_fields'])
    calibrated = set()
    calibrated_labels = {}
    from src.alice_personality.teaching.trainer import _json, _path
    for pin in consumer.checkpoints:
        manifest = _json(_path(_path(pin.path, directory=True) / 'manifest.json'))
        if manifest['phase'] in ('n1', 'calibration'):
            coverage = n1_taught if manifest['phase'] == 'n1' else calibrated
            coverage.update(field for event in manifest['events']
                for field, count in event.get('effective_target_coverage', {}).items() if count > 0)
        if manifest['phase'] == 'calibration':
            for event in manifest['events']:
                for field, labels in event.get('reviewed_label_coverage', {}).items():
                    calibrated_labels.setdefault(field, set()).update(labels)
    output_fields = ('preferences', 'co_valid_probabilities', 'contraindications',
        'uncertainty', 'evidence_sufficiency', 'failure_tail_risk', 'owner_fidelity_uncertainty',
        'voice_control_values', 'voice_control_confidence', 'candidate_pair_probabilities',
        'value_tradeoff_probabilities', 'evidence_pointers', 'historical_evidence_pointers',
        'alice_experience_pointers', 'concept_pointers',
        'concept_applicability', 'concept_uncertainty', 'concept_source_support',
        'concept_source_disconfirmation',
        *("head:" + family for family in packet.heads if family != 'voice'))
    scope = {field: {'taught': field in n1_taught or field in taught or field in calibrated,
                     'n1_taught': field in n1_taught,
                     'n2_taught': field in taught, 'n3_reviewed': field in calibrated}
             for field in output_fields}
    for field, entry in scope.items():
        family = field[5:] if field.startswith('head:') else \
                 'voice' if field.startswith('voice_control_') else None
        if family is None:
            continue
        specs = [spec for row in packet.heads[family].specs for spec in row if spec is not None]
        ids = lambda reviewed: sorted({spec.label_id for spec in specs
            if label_semantics_fingerprint(family, spec) in reviewed})
        n2_ids = ids(evidence['reviewed_labels'].get(field, set()))
        n3_ids = ids(calibrated_labels.get(field, set()))
        entry.update({'global_n2_coverage': field in taught, 'global_n3_coverage': field in calibrated,
            'reviewed_current_label_ids': sorted(set(n2_ids) | set(n3_ids)),
            'n2_taught': field in taught and bool(n2_ids),
            'n3_reviewed': field in calibrated and bool(n3_ids)})
        # Calibration reviews refine an existing semantic head; temperatures
        # alone do not teach a new head's meaning or new portable voice units.
        entry['taught'] = entry['n2_taught']
    preferred = list(original['selected_candidate_ids'])
    valid_taught = 'co_valid_probabilities' in taught
    acceptable = [intent['candidate_id'] for intent in original['intents']
        if valid_taught and intent['co_valid_probability'] is not None
        and intent['co_valid_probability'] >= .5]
    conflict = valid_taught and any(candidate not in acceptable for candidate in preferred)
    status = 'UNRESOLVED_GOVERNING_CONFLICT' if conflict else original['status']
    result = {'schema': 'alice-personality-governed-response-v1',
        'status': status, 'validity_status': 'TAUGHT' if valid_taught else 'UNTAUGHT',
        'preferred_candidate_ids': preferred,
        'selected_candidate_ids': [] if conflict else preferred,
        'acceptable_candidate_ids': acceptable,
        'response_text': None if conflict else original['response_text'],
        'preference_selection': original, 'decision': decision,
        'packet_sha256': original['packet_sha256'], 'output_scope': scope,
        'reviewed_label_fingerprints': {field: sorted(values)
            for field, values in evidence['reviewed_labels'].items()},
        'rendering': deepcopy(original['rendering']),
        'qualification': 'UNQUALIFIED', 'acceptance_authority': False,
        'owner_fidelity_approved': False, 'spoken_fidelity_verified': False}
    if renderer is not None and status == 'SELECTED':
        # A replaceable renderer receives full meaning and declared unknowns.
        rendered = renderer(deepcopy(result))
        if (type(rendered) is not RenderedResponse or not isinstance(rendered.text, str)
                or not rendered.text.strip() or type(rendered.delivery) is not dict
                or not isinstance(rendered.renderer_id, str) or not rendered.renderer_id.strip()):
            raise IdentityError('renderer must return an explicit actual realization')
        canonical(rendered.delivery)
        if rendered.artifact_sha256 is not None:
            import re
            if not isinstance(rendered.artifact_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', rendered.artifact_sha256):
                raise IdentityError('renderer artifact requires an exact digest or explicit absence')
        intent = next(item for item in original['intents'] if item['candidate_id'] == preferred[0])
        equivalent = rendered.text == result['response_text'] and rendered.delivery == intent['voice_controls']
        result['rendering'] = {'status': 'TEXT_REALIZED_DELIVERY_UNVERIFIED' if equivalent else 'UNRESOLVED_RENDERER',
                               **vars(rendered)}
        if not equivalent:
            result['status'] = 'UNRESOLVED_RENDERER'
    if fingerprint(consumer.model.state_dict()) != original['model_state_sha256']:
        raise IdentityError('governing model changed during response realization')
    # Recheck actual checkpoint/source evidence after the external callback.
    consumer._evidence()
    result['receipt_sha256'] = sha256(canonical(result)).hexdigest()
    return result
