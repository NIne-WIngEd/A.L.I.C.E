"""Recover completed measurements without substituting evidence for learning."""
import importlib
import json
from copy import deepcopy
from hashlib import sha256

import pytest


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def seal(value):
    value['receipt_sha256'] = sha256(canonical(value)).hexdigest()
    return value


def fixture():
    selected = seal({'document_sha256': 'doc', 'frame_sha256': 'frame',
        'feature_binding_sha256': 'binding', 'closed_session_sha256': 'session',
        'packet_sha256': 'packet', 'model_state_sha256': 'model',
        'checkpoint_manifest_sha256': ['n1', 'n2', 'n3']})
    response = seal({'schema': 'alice-personality-governed-response-v1',
        'packet_sha256': 'packet', 'preference_selection': selected,
        'decision': {'complete_saved_packet': True}, 'qualification': 'UNQUALIFIED',
        'acceptance_authority': False})
    rows = [{'packet': 'original', 'document_id': 'scene', 'document_sha256': 'doc', 'response': response}]
    losses = [{'packet': 'original', 'case_id': 'case', 'case_sha256': 'case-sha',
        'phase': 'n1', 'batch_sha256': 'batch', 'total_loss': .7,
        'field_loss': {'concept_applicability': .7}, 'available_counts': {'concept_applicability': 2},
        'descriptive_training_targets_not_independent_qualification': True}]
    expected_documents = {('original', 'scene'): {'document_sha256': 'doc',
        'frame_sha256': 'frame', 'feature_binding_sha256': 'binding', 'closed_session_sha256': 'session'}}
    expected_batches = {'batch': {key: losses[0][key] for key in ('packet', 'case_id', 'case_sha256', 'phase')}}
    return rows, losses, expected_documents, expected_batches


def recover(rows, losses, documents, batches, **overrides):
    try:
        api = importlib.import_module('scripts.eipm.gemma_n0.cumulative_recovery')
    except ModuleNotFoundError as error:
        if error.name != 'scripts.eipm.gemma_n0.cumulative_recovery':
            raise
        pytest.fail('The exact completed-baseline recovery implementation is absent')
    responses = b''.join(canonical(row)+b'\n' for row in rows)
    objectives = b''.join(canonical(row)+b'\n' for row in losses)
    parameters = dict(response_sha256=sha256(responses).hexdigest(),
        objective_sha256=sha256(objectives).hexdigest(), expected_documents=documents,
        expected_batches=batches, parent_model_sha256='model',
        parent_checkpoint_sha256=['n1', 'n2', 'n3'])
    parameters.update(overrides)
    return api.recover_parent_measurements(responses, objectives, **parameters)


def test_complete_baseline_recovered_without_recomputing_or_claiming_learning():
    values = fixture()
    before = deepcopy(values)
    result = recover(*values)
    assert result['count'] == result['measured_case_count'] == 1
    assert result['available_fact_counts'] == {'concept_applicability': 2}
    assert result['reused_completed_measurements'] is True
    assert values == before


@pytest.mark.parametrize('mutation', ['model', 'checkpoint', 'document', 'frame', 'session', 'batch', 'case', 'response_seal', 'loss_file_hash', 'duplicate_response', 'missing_case'])
def test_changed_source_model_or_measurement_is_refused(mutation):
    rows, losses, documents, batches = fixture()
    options = {}
    if mutation == 'model':
        options['parent_model_sha256'] = 'changed'
    elif mutation == 'checkpoint':
        options['parent_checkpoint_sha256'] = ['n1', 'different-n2', 'n3']
    elif mutation in ('document', 'frame', 'session'):
        field = {'document': 'document_sha256', 'frame': 'frame_sha256', 'session': 'closed_session_sha256'}[mutation]
        documents[('original', 'scene')][field] = 'changed'
    elif mutation == 'batch':
        batches['other'] = batches.pop('batch')
    elif mutation == 'case':
        batches['batch']['case_sha256'] = 'changed'
    elif mutation == 'response_seal':
        rows[0]['response']['decision']['complete_saved_packet'] = False
    elif mutation == 'loss_file_hash':
        options['objective_sha256'] = 'wrong'
    elif mutation == 'duplicate_response':
        rows.append(deepcopy(rows[0]))
    elif mutation == 'missing_case':
        losses.clear()
    with pytest.raises(ValueError):
        recover(rows, losses, documents, batches, **options)
