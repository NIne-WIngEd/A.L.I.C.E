"""Reuse exact completed baseline measurements bound to the current construction.

The caller must supply digests from the owner-isolated failed-run readback.
This performs no inference, training, source acceptance or qualification.
"""
import json
import math
from hashlib import sha256


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def recover_parent_measurements(response_data, objective_data, *, response_sha256,
        objective_sha256, expected_documents, expected_batches, parent_model_sha256,
        parent_checkpoint_sha256):
    require(sha256(response_data).hexdigest() == response_sha256, 'completed response file changed')
    require(sha256(objective_data).hexdigest() == objective_sha256, 'completed objective file changed')
    require(bool(expected_documents) and bool(expected_batches), 'completed baseline cannot be empty')
    seen_documents = set()
    for line in response_data.splitlines():
        row = json.loads(line)
        key = (row['packet'], row['document_id'])
        require(key in expected_documents and key not in seen_documents, 'unexpected or duplicate baseline document')
        expected = expected_documents[key]
        response = row['response']
        selected = response['preference_selection']
        for sealed in (response, selected):
            require(sha256(canonical({k: v for k, v in sealed.items() if k != 'receipt_sha256'})).hexdigest() == sealed['receipt_sha256'], 'baseline response seal changed')
        require(response['schema'] == 'alice-personality-governed-response-v1', 'wrong baseline response schema')
        require(row['document_sha256'] == expected['document_sha256'], 'baseline document changed')
        require(all(selected.get(field) == value for field, value in expected.items()), 'baseline source/frame/session binding changed')
        require(selected['model_state_sha256'] == parent_model_sha256, 'baseline learned model changed')
        require(selected['checkpoint_manifest_sha256'] == list(parent_checkpoint_sha256), 'baseline checkpoint chain changed')
        require(response['packet_sha256'] == selected['packet_sha256'], 'baseline native packet binding changed')
        require(response['qualification'] == 'UNQUALIFIED' and response['acceptance_authority'] is False, 'baseline cannot grant acceptance')
        seen_documents.add(key)
    require(seen_documents == set(expected_documents), 'baseline document coverage is incomplete')

    seen_batches = set()
    coverage = {}
    for line in objective_data.splitlines():
        row = json.loads(line)
        digest = row['batch_sha256']
        require(digest in expected_batches and digest not in seen_batches, 'unexpected or duplicate baseline batch')
        require(all(row.get(field) == value for field, value in expected_batches[digest].items()), 'baseline case/source/phase binding changed')
        require(row['descriptive_training_targets_not_independent_qualification'] is True, 'baseline target fit is descriptive')
        require(math.isfinite(row['total_loss']) and all(math.isfinite(value) for value in row['field_loss'].values()), 'baseline objective is nonfinite')
        for field, count in row['available_counts'].items():
            require(type(count) is int and count >= 0, 'baseline target count is invalid')
            coverage[field] = coverage.get(field, 0) + count
        seen_batches.add(digest)
    require(seen_batches == set(expected_batches), 'baseline case coverage is incomplete')
    return {'count': len(seen_documents), 'file_sha256': response_sha256,
        'measured_case_count': len(seen_batches), 'all_reviewed_objective_file_sha256': objective_sha256,
        'available_fact_counts': coverage, 'reused_completed_measurements': True}
