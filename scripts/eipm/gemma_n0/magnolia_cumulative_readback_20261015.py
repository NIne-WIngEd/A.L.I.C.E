"""Recover completed prelearning measurements from failed owner job 577303.

No model construction, optimization, protected export, or acceptance authority.
"""
import importlib.util
import json
import math
import os
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path('/homes/01/mxrayan/rayan-compute')
REV = '333fcd0c8239fc81a6459b13d53bbe592b33fda6'
GUARD = 'f5822c3657b246a5b232f42bddf93b8026bda0b6a7e573835c1cd8c90136c1e4'
NAME = 'rayan-cumulative-readback-20261015'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def run(namespace):
    isolation = namespace._p2_guard()
    os.umask(0o077)
    code = namespace._bounded_path(os.environ['PERSONALITY_REPO_ROOT'], ROOT, file=False)
    namespace._clean_code(code, REV)
    output = namespace._bounded_path(str(ROOT/'rayan-personality/private-compile/rayan-cumulative-learning-r2-577303'), ROOT, file=False)

    def read(path):
        path = namespace._bounded_path(str(path), ROOT, file=True)
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.stat().st_size < 512 * 1024**2
        return path.read_bytes()

    plan_raw = read(ROOT/'rayan-personality/private-inputs/rayan-cumulative-learning-r2-20261013/owner-cumulative-run.protected.json')
    assert sha256(plan_raw).hexdigest() == 'c40303c2823d84c9246f92873713859485c2ab17460801f386f8d2ab051032fa'
    plan = json.loads(plan_raw)
    failure_raw = read(output/'failure-detail.protected.json')
    failure = json.loads(failure_raw)
    assert failure['exception_type'] == 'ValueError'
    assert failure['message'] == 'report requires one row with unique pointer IDs'
    assert failure['public_code_frames'][-1]['function'] == 'reviewed_region_fit'

    summaries = {}
    for name in ('parent-governed-responses.protected.jsonl', 'parent-all-reviewed-objectives.protected.jsonl', 'parent-original-conditional-failures.protected.jsonl'):
        data = read(output/name)
        rows = [json.loads(line) for line in data.splitlines()]
        summaries[name] = {'file_sha256': sha256(data).hexdigest(), 'bytes': len(data), 'rows': len(rows)}
        if name == 'parent-governed-responses.protected.jsonl':
            assert len(rows) == 236
            assert len({(row['packet'], row['document_id']) for row in rows}) == 236
            packet_counts = {}
            for row in rows:
                response = row['response']
                selected = response['preference_selection']
                for sealed in (response, selected):
                    assert sha256(canonical({k: v for k, v in sealed.items() if k != 'receipt_sha256'})).hexdigest() == sealed['receipt_sha256']
                assert row['document_sha256'] == selected['document_sha256']
                assert response['packet_sha256'] == selected['packet_sha256']
                assert selected['model_state_sha256'] == plan['parent_model_state_sha256']
                assert selected['checkpoint_manifest_sha256'] == [pin['manifest_sha256'] for pin in plan['parent_checkpoints']]
                assert response['qualification'] == 'UNQUALIFIED' and response['acceptance_authority'] is False
                packet_counts[row['packet']] = packet_counts.get(row['packet'], 0) + 1
            assert packet_counts == {'original': 60, 'conditional': 80, 'broader': 80, 'voice': 16}
            summaries[name]['packet_counts'] = packet_counts
            summaries[name]['every_native_and_governed_response_seal_valid'] = True
        elif name == 'parent-all-reviewed-objectives.protected.jsonl':
            assert len(rows) == 394
            assert len({row['batch_sha256'] for row in rows}) == 394
            phase_counts = {}
            coverage = {}
            for row in rows:
                assert math.isfinite(row['total_loss'])
                assert all(math.isfinite(value) for value in row['field_loss'].values())
                assert row['descriptive_training_targets_not_independent_qualification'] is True
                phase_counts[row['phase']] = phase_counts.get(row['phase'], 0) + 1
                for field, count in row['available_counts'].items():
                    assert type(count) is int and count >= 0
                    coverage[field] = coverage.get(field, 0) + count
            assert phase_counts == {'n1': 178, 'n2': 216}
            summaries[name].update(phase_counts=phase_counts, available_fact_counts=coverage)
        else:
            assert len(rows) == 0

    no_updates = all(not (output/phase/name).exists() for phase in ('n1', 'n2') for name in ('current-authorization.protected.json', 'final-checkpoint', 'successor-lineage.protected.json'))
    assert no_updates
    summary = {
        'schema': 'alice-personality-failed-cumulative-readback-v1',
        'job_id': int(os.environ['SLURM_JOB_ID']), 'owner': 'mxrayan',
        'source_job_id': 577303, 'source_state': 'FAILED_BEFORE_OPTIMIZER',
        'isolation': isolation, 'namespace_sha256': GUARD,
        'worker_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_worker_sha256': plan['worker_sha256'],
        'source_plan_sha256': sha256(plan_raw).hexdigest(),
        'failure_file_sha256': sha256(failure_raw).hexdigest(),
        'known_report_pointer_representation_refusal_confirmed': True,
        'retained_measurements': summaries, 'new_private_gradient_steps': 0,
        'parent_model_state_sha256': plan['parent_model_state_sha256'],
        'no_phase_authorization_or_checkpoint_created': no_updates,
        'baseline_fit_review_complete': False, 'n0_approved': False,
        'personality_approved': False, 'private_payload_exported': False,
    }
    summary['receipt_sha256'] = sha256(canonical(summary)).hexdigest()
    destination = ROOT/'jobs'/(NAME+'.json')
    with destination.open('xb') as stream:
        stream.write(canonical(summary)+b'\n')
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == '__main__':
    path = Path(os.environ['PERSONALITY_REPO_ROOT'])/'scripts/eipm/n1/enter_private_network_namespace.py'
    assert sha256(path.read_bytes()).hexdigest() == GUARD
    spec = importlib.util.spec_from_file_location('owner_readback_namespace', path)
    namespace = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = namespace
    spec.loader.exec_module(namespace)
    try:
        run(namespace)
    except BaseException as error:
        import traceback
        print(json.dumps({'state': 'READBACK_REFUSED', 'exception_type': type(error).__name__, 'public_code_frames': [{'module': Path(frame.filename).name, 'line': frame.lineno, 'function': frame.name} for frame in traceback.extract_tb(error.__traceback__)]}), flush=True)
        raise SystemExit(2)
