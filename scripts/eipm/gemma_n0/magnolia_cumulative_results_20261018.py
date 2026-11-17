"""Owner-isolated read-only comparison of the complete retained 577714 result."""
import importlib.util
import json
import math
import os
import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path

ROOT = Path('/homes/01/mxrayan/rayan-compute')
REV = '333fcd0c8239fc81a6459b13d53bbe592b33fda6'
GUARD = 'f5822c3657b246a5b232f42bddf93b8026bda0b6a7e573835c1cd8c90136c1e4'
NAME = 'rayan-cumulative-results-20261018'
PLAN_SHA = '57145c3e0dab55644f55d0d4e3d533a0241bfd84d55821fcc3d711a059a84ee7'
SUMMARY_SHA = '6e92df79abe4a13a17d6c4e92bb275db8250ad5e1d3de79d4c2006a1388e4533'
ANALYZER_SHA = '8e0982e758ad81d735c474101fdd2e271d4dee3e124d33fc7f1e129e0c50b6c7'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def seal(value, field='receipt_sha256'):
    assert sha256(canonical({k: v for k, v in value.items() if k != field})).hexdigest() == value[field]
    return value


def run(namespace):
    isolation = namespace._p2_guard()  # FIRST, before protected path/stat/hash/read.
    os.umask(0o077)
    sys.dont_write_bytecode = True
    code = namespace._bounded_path(os.environ['PERSONALITY_REPO_ROOT'], ROOT, file=False)
    namespace._clean_code(code, REV)
    held = {}

    def read(path, expected=None, limit=256*1024**2):
        path = namespace._bounded_path(str(path), ROOT, file=True)
        assert path.stat().st_mode & 0o777 == 0o600 and path.stat().st_size <= limit
        raw = path.read_bytes()
        digest = sha256(raw).hexdigest()
        if expected is not None:
            assert digest == expected
        held[path] = digest
        return raw

    def jread(path, expected=None):
        return json.loads(read(path, expected))

    def rows(path, expected):
        return [json.loads(line) for line in read(path, expected).splitlines()]

    summary = seal(jread(ROOT/'jobs/rayan-cumulative-learning-r3-20261015.json', SUMMARY_SHA))
    plan = jread(ROOT/'rayan-personality/private-inputs/rayan-cumulative-learning-r3-20261015/owner-cumulative-run.protected.json', PLAN_SHA)
    assert summary['owner_run_plan_sha256'] == PLAN_SHA and summary['phase_updates'] == {'n1': 178, 'n2': 216}
    output = namespace._bounded_path(str(ROOT/'rayan-personality/private-compile/rayan-cumulative-learning-r3-577714'), ROOT, file=False)
    original_plan = jread(ROOT/'rayan-personality/private-inputs/rayan-primary-governing-single-thread-20261008/owner-connected-run.protected.json',
                         '09ea1efff8dcb62e7c0b391ec8fe00a30204691aea93f01cf84a72625ffb2f2b')
    feature_ref = original_plan['completed_features']
    config_mapping = jread(Path(feature_ref['directory'])/'model-config.protected.json', feature_ref['file_sha256']['model-config.protected.json'])
    overlay = namespace._bounded_path(str(ROOT/'rayan-personality/runs/public-forward-576598/processor-deps'), ROOT, file=False)
    sys.path[:0] = [str(overlay), str(code)]
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from src.alice_personality.identity.contracts import IdentityModelConfig
    from src.alice_personality.identity.consumer import CheckpointPin
    from src.alice_personality.teaching.trainer import model_fingerprint
    loaded = {}
    for name in ('governed_response.py', 'cumulative_successor.py', 'conditional_successor_report.py'):
        path = ROOT/'jobs'/('rayan-20261015-r3-'+name)
        read(path, plan['helper_sha256'][name])
        loaded[name] = module('scripts.eipm.gemma_n0.'+name[:-3], path)
    analyzer_path = ROOT/'jobs'/(NAME+'-analysis.py')
    read(analyzer_path, ANALYZER_SHA)
    analyzer = module('saved_cumulative_result_analysis', analyzer_path)
    successor = loaded['cumulative_successor.py']
    parents = tuple(CheckpointPin(**item) for item in plan['parent_checkpoints'])
    lineage_pins = tuple(successor.SuccessorPin(str(output/phase/'successor-lineage.protected.json'), digest)
        for phase, digest in zip(('n1', 'n2'), summary['successor_lineage_file_sha256']))
    for pin in (*parents, *lineage_pins):
        namespace._bounded_path(pin.path, ROOT, file=isinstance(pin, successor.SuccessorPin))
    print(json.dumps({'stage': 'complete_native_checkpoint_lineage_readback', 'source_job': 577714}), flush=True)
    construction = successor.CumulativeSuccessor.load(IdentityModelConfig(**config_mapping), parents, successors=lineage_pins)
    assert model_fingerprint(construction.model) == summary['model_state_sha256']
    verified = construction._verify()
    assert len(verified['valid']) == summary['cumulative_valid_batch_count'] == 394
    assert [p.manifest_sha256 for p in verified['checkpoints']][-2:] == [summary['phase_checkpoint_manifest_sha256'][phase] for phase in ('n1', 'n2')]
    print(json.dumps({'stage': 'native_checkpoints_lineage_and_394_batches_verified'}), flush=True)

    objective_rows, response_rows = {}, {}
    all_response_sha = {}
    for tag, root in (('parent', Path(summary['baseline_full_governed_responses']['source_directory'])), ('successor', output)):
        meta = summary['baseline_full_governed_responses' if tag == 'parent' else 'successor_full_governed_responses']
        objective_rows[tag] = rows(root/(tag+'-all-reviewed-objectives.protected.jsonl'), meta['all_reviewed_objective_file_sha256'])
        values = rows(root/(tag+'-governed-responses.protected.jsonl'), meta['file_sha256'])
        expected_model = summary['parent_model_state_sha256'] if tag == 'parent' else summary['model_state_sha256']
        expected_checkpoints = [p.manifest_sha256 for p in (parents if tag == 'parent' else verified['checkpoints'])]
        index = {}
        for row in values:
            response = seal(row['response']); selected = seal(response['preference_selection'])
            assert row['document_sha256'] == selected['document_sha256']
            assert response['packet_sha256'] == selected['packet_sha256']
            assert selected['model_state_sha256'] == expected_model
            assert selected['checkpoint_manifest_sha256'] == expected_checkpoints
            assert response['qualification'] == 'UNQUALIFIED' and response['acceptance_authority'] is False
            if tag == 'successor':
                assert response['calibration_applicability']['applicable'] is False
                assert response['calibration_applicability']['fresh_outcome_calibration_required'] is True
                assert response['successor_lineage']['successor_receipt_file_sha256'] == summary['successor_lineage_file_sha256']
            key = (row['packet'], row['document_id'])
            assert key not in index
            index[key] = row
        assert len(values) == len(index) == 236
        assert Counter(row['packet'] for row in values) == {'original': 60, 'conditional': 80, 'broader': 80, 'voice': 16}
        assert len(objective_rows[tag]) == 394
        assert Counter(row['phase'] for row in objective_rows[tag]) == {'n1': 178, 'n2': 216}
        response_rows[tag] = index
        all_response_sha[tag] = meta['file_sha256']
    assert response_rows['parent'].keys() == response_rows['successor'].keys()
    changes = {}
    for key, before in response_rows['parent'].items():
        after = response_rows['successor'][key]
        assert before['document_sha256'] == after['document_sha256']
        left, right = before['response'], after['response']
        for binding in ('frame_sha256', 'feature_binding_sha256', 'closed_session_sha256'):
            assert left['preference_selection'][binding] == right['preference_selection'][binding]
        group = changes.setdefault(key[0], {'documents': 0, 'before_status': Counter(), 'after_status': Counter(),
            'changed_preference_sets': 0, 'changed_acceptable_sets': 0, 'changed_decision_packets': 0})
        group['documents'] += 1
        group['before_status'][left['status']] += 1; group['after_status'][right['status']] += 1
        group['changed_preference_sets'] += left['preferred_candidate_ids'] != right['preferred_candidate_ids']
        group['changed_acceptable_sets'] += left['acceptable_candidate_ids'] != right['acceptable_candidate_ids']
        group['changed_decision_packets'] += left['packet_sha256'] != right['packet_sha256']

    diagnostic_summary = seal(jread(ROOT/'jobs/rayan-n0-authentic-20261011.json'))
    assert diagnostic_summary['receipt_sha256'] == 'f0164851db42959919095c92ea020ef49471a1590e50670b0924ea9b95ca0624'
    diagnostic_root = ROOT/'rayan-personality/private-inputs/rayan-n0-semantic-20261011'
    # Resolve the exact input path used by the immutable original worker.
    diagnostic_index = jread(diagnostic_root/'n0-probe-index.protected.json', diagnostic_summary['protected_input_index_sha256'])
    targets = jread(diagnostic_root/'primary-directional-targets.protected.json', diagnostic_index['files']['primary-directional-targets.protected.json'])
    diagnostic = {}
    for tag in ('parent', 'successor'):
        values = rows(output/(tag+'-original-conditional-failures.protected.jsonl'), summary['original_conditional_'+tag+'_readback']['file_sha256'])
        assert len(values) == len({row['document_id'] for row in values}) == 10
        groups = {}
        for row in values:
            response = seal(row['response']); selected = seal(response['preference_selection'])
            assert selected['model_state_sha256'] == summary['parent_model_state_sha256' if tag == 'parent' else 'model_state_sha256']
            target = targets[row['document_id']]
            assert target['document_sha256'] == row['document_sha256'] == selected['document_sha256']
            regions = target['review']['candidate_regions']
            target_rows = [{'field': field, 'facts': [{'pointers': [region['candidate_id']], 'available': True,
                            'value': int(region[key] == 'POSITIVE')} for region in regions]}
                           for field, key in (('co_valid_probabilities', 'co_valid_region'), ('contraindications', 'contraindication_region'))]
            fit = loaded['conditional_successor_report.py'].reviewed_region_fit(response['decision'], target_rows)
            assert fit == row['original_region_fit']
            group = groups.setdefault(row['comparison_kind'], {'documents': 0, 'fields': {}})
            group['documents'] += 1
            for field in ('co_valid_probabilities', 'contraindications'):
                metric = group['fields'].setdefault(field, {'comparisons': 0, 'errors': 0, 'absolute_error_sum': 0.})
                metric['comparisons'] += fit[field]['comparisons']
                metric['errors'] += fit[field]['errors']
                metric['absolute_error_sum'] += fit[field]['mean_absolute_error'] * fit[field]['comparisons']
        assert sum(m['comparisons'] for g in groups.values() for m in g['fields'].values()) == 60
        assert sum(m['errors'] for g in groups.values() for m in g['fields'].values()) == summary['original_conditional_'+tag+'_readback']['binary_region_errors']
        diagnostic[tag] = groups

    training = {}
    for phase, manifest in zip(('n1', 'n2'), verified['manifests'][-2:]):
        assert manifest['phase'] == phase and manifest['step'] == summary['phase_updates'][phase]
        events = manifest['events']
        assert len(events) == manifest['step']
        norms = [e['gradient_norm_before_clip'] for e in events]
        assert all(type(x) in (int, float) and math.isfinite(x) for x in norms)
        training[phase] = {'updates': manifest['step'], 'recipe': manifest['recipe'],
            'gradient_norm_min': min(norms), 'gradient_norm_max': max(norms),
            'gradient_norm_mean': math.fsum(norms)/len(norms),
            'effective_target_coverage': dict(sum((Counter(e['effective_target_coverage']) for e in events), Counter()))}
    metrics = analyzer.compare_objectives(objective_rows['parent'], objective_rows['successor'])
    for path, digest in held.items():
        assert sha256(path.read_bytes()).hexdigest() == digest
    namespace._clean_code(code, REV)
    result = {'schema': 'alice-personality-complete-cumulative-readback-v1', 'job_id': int(os.environ['SLURM_JOB_ID']),
        'source_job_id': 577714, 'owner': 'mxrayan', 'isolation': isolation, 'namespace_sha256': GUARD,
        'worker_sha256': sha256(Path(__file__).read_bytes()).hexdigest(), 'source_summary_sha256': SUMMARY_SHA,
        'source_plan_sha256': PLAN_SHA, 'source_response_sha256': all_response_sha,
        'complete_native_lineage_verified': True, 'valid_batch_count': len(verified['valid']),
        'all_472_response_seals_verified': True, 'objective_comparison': metrics, 'response_changes': changes,
        'original_diagnostic_recomputed': diagnostic, 'training': training,
        'new_private_gradient_steps': 0, 'n0_approved': False, 'personality_approved': False,
        'independent_qualification': False, 'private_payload_exported': False}
    result['receipt_sha256'] = sha256(canonical(result)).hexdigest()
    with (ROOT/'jobs'/(NAME+'.json')).open('xb') as stream:
        stream.write(canonical(result)+b'\n')
    print(json.dumps({'stage': 'complete_readback_saved', 'receipt_sha256': result['receipt_sha256']}), flush=True)


if __name__ == '__main__':
    path = Path(os.environ['PERSONALITY_REPO_ROOT'])/'scripts/eipm/n1/enter_private_network_namespace.py'
    assert sha256(path.read_bytes()).hexdigest() == GUARD
    namespace = module('owner_result_namespace', path)
    try:
        run(namespace)
    except BaseException as error:
        import traceback
        print(json.dumps({'state': 'RESULT_READBACK_REFUSED', 'exception_type': type(error).__name__,
            'public_code_frames': [{'module': Path(frame.filename).name, 'line': frame.lineno, 'function': frame.name}
                for frame in traceback.extract_tb(error.__traceback__)]}), flush=True)
        raise SystemExit(2)
