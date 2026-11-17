"""Zero-update native parent -> retained N1 -> final N2 phase attribution.

Complete source replay and all reviewed objectives; never a model correction.
The scalar hook observes the actual N2 forward without modifying its outputs.
"""
import gc
import importlib.util
import json
import math
import os
import stat
import sys
from collections import Counter
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

ROOT = Path('/homes/01/mxrayan/rayan-compute')
REV = '333fcd0c8239fc81a6459b13d53bbe592b33fda6'
GUARD = 'f5822c3657b246a5b232f42bddf93b8026bda0b6a7e573835c1cd8c90136c1e4'
NAME = 'rayan-cumulative-phase-attribution-20261018'
ORIGINAL_NAME = 'rayan-cumulative-learning-r3-20261015'
PLAN_SHA = '57145c3e0dab55644f55d0d4e3d533a0241bfd84d55821fcc3d711a059a84ee7'
SUMMARY_SHA = '6e92df79abe4a13a17d6c4e92bb275db8250ad5e1d3de79d4c2006a1388e4533'
READBACK_SHA = 'd1eb676108a800f81e603ea45c7ba0b77298d88984ee0f4966cf68789efd5839'
READBACK_RECEIPT = '12ee55ddefe518a5f817023109aa17e32725e65ed6ef699e6e9e5af89eac3949'
ANALYZER_SHA = '8e0982e758ad81d735c474101fdd2e271d4dee3e124d33fc7f1e129e0c50b6c7'
KINDS = ('original', 'decisive_condition', 'paraphrase', 'irrelevant_social', 'neutral_extra_text')
SCALARS = {'co_valid_probabilities': (1, 'co_valid_region'), 'contraindications': (4, 'contraindication_region')}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def write(path, value):
    with path.open('xb') as stream:
        stream.write(canonical(value) + b'\n')
    return sha256(path.read_bytes()).hexdigest()


def verify_seal(value):
    if sha256(canonical({k: v for k, v in value.items() if k != 'receipt_sha256'})).hexdigest() != value['receipt_sha256']:
        raise ValueError('retained receipt seal changed')
    return value


def capture_native_response(respond, consumer, document, produced, session, *, expected_packet_sha256=None):
    """Observe selection and governing-verification forwards without changing either.

    DecisionConsumer runs the selected-response forward. The unchanged governing
    wrapper then runs another forward and verifies its full packet against that
    selection. Both observed packets and raw N2 scalars must be exactly equal.
    """
    import torch
    from src.alice_personality.identity.codec import fingerprint
    packets, scalars = [], []
    def packet_hook(_module, _inputs, output):
        packets.append(output)
    def scalar_hook(_module, _inputs, output):
        scalars.append(output['scalar'].detach().clone())
    packet_handle = consumer.model.register_forward_hook(packet_hook)
    scalar_handle = consumer.model.n2.register_forward_hook(scalar_hook)
    try:
        with torch.no_grad():
            response = respond(consumer, document, produced, session)
    finally:
        scalar_handle.remove()
        packet_handle.remove()
    if len(packets) != 2 or len(scalars) != 2:
        raise ValueError('native capture requires both selection and governing-verification forwards')
    packet, scalar = packets[-1], scalars[-1]
    if scalar.ndim != 3 or scalar.shape[0] != 1 or scalar.shape[-1] != 6 or not bool(torch.isfinite(scalar).all()):
        raise ValueError('actual N2 scalar output is invalid')
    if any(fingerprint(value) != response['packet_sha256'] for value in packets):
        raise ValueError('actual native packet differs from its governing response')
    if not torch.equal(scalars[0], scalars[1]):
        raise ValueError('selection and governing-verification raw scalars differ')
    if expected_packet_sha256 is not None and response['packet_sha256'] != expected_packet_sha256:
        raise ValueError('endpoint packet reproduction failed; no phase conclusion is valid')
    return response, packet, scalar.cpu().tolist()


def capture_diagnostic_response(respond, consumer, document, produced, session, *, expected_packet_sha256=None):
    response, _, scalar = capture_native_response(respond, consumer, document, produced, session,
        expected_packet_sha256=expected_packet_sha256)
    return response, scalar


def verify_n1_phase_invariance(after_n1, successor, intermediate_auxiliary_sha256, final_auxiliary_sha256):
    """N2-only updates must preserve every matched N1 objective and its auxiliary state."""
    if intermediate_auxiliary_sha256 != final_auxiliary_sha256:
        raise ValueError('N1 auxiliary state changed during the N2-only phase')
    def index(rows):
        selected = {}
        for row in rows:
            if row['phase'] != 'n1':
                continue
            key = (row['packet'], row['case_id'], row['batch_sha256'])
            if key in selected:
                raise ValueError('duplicate N1 objective row')
            selected[key] = row
        return selected
    left, right = index(after_n1), index(successor)
    if not left or left != right:
        raise ValueError('matched N1 objective rows changed during the N2-only phase')
    return {'n1_case_count':len(left), 'all_n1_objective_rows_exact':True,
            'auxiliary_state_sha256':intermediate_auxiliary_sha256}


def summarize_diagnostics(stages, targets):
    """Aggregate matched region transitions and within-family conditional deltas."""
    def require(condition, reason):
        if not condition:
            raise ValueError(reason)
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)
    def mean(values):
        return math.fsum(values) / len(values) if values else None
    def movements(values, prefix):
        return {'mean_'+prefix: mean(values), 'mean_absolute_'+prefix: mean([abs(v) for v in values]),
                'max_absolute_'+prefix: max(map(abs, values)) if values else None}
    require(set(stages) == {'parent', 'after_n1', 'successor'}, 'all three exact phases are required')
    indexed = {}
    for stage, rows in stages.items():
        require(len(rows) == 10, 'complete ten-context diagnostic required')
        index = {}
        for row in rows:
            name = row['document_id']
            require(name not in index and name in targets, 'duplicate or unknown diagnostic context')
            require(row['comparison_kind'] in KINDS, 'unreviewed comparison kind')
            target = targets[name]
            require(row['document_sha256'] == target['document_sha256'], 'diagnostic target/document mismatch')
            decision = row['response']['decision']
            ids = decision['candidate_ids']
            require(len(ids) == 1 and len(ids[0]) == 3 and len(set(ids[0])) == 3, 'exact native candidate row required')
            regions = target['review']['candidate_regions']
            require([r['candidate_id'] for r in regions] == list(ids[0]), 'diagnostic candidate pointer mismatch')
            raw = row['raw_n2_scalar_logits']
            require(len(raw) == 1 and len(raw[0]) == 3 and all(len(v) == 6 and all(finite(x) for x in v) for v in raw[0]), 'nonfinite or incomplete raw N2 scalar row')
            for field, (_, region_key) in SCALARS.items():
                require(len(decision[field]) == 1 and len(decision[field][0]) == 3
                    and all(finite(v) and 0 <= v <= 1 for v in decision[field][0]), 'invalid native probabilities')
                require(all(r[region_key] in ('POSITIVE', 'NEGATIVE') for r in regions), 'unknown binary target region')
            index[name] = row
        families = {}
        for row in rows:
            kinds = families.setdefault(row['parent_document_id'], [])
            kinds.append(row['comparison_kind'])
        require(len(families) == 2 and all(sorted(k) == sorted(KINDS) for k in families.values()), 'matched two-family diagnostic is incomplete')
        indexed[stage] = index
    reference = indexed['parent']
    for index in indexed.values():
        require(index.keys() == reference.keys(), 'phase diagnostic context coverage differs')
        for name, row in index.items():
            require(all(row[key] == reference[name][key] for key in ('document_sha256','parent_document_id','comparison_kind')), 'phase diagnostic binding changed')
            require(row['response']['decision']['candidate_ids'] == reference[name]['response']['decision']['candidate_ids'], 'phase candidate pointers changed')
    result = {'independent_qualification': False, 'regions': {}, 'transitions': {}, 'context_deltas': {},
              'raw_logit_source': 'actual_n2_forward_scalar_slots_1_and_4_before_n3_temperature'}
    for stage, index in indexed.items():
        result['regions'][stage], result['context_deltas'][stage] = {}, {}
        for kind in KINDS:
            rows = [row for row in index.values() if row['comparison_kind'] == kind]
            result['regions'][stage][kind] = {}
            if kind != 'original': result['context_deltas'][stage][kind] = {}
            for field, (slot, region_key) in SCALARS.items():
                errors, absolute = 0, []
                deltas, raw_deltas, signed, raw_signed = [], [], [], []
                for row in rows:
                    p = row['response']['decision'][field][0]
                    desired = [int(r[region_key] == 'POSITIVE') for r in targets[row['document_id']]['review']['candidate_regions']]
                    errors += sum((v >= .5) != bool(y) for v, y in zip(p, desired))
                    absolute.extend(abs(v-y) for v,y in zip(p,desired))
                    if kind == 'original': continue
                    original = next(r for r in index.values() if r['parent_document_id'] == row['parent_document_id'] and r['comparison_kind'] == 'original')
                    require(original['response']['decision']['candidate_ids'] == row['response']['decision']['candidate_ids'], 'within-family candidate pointers changed')
                    base_p = original['response']['decision'][field][0]
                    base_y = [int(r[region_key] == 'POSITIVE') for r in targets[original['document_id']]['review']['candidate_regions']]
                    for i, (value, base, y, old_y) in enumerate(zip(p, base_p, desired, base_y)):
                        delta = value-base
                        raw_delta = row['raw_n2_scalar_logits'][0][i][slot] - original['raw_n2_scalar_logits'][0][i][slot]
                        deltas.append(delta); raw_deltas.append(raw_delta)
                        if y != old_y:
                            signed.append(delta*(y-old_y)); raw_signed.append(raw_delta*(y-old_y))
                result['regions'][stage][kind][field] = {'comparisons':len(absolute), 'errors':errors, 'mean_absolute_error':mean(absolute)}
                if kind != 'original':
                    result['context_deltas'][stage][kind][field] = {'comparisons':len(deltas), **movements(deltas,'probability_delta'), **movements(raw_deltas,'raw_logit_delta'),
                        'changed_target_comparisons':len(signed), 'mean_changed_target_signed_probability_delta':mean(signed), 'mean_changed_target_signed_raw_logit_delta':mean(raw_signed)}
    for first, second in (('parent','after_n1'),('after_n1','successor'),('parent','successor')):
        result['transitions'][first+'_to_'+second] = groups = {}
        for kind in KINDS:
            groups[kind] = {}
            for field, (slot, region_key) in SCALARS.items():
                deltas, signed, raw_deltas = [], [], []
                fixed = broken = remaining_wrong = remaining_right = 0
                for name, left in indexed[first].items():
                    if left['comparison_kind'] != kind: continue
                    right = indexed[second][name]
                    desired = [int(r[region_key] == 'POSITIVE') for r in targets[name]['review']['candidate_regions']]
                    for i,(a,b,y) in enumerate(zip(left['response']['decision'][field][0],right['response']['decision'][field][0],desired)):
                        was, now = (a>=.5)==bool(y), (b>=.5)==bool(y)
                        fixed += not was and now; broken += was and not now
                        remaining_wrong += not was and not now; remaining_right += was and now
                        deltas.append(b-a); signed.append((2*y-1)*(b-a))
                        raw_deltas.append(right['raw_n2_scalar_logits'][0][i][slot]-left['raw_n2_scalar_logits'][0][i][slot])
                groups[kind][field] = {'comparisons':len(deltas),'wrong_to_right':fixed,'right_to_wrong':broken,
                    'remaining_wrong':remaining_wrong,'remaining_right':remaining_right,
                    **movements(deltas,'probability_change'), **movements(raw_deltas,'raw_logit_change'),
                    'mean_target_signed_probability_change':mean(signed)}
    return result


def run(namespace):
    isolation = namespace._p2_guard()  # FIRST: before any private path/stat/hash/read.
    os.umask(0o077)
    sys.dont_write_bytecode = True
    code = namespace._bounded_path(os.environ['PERSONALITY_REPO_ROOT'], ROOT, file=False)
    namespace._clean_code(code, REV)
    assert sha256(Path(namespace.__file__).read_bytes()).hexdigest() == GUARD
    overlay = namespace._bounded_path(str(ROOT/'rayan-personality/runs/public-forward-576598/processor-deps'), ROOT, file=False)
    sys.path[:0] = [str(overlay), str(code)]
    import torch
    import PIL
    import torchvision
    import transformers
    assert (torch.__version__, PIL.__version__, torchvision.__version__, transformers.__version__) == ('2.7.1+cu118', '11.3.0', '0.22.1+cu118', '5.17.0')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from src.alice_personality.n1.manifest_structure_audit import _PinnedFile
    from src.alice_personality.n1 import compiler, raw_inference_lineage as raw
    from src.alice_personality.identity.codec import CognitiveFrameDocument, fingerprint
    from src.alice_personality.identity.consumer import CheckpointPin
    from src.alice_personality.identity.feature_producer import FrozenFrameProducer
    from src.alice_personality.identity.feature_store import FrozenFeatureStore
    from src.alice_personality.gemma_n0.preparation import verify_prepared
    from src.alice_personality.teaching.cases import ReviewedTeachingCase
    from src.alice_personality.teaching.contracts import TrainingRecipe
    from src.alice_personality.teaching.materialize import materialize_case
    from src.alice_personality.teaching.lineage import verify_allocation, validate_allocation
    from src.alice_personality.teaching.objectives import IdentityObjectives
    from src.alice_personality.teaching.trainer import model_fingerprint
    from transformers import AutoProcessor
    helper = module('unchanged_cumulative_scientific_construction', code/'scripts/eipm/gemma_n0/construct_personality.py')
    overlay_sha = helper.inventory(overlay)
    output = ROOT/'rayan-personality/private-compile'/('rayan-cumulative-phase-attribution-'+os.environ['SLURM_JOB_ID'])
    output.mkdir(mode=0o700)
    output = namespace._bounded_path(str(output), ROOT, file=False)

    with ExitStack() as stack:
        held = []

        def pin(path, digest, limit=64*1024**2):
            path = namespace._bounded_path(str(path), ROOT, file=True)
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            value = stack.enter_context(_PinnedFile(path, digest, limit))
            held.append(value)
            value.stream.seek(0)
            return value

        def jpin(path, digest):
            return raw._json(pin(path, digest).stream.read())

        def lines(value):
            value.stream.seek(0)
            return [raw._json(line) for line in value.stream]

        plan_path = ROOT/'rayan-personality/private-inputs'/ORIGINAL_NAME/'owner-cumulative-run.protected.json'
        plan_sha = PLAN_SHA
        plan = jpin(plan_path, plan_sha)
        assert plan['schema'] == 'alice-personality-owner-cumulative-run-v1'
        assert plan['owner'] == 'mxrayan' and plan['purpose'] == 'development_construction'
        assert plan['private_gradient_authorized'] is True and plan['acceptance_authority'] is False
        assert plan['epochs'] == {'n1': 1, 'n2': 1} and plan['expected_updates'] == {'n1': 178, 'n2': 216}
        assert plan['cpu_threads'] == 1 == int(os.environ['SLURM_CPUS_PER_TASK'])
        assert all(plan[key] is False for key in ('n3_updates_authorized', 'n0_approved', 'owner_fidelity_approved', 'full_personality_complete', 'flora_work'))
        worker_sha = sha256(Path(__file__).read_bytes()).hexdigest()
        assert sha256(namespace._bounded_path(str(ROOT/'jobs/rayan-cumulative-learning-r3-20261015.py'), ROOT, file=True).read_bytes()).hexdigest() == plan['worker_sha256']
        summary = verify_seal(jpin(ROOT/'jobs/rayan-cumulative-learning-r3-20261015.json', SUMMARY_SHA))
        readback = verify_seal(jpin(ROOT/'jobs/rayan-cumulative-results-20261018.json', READBACK_SHA))
        assert readback['receipt_sha256'] == READBACK_RECEIPT and readback['job_id'] == 578457
        assert readback['source_summary_sha256'] == SUMMARY_SHA and readback['source_plan_sha256'] == PLAN_SHA
        assert readback['complete_native_lineage_verified'] is True and readback['all_472_response_seals_verified'] is True
        assert readback['new_private_gradient_steps'] == 0 and readback['valid_batch_count'] == 394
        assert summary['job_id'] == 577714 and summary['owner_run_plan_sha256'] == PLAN_SHA
        assert summary['phase_updates'] == {'n1': 178, 'n2': 216}
        source_output = namespace._bounded_path(str(ROOT/'rayan-personality/private-compile/rayan-cumulative-learning-r3-577714'), ROOT, file=False)
        analyzer_path = namespace._bounded_path(str(ROOT/'jobs/rayan-cumulative-results-20261018-analysis.py'), ROOT, file=True)
        pin(analyzer_path, ANALYZER_SHA)
        analyzer = module('phase_attribution_pinned_analysis', analyzer_path)
        public_paths = {}
        loaded = {}
        # Preserve the old scientific package; inject only exact separately pinned helpers.
        for filename in ('governed_response.py', 'cumulative_successor.py', 'cumulative_bindings.py', 'connected_voice_references.py', 'n0_semantic_contrasts.py', 'conditional_successor_report.py', 'cumulative_recovery.py'):
            public_path = namespace._bounded_path(str(ROOT/'jobs'/('rayan-20261015-r3-'+filename)), ROOT, file=True)
            assert sha256(public_path.read_bytes()).hexdigest() == plan['helper_sha256'][filename]
            public_paths[filename] = public_path
            loaded[filename] = module('scripts.eipm.gemma_n0.'+filename[:-3], public_path)
        successor_api, binding_api = loaded['cumulative_successor.py'], loaded['cumulative_bindings.py']
        original_plan = jpin(ROOT/'rayan-personality/private-inputs/rayan-primary-governing-single-thread-20261008/owner-connected-run.protected.json', '09ea1efff8dcb62e7c0b391ec8fe00a30204691aea93f01cf84a72625ffb2f2b')
        selection = jpin(ROOT/'rayan-personality/private-inputs'/ORIGINAL_NAME/'case-selection.protected.json', plan['selection_file_sha256'])
        assert selection['parent_model_state_sha256'] == plan['parent_model_state_sha256']

        packets = {}
        for packet_name in ('original', 'conditional', 'broader', 'voice'):
            reference = plan['packets'][packet_name]
            directory = original_plan['packet_directory'] if packet_name == 'original' else reference['directory']
            root = namespace._bounded_path(directory, ROOT, file=False)
            index = jpin(root/'packet-index.protected.json', reference['index_sha256'])
            assert selection['packet_index_file_sha256'][packet_name] == reference['index_sha256']
            inputs = {name: pin(root/name, digest) for name, digest in index['files'].items()}
            artifacts = index['artifact_sha256']
            protocol = jpin(root/'annotation-protocol.protected.json', artifacts['annotation_protocol'])
            review = jpin(root/('primary-review.protected.json' if packet_name in ('broader', 'voice') else 'author-review.protected.json'), artifacts['review_record'])
            assert protocol['other_teacher_used'] is False and protocol['source_complete_authoring_sha256'] == helper.SOURCE_CONTEXT_SHA
            assert review['annotation_protocol_sha256'] == artifacts['annotation_protocol']
            if 'teacher_spec_sha256' in protocol:
                jpin(root/'teacher-spec.protected.json', protocol['teacher_spec_sha256'])
            allocation = raw._json(inputs['construction-allocation.protected.json'].stream.read())
            validate_allocation(allocation)
            doc_rows = lines(inputs['target-free-documents.protected.jsonl'])
            assert all(set(row) == {'document_id', 'split', 'document'} for row in doc_rows)
            all_documents = {row['document_id']: CognitiveFrameDocument.from_mapping(row['document']) for row in doc_rows}
            assert len(all_documents) == len(doc_rows)
            assert {name: doc.sha256 for name, doc in all_documents.items()} == index['document_sha256']
            case_rows = lines(inputs['reviewed-teaching-cases.protected.jsonl'])
            cases = {case.case_id: case for case in map(ReviewedTeachingCase.from_mapping, case_rows)}
            assert len(cases) == len(case_rows)
            assert {name: case.sha256 for name, case in cases.items()} == index['case_sha256']
            assert all(case.split == 'TRAIN' and case.phase in ('n1', 'n2') and case.source_class == 'governed_identity_targets' and dict(case.artifact_sha256) == artifacts for case in cases.values())
            documents = all_documents if packet_name == 'original' else {row['document_id']: all_documents[row['document_id']] for row in doc_rows if row['split'] == 'TRAIN'}
            assert all(case.document.sha256 in {doc.sha256 for doc in documents.values()} for case in cases.values())
            packets[packet_name] = dict(index=index, cases=cases, documents=documents, all_documents=all_documents, doc_rows=doc_rows, artifacts=artifacts, protocol=protocol, review=review, allocation=allocation)
        original = packets['original']
        assert len(original['cases']) == 80 and len(original['documents']) == 60
        assert [row for row in original['doc_rows'] if row['split'] == 'CALIBRATION_TRAIN'] == [row for row in packets['conditional']['doc_rows'] if row['split'] == 'CALIBRATION_TRAIN']
        store, config, session, bindings, profile = helper.restore_completed_features(original_plan['completed_features'], documents=original['documents'], source_package_sha256=original['artifacts']['source_package'], prepared_receipt_file_sha256=original['artifacts']['prepared_receipt'], read_pinned_json=jpin, bounded_directory=lambda path: namespace._bounded_path(path, ROOT, file=False))
        original.update(store=store, session=session, bindings=bindings, replay=FrozenFrameProducer.replay(store, config, session, expected_session_sha256=session['receipt_sha256']))
        checkpoints = tuple(CheckpointPin(**value) for value in plan['parent_checkpoints'])
        for checkpoint in checkpoints:
            namespace._bounded_path(checkpoint.path, ROOT, file=False)
        construction = successor_api.CumulativeSuccessor.load(config, checkpoints, device='cpu')
        assert model_fingerprint(construction.model) == plan['parent_model_state_sha256']
        verified = construction._verify()
        manifests = verified['manifests']
        source_artifacts = manifests[-1]['binding']['artifacts']
        common = {key: original['artifacts'][key] for key in ('source_package', 'compiled_receipt', 'prepared_receipt')}
        assert all(all(packet['artifacts'][key] == value for key, value in common.items()) for packet in packets.values())
        for key, value in common.items():
            assert source_artifacts[key]['file_sha256'] == value
            pin(source_artifacts[key]['path'], value, raw.MAX_ARCHIVE_BYTES if key == 'source_package' else 64*1024**2)
        compiled = namespace._bounded_path(helper.COMPILED, ROOT, file=False)
        compiled_receipt = compiler.verify_compiled(compiled, expected_source_archive_sha256=common['source_package'], expected_receipt_sha256=helper.COMPILE_SEAL)
        originals_list = lines(pin(helper.SOURCE_CONTEXT, helper.SOURCE_CONTEXT_SHA))
        originals = {row['draft_workitem']['record_id']: row for row in originals_list}
        assert len(originals) == len(originals_list) == 2556
        family_rows = [raw._json(line) for line in (compiled/'split_families.jsonl').read_bytes().splitlines()]
        voice_families = loaded['connected_voice_references.py'].connected_voice_reference_families(family_rows, originals, expected_voice_count=96)
        for name, packet in packets.items():
            for document in packet['all_documents'].values():
                helper.verify_reference_families(document, packet['allocation'], voice_families if name == 'voice' else family_rows, originals, packet['protocol'])
            for case in packet['cases'].values():
                verify_allocation(case.lineage, tuple(entry.record for entry in case.document.sources), case.split, packet['allocation'])
        merged = binding_api.merge_allocations([packet['allocation'] for packet in packets.values()])
        allocation_path = output/'construction-allocation.protected.json'
        allocation_sha = write(allocation_path, merged)
        for name in ('conditional', 'broader', 'voice'):
            packet = packets[name]
            reference = plan['completed_features'][name]
            feature_summary = jpin(reference['summary_path'], reference['summary_file_sha256'])
            assert sha256(canonical({k: v for k, v in feature_summary.items() if k != 'receipt_sha256'})).hexdigest() == feature_summary['receipt_sha256'] == reference['summary_receipt_sha256']
            assert feature_summary['state'] == 'REAL_FEATURES_MATERIALIZED_UNADMITTED' and feature_summary['private_gradient_steps'] == 0
            assert feature_summary['packet_index_file_sha256'] == plan['packets'][name]['index_sha256']
            root = namespace._bounded_path(reference['directory'], ROOT, file=False)
            session = jpin(root/'feature-session.protected.json', feature_summary['closed_session_file_sha256'])
            bindings = jpin(root/'frame-bindings.protected.json', feature_summary['frame_bindings_file_sha256'])
            store = FrozenFeatureStore.open(namespace._bounded_path(str(root/'feature-store'), ROOT, file=False), expected_manifest_sha256=feature_summary['feature_store_manifest_sha256'])
            assert store.manifest['source_package_sha256'] == common['source_package']
            assert session['prepared_receipt_file_sha256'] == common['prepared_receipt'] and session['receipt_sha256'] == feature_summary['closed_session_sha256']
            assert set(bindings) == set(packet['documents']) and len(packet['cases']) == feature_summary['materialized_case_count']
            assert {binding['binding_sha256'] for binding in bindings.values()} == set(session['frame_binding_sha256'])
            packet.update(store=store, session=session, bindings=bindings, replay=FrozenFrameProducer.replay(store, config, session, expected_session_sha256=session['receipt_sha256']))
        preparation = verify_prepared(source_artifacts['prepared_receipt']['path'])
        processor = AutoProcessor.from_pretrained(preparation['snapshot_path'], local_files_only=True, trust_remote_code=False)
        costs = {name: helper.token_cost_profile(list(packet['documents'].values()), processor.tokenizer, layers=49, width=3840, max_tokens=preparation['model_geometry']['max_position_embeddings']) for name, packet in packets.items()}
        estimate = 48*1024**3 + 12*max(cost['largest_padded_frame_value_bytes'] for cost in costs.values())
        helper.verify_memory_allocation(plan['memory_budget_bytes'], estimate, os.environ['SLURM_MEM_PER_NODE'])
        write(output/'token-cost-profiles.protected.json', costs)
        del processor
        gc.collect()

        lineages = tuple(successor_api.SuccessorPin(str(source_output/phase/'successor-lineage.protected.json'), digest)
            for phase, digest in zip(('n1', 'n2'), summary['successor_lineage_file_sha256']))
        assert len(lineages) == 2
        for value in lineages: pin(value.path, value.file_sha256)
        intermediate = successor_api.CumulativeSuccessor.load(config, checkpoints, successors=lineages[:1], device='cpu')
        final = successor_api.CumulativeSuccessor.load(config, checkpoints, successors=lineages, device='cpu')
        final_verified = final._verify()
        intermediate_verified = intermediate._verify()
        assert len(final_verified['valid']) == 394 and len(intermediate_verified['valid']) == 218
        assert model_fingerprint(final.model) == summary['model_state_sha256']
        assert [pin.manifest_sha256 for pin in final_verified['checkpoints']][-2:] == [summary['phase_checkpoint_manifest_sha256'][phase] for phase in ('n1', 'n2')]
        models = {'parent': construction, 'after_n1': intermediate, 'successor': final}
        model_states = {tag: model_fingerprint(value.model) for tag, value in models.items()}
        assert model_states['parent'] == summary['parent_model_state_sha256']
        assert all(torch.equal(value, intermediate.model.state_dict()[key]) for key, value in construction.model.state_dict().items() if key.startswith(('n2.', 'n3.')))
        assert all(torch.equal(value, final.model.state_dict()[key]) for key, value in intermediate.model.state_dict().items() if key.startswith(('n1.', 'n3.')))
        consumers = {tag: value.consumer() for tag, value in models.items()}

        def materialize(packet_name, case):
            packet = packets[packet_name]
            produced = packet['replay'].produce(case.document)
            expected = next(binding for key, binding in packet['bindings'].items() if packet['documents'][key].sha256 == case.document.sha256)
            assert produced.binding == expected
            verify_allocation(case.lineage, tuple(entry.record for entry in case.document.sources), 'TRAIN', merged)
            return materialize_case(case, produced, packet['session'], config, expected_case_sha256=case.sha256, expected_artifact_sha256=packet['artifacts'], family_splits=packet['allocation']).batch

        class OriginalBatches(Mapping):
            def __iter__(self): return iter(original['cases'])
            def __len__(self): return len(original['cases'])
            def __getitem__(self, key): return materialize('original', original['cases'][key])

        scope = binding_api.bind_parent_scope(selection, original['cases'], OriginalBatches(), manifests)
        assert len(scope['retained_parent_batches']) == 42 and len(scope['superseded_parent_batches']) == 38
        first_lineage = intermediate_verified['receipts'][0]
        assert set(scope['retained_parent_batches']) == set(first_lineage['retained_parent_batches'])
        assert scope['superseded_parent_batches'] == first_lineage['superseded_parent_batches']
        write(output/'parent-case-batch-binding.protected.json', scope)
        phase_data = {}
        selected_cases = set()
        for phase, manifest in zip(('n1', 'n2'), final_verified['manifests'][-2:]):
            assert manifest['phase'] == phase
            selected_rows = selection['phase_case_selection'][phase]
            selected = [(row['packet'], packets[row['packet']]['cases'][row['case_id']]) for row in selected_rows]
            assert len(selected) == plan['expected_updates'][phase]
            assert len({(name, case.case_id) for name, case in selected}) == len(selected)
            assert all(case.phase == phase and case.sha256 == row['case_sha256'] for (_, case), row in zip(selected, selected_rows))
            expected = {(name, case.case_id) for name, packet in packets.items() if name != 'original' for case in packet['cases'].values() if case.phase == phase}
            expected |= {('original', row['case_id']) for row in selection['retained_original_cases'] if row['phase'] == phase}
            assert {(name, case.case_id) for name, case in selected} == expected
            selected_cases |= expected
            target_pin = manifest['binding']['artifacts']['reviewed_targets']
            reviewed = jpin(target_pin['path'], target_pin['file_sha256'])
            assert reviewed['reviewed_case_artifact_sha256'] == {case.sha256: dict(case.artifact_sha256) for _, case in selected}
            assert len(reviewed['reviewed_batches']) == len(selected)
            arguments = dict(manifest['recipe']); arguments['betas'] = tuple(arguments['betas'])
            recipe = TrainingRecipe(**arguments); recipe.validate()
            assert recipe.phase == phase and recipe.adapt_n1 is False
            assert manifest['recipe'] == next(value['recipe'] for value in manifests if value['phase'] == phase)
            phase_data[phase] = {'recipe': recipe, 'reviewed': reviewed}
        assert len(selected_cases) == 394
        _, _, auxiliary = successor_api._checkpoint(config, intermediate_verified['checkpoints'][-1])
        objectives = IdentityObjectives(config.learned_width)
        objectives.load_state_dict(auxiliary, strict=True)
        objectives.requires_grad_(False); objectives.eval()

        # Already sealed endpoint bytes establish disk needs; no checkpoints are written.
        saved = {}
        for tag, root in (('parent', Path(summary['baseline_full_governed_responses']['source_directory'])), ('successor', source_output)):
            meta = summary['baseline_full_governed_responses' if tag == 'parent' else 'successor_full_governed_responses']
            saved[tag] = {'responses': pin(root/(tag+'-governed-responses.protected.jsonl'), meta['file_sha256'], 256*1024**2).stream.read(),
                          'objectives': pin(root/(tag+'-all-reviewed-objectives.protected.jsonl'), meta['all_reviewed_objective_file_sha256']).stream.read(), 'meta': meta}
        output_budget = 2*sum(len(value[key]) for value in saved.values() for key in ('responses','objectives')) + 64*1024**2
        disk = os.statvfs(output)
        assert disk.f_bavail*disk.f_frsize >= output_budget
        print(json.dumps({'stage':'all_native_phases_and_original_case_scope_verified', 'new_gradients':0, 'documents':236, 'selected_cases':394}), flush=True)

        expected_documents, expected_batches, measured = {}, {}, []
        response_path = output/'after-n1-governed-responses.protected.jsonl'
        objective_path = output/'after-n1-all-reviewed-objectives.protected.jsonl'
        documents_measured = 0
        with response_path.open('xb') as response_stream, objective_path.open('xb') as objective_stream:
            for packet_name, packet in packets.items():
                for name, document in packet['documents'].items():
                    produced = packet['replay'].produce(document)
                    assert produced.binding == packet['bindings'][name]
                    binding = produced.binding
                    expected_documents[(packet_name,name)] = {'document_sha256':document.sha256, 'frame_sha256':binding['frame_sha256'], 'feature_binding_sha256':binding['binding_sha256'], 'closed_session_sha256':packet['session']['receipt_sha256']}
                    response, decision, _ = capture_native_response(successor_api.respond_with_governance, consumers['after_n1'], document, produced, packet['session'])
                    response_stream.write(canonical({'packet':packet_name, 'document_id':name, 'document_sha256':document.sha256, 'response':response})+b'\n')
                    with torch.no_grad():
                        for case in packet['cases'].values():
                            if case.document.sha256 != document.sha256 or (packet_name,case.case_id) not in selected_cases: continue
                            verify_allocation(case.lineage, tuple(entry.record for entry in case.document.sources), 'TRAIN', merged)
                            batch = materialize_case(case, produced, packet['session'], config, expected_case_sha256=case.sha256, expected_artifact_sha256=packet['artifacts'], family_splits=packet['allocation']).batch
                            digest = batch.fingerprint()
                            reviewed = phase_data[case.phase]['reviewed']['reviewed_batches']
                            assert digest not in expected_batches and digest not in scope['superseded_parent_batches']
                            assert reviewed[digest] == {'split':'TRAIN', 'case_ids':list(batch.case_ids), 'source_family_ids':[list(value) for value in batch.source_family_ids], 'construction_lineage_sha256':fingerprint(batch.lineage)}
                            expected_batches[digest] = {'packet':packet_name, 'case_id':case.case_id, 'case_sha256':case.sha256, 'phase':case.phase}
                            total, losses, coverage = objectives(decision, batch, loss_weights=phase_data[case.phase]['recipe'].loss_weights)
                            assert bool(torch.isfinite(total)) and all(bool(torch.isfinite(value)) for value in losses.values())
                            row = {**expected_batches[digest], 'batch_sha256':digest, 'total_loss':float(total), 'field_loss':{field:float(value) for field,value in losses.items()}, 'available_counts':coverage, 'descriptive_training_targets_not_independent_qualification':True}
                            objective_stream.write(canonical(row)+b'\n'); measured.append(row)
                            del batch, total, losses
                    documents_measured += 1
                    del produced, response, decision
                    gc.collect()
                    if documents_measured % 10 == 0:
                        print(json.dumps({'stage':'after_n1_native_replay', 'documents_completed':documents_measured, 'documents_total':236, 'objectives_completed':len(measured), 'objectives_total':394, 'new_gradients':0}), flush=True)
        assert documents_measured == 236 and len(measured) == len(expected_batches) == 394
        assert set(expected_batches) == set(final_verified['valid'])
        for phase, data in phase_data.items():
            assert {digest for digest,row in expected_batches.items() if row['phase'] == phase} == set(data['reviewed']['reviewed_batches'])
        objective_rows = {'after_n1':measured}
        for tag, values in saved.items():
            meta = values['meta']
            loaded['cumulative_recovery.py'].recover_parent_measurements(values['responses'], values['objectives'], response_sha256=meta['file_sha256'], objective_sha256=meta['all_reviewed_objective_file_sha256'], expected_documents=expected_documents, expected_batches=expected_batches, parent_model_sha256=model_states[tag], parent_checkpoint_sha256=[value.manifest_sha256 for value in models[tag]._verify()['checkpoints']])
            objective_rows[tag] = [raw._json(line) for line in values['objectives'].splitlines()]
        objective_comparison = {left+'_to_'+right:analyzer.compare_objectives(objective_rows[left],objective_rows[right]) for left,right in (('parent','after_n1'),('after_n1','successor'),('parent','successor'))}
        assert objective_comparison['parent_to_successor'] == readback['objective_comparison']
        n1_phase_control = verify_n1_phase_invariance(objective_rows['after_n1'], objective_rows['successor'],
            fingerprint(auxiliary), final_verified['receipts'][-1]['objective_final_state_sha256'])
        assert n1_phase_control['n1_case_count'] == 178
        del saved
        gc.collect()

        def sealed(path, expected):
            path = namespace._bounded_path(str(path), ROOT, file=True)
            value = jpin(path, sha256(path.read_bytes()).hexdigest())
            verify_seal(value)
            assert value['receipt_sha256'] == expected
            return value

        diagnostic_summary = sealed(ROOT/'jobs/rayan-n0-authentic-20261011.json', 'f0164851db42959919095c92ea020ef49471a1590e50670b0924ea9b95ca0624')
        diagnostic_root = namespace._bounded_path(str(ROOT/'rayan-personality/private-inputs/rayan-n0-semantic-20261011'), ROOT, file=False)
        diagnostic_index = jpin(diagnostic_root/'n0-probe-index.protected.json', diagnostic_summary['protected_input_index_sha256'])
        diagnostic_rows = lines(pin(diagnostic_root/'target-free-documents.protected.jsonl', diagnostic_index['files']['target-free-documents.protected.jsonl']))
        diagnostic_bindings = jpin(diagnostic_root/'contrast-bindings.protected.json', diagnostic_index['files']['contrast-bindings.protected.json'])
        diagnostic_targets = jpin(diagnostic_root/'primary-directional-targets.protected.json', diagnostic_index['files']['primary-directional-targets.protected.json'])
        diagnostic_documents = {}
        for row in diagnostic_rows:
            assert row['split'] == 'TRAIN' and row['parent_document_id'] in original['documents']
            document = CognitiveFrameDocument.from_mapping(row['document'])
            assert document.sha256 == diagnostic_index['document_sha256'][row['document_id']]
            parent = original['documents'][row['parent_document_id']]
            binding = diagnostic_bindings[row['document_id']]
            if row['comparison_kind'] == 'original':
                assert document.sha256 == parent.sha256 == binding['document_sha256'] == binding['parent_document_sha256']
                assert sha256(canonical({key:value for key,value in binding.items() if key != 'binding_sha256'})).hexdigest() == binding['binding_sha256']
            else:
                loaded['n0_semantic_contrasts.py'].verify_context_contrast(parent, document, binding, expected_binding_sha256=binding['binding_sha256'])
            target = diagnostic_targets[row['document_id']]
            assert target['document_sha256'] == document.sha256
            assert all(target[key] is False for key in ('training_eligible','evaluation_eligible','independent_qualification'))
            diagnostic_documents[row['document_id']] = document
        assert len(diagnostic_rows) == len(diagnostic_documents) == len(diagnostic_bindings) == len(diagnostic_targets) == 10
        diagnostic_output = namespace._bounded_path(str(ROOT/'rayan-personality/private-compile/rayan-n0-semantic-577240'), ROOT, file=False)
        diagnostic_session = sealed(diagnostic_output/'feature-session.protected.json', diagnostic_summary['new_closed_feature_session_sha256'])
        diagnostic_originals = lines(pin(diagnostic_output/'authentic-governed-contrasts.protected.jsonl', diagnostic_summary['protected_responses_sha256']))
        diagnostic_frames = {row['document_id']:row['feature_binding'] for row in diagnostic_originals}
        assert set(diagnostic_frames) == set(diagnostic_documents)
        diagnostic_store = FrozenFeatureStore.open(namespace._bounded_path(str(diagnostic_output/'feature-store'), ROOT, file=False), expected_manifest_sha256=diagnostic_session['feature_store']['manifest_sha256'])
        diagnostic_replay = FrozenFrameProducer.replay(diagnostic_store, config, diagnostic_session, expected_session_sha256=diagnostic_session['receipt_sha256'])
        saved_diagnostics = {}
        for tag in ('parent','successor'):
            values = lines(pin(source_output/(tag+'-original-conditional-failures.protected.jsonl'), summary['original_conditional_'+tag+'_readback']['file_sha256']))
            assert len(values) == 10
            saved_diagnostics[tag] = {row['document_id']:row for row in values}
            assert set(saved_diagnostics[tag]) == set(diagnostic_documents)
        stages = {tag:[] for tag in models}
        private_diagnostic_hashes = {}
        with ExitStack() as diagnostic_stack:
            streams = {tag:diagnostic_stack.enter_context((output/(tag+'-raw-native-diagnostics.protected.jsonl')).open('xb')) for tag in models}
            for row in diagnostic_rows:
                name = row['document_id']; document = diagnostic_documents[name]
                produced = diagnostic_replay.produce(document)
                assert produced.binding == diagnostic_frames[name]
                for tag in models:
                    expected_packet = None
                    if tag in saved_diagnostics:
                        prior = saved_diagnostics[tag][name]
                        verify_seal(prior['response']); verify_seal(prior['response']['preference_selection'])
                        assert prior['document_sha256'] == document.sha256
                        assert prior['response']['preference_selection']['model_state_sha256'] == model_states[tag]
                        expected_packet = prior['response']['packet_sha256']
                    response, raw_logits = capture_diagnostic_response(successor_api.respond_with_governance, consumers[tag], document, produced, diagnostic_session, expected_packet_sha256=expected_packet)
                    regions = diagnostic_targets[name]['review']['candidate_regions']
                    targets = [{'field':field,'facts':[{'pointers':[value['candidate_id']], 'available':True, 'value':int(value[key]=='POSITIVE')} for value in regions]} for field,(_,key) in SCALARS.items()]
                    fit = loaded['conditional_successor_report.py'].reviewed_region_fit(response['decision'], targets)
                    if tag in saved_diagnostics: assert fit == prior['original_region_fit']
                    record = {'document_id':name,'comparison_kind':row['comparison_kind'],'parent_document_id':row['parent_document_id'],'document_sha256':document.sha256,'response':response,'raw_n2_scalar_logits':raw_logits,'original_region_fit':fit,'independent_qualification':False}
                    streams[tag].write(canonical(record)+b'\n'); stages[tag].append(record)
                del produced
                gc.collect()
                print(json.dumps({'stage':'matched_three_phase_diagnostic', 'completed':len(stages['after_n1']), 'total':10, 'endpoint_reproduction_required':True,'new_gradients':0}), flush=True)
        diagnostics = summarize_diagnostics(stages, diagnostic_targets)
        for tag in models:
            private_diagnostic_hashes[tag] = sha256((output/(tag+'-raw-native-diagnostics.protected.jsonl')).read_bytes()).hexdigest()
        for tag in ('parent','successor'):
            assert sum(metric['errors'] for group in diagnostics['regions'][tag].values() for metric in group.values()) == summary['original_conditional_'+tag+'_readback']['binary_region_errors']

        # Close every original custody chain and prove all three observed models unchanged.
        closing = {}
        for tag, value in models.items():
            verified_now = value._verify(deep_custody=True)
            assert model_fingerprint(value.model) == model_states[tag]
            assert all(parameter.grad is None and not parameter.requires_grad for parameter in value.model.parameters())
            closing[tag] = {'model_state_sha256':model_states[tag], 'checkpoint_manifest_sha256':[pin.manifest_sha256 for pin in verified_now['checkpoints']], 'checkpoint_model_state_sha256':[manifest['model_state_sha256'] for manifest in verified_now['manifests']], 'checkpoint_resume_sha256':[manifest['resume_sha256'] for manifest in verified_now['manifests']]}
        for value in held: value.verify()
        for packet in packets.values():
            FrozenFeatureStore.open(packet['store'].path, expected_manifest_sha256=packet['store'].manifest['manifest_sha256'])
        FrozenFeatureStore.open(diagnostic_store.path, expected_manifest_sha256=diagnostic_store.manifest['manifest_sha256'])
        assert helper.inventory(overlay) == overlay_sha
        assert verify_prepared(source_artifacts['prepared_receipt']['path']) == preparation
        assert compiler.verify_compiled(compiled, expected_source_archive_sha256=common['source_package'], expected_receipt_sha256=helper.COMPILE_SEAL) == compiled_receipt
        for filename, path in public_paths.items(): assert sha256(path.read_bytes()).hexdigest() == plan['helper_sha256'][filename]
        assert sha256(Path(__file__).read_bytes()).hexdigest() == worker_sha
        namespace._clean_code(code, REV)
        result = {'schema':'alice-personality-cumulative-phase-attribution-v1','job_id':int(os.environ['SLURM_JOB_ID']),'source_job_id':577714,'source_readback_job_id':578457,'owner':'mxrayan','isolation':isolation,'namespace_sha256':GUARD,'worker_sha256':worker_sha,'scientific_revision':REV,'source_summary_sha256':SUMMARY_SHA,'source_plan_sha256':PLAN_SHA,'source_readback_sha256':READBACK_SHA,'source_readback_receipt_sha256':READBACK_RECEIPT,'analyzer_sha256':ANALYZER_SHA,
            'complete_source_and_checkpoint_custody_verified':True,'phase_states':closing,'after_n1_responses':236,'after_n1_objectives':394,'after_n1_response_file_sha256':sha256(response_path.read_bytes()).hexdigest(),'after_n1_objective_file_sha256':sha256(objective_path.read_bytes()).hexdigest(),'diagnostic_private_file_sha256':private_diagnostic_hashes,'endpoint_diagnostic_packet_reproductions':20,'objective_comparison':objective_comparison,'n1_after_n2_invariance':n1_phase_control,'diagnostic_comparison':diagnostics,'estimated_live_memory_bytes':estimate,'output_disk_budget_bytes':output_budget,'new_private_gradient_steps':0,'n0_approved':False,'owner_fidelity_approved':False,'personality_approved':False,'independent_qualification':False,'private_payload_exported':False,'model_correction_applied':False}
        result['receipt_sha256'] = sha256(canonical(result)).hexdigest()
        write(ROOT/'jobs'/(NAME+'.json'), result)
        print(json.dumps({'stage':'complete_zero_update_phase_attribution_saved','receipt_sha256':result['receipt_sha256']}), flush=True)


if __name__ == '__main__':
    path = Path(os.environ['PERSONALITY_REPO_ROOT'])/'scripts/eipm/n1/enter_private_network_namespace.py'
    assert sha256(path.read_bytes()).hexdigest() == GUARD
    namespace = module('owner_phase_attribution_namespace', path)
    try:
        run(namespace)
    except BaseException as error:
        import traceback
        print(json.dumps({'state':'PHASE_ATTRIBUTION_REFUSED','exception_type':type(error).__name__,
            'public_code_frames':[{'module':Path(frame.filename).name,'line':frame.lineno,'function':frame.name} for frame in traceback.extract_tb(error.__traceback__)]}), flush=True)
        raise SystemExit(2)

