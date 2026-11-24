"""Owner-isolated compatible learning from exact retained and new full features.

One full N1 sweep and one full N2 sweep; new admissions/fresh AdamW. This is
construction, not convergence, calibration, N0 approval or personality fidelity.
"""
import gc
import importlib.util
import json
import os
import random
import stat
import sys
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

ROOT = Path('/homes/01/mxrayan/rayan-compute')
REV = '333fcd0c8239fc81a6459b13d53bbe592b33fda6'
GUARD = 'f5822c3657b246a5b232f42bddf93b8026bda0b6a7e573835c1cd8c90136c1e4'
NAME = 'rayan-cumulative-learning-r2-20261013'


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
    from src.alice_personality.identity.codec import CognitiveFrameDocument, fingerprint, decision_to_mapping
    from src.alice_personality.identity.consumer import CheckpointPin
    from src.alice_personality.identity.feature_producer import FrozenFrameProducer
    from src.alice_personality.identity.feature_store import FrozenFeatureStore
    from src.alice_personality.gemma_n0.preparation import verify_prepared
    from src.alice_personality.teaching.cases import ReviewedTeachingCase
    from src.alice_personality.teaching.contracts import AUTH_SCHEMA, REVIEW_SCHEMA, TARGET_SCHEMA, TrainingRecipe
    from src.alice_personality.teaching.materialize import materialize_case
    from src.alice_personality.teaching.lineage import verify_allocation, validate_allocation
    from src.alice_personality.teaching.objectives import IdentityObjectives
    from src.alice_personality.teaching.trainer import ArtifactPin, TeachingAdmission, current_code_hashes, model_fingerprint
    from transformers import AutoProcessor
    helper = module('unchanged_cumulative_scientific_construction', code/'scripts/eipm/gemma_n0/construct_personality.py')
    overlay_sha = helper.inventory(overlay)
    output = ROOT/'rayan-personality/private-compile'/('rayan-cumulative-learning-r2-'+os.environ['SLURM_JOB_ID'])
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

        plan_path = ROOT/'rayan-personality/private-inputs'/NAME/'owner-cumulative-run.protected.json'
        plan_sha = os.environ['PERSONALITY_CUMULATIVE_PLAN_SHA256']
        plan = jpin(plan_path, plan_sha)
        assert plan['schema'] == 'alice-personality-owner-cumulative-run-v1'
        assert plan['owner'] == 'mxrayan' and plan['purpose'] == 'development_construction'
        assert plan['private_gradient_authorized'] is True and plan['acceptance_authority'] is False
        assert plan['epochs'] == {'n1': 1, 'n2': 1} and plan['expected_updates'] == {'n1': 178, 'n2': 216}
        assert plan['cpu_threads'] == 1 == int(os.environ['SLURM_CPUS_PER_TASK'])
        assert all(plan[key] is False for key in ('n3_updates_authorized', 'n0_approved', 'owner_fidelity_approved', 'full_personality_complete', 'flora_work'))
        worker_sha = sha256(Path(__file__).read_bytes()).hexdigest()
        assert worker_sha == plan['worker_sha256']
        public_paths = {}
        loaded = {}
        # Preserve the old scientific package; inject only exact separately pinned helpers.
        for filename in ('governed_response.py', 'cumulative_successor.py', 'cumulative_bindings.py', 'connected_voice_references.py', 'n0_semantic_contrasts.py', 'conditional_successor_report.py'):
            public_path = namespace._bounded_path(str(ROOT/'jobs'/('rayan-20261013-r2-'+filename)), ROOT, file=True)
            assert sha256(public_path.read_bytes()).hexdigest() == plan['helper_sha256'][filename]
            public_paths[filename] = public_path
            loaded[filename] = module('scripts.eipm.gemma_n0.'+filename[:-3], public_path)
        successor_api, binding_api = loaded['cumulative_successor.py'], loaded['cumulative_bindings.py']
        original_plan = jpin(ROOT/'rayan-personality/private-inputs/rayan-primary-governing-single-thread-20261008/owner-connected-run.protected.json', '09ea1efff8dcb62e7c0b391ec8fe00a30204691aea93f01cf84a72625ffb2f2b')
        selection = jpin(ROOT/'rayan-personality/private-inputs'/NAME/'case-selection.protected.json', plan['selection_file_sha256'])
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
            summary = jpin(reference['summary_path'], reference['summary_file_sha256'])
            assert sha256(canonical({k: v for k, v in summary.items() if k != 'receipt_sha256'})).hexdigest() == summary['receipt_sha256'] == reference['summary_receipt_sha256']
            assert summary['state'] == 'REAL_FEATURES_MATERIALIZED_UNADMITTED' and summary['private_gradient_steps'] == 0
            assert summary['packet_index_file_sha256'] == plan['packets'][name]['index_sha256']
            root = namespace._bounded_path(reference['directory'], ROOT, file=False)
            session = jpin(root/'feature-session.protected.json', summary['closed_session_file_sha256'])
            bindings = jpin(root/'frame-bindings.protected.json', summary['frame_bindings_file_sha256'])
            store = FrozenFeatureStore.open(namespace._bounded_path(str(root/'feature-store'), ROOT, file=False), expected_manifest_sha256=summary['feature_store_manifest_sha256'])
            assert store.manifest['source_package_sha256'] == common['source_package']
            assert session['prepared_receipt_file_sha256'] == common['prepared_receipt'] and session['receipt_sha256'] == summary['closed_session_sha256']
            assert set(bindings) == set(packet['documents']) and len(packet['cases']) == summary['materialized_case_count']
            assert {binding['binding_sha256'] for binding in bindings.values()} == set(session['frame_binding_sha256'])
            packet.update(store=store, session=session, bindings=bindings, replay=FrozenFrameProducer.replay(store, config, session, expected_session_sha256=session['receipt_sha256']))
        preparation = verify_prepared(source_artifacts['prepared_receipt']['path'])
        processor = AutoProcessor.from_pretrained(preparation['snapshot_path'], local_files_only=True, trust_remote_code=False)
        costs = {name: helper.token_cost_profile(list(packet['documents'].values()), processor.tokenizer, layers=49, width=3840, max_tokens=preparation['model_geometry']['max_position_embeddings']) for name, packet in packets.items()}
        estimate = 48*1024**3 + 12*max(cost['largest_padded_frame_value_bytes'] for cost in costs.values())
        helper.verify_memory_allocation(plan['memory_budget_bytes'], estimate, os.environ['SLURM_MEM_PER_NODE'])
        disk = os.statvfs(output)
        assert disk.f_bavail*disk.f_frsize >= plan['checkpoint_disk_budget_bytes']
        write(output/'token-cost-profiles.protected.json', costs)
        del processor
        gc.collect()

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
        write(output/'parent-case-batch-binding.protected.json', scope)
        phase_data = {}
        for phase in ('n1', 'n2'):
            rows = selection['phase_case_selection'][phase]
            assert len(rows) == plan['expected_updates'][phase]
            selected = [(row['packet'], packets[row['packet']]['cases'][row['case_id']]) for row in rows]
            assert len({(packet, case.case_id) for packet, case in selected}) == len(selected)
            assert all(case.phase == phase and case.sha256 == row['case_sha256'] for (_, case), row in zip(selected, rows))
            expected = {(name, case.case_id) for name, packet in packets.items() if name != 'original' for case in packet['cases'].values() if case.phase == phase}
            expected |= {('original', row['case_id']) for row in selection['retained_original_cases'] if row['phase'] == phase}
            assert {(name, case.case_id) for name, case in selected} == expected
            phase_root = output/phase
            phase_root.mkdir(mode=0o700)
            reviewed = {'schema': REVIEW_SCHEMA, 'target_schema': TARGET_SCHEMA, 'reviewed': True, 'source_class': 'governed_identity_targets', 'reviewed_batches': {}, 'batch_order': {'TRAIN': []}, 'reviewed_case_artifact_sha256': {case.sha256: dict(case.artifact_sha256) for _, case in selected}}
            batch_cases = {}
            for count, (packet_name, case) in enumerate(selected, 1):
                batch = materialize(packet_name, case)
                digest = batch.fingerprint()
                assert digest not in batch_cases and digest not in scope['superseded_parent_batches']
                batch_cases[digest] = (packet_name, case)
                reviewed['reviewed_batches'][digest] = {'split': 'TRAIN', 'case_ids': list(batch.case_ids), 'source_family_ids': [list(value) for value in batch.source_family_ids], 'construction_lineage_sha256': fingerprint(batch.lineage)}
                reviewed['batch_order']['TRAIN'].append(digest)
                del batch
                gc.collect()
                if count % 20 == 0: print(json.dumps({'stage': 'actual_cumulative_batch_binding', 'phase': phase, 'completed': count, 'total': len(selected)}), flush=True)
            arguments = dict(next(manifest['recipe'] for manifest in manifests if manifest['phase'] == phase))
            arguments['betas'] = tuple(arguments['betas'])
            recipe = TrainingRecipe(**arguments)
            recipe.validate()
            assert recipe.phase == phase and recipe.adapt_n1 is False
            random.Random(recipe.seed).shuffle(reviewed['batch_order']['TRAIN'])
            target_path = phase_root/'reviewed-batches.protected.json'
            target_sha = write(target_path, reviewed)
            phase_data[phase] = dict(recipe=recipe, reviewed=reviewed, batch_cases=batch_cases, root=phase_root, target_path=target_path, target_sha=target_sha)
        print(json.dumps({'stage': 'actual_parent_and_cumulative_cases_bound', 'phase_cases': plan['expected_updates'], 'retained_parent_batches': 42, 'superseded_parent_batches': 38, 'publisher_reloaded': False, 'estimated_live_memory_bytes': estimate, 'measured_peak': False}), flush=True)

        # Reuse the original failed-condition store. Diagnostic facts never enter admissions.
        def sealed(path, expected):
            path = namespace._bounded_path(str(path), ROOT, file=True)
            payload = path.read_bytes()
            value = jpin(path, sha256(payload).hexdigest())
            assert sha256(canonical({key: item for key, item in value.items() if key != 'receipt_sha256'})).hexdigest() == value['receipt_sha256'] == expected
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
                assert sha256(canonical({key: value for key, value in binding.items() if key != 'binding_sha256'})).hexdigest() == binding['binding_sha256']
            else:
                loaded['n0_semantic_contrasts.py'].verify_context_contrast(parent, document, binding, expected_binding_sha256=binding['binding_sha256'])
            target = diagnostic_targets[row['document_id']]
            assert target['document_sha256'] == document.sha256
            assert all(target[key] is False for key in ('training_eligible', 'evaluation_eligible', 'independent_qualification'))
            diagnostic_documents[row['document_id']] = document
        assert len(diagnostic_rows) == len(diagnostic_documents) == len(diagnostic_bindings) == len(diagnostic_targets) == 10
        diagnostic_output = namespace._bounded_path(str(ROOT/'rayan-personality/private-compile/rayan-n0-semantic-577240'), ROOT, file=False)
        diagnostic_session = sealed(diagnostic_output/'feature-session.protected.json', diagnostic_summary['new_closed_feature_session_sha256'])
        diagnostic_originals = lines(pin(diagnostic_output/'authentic-governed-contrasts.protected.jsonl', diagnostic_summary['protected_responses_sha256']))
        diagnostic_frames = {row['document_id']: row['feature_binding'] for row in diagnostic_originals}
        assert set(diagnostic_frames) == set(diagnostic_documents)
        diagnostic_store = FrozenFeatureStore.open(namespace._bounded_path(str(diagnostic_output/'feature-store'), ROOT, file=False), expected_manifest_sha256=diagnostic_session['feature_store']['manifest_sha256'])
        diagnostic_replay = FrozenFrameProducer.replay(diagnostic_store, config, diagnostic_session, expected_session_sha256=diagnostic_session['receipt_sha256'])

        def diagnostics(tag):
            consumer = construction.consumer()
            results_by_name = {}
            path = output/(tag+'-original-conditional-failures.protected.jsonl')
            compared = errors = 0
            with path.open('xb') as stream:
                for row in diagnostic_rows:
                    name = row['document_id']
                    document = diagnostic_documents[name]
                    produced = diagnostic_replay.produce(document)
                    assert produced.binding == diagnostic_frames[name]
                    response = successor_api.respond_with_governance(consumer, document, produced, diagnostic_session)
                    regions = diagnostic_targets[name]['review']['candidate_regions']
                    assert [value['candidate_id'] for value in regions] == [entry.candidate_id for entry in document.candidates]
                    targets = [{'field': field, 'facts': [{'pointers': [value['candidate_id']], 'available': True, 'value': int(value[key] == 'POSITIVE')} for value in regions]} for field, key in (('co_valid_probabilities', 'co_valid_region'), ('contraindications', 'contraindication_region'))]
                    fit = loaded['conditional_successor_report.py'].reviewed_region_fit(response['decision'], targets)
                    compared += sum(fit[field]['comparisons'] for field in ('co_valid_probabilities', 'contraindications'))
                    errors += sum(fit[field]['errors'] for field in ('co_valid_probabilities', 'contraindications'))
                    record = {'document_id': name, 'comparison_kind': row['comparison_kind'], 'parent_document_id': row['parent_document_id'], 'document_sha256': document.sha256, 'response': response, 'original_region_fit': fit, 'independent_qualification': False}
                    stream.write(canonical(record)+b'\n')
                    results_by_name[name] = response
                    del produced, response
                    gc.collect()
            control_preference_changes = control_acceptable_changes = 0
            for row in diagnostic_rows:
                if row['comparison_kind'] in ('original', 'decisive_condition'): continue
                parent_name = next(parent['document_id'] for parent in diagnostic_rows if parent['comparison_kind'] == 'original' and parent['parent_document_id'] == row['parent_document_id'])
                first, second = results_by_name[parent_name], results_by_name[row['document_id']]
                control_preference_changes += first['preferred_candidate_ids'] != second['preferred_candidate_ids']
                control_acceptable_changes += first['acceptable_candidate_ids'] != second['acceptable_candidate_ids']
            assert compared == 60
            return {'responses': 10, 'file_sha256': sha256(path.read_bytes()).hexdigest(), 'binary_region_comparisons': compared, 'binary_region_errors': errors, 'irrelevant_control_preference_changes': control_preference_changes, 'irrelevant_control_acceptable_set_changes': control_acceptable_changes, 'independent_qualification': False}

        # Save complete pre-update judgments for every original/new scene. No gold is invented.
        def responses(tag):
            construction.model.set_phase('inference')
            construction.model.eval()
            consumer = construction.consumer()
            count = 0
            path = output/(tag+'-governed-responses.protected.jsonl')
            auxiliary_pin = checkpoints[0] if tag == 'parent' else CheckpointPin(**results['n1']['lineage']['checkpoint'])
            _, _, auxiliary_state = successor_api._checkpoint(config, auxiliary_pin)
            objectives = IdentityObjectives(config.learned_width)
            objectives.load_state_dict(auxiliary_state, strict=True)
            objectives.requires_grad_(False)
            objectives.eval()
            selected_cases = {(packet, case.case_id) for data in phase_data.values() for packet, case in data['batch_cases'].values()}
            measured_cases = 0
            available_counts = {}
            loss_path = output/(tag+'-all-reviewed-objectives.protected.jsonl')
            with path.open('xb') as stream, loss_path.open('xb') as loss_stream:
                for packet_name, packet in packets.items():
                    for name, document in packet['documents'].items():
                        produced = packet['replay'].produce(document)
                        assert produced.binding == packet['bindings'][name]
                        response = successor_api.respond_with_governance(consumer, document, produced, packet['session'])
                        stream.write(canonical({'packet': packet_name, 'document_id': name, 'document_sha256': document.sha256, 'response': response})+b'\n')
                        with torch.no_grad():
                            decision = construction.model(produced.frame)
                            assert fingerprint(decision) == response['packet_sha256']
                            for case in packet['cases'].values():
                                if case.document.sha256 != document.sha256 or (packet_name, case.case_id) not in selected_cases:
                                    continue
                                batch = materialize_case(case, produced, packet['session'], config, expected_case_sha256=case.sha256, expected_artifact_sha256=packet['artifacts'], family_splits=packet['allocation']).batch
                                total, losses, coverage = objectives(decision, batch, loss_weights=phase_data[case.phase]['recipe'].loss_weights)
                                assert bool(torch.isfinite(total)) and all(bool(torch.isfinite(value)) for value in losses.values())
                                for field, value in coverage.items(): available_counts[field] = available_counts.get(field, 0) + value
                                loss_stream.write(canonical({'packet': packet_name, 'case_id': case.case_id, 'case_sha256': case.sha256, 'phase': case.phase, 'batch_sha256': batch.fingerprint(), 'total_loss': float(total), 'field_loss': {field: float(value) for field, value in losses.items()}, 'available_counts': coverage, 'descriptive_training_targets_not_independent_qualification': True})+b'\n')
                                measured_cases += 1
                                del batch, total, losses
                            del decision
                        count += 1
                        del produced, response
                        gc.collect()
                        if count % 20 == 0: print(json.dumps({'stage': tag+'_complete_governed_response', 'completed': count, 'total': 236}), flush=True)
            assert count == 236
            assert measured_cases == 394
            return {'count': count, 'file_sha256': sha256(path.read_bytes()).hexdigest(), 'measured_case_count': measured_cases, 'all_reviewed_objective_file_sha256': sha256(loss_path.read_bytes()).hexdigest(), 'available_fact_counts': available_counts}

        baseline = responses('parent')
        baseline_diagnostics = diagnostics('parent')
        assert baseline_diagnostics['binary_region_errors'] == 8
        results = {}
        for phase in ('n1', 'n2'):
            data = phase_data[phase]
            recipe = data['recipe']
            initial = model_fingerprint(construction.model)
            code_sha = current_code_hashes()
            admitted = {key: ArtifactPin(source_artifacts[key]['path'], common[key]) for key in common}
            admitted.update(reviewed_targets=ArtifactPin(data['target_path'], data['target_sha']), source_families=ArtifactPin(allocation_path, allocation_sha))
            config_sha = sha256(canonical(asdict(config))).hexdigest()
            auth = {'schema': AUTH_SCHEMA, 'authorization_id': plan['authorization_id']+':'+phase, 'phase': phase, 'purpose': 'development_construction', 'source_class': 'governed_identity_targets', 'private_gradient_authorized': True, 'mechanical_steps_authorized': False, 'artifact_file_sha256': {key: value.sha256 for key, value in admitted.items()}, 'model_config_sha256': config_sha, 'model_initial_state_sha256': initial, 'recipe_sha256': recipe.fingerprint(), 'code_sha256': code_sha, 'owner_run_plan_sha256': plan_sha, 'case_selection_file_sha256': plan['selection_file_sha256'], 'acceptance_authority': False, 'n0_approved': False, 'owner_fidelity_approved': False}
            auth_path = data['root']/'current-authorization.protected.json'
            admitted['authorization'] = ArtifactPin(auth_path, write(auth_path, auth))
            admission = TeachingAdmission(admitted, config_sha, initial, recipe.fingerprint(), code_sha)
            live_scope = construction._verify()['valid']
            retained = scope['retained_parent_batches'] if phase == 'n1' else tuple(sorted(live_scope))
            superseded = scope['superseded_parent_batches'] if phase == 'n1' else {}
            auxiliary = checkpoints[0] if phase == 'n1' else CheckpointPin(**results['n1']['lineage']['checkpoint'])
            load_count = 0

            def loader(digest):
                nonlocal load_count
                load_count += 1
                if load_count % 20 == 0: print(json.dumps({'stage': 'cumulative_streamed_batch_load', 'phase': phase, 'load_calls': load_count, 'optimizer_progress_not_inferred': True}), flush=True)
                packet_name, case = data['batch_cases'][digest]
                return materialize(packet_name, case)

            print(json.dumps({'stage': 'actual_cumulative_phase_admission_and_learning', 'phase': phase, 'epochs': 1, 'planned_updates': plan['expected_updates'][phase], 'new_admission_fresh_adamw': True}), flush=True)
            results[phase] = construction.run_phase(recipe, admission, batch_loader=loader, epochs=1, destination=data['root']/'final-checkpoint', lineage_destination=data['root']/'successor-lineage.protected.json', retained_parent_batches=retained, superseded_parent_batches=superseded, objective_transfer=auxiliary)
            assert results[phase]['checkpoint']['step'] == plan['expected_updates'][phase]
            print(json.dumps({'stage': 'actual_cumulative_phase_checkpoint_retained', 'phase': phase, 'updates': results[phase]['checkpoint']['step']}), flush=True)
        after = responses('successor')
        successor_diagnostics = diagnostics('successor')
        assert len(construction._verify()['valid']) == 394
        assert all(torch.equal(value, construction.model.state_dict()[name]) for name, value in verified['model'].state_dict().items() if name.startswith('n3.'))
        for value in held: value.verify()
        assert helper.inventory(overlay) == overlay_sha
        assert verify_prepared(source_artifacts['prepared_receipt']['path']) == preparation
        assert compiler.verify_compiled(compiled, expected_source_archive_sha256=common['source_package'], expected_receipt_sha256=helper.COMPILE_SEAL) == compiled_receipt
        for filename, path in public_paths.items(): assert sha256(path.read_bytes()).hexdigest() == plan['helper_sha256'][filename]
        assert sha256(Path(__file__).read_bytes()).hexdigest() == worker_sha
        namespace._clean_code(code, REV)
        summary = {'schema': 'alice-personality-actual-cumulative-learning-v1', 'state': 'CUMULATIVE_N1_N2_CONSTRUCTION_RETAINED_UNQUALIFIED', 'job_id': int(os.environ['SLURM_JOB_ID']), 'owner': 'mxrayan', 'scientific_revision': REV, 'isolation': isolation, 'namespace_sha256': GUARD, 'worker_sha256': worker_sha, 'helper_sha256': plan['helper_sha256'], 'owner_run_plan_sha256': plan_sha, 'selection_file_sha256': plan['selection_file_sha256'], 'source_geometry': [49, 3840], 'parent_model_state_sha256': plan['parent_model_state_sha256'], 'model_state_sha256': model_fingerprint(construction.model), 'phase_updates': {phase: result['checkpoint']['step'] for phase, result in results.items()}, 'phase_checkpoint_manifest_sha256': {phase: result['checkpoint']['manifest_sha256'] for phase, result in results.items()}, 'successor_lineage_file_sha256': [pin.file_sha256 for pin in construction.successors], 'retained_original_n1_cases': 2, 'retained_original_n2_cases': 40, 'superseded_original_n1_annotations': 38, 'cumulative_valid_batch_count': 394, 'baseline_full_governed_responses': baseline, 'successor_full_governed_responses': after, 'upstream_gradient': False, 'base_generation_or_chat_template': False, 'source_truncation': False, 'n3_updates': 0, 'historical_n3_numerical_state_preserved': True, 'successor_n3_calibration_applicable': False, 'governing_learning_repair_verified': False, 'full_curriculum_complete': False, 'owner_fidelity_approved': False, 'spoken_fidelity_verified': False, 'n0_approved': False, 'full_personality_complete': False, 'qualification': 'UNQUALIFIED', 'acceptance_authority': False, 'private_payload_exported': False, 'flora_work': False}
        summary['original_conditional_parent_readback'] = baseline_diagnostics
        summary['original_conditional_successor_readback'] = successor_diagnostics
        summary['descriptive_fit_is_independent_qualification'] = False
        summary['receipt_sha256'] = sha256(canonical(summary)).hexdigest()
        write(output/'cumulative-construction-summary.json', summary)
        write(ROOT/'jobs'/(NAME+'.json'), summary)
        print(json.dumps(summary, sort_keys=True), flush=True)


code = Path(os.environ['PERSONALITY_REPO_ROOT'])
guard_path = code/'scripts/eipm/n1/enter_private_network_namespace.py'
assert sha256(guard_path.read_bytes()).hexdigest() == GUARD
namespace = module('owner_cumulative_original_namespace', guard_path)
try:
    run(namespace)
except BaseException as error:
    import traceback
    frames = [{'module': Path(frame.filename).name, 'line': frame.lineno, 'function': frame.name} for frame in traceback.extract_tb(error.__traceback__)]
    try:
        namespace._p2_guard()
        output = namespace._bounded_path(str(ROOT/'rayan-personality/private-compile'/('rayan-cumulative-learning-r2-'+os.environ['SLURM_JOB_ID'])), ROOT, file=False)
        write(output/'failure-detail.protected.json', {'exception_type': type(error).__name__, 'message': str(error), 'public_code_frames': frames})
    except BaseException:
        pass
    print(json.dumps({'state': 'CUMULATIVE_CONSTRUCTION_REFUSED', 'exception_type': type(error).__name__, 'public_code_frames': frames}), flush=True)
    raise SystemExit(2)
