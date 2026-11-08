"""Read-only source/frame/target admission reused from completed job579741.

Static reviewed copy: no dynamic source transformation, optimizer or candidate
export. Original worker remains immutable. The yielded context closes full
custody even when its diagnostic consumer raises.
"""
import copy
import gc
import importlib.util
import json
import math
import os
import stat
import sys
from collections import Counter
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

ROOT = Path('/homes/01/mxrayan/rayan-compute')
REV = '333fcd0c8239fc81a6459b13d53bbe592b33fda6'
GUARD = 'f5822c3657b246a5b232f42bddf93b8026bda0b6a7e573835c1cd8c90136c1e4'
NAME = 'rayan-conditional-learning-20261025'
ORIGINAL_NAME = 'rayan-cumulative-learning-r3-20261015'
PLAN_SHA = '57145c3e0dab55644f55d0d4e3d533a0241bfd84d55821fcc3d711a059a84ee7'
SUMMARY_SHA = '6e92df79abe4a13a17d6c4e92bb275db8250ad5e1d3de79d4c2006a1388e4533'
READBACK_SHA = 'd1eb676108a800f81e603ea45c7ba0b77298d88984ee0f4966cf68789efd5839'
READBACK_RECEIPT = '12ee55ddefe518a5f817023109aa17e32725e65ed6ef699e6e9e5af89eac3949'
ANALYZER_SHA = '8e0982e758ad81d735c474101fdd2e271d4dee3e124d33fc7f1e129e0c50b6c7'
PHASE_SHA = '34f0e49703985d022b92846cecc297e2d69eee7e6e81248e911d01e7a47ad8f1'
SAVED_PATH_SHA = '7f9dad78ebe3a59663885a905d9fa11aca160d813287b3c9a3369fabdfde8410'
CURRENT_MODEL_SHA = '9f3de2e625f267a080f2bc0c2632442dfee88fc4e1f45dead09a80692c5ea8b9'
EXPERIMENT_HELPER_SHA = 'a678d7921cc11833ca956e3953fbeaa88636963d45f57f4001069de53c635a28'
TEACHING_HELPER_SHA = '923eef65f9b37b28ade4b72fc89cfa33e4a7cecfbf53b5b408e43f8ab60862b5'
ORIGINAL_EXPERIMENT_HELPER_SHA = 'd3d9b52b7abbd702fd55fd39ab2c0e72c19b122d500f39953c8f5cfcba042bd2'
EXPERIMENT_OWNER_PLAN_SHA = 'c45cc7671dec34be11f9b9fbe8ae0eda99c21674bc45a3c20115cb104752574f'
LIVE_CUSTODY_SHA = '7bde912edf49ffd1227f47d2bbf77068d6c0d79c7f65159b9e49a36522505533'
BROADER_PRIMARY_SPEC_SHA = 'eda4a181c569c9715494d7f1dada2b193520d4631edd20e83b7a9da7330d8503'
KINDS = ('original', 'decisive_condition', 'paraphrase', 'irrelevant_social', 'neutral_extra_text')
SCALARS = {'co_valid_probabilities': (1, 'co_valid_region'), 'contraindications': (4, 'contraindication_region')}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()

def require(condition, reason):
    if not condition:
        raise ValueError(reason)

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
    require(sha256(canonical({k: v for k, v in value.items() if k != 'receipt_sha256'})).hexdigest()
            == value['receipt_sha256'], 'retained receipt seal changed')
    return value

def source_pairs(cases, allocation, *, expected_pairs=40):
    """Derive pairs from reviewed construction ancestry plus full semantics.

    The source compiler reverses candidate presentation in the second context.
    Canonical pointer-to-meaning equality deliberately preserves that reversal.
    Ancestry alone, matching IDs alone, or model mistakes never select a pair.
    """
    index = {}
    for case in cases:
        case.validate()
        require(case.phase == 'n2' and case.split == 'TRAIN'
                and case.lineage is not None, 'pair requires reviewed N2 TRAIN lineage')
        families = case.lineage.supervision_family_ids
        require(len(families) == 1 and families[0] not in index, 'ambiguous supervision occurrence')
        index[families[0]] = case
    require(len(index) == expected_pairs * 2, 'complete broader occurrence set required')
    parents = allocation['supervision_parents']
    pairs, used, meanings = [], set(), set()
    for family, child in index.items():
        ancestry = parents.get(family)
        require(type(ancestry) is list, 'missing explicit reviewed ancestry')
        if not ancestry:
            continue
        require(len(ancestry) == 1 and ancestry[0] in index and not parents[ancestry[0]],
                'pair needs exactly one direct root construction parent')
        parent = index[ancestry[0]]
        require(family not in used and ancestry[0] not in used, 'ambiguous repeated source pair')
        left, right = parent.document, child.document
        candidate = lambda doc: sorted((asdict(entry) for entry in doc.candidates), key=lambda v: v['candidate_id'])
        require(canonical(candidate(left)) == canonical(candidate(right)),
                'paired candidate meanings or authority differ')
        require(canonical({k: [asdict(v) for v in values] for k, values in left.labels.items()})
                == canonical({k: [asdict(v) for v in values] for k, values in right.labels.items()}),
                'paired label meanings differ')
        require(parent.lineage.reference_family_ids == child.lineage.reference_family_ids
                and parent.lineage.context_family_ids != child.lineage.context_family_ids,
                'paired fixed source or current-context ancestry differs')
        refs = set(parent.lineage.reference_family_ids)
        static = lambda doc: [asdict(entry) for entry in doc.sources if entry.record.source_family_id in refs]
        require(canonical(static(left)) == canonical(static(right)), 'paired source meanings differ')
        require(parent.artifact_sha256 == child.artifact_sha256
                and parent.source_class == child.source_class == 'governed_identity_targets',
                'paired source review differs')
        scenes = lambda doc: [(field.semantic_role, field.text, field.available)
                             for field in doc.fields if field.semantic_role == 'situation']
        require(scenes(left) != scenes(right) and left.sha256 != right.sha256,
                'pair must contain distinct supplied current conditions')
        meaning = sha256(canonical(candidate(left))).hexdigest()
        require(meaning not in meanings, 'same semantic candidate tuple occurs in multiple pairs')
        meanings.add(meaning); used.update((family, ancestry[0]))
        pairs.append((meaning, parent, child))
    require(len(pairs) == expected_pairs and used == set(index), 'nonambiguous full source pair coverage required')
    return [(parent, child) for _, parent, child in sorted(pairs, key=lambda value: value[0])]

def matched_group_geometry(pairs, remainder, batch_by_case):
    require(len(pairs) == 40 and {k: len(v) for k, v in remainder.items()}
            == {'conditional': 80, 'original': 40, 'voice': 16}, 'complete 216-case geometry required')
    remainder = {k: sorted(v, key=lambda c: c.sha256) for k, v in remainder.items()}
    groups = []
    for i, pair in enumerate(pairs):
        members = [('broader', c) for c in pair]
        members += [('conditional', c) for c in remainder['conditional'][2*i:2*i+2]]
        members += [('original', remainder['original'][i])]
        if i < 16:
            members += [('voice', remainder['voice'][i])]
        groups.append([batch_by_case[(packet, case.case_id)] for packet, case in members])
    require(len({v for group in groups for v in group}) == 216, 'group occurrence duplicated or absent')
    # A moved case must keep its 1/group_size coefficient in both arms.
    permutation = list(range(1, 16)) + [0] + list(range(17, 40)) + [16]
    require(all(len(groups[i]) == len(groups[j]) for i, j in enumerate(permutation)),
            'control changed an endpoint loss coefficient')
    return groups, permutation

@contextmanager
def open_context(namespace):
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
    from src.alice_personality.identity.codec import CognitiveFrameDocument, decision_to_mapping, fingerprint
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
    output = ROOT/'rayan-personality/private-compile'/('rayan-conditional-gradient-attribution-20261028-'+os.environ['SLURM_JOB_ID'])
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

        # Startup isolation cleanup has already run. This separately pinned
        # predicate verifies live isolation without closing the files we own.
        live_path = ROOT/'jobs'/(NAME+'-live.py')
        pin(live_path, LIVE_CUSTODY_SHA)
        live_api = module('paired_comparison_live_custody', live_path)
        live_api.verify_active_namespace(namespace, isolation)

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
        assert (config.provider_state_count, config.provider_width) == (49, 3840)
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
        assert model_fingerprint(final.model) == CURRENT_MODEL_SHA
        assert fingerprint(auxiliary) == final_verified['receipts'][-1]['objective_final_state_sha256']

        # A new explicit owner plan grants this bounded comparison, not promotion.
        experiment_owner = jpin(ROOT/'jobs'/(NAME+'-owner-plan.json'),
                                EXPERIMENT_OWNER_PLAN_SHA)
        assert experiment_owner['schema'] == 'alice-personality-owner-conditional-learning-v1'
        assert experiment_owner['owner'] == 'mxrayan' and experiment_owner['purpose'] == 'development_construction'
        assert experiment_owner['private_gradient_authorized'] is True and experiment_owner['acceptance_authority'] is False
        assert experiment_owner['model_state_sha256'] == CURRENT_MODEL_SHA
        assert experiment_owner['helper_sha256'] == EXPERIMENT_HELPER_SHA
        assert experiment_owner['source_plan_sha256'] == PLAN_SHA and experiment_owner['source_summary_sha256'] == SUMMARY_SHA
        assert experiment_owner['source_phase_sha256'] == PHASE_SHA
        assert experiment_owner['expected_updates'] == {'intervention': 40, 'control': 40}
        assert experiment_owner['expected_case_occurrences'] == {'intervention': 216, 'control': 216}
        assert experiment_owner['epochs'] == 1 and experiment_owner['adapt_n1'] is False
        assert all(experiment_owner[key] is False for key in ('n1_updates_authorized', 'n3_updates_authorized',
                   'n0_approved', 'owner_fidelity_approved', 'full_personality_complete', 'flora_work'))
        assert experiment_owner['live_custody_helper_sha256'] == LIVE_CUSTODY_SHA
        assert experiment_owner['reuse_completed_baseline'] is True
        assert experiment_owner['persist_each_completed_group'] is True
        assert experiment_owner['teaching_helper_sha256'] == TEACHING_HELPER_SHA
        assert experiment_owner['correction_contexts'] == 80
        assert experiment_owner['native_forward_passes_per_endpoint'] == 2
        assert experiment_owner['natural_loss_weight'] == experiment_owner['replay_loss_weight'] == .5
        teaching_path = namespace._bounded_path(str(ROOT/'jobs'/(NAME+'-teaching.py')), ROOT, file=True)
        pin(teaching_path, TEACHING_HELPER_SHA)
        teaching_api = module('scripts.eipm.gemma_n0.conditional_teaching', teaching_path)
        original_experiment_path = namespace._bounded_path(str(ROOT/'jobs/rayan-paired-update-comparison-20261020-helper.py'), ROOT, file=True)
        pin(original_experiment_path, ORIGINAL_EXPERIMENT_HELPER_SHA)
        module('scripts.eipm.gemma_n0.paired_learning_experiment', original_experiment_path)
        experiment_path = namespace._bounded_path(str(ROOT/'jobs'/(NAME+'-helper.py')), ROOT, file=True)
        pin(experiment_path, EXPERIMENT_HELPER_SHA)
        experiment_api = module('conditional_learning_pinned_experiment', experiment_path)
        phase_summary = verify_seal(jpin(ROOT/'jobs/rayan-cumulative-phase-attribution-20261018.json', PHASE_SHA))
        assert phase_summary['job_id'] == 578460 and phase_summary['after_n1_responses'] == 236
        assert phase_summary['after_n1_objectives'] == 394 and phase_summary['endpoint_diagnostic_packet_reproductions'] == 20
        assert phase_summary['complete_source_and_checkpoint_custody_verified'] is True
        saved_path_summary = verify_seal(jpin(ROOT/'jobs/rayan-saved-path-readback-20261020.json', SAVED_PATH_SHA))
        assert saved_path_summary['source_phase_sha256'] == PHASE_SHA and saved_path_summary['source_record_seals_verified'] is True
        phase_path = namespace._bounded_path(str(ROOT/'jobs/rayan-cumulative-phase-attribution-20261018.py'), ROOT, file=True)
        pin(phase_path, phase_summary['worker_sha256'])
        phase_api = module('paired_comparison_original_phase_measurements', phase_path)
        assert phase_summary['phase_states']['successor']['model_state_sha256'] == CURRENT_MODEL_SHA
        assert phase_summary['source_plan_sha256'] == PLAN_SHA and phase_summary['source_summary_sha256'] == SUMMARY_SHA
        assert packets['broader']['protocol']['original_primary_spec_sha256'] == BROADER_PRIMARY_SPEC_SHA
        assert packets['broader']['review']['reviewed_policy_count'] == 40
        assert packets['broader']['review']['calibration_or_diagnostic_rows_used'] is False
        recipe = phase_data['n2']['recipe']
        assert recipe.phase == 'n2' and recipe.adapt_n1 is False
        assert recipe.learning_rate == .0003 and recipe.max_grad_norm == 1.
        print(json.dumps({'stage': 'native_source_and_new_experiment_owner_plan_verified',
            'current_model_sha256': CURRENT_MODEL_SHA, 'owner_plan_sha256': EXPERIMENT_OWNER_PLAN_SHA,
            'complete_valid_teaching_ancestry': 394, 'completed_baseline_packets_to_rebind': 236,
            'new_private_gradient_steps': 0}), flush=True)

        # Current saved endpoints are reusable only after exact complete rebinding.
        baseline_meta = summary['successor_full_governed_responses']
        baseline_response_bytes = pin(source_output/'successor-governed-responses.protected.jsonl',
                                     baseline_meta['file_sha256'], 256*1024**2).stream.read()
        baseline_objective_bytes = pin(source_output/'successor-all-reviewed-objectives.protected.jsonl',
                                      baseline_meta['all_reviewed_objective_file_sha256']).stream.read()
        baseline_responses = [raw._json(line) for line in baseline_response_bytes.splitlines()]
        baseline_index = {(r['packet'], r['document_id']): r for r in baseline_responses}
        assert len(baseline_index) == len(baseline_responses) == 236
        baseline_objectives = [raw._json(line) for line in baseline_objective_bytes.splitlines()]
        expected_documents, expected_batches, batch_by_case = {}, {}, {}
        baseline_broader, correction_records = {}, {}
        for packet_name, packet in packets.items():
            for name, document in packet['documents'].items():
                produced = packet['replay'].produce(document)
                assert produced.binding == packet['bindings'][name]
                binding = produced.binding
                expected_documents[(packet_name, name)] = {
                    'document_sha256': document.sha256, 'frame_sha256': binding['frame_sha256'],
                    'feature_binding_sha256': binding['binding_sha256'],
                    'closed_session_sha256': packet['session']['receipt_sha256']}
                prior = baseline_index[(packet_name, name)]['response']
                # Reuse the complete native baseline already published by 577714
                # and revalidated by 578457. Full source/frame/session/checkpoint
                # rebinding below is mandatory; no fresh inference is claimed.
                document_batches = {}
                for case in packet['cases'].values():
                    if case.document.sha256 != document.sha256 or (packet_name, case.case_id) not in selected_cases:
                        continue
                    verify_allocation(case.lineage, tuple(entry.record for entry in case.document.sources), 'TRAIN', merged)
                    batch = materialize_case(case, produced, packet['session'], config,
                        expected_case_sha256=case.sha256, expected_artifact_sha256=packet['artifacts'],
                        family_splits=packet['allocation']).batch
                    digest = batch.fingerprint()
                    assert digest not in expected_batches and digest not in scope['superseded_parent_batches']
                    assert phase_data[case.phase]['reviewed']['reviewed_batches'][digest] == {
                        'split': 'TRAIN', 'case_ids': list(batch.case_ids),
                        'source_family_ids': [list(value) for value in batch.source_family_ids],
                        'construction_lineage_sha256': fingerprint(batch.lineage)}
                    expected_batches[digest] = {'packet': packet_name, 'case_id': case.case_id,
                                               'case_sha256': case.sha256, 'phase': case.phase}
                    batch_by_case[(packet_name, case.case_id)] = digest
                    if packet_name == 'broader' and case.phase == 'n2':
                        baseline_broader[case.case_id] = prior['decision']
                    if packet_name == 'broader':
                        assert case.phase not in document_batches
                        document_batches[case.phase] = (case, batch)
                    del batch
                if packet_name == 'broader':
                    assert set(document_batches) == {'n1', 'n2'}
                    n1_case, n1_batch = document_batches['n1']
                    n2_case, n2_batch = document_batches['n2']
                    record = teaching_api.build_correction_record(n1_case, n2_case, n1_batch, n2_batch,
                        frame_binding=binding, reviewed_n1_batches=phase_data['n1']['reviewed']['reviewed_batches'])
                    correction_records[n2_batch.fingerprint()] = record
                    del n1_case, n1_batch, n2_case, n2_batch, record
                del document_batches
                del produced
                gc.collect()
                print(json.dumps({'stage': 'completed_baseline_source_rebinding',
                    'documents_completed': len(expected_documents), 'documents_total': 236,
                    'bound_objectives': len(expected_batches), 'objective_total': 394,
                    'new_private_gradient_steps': 0}), flush=True)
        assert len(correction_records) == 80
        correction_digests = {key: teaching_api.record_sha256(value) for key, value in correction_records.items()}
        correction_file_sha = write(output/'admitted-corrections.protected.json', correction_records)
        assert len(expected_documents) == 236 and len(expected_batches) == 394
        assert set(expected_batches) == set(final_verified['valid'])
        baseline_recovery = loaded['cumulative_recovery.py'].recover_parent_measurements(
            baseline_response_bytes, baseline_objective_bytes, response_sha256=baseline_meta['file_sha256'],
            objective_sha256=baseline_meta['all_reviewed_objective_file_sha256'],
            expected_documents=expected_documents, expected_batches=expected_batches,
            parent_model_sha256=CURRENT_MODEL_SHA,
            parent_checkpoint_sha256=[value.manifest_sha256 for value in final_verified['checkpoints']])
        selected_n2 = [(row['packet'], packets[row['packet']]['cases'][row['case_id']])
                       for row in selection['phase_case_selection']['n2']]
        n2_digests = [batch_by_case[(name, case.case_id)] for name, case in selected_n2]
        assert len(n2_digests) == len(set(n2_digests)) == 216
        assert set(n2_digests) == set(phase_data['n2']['reviewed']['reviewed_batches'])
        pairs = source_pairs([case for name, case in selected_n2 if name == 'broader'], packets['broader']['allocation'])
        groups, permutation = matched_group_geometry(pairs,
            {name: [case for packet_name, case in selected_n2 if packet_name == name]
             for name in ('conditional', 'original', 'voice')}, batch_by_case)
        n2_lookup = {batch_by_case[(name, case.case_id)]: (name, case) for name, case in selected_n2}
        private_pairs = [{'left_case_sha256': a.sha256, 'right_case_sha256': b.sha256,
            'left_batch_sha256': batch_by_case[('broader', a.case_id)],
            'right_batch_sha256': batch_by_case[('broader', b.case_id)],
            'left_lineage': a.lineage.to_mapping(), 'right_lineage': b.lineage.to_mapping()} for a, b in pairs]
        pair_file_sha = write(output/'source-pair-bindings.protected.json', private_pairs)

        assert baseline_recovery['count'] == 236 and baseline_recovery['measured_case_count'] == 394

        # Native validation has a fresh exact admission for this current state.
        # Its optimizer is never stepped; grouped helper owns both separate clones.
        from src.alice_personality.teaching.trainer import ArtifactPin, TeachingAdmission, IdentityTrainer, current_code_hashes
        from src.alice_personality.teaching.contracts import AUTH_SCHEMA
        code_hashes = current_code_hashes()
        artifacts = {role: ArtifactPin(source_artifacts[role]['path'], source_artifacts[role]['file_sha256'])
                     for role in ('source_package', 'compiled_receipt', 'prepared_receipt')}
        target_pin = final_verified['manifests'][-1]['binding']['artifacts']['reviewed_targets']
        artifacts.update(reviewed_targets=ArtifactPin(target_pin['path'], target_pin['file_sha256']),
                         source_families=ArtifactPin(allocation_path, allocation_sha))
        authorization = {'schema': AUTH_SCHEMA, 'authorization_id': 'rayan-conditional-gradient-attribution-20261028-readonly', 'phase': 'n2',
            'purpose': 'development_construction', 'source_class': 'governed_identity_targets',
            'private_gradient_authorized': True, 'mechanical_steps_authorized': False,
            'artifact_file_sha256': {k: v.sha256 for k, v in artifacts.items()},
            'model_config_sha256': sha256(canonical(asdict(config))).hexdigest(),
            'model_initial_state_sha256': CURRENT_MODEL_SHA, 'recipe_sha256': recipe.fingerprint(),
            'code_sha256': code_hashes, 'owner_comparison_plan_sha256': EXPERIMENT_OWNER_PLAN_SHA,
            'native_checkpoint_authorized': False, 'acceptance_authority': False}
        authorization_path = output/'experiment-authorization.protected.json'
        authorization_sha = write(authorization_path, authorization)
        artifacts['authorization'] = ArtifactPin(authorization_path, authorization_sha)
        for role in ('source_families', 'authorization'):
            pin(artifacts[role].path, artifacts[role].sha256)
        admission = TeachingAdmission(artifacts, authorization['model_config_sha256'], CURRENT_MODEL_SHA,
                                      recipe.fingerprint(), code_hashes)
        first_batch = materialize(*n2_lookup[n2_digests[0]])
        validation_trainer = IdentityTrainer(copy.deepcopy(final.model), recipe, admission, first_batch=first_batch)
        assert validation_trainer.steps == 0
        del first_batch
        baseline_n2_objectives = [row for row in baseline_objectives if row['phase'] == 'n2']
        assert len(baseline_n2_objectives) == 216

        def custody_guard():
            live_api.verify_active_namespace(namespace, isolation)
            # Full hashes were checked at acquisition and close. Between them,
            # descriptor/path identity+ctime/mtime/size prevent stale-file reuse.
            for value in held:
                assert raw._file_state(os.fstat(value.stream.fileno())) == raw._file_state(value.descriptor_before)
                assert raw._file_state(raw._archive_path(value.path).stat()) == raw._file_state(value.before)
            assert sha256(live_path.read_bytes()).hexdigest() == LIVE_CUSTODY_SHA
            assert sha256(experiment_path.read_bytes()).hexdigest() == EXPERIMENT_HELPER_SHA
            assert sha256(teaching_path.read_bytes()).hexdigest() == TEACHING_HELPER_SHA
            assert sha256(original_experiment_path.read_bytes()).hexdigest() == ORIGINAL_EXPERIMENT_HELPER_SHA
            assert sha256(Path(__file__).read_bytes()).hexdigest() == worker_sha

        def batch_loader(digest):
            custody_guard()
            name, case = n2_lookup[digest]
            batch = materialize(name, case)
            assert batch.fingerprint() == digest
            batch_loader.loaded += 1
            print(json.dumps({'stage': 'experimental_batch_materialized',
                'materializations_completed': batch_loader.loaded, 'materializations_expected': 216}), flush=True)
            return batch
        batch_loader.loaded = 0

        def validate_batch(batch):
            digest = validation_trainer._check_batch(batch)
            assert digest in n2_lookup and digest not in scope['superseded_parent_batches']
            return digest

        try:
            yield {'model':final.model,'objectives':objectives,'recipe':recipe,
                'groups':groups,'batch_loader':batch_loader,'validate_batch':validate_batch,
                'correction_records':correction_records,'correction_digests':correction_digests,
                'n2_lookup':n2_lookup,'expected_batches':expected_batches,'custody_guard':custody_guard,
                'pin':pin,'jpin':jpin,'output':output,'isolation':isolation,
                'scientific_revision':REV,'context_sha256':worker_sha,
                'original_model_sha256':CURRENT_MODEL_SHA,'source_plan_sha256':PLAN_SHA}
        finally:
            assert validation_trainer.steps == 0
            closing = {}
            for tag, value in models.items():
                verified_now = value._verify(deep_custody=True)
                assert model_fingerprint(value.model) == model_states[tag]
                assert all(parameter.grad is None and not parameter.requires_grad for parameter in value.model.parameters())
                closing[tag] = {'model_state_sha256': model_states[tag],
                    'checkpoint_manifest_sha256': [pin.manifest_sha256 for pin in verified_now['checkpoints']],
                    'checkpoint_model_state_sha256': [m['model_state_sha256'] for m in verified_now['manifests']],
                    'checkpoint_resume_sha256': [m['resume_sha256'] for m in verified_now['manifests']]}
            for value in held:
                value.verify()
            for packet in packets.values():
                FrozenFeatureStore.open(packet['store'].path, expected_manifest_sha256=packet['store'].manifest['manifest_sha256'])
            assert helper.inventory(overlay) == overlay_sha
            assert verify_prepared(source_artifacts['prepared_receipt']['path']) == preparation
            assert compiler.verify_compiled(compiled, expected_source_archive_sha256=common['source_package'],
                                           expected_receipt_sha256=helper.COMPILE_SEAL) == compiled_receipt
            for filename, path in public_paths.items():
                assert sha256(path.read_bytes()).hexdigest() == plan['helper_sha256'][filename]
            custody_guard()
            namespace._clean_code(code, REV)
