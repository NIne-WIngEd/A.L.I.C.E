"""Frozen native N1 conditional-channel intervention over all40 broader pairs.

Reviewed substitutions are explicitly target-bearing diagnostics, never learned
behavior or promotion. Reuse complete original source/checkpoint custody.
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
NAME = 'rayan-conditional-channel-probe-20261023'
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


def changed_channel_values(current, partner, donor_decision, saved_api):
    """Known changed facts only; donor source roles map to recipient IDs."""
    aliases = [saved_api.source_aliases(c) for c in (current, partner)]
    assert set(aliases[0].values()) == set(aliases[1].values())
    inverse = [{v:k for k,v in alias.items()} for alias in aliases]
    targets = []
    for case, alias in zip((current, partner), aliases):
        fields = {}
        for target in case.targets:
            if target.field not in saved_api.CONDITIONAL: continue
            assert target.field not in fields
            fields[target.field] = {tuple(alias.get(p,p) for p in fact.pointers):fact.value for fact in target.facts if fact.available}
        assert set(fields) == set(saved_api.CONDITIONAL)
        targets.append(fields)
    values = {tag:{field:{} for field in saved_api.CONDITIONAL} for tag in ('natural_partner','reviewed_current','reviewed_partner')}
    for field in saved_api.CONDITIONAL:
        for key in targets[0][field].keys() & targets[1][field].keys():
            a,b = [row[field][key] for row in targets]
            if a == b: continue
            recipient = tuple(inverse[0].get(p,p) for p in key)
            donor = tuple(inverse[1].get(p,p) for p in key)
            concept = donor_decision['concept_ids'][0].index(donor[0])
            observed = donor_decision['conditional_concepts'][field][0][concept]
            if len(donor) == 2:
                source = [r['record_id'] for r in donor_decision['sources'][0]].index(donor[1])
                observed = observed[source]
            values['reviewed_current'][field][recipient] = a
            values['reviewed_partner'][field][recipient] = b
            values['natural_partner'][field][recipient] = observed
    return values


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
    from src.alice_personality.teaching.contracts import TrainingRecipe
    from src.alice_personality.teaching.materialize import materialize_case
    from src.alice_personality.teaching.lineage import verify_allocation, validate_allocation
    from src.alice_personality.teaching.objectives import IdentityObjectives
    from src.alice_personality.teaching.trainer import model_fingerprint
    from transformers import AutoProcessor
    helper = module('unchanged_cumulative_scientific_construction', code/'scripts/eipm/gemma_n0/construct_personality.py')
    overlay_sha = helper.inventory(overlay)
    output = ROOT/'rayan-personality/private-compile'/(NAME+'-'+os.environ['SLURM_JOB_ID'])
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

        dependencies = {
            'live': ('rayan-paired-update-live-custody-20261021-live.py', '7bde912edf49ffd1227f47d2bbf77068d6c0d79c7f65159b9e49a36522505533'),
            'pair': ('rayan-paired-update-live-custody-20261021.py', '7675d14a2c04efc6a4244b5c6a22db6aa8be2b109e6e4b7cfaf8f3317b6c1537'),
            'saved': ('rayan-paired-readback-20261023.py', 'd9ccdcda05aea5ff2d81252ece733be676fe42539b42a16b2f0601e6e496f59b'),
            'probe': ('rayan-conditional-channel-probe-20261023-helper.py', 'a9bebc3635e4578f1c8c69343ef5ec1b988367a15a693cafd28784782f5a41af'),
        }
        apis = {}
        for key, (filename, digest) in dependencies.items():
            path = ROOT/'jobs'/filename
            pin(path, digest)
            apis[key] = module('channel_probe_'+key, path)
        live, pair_api, saved_api, probe = [apis[k] for k in ('live','pair','saved','probe')]
        live.verify_active_namespace(namespace, isolation)
        saved_result = verify_seal(jpin(ROOT/'jobs/rayan-paired-readback-20261023.json', '0d9e22bb3513f4cbb9eb166d39a9369d1c98cc1f3ec9309fe7e197667a768bde'))
        assert saved_result['job_id'] == 579462 and saved_result['full_objective_and_pair_comparisons_reproduced']
        assert saved_result['new_forward_passes'] == saved_result['new_gradient_steps'] == 0

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

        # No new admission, optimizer update or independent evaluation. The
        # existing source/checkpoint loader above remains the native authority.
        model = final.model
        assert config.provider_state_count == 49 and config.provider_width == 3840
        assert model_fingerprint(model) == pair_api.CURRENT_MODEL_SHA
        packet = packets['broader']
        n2_cases = [c for c in packet['cases'].values() if c.phase == 'n2']
        n1_cases = {c.document.sha256: c for c in packet['cases'].values() if c.phase == 'n1'}
        assert len(n2_cases) == len(n1_cases) == 80
        pairs = pair_api.source_pairs(n2_cases, packet['allocation'])
        meta = summary['successor_full_governed_responses']
        response_pin = pin(source_output/'successor-governed-responses.protected.jsonl', meta['file_sha256'], 256*1024**2)
        saved_rows = lines(response_pin)
        assert len(saved_rows) == 236
        baseline_by_document = {}
        for row in saved_rows:
            if row['packet'] != 'broader': continue
            response = verify_seal(row['response'])
            selected = verify_seal(response['preference_selection'])
            assert selected['model_state_sha256'] == model_fingerprint(model)
            assert response['packet_sha256'] == selected['packet_sha256']
            baseline_by_document[selected['document_sha256']] = response
        assert len(baseline_by_document) == 80
        saved_objectives = lines(pin(source_output/'successor-all-reviewed-objectives.protected.jsonl', meta['all_reviewed_objective_file_sha256']))
        baseline_objectives = [r for r in saved_objectives if r['packet'] == 'broader' and r['phase'] == 'n2']
        assert len(baseline_objectives) == 80
        baseline_objective_index = {r['case_id']: r for r in baseline_objectives}
        target_pin = final_verified['manifests'][-1]['binding']['artifacts']['reviewed_targets']
        reviewed = jpin(target_pin['path'], target_pin['file_sha256'])
        _, _, auxiliary = successor_api._checkpoint(config, intermediate_verified['checkpoints'][-1])
        objectives = IdentityObjectives(config.learned_width)
        objectives.load_state_dict(auxiliary, strict=True)
        objectives.requires_grad_(False); objectives.eval()
        recipe_data = dict(final_verified['manifests'][-1]['recipe'])
        recipe_data['betas'] = tuple(recipe_data['betas'])
        recipe = TrainingRecipe(**recipe_data); recipe.validate()
        assert recipe.phase == 'n2' and recipe.adapt_n1 is False
        tags = ('baseline','sham','natural_partner','reviewed_current','reviewed_partner')
        output_budget = 6*response_pin.before.st_size + 128*1024**2
        disk = os.statvfs(output)
        assert disk.f_bavail*disk.f_frsize >= output_budget
        predictions = {tag:{} for tag in tags}
        objective_rows = {tag:[] for tag in tags}
        files = {tag:output/(tag+'-channel-interventions.protected.jsonl') for tag in tags}
        streams = {tag:stack.enter_context(path.open('xb')) for tag,path in files.items()}
        baseline_lookup = {c.case_id: baseline_by_document[c.document.sha256]['decision'] for c in n2_cases}
        eligible_counts = Counter()
        contexts = 0

        print(json.dumps({'stage':'conditional_channel_probe_admitted','pairs':40,'contexts':80,
            'model_state_sha256':model_fingerprint(model),'new_gradient_steps':0,
            'reviewed_value_outputs_are_target_bearing_diagnostics':True}),flush=True)
        for pair_index, (left,right) in enumerate(pairs):
            for current, partner in ((left,right),(right,left)):
                live.verify_active_namespace(namespace,isolation)
                current_n1, partner_n1 = [n1_cases[c.document.sha256] for c in (current,partner)]
                assert current_n1.lineage == current.lineage and partner_n1.lineage == partner.lineage
                aliases = [saved_api.source_aliases(c) for c in (current,partner)]
                assert set(aliases[0].values()) == set(aliases[1].values())
                authority = [{alias[e.record.record_id]:{k:v for k,v in asdict(e.record).items() if k not in ('record_id','source_family_id')}
                    for e in case.document.sources} for case,alias in zip((current,partner),aliases)]
                assert authority[0] == authority[1]
                values = changed_channel_values(current_n1,partner_n1,baseline_by_document[partner.document.sha256]['decision'],saved_api)
                eligible_counts.update({field:len(facts) for field,facts in values['reviewed_current'].items()})
                document_id = next(k for k,v in packet['documents'].items() if v.sha256 == current.document.sha256)
                produced = packet['replay'].produce(current.document)
                assert produced.binding == packet['bindings'][document_id]
                verify_allocation(current.lineage, tuple(e.record for e in current.document.sources),'TRAIN',merged)
                batch = materialize_case(current,produced,packet['session'],config,
                    expected_case_sha256=current.sha256,expected_artifact_sha256=packet['artifacts'],family_splits=packet['allocation']).batch
                batch_sha = batch.fingerprint()
                prior_objective = baseline_objective_index[current.case_id]
                assert batch_sha == prior_objective['batch_sha256'] and batch_sha in reviewed['reviewed_batches']
                prior = baseline_by_document[current.document.sha256]
                captured = []
                hook = model.n1.register_forward_hook(lambda _m,_i,value:captured.append(value))
                try:
                    actual,decision,raw_logits,digest = pair_api.capture_candidate(model,produced.frame,expected_packet_sha256=prior['packet_sha256'])
                finally:
                    hook.remove()
                assert len(captured) == 1
                representation,query = captured.pop()
                saved_api.validate_saved_authority(current,decision)
                assert canonical(decision) == canonical(prior['decision'])
                for tag in tags:
                    intervention = {'applied_facts':0,'target_bearing':False}
                    if tag != 'baseline':
                        patched = representation
                        if tag != 'sham':
                            patched,intervention = probe.patch_conditionals(representation,produced.frame,values[tag])
                            intervention['target_bearing_diagnostic'] = tag.startswith('reviewed_')
                            intervention['values_from_reviewed_targets'] = tag.startswith('reviewed_')
                            intervention['coordinate_selection_uses_reviewed_target_changes'] = True
                        scalars = []
                        scalar_hook = model.n2.register_forward_hook(lambda _m,_i,value:scalars.append(value['scalar'].detach().clone()))
                        try:
                            actual = probe.replay_with_representation(model,produced.frame,patched,query)
                        finally:
                            scalar_hook.remove()
                        assert len(scalars) == 1
                        raw_logits = scalars[0].cpu().tolist()
                        decision,digest = decision_to_mapping(actual),fingerprint(actual)
                        if tag == 'sham': assert digest == prior['packet_sha256'] and canonical(decision) == canonical(prior['decision'])
                    with torch.no_grad():
                        total,losses,coverage = objectives(actual,batch,loss_weights=recipe.loss_weights)
                    objective = {k:prior_objective[k] for k in ('packet','phase','case_id','case_sha256','batch_sha256')}
                    objective.update(total_loss=float(total),field_loss={k:float(v) for k,v in losses.items()},available_counts=coverage,
                        descriptive_training_targets_not_independent_qualification=True)
                    if tag in ('baseline','sham'): assert objective == prior_objective
                    objective_rows[tag].append(objective)
                    predictions[tag][current.case_id] = decision
                    streams[tag].write(canonical({'schema':'alice-personality-conditional-channel-diagnostic-v1',
                        'variant':tag,'pair_index':pair_index,'case_sha256':current.sha256,'document_sha256':current.document.sha256,
                        'frame_binding':produced.binding,'source_packet_sha256':prior['packet_sha256'],'packet_sha256':digest,
                        'decision':decision,'raw_n2_scalar_logits':raw_logits,'objective':objective,'intervention':intervention,
                        'native_checkpoint':False,'learned_capability_evidence':False,'independent_qualification':False})+b'\n')
                    streams[tag].flush()
                contexts += 1
                del actual,decision,produced,batch,representation,query,patched,scalars
                gc.collect()
                print(json.dumps({'stage':'conditional_channel_context_complete','completed':contexts,'total':80,'new_gradient_steps':0}),flush=True)
        assert contexts == 80 and all(len(rows) == 80 for rows in objective_rows.values())
        assert all(len(rows) == 80 for rows in predictions.values())
        assert sum(path.stat().st_size for path in files.values()) <= output_budget
        metrics = {tag:{'full_n2_objectives':analyzer.compare_objectives(baseline_objectives,objective_rows[tag]),
            'broader_pairs':pair_api.paired_region_comparison(pairs,baseline_lookup,predictions[tag])} for tag in tags}
        closing = {}
        live.verify_active_namespace(namespace,isolation)
        for tag,value in models.items():
            verified_now = value._verify(deep_custody=True)
            assert model_fingerprint(value.model) == model_states[tag]
            assert all(p.grad is None and not p.requires_grad for p in value.model.parameters())
            closing[tag] = {'model_state_sha256':model_states[tag], 'checkpoint_manifest_sha256':[p.manifest_sha256 for p in verified_now['checkpoints']]}
        for value in held:value.verify()
        for source_packet in packets.values():
            FrozenFeatureStore.open(source_packet['store'].path,expected_manifest_sha256=source_packet['store'].manifest['manifest_sha256'])
        assert helper.inventory(overlay) == overlay_sha
        assert verify_prepared(source_artifacts['prepared_receipt']['path']) == preparation
        assert compiler.verify_compiled(compiled,expected_source_archive_sha256=common['source_package'],expected_receipt_sha256=helper.COMPILE_SEAL) == compiled_receipt
        for filename,path in public_paths.items():assert sha256(path.read_bytes()).hexdigest() == plan['helper_sha256'][filename]
        assert sha256(Path(__file__).read_bytes()).hexdigest() == worker_sha
        namespace._clean_code(code,REV)
        result = {'schema':'alice-personality-conditional-channel-diagnostic-v1','job_id':int(os.environ['SLURM_JOB_ID']),
            'owner':'mxrayan','namespace_sha256':GUARD,'isolation':isolation,'worker_sha256':worker_sha,'dependencies':dependencies,
            'source_model_state_sha256':pair_api.CURRENT_MODEL_SHA,'source_saved_readback_receipt_sha256':saved_result['receipt_sha256'],
            'complete_source_and_checkpoint_custody_verified':True,'closed_native_checkpoints':closing,
            'output_disk_budget_bytes':output_budget,'estimated_live_memory_bytes':estimate,
            'baseline_exact_native_packets':80,'sham_exact_native_packets':80,'source_pairs':40,'contexts':80,
            'full_native_model_forwards':80,'cached_n1_native_n2_n3_forwards':320,'new_gradient_steps':0,
            'changed_known_fact_occurrences':dict(eligible_counts),'metrics':metrics,
            'private_output_file_sha256':{tag:sha256(path.read_bytes()).hexdigest() for tag,path in files.items()},
            'reviewed_interventions_contain_targets':True,'natural_partner_is_observed_prediction':True,
            'mechanistic_scope':'Explicit N1 conditional channel only; latent views, tokens, query and all parameters retained.',
            'learned_capability_evidence':False,'native_checkpoint':False,'n0_approved':False,'personality_approved':False,
            'independent_qualification':False,'private_payload_exported':False}
        result['receipt_sha256'] = sha256(canonical(result)).hexdigest()
        write(ROOT/'jobs'/(NAME+'.json'),result)
        print(json.dumps({'stage':'complete_conditional_channel_probe','receipt_sha256':result['receipt_sha256']}),flush=True)


if __name__ == '__main__':
    try:
        path = Path(os.environ['PERSONALITY_REPO_ROOT'])/'scripts/eipm/n1/enter_private_network_namespace.py'
        assert sha256(path.read_bytes()).hexdigest() == GUARD
        run(module('owner_conditional_channel_namespace',path))
    except BaseException as error:
        try:
            path = ROOT/'jobs/rayan-paired-update-live-custody-20261021-live.py'
            assert sha256(path.read_bytes()).hexdigest() == '7bde912edf49ffd1227f47d2bbf77068d6c0d79c7f65159b9e49a36522505533'
            reporter = module('conditional_channel_failure_reporter',path)
            report = {'state':'CONDITIONAL_CHANNEL_PROBE_REFUSED','exception_chain':reporter.public_exception_chain(error)}
        except BaseException as reporter_error:
            report = {'state':'CONDITIONAL_CHANNEL_PROBE_REFUSED','exception_type':type(error).__name__,'reporter_exception_type':type(reporter_error).__name__}
        try:print(json.dumps(report),flush=True)
        finally:raise SystemExit(2) from None
