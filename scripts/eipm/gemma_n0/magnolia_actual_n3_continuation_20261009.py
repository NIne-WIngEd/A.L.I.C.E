"""Actual guarded model reconstruction; fixed aggregate receipt only."""
import importlib.util
import json
import os
from pathlib import Path
from hashlib import sha256
import sys
from contextlib import ExitStack
ROOT=Path('/homes/01/mxrayan/rayan-compute')
REV='333fcd0c8239fc81a6459b13d53bbe592b33fda6'
GUARD='f5822c3657b246a5b232f42bddf93b8026bda0b6a7e573835c1cd8c90136c1e4'
INVENTORY='5babc6eddd82cf45da2383e3f1faf81eab353b434ad63698ff9099aeea26a5c5'
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
def reload(namespace):
    isolation=namespace._p2_guard()  # FIRST protected path operation below.
    os.umask(0o077);sys.dont_write_bytecode=True
    code=Path(os.environ['PERSONALITY_REPO_ROOT']);namespace._clean_code(code,REV)
    assert sha256(Path(namespace.__file__).read_bytes()).hexdigest()==GUARD
    overlay=namespace._bounded_path(str(ROOT/'rayan-personality/runs/public-forward-576598/processor-deps'),ROOT,file=False)
    sys.path[:0]=[str(overlay),str(code)]
    import torch
    assert torch.__version__=='2.7.1+cu118'
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from src.alice_personality.n1.manifest_structure_audit import _PinnedFile
    from src.alice_personality.identity.codec import CognitiveFrameDocument,fingerprint,decision_to_mapping
    from src.alice_personality.identity.feature_producer import FrozenFrameProducer
    from src.alice_personality.identity.model import IdentityModel
    from src.alice_personality.identity.consumer import CheckpointPin
    from src.alice_personality.teaching.construction import ConnectedConstruction
    helper=module('unchanged_connected_helper',code/'scripts/eipm/gemma_n0/construct_personality.py')
    old=namespace._bounded_path(str(ROOT/'rayan-personality/private-compile/rayan-connected-personality-577095'),ROOT,file=False)
    inventory=json.loads((ROOT/'jobs/rayan-connected-577095-completion-recovery-20261009.json').read_bytes())
    assert sha256(canonical({k:v for k,v in inventory.items() if k!='receipt_sha256'})).hexdigest()==inventory['receipt_sha256']==INVENTORY
    with ExitStack() as stack:
        held=[]
        def pin(path,digest,limit=64*1024**2):
            path=namespace._bounded_path(str(path),ROOT,file=True)
            value=stack.enter_context(_PinnedFile(path,digest,limit));held.append(value)
            value.stream.seek(0);return json.loads(value.stream.read())
        plan=pin(ROOT/'rayan-personality/private-inputs/rayan-primary-governing-single-thread-20261008/owner-connected-run.protected.json',
            '09ea1efff8dcb62e7c0b391ec8fe00a30204691aea93f01cf84a72625ffb2f2b')
        packet=namespace._bounded_path(plan['packet_directory'],ROOT,file=False)
        index=pin(packet/'packet-index.protected.json',plan['packet_index_sha256'])
        docpath=namespace._bounded_path(str(packet/'target-free-documents.protected.jsonl'),ROOT,file=True)
        docpin=stack.enter_context(_PinnedFile(docpath,index['files'][docpath.name],64*1024**2));held.append(docpin)
        docpin.stream.seek(0);rows=[json.loads(line) for line in docpin.stream]
        documents={r['document_id']:CognitiveFrameDocument.from_mapping(r['document']) for r in rows}
        assert {k:v.sha256 for k,v in documents.items()}==index['document_sha256']
        artifacts=index['artifact_sha256']
        store,config,session,bindings,cost=helper.restore_completed_features(plan['completed_features'],documents=documents,
            source_package_sha256=artifacts['source_package'],prepared_receipt_file_sha256=artifacts['prepared_receipt'],
            read_pinned_json=lambda path,digest:pin(path,digest),bounded_directory=lambda path:namespace._bounded_path(path,ROOT,file=False))
        pins=[]
        for phase in ('n1','n2'):
            row=next(r for r in inventory['checkpoints'][phase] if r['name']=='final-checkpoint')
            checkpoint=namespace._bounded_path(str(old/phase/'final-checkpoint'),ROOT,file=False)
            manifest=pin(checkpoint/'manifest.json',row['manifest_file_sha256'])
            assert manifest['manifest_sha256']==row['manifest_sha256']
            assert sha256(canonical({k:v for k,v in manifest.items() if k!='manifest_sha256'})).hexdigest()==row['manifest_sha256']
            pins.append(CheckpointPin(str(checkpoint),row['manifest_sha256']))
        payload=namespace._bounded_path(str(old/'n2/final-checkpoint/resume.pt'),ROOT,file=True)
        payloadpin=stack.enter_context(_PinnedFile(payload,manifest['resume_sha256'],512*1024**2));held.append(payloadpin)
        payloadpin.stream.seek(0)
        state=torch.load(payloadpin.stream,map_location='cpu',weights_only=True)
        assert set(state)=={'model','objectives','optimizer','torch_rng','cuda_rng','python_rng','numpy_rng','steps','epoch','batch_index'}
        assert state['steps']==320 and state['epoch']==8 and state['batch_index']==0
        assert isinstance(state['model'],dict) and all(isinstance(t,torch.Tensor) and bool(torch.isfinite(t).all()) for t in state['model'].values())
        assert fingerprint(state['model'])==manifest['model_state_sha256']
        model=IdentityModel(config);model.load_state_dict(state['model'],strict=True);model.set_phase('inference');model.eval()
        assert fingerprint(model.state_dict())==manifest['model_state_sha256']
        construction=ConnectedConstruction(model,checkpoints=tuple(pins));consumer=construction.consumer()
        responsepath=namespace._bounded_path(str(old/'connected-responses.protected.jsonl'),ROOT,file=True)
        responsepin=stack.enter_context(_PinnedFile(responsepath,inventory['response_file_sha256'],64*1024**2));held.append(responsepin)
        responsepin.stream.seek(0);responses=[json.loads(line) for line in responsepin.stream]
        selected=next(r for r in responses if r['split']=='CALIBRATION_TRAIN')
        document=documents[selected['document_id']]
        replay=FrozenFrameProducer.replay(store,config,session,expected_session_sha256=session['receipt_sha256'])
        produced=replay.produce(document);response=consumer.respond(document,produced,session)
        assert response==selected['response']
        packet=model(produced.frame)
        assert fingerprint(packet)==response['packet_sha256']
        assert canonical(decision_to_mapping(packet))==canonical(selected['decision'])
        from dataclasses import asdict
        import gc
        from src.alice_personality.teaching.cases import ReviewedTeachingCase
        from src.alice_personality.teaching.contracts import AUTH_SCHEMA,TrainingRecipe
        from src.alice_personality.teaching.trainer import ArtifactPin,current_code_hashes,model_fingerprint
        from src.alice_personality.teaching.lineage import validate_allocation
        from src.alice_personality.identity.codec import fingerprint as full_fingerprint
        plan3=pin(ROOT/'rayan-personality/private-inputs/rayan-actual-personality-n3-20261009/owner-n3-run.protected.json','ef7d5a78b2c01a5b013f601c1865c2033ce70b8e3c72be2d858f2bbdc0b2b4be')
        assert plan3['schema']=='alice-personality-owner-actual-n3-run-v1' and plan3['owner']=='mxrayan'
        assert plan3['private_gradient_authorized'] is True and plan3['n3_targets_authorized'] is True
        assert plan3['purpose']=='development_construction' and plan3['acceptance_authority'] is False
        assert plan3['scientific_revision']==REV and plan3['cpu_threads']==1
        assert all(plan3[k] is False for k in ('owner_fidelity_approved','spoken_fidelity_verified','n0_approved','flora_work'))
        reload_receipt=json.loads((ROOT/'jobs/rayan-connected-model-reload-r2-20261009.json').read_bytes())
        assert sha256(canonical({k:v for k,v in reload_receipt.items() if k!='receipt_sha256'})).hexdigest()==reload_receipt['receipt_sha256']==plan3['verified_reload_receipt_sha256']
        assert reload_receipt['actual_complete_model_tensor_reload_verified'] is True and reload_receipt['full_decision_packet_equivalent'] is True
        packet3=namespace._bounded_path(plan3['packet_directory'],ROOT,file=False)
        index3=pin(packet3/'packet-index.protected.json',plan3['packet_index_sha256'])
        assert index3['parent_packet_index_sha256']==plan['packet_index_sha256']
        assert index3['actual_response_file_sha256']==inventory['response_file_sha256']
        assert index3['document_sha256']==index['document_sha256'] and index3['files']['target-free-documents.protected.jsonl']==index['files']['target-free-documents.protected.jsonl']
        doc3pin=stack.enter_context(_PinnedFile(namespace._bounded_path(str(packet3/'target-free-documents.protected.jsonl'),ROOT,file=True),index3['files']['target-free-documents.protected.jsonl'],64*1024**2));held.append(doc3pin)
        artifacts3=index3['artifact_sha256']
        assert all(artifacts3[k]==artifacts[k] for k in ('source_package','compiled_receipt','prepared_receipt'))
        protocol=pin(packet3/'annotation-protocol.protected.json',artifacts3['annotation_protocol'])
        review=pin(packet3/'author-review.protected.json',artifacts3['review_record'])
        pin(packet3/'teacher-spec.protected.json',protocol['teacher_spec_sha256'])
        assert protocol['other_teacher_used'] is False and protocol['actual_response_file_sha256']==inventory['response_file_sha256']
        assert review['development_targets_reviewed'] is True and review['owner_acceptance_recorded'] is False
        allocation=pin(packet3/'construction-allocation.protected.json',index3['files']['construction-allocation.protected.json']);validate_allocation(allocation)
        original_allocation=pin(old/'construction-allocation.protected.json','d2aa703171da58d1269345ae2fa44b0e3798f4617c297b1d908899a960d90446')
        assert allocation['family_splits']==original_allocation['family_splits'] and allocation['reference_family_ids']==original_allocation['reference_family_ids']
        assert all(allocation['supervision_family_splits'][k]==v for k,v in original_allocation['supervision_family_splits'].items())
        assert all(allocation['supervision_parents'][k]==v for k,v in original_allocation['supervision_parents'].items())
        casepin=stack.enter_context(_PinnedFile(namespace._bounded_path(str(packet3/'reviewed-teaching-cases.protected.jsonl'),ROOT,file=True),index3['files']['reviewed-teaching-cases.protected.jsonl'],64*1024**2));held.append(casepin)
        casepin.stream.seek(0);cases=tuple(ReviewedTeachingCase.from_mapping(json.loads(line)) for line in casepin.stream)
        assert len(cases)==20 and {c.case_id:c.sha256 for c in cases}==index3['case_sha256']
        for c in cases:
            assert c.phase=='calibration' and c.split=='CALIBRATION_TRAIN' and c.source_class=='governed_calibration_targets'
            assert c.document.sha256 in set(index['document_sha256'].values()) and dict(c.artifact_sha256)==artifacts3
            assert all(not f.available for t in c.targets if t.field in {'owner_fidelity_uncertainty','failure_tail_risk','evidence_sufficiency','voice_control_confidence'} for f in t.facts)
        helper_path=namespace._bounded_path(str(ROOT/'jobs/rayan-actual-personality-n3-20261009-helper.py'),ROOT,file=True)
        assert sha256(helper_path.read_bytes()).hexdigest()==plan3['calibration_helper_sha256']=='4ddbb3a8f3ef4407c7eece4bd226a65f4f7034562764799e158f0e4c92c4be1e'
        calibration=module('pinned_actual_calibration_helper',helper_path)
        from src.alice_personality.n1 import compiler
        from src.alice_personality.gemma_n0.preparation import verify_prepared
        source_path=namespace._bounded_path(str(ROOT/'rayan-personality/private-inputs/ALICE_EIPM_CURATED_FRONTIER_v2-3867ff04.zip'),ROOT,file=True)
        sourcepin=stack.enter_context(_PinnedFile(source_path,artifacts3['source_package'],512*1024**2));held.append(sourcepin)
        compiled=namespace._bounded_path(helper.COMPILED,ROOT,file=False)
        compiled_path=compiled/'compile_receipt.json';pin(compiled_path,artifacts3['compiled_receipt'])
        compile_verified=compiler.verify_compiled(compiled,expected_source_archive_sha256=artifacts3['source_package'],expected_receipt_sha256=helper.COMPILE_SEAL)
        prepared_path=namespace._bounded_path(helper.PREPARED,ROOT,file=True);pin(prepared_path,artifacts3['prepared_receipt']);verify_prepared(prepared_path)
        runroot=ROOT/'rayan-personality/private-compile'/('rayan-actual-personality-n3-'+os.environ['SLURM_JOB_ID'])
        assert not runroot.exists() and not runroot.is_symlink();runroot.mkdir(mode=0o700)
        output=namespace._bounded_path(str(runroot),ROOT,file=False)
        helper.verify_memory_allocation(plan3['memory_budget_bytes'],cost['estimated_live_memory_budget_bytes'],os.environ['SLURM_MEM_PER_NODE'])
        assert os.statvfs(output).f_bavail*os.statvfs(output).f_frsize>=plan3['checkpoint_disk_budget_bytes']
        print(json.dumps({'stage':'actual_n3_outcome_materialization','reviewed_outcomes':20,'source_document_count':60,'n1_n2_retrained':False}),flush=True)
        response_receipts={r['response']['receipt_sha256']:r['response'] for r in responses}
        n3=calibration.prepare_calibration(construction,cases=cases,expected_case_sha256=index3['case_sha256'],response_receipts=response_receipts,
            artifact_sha256=artifacts3,family_allocation=allocation,feature_store=store,closed_session=session,expected_session_sha256=session['receipt_sha256'])
        def write(path,value):
            with path.open('xb') as f:f.write(canonical(value)+b'\n')
        allocation_path=output/'construction-allocation.protected.json';write(allocation_path,allocation)
        target_path=output/'reviewed-batches.protected.json';write(target_path,n3.reviewed_targets)
        admitted={'source_package':ArtifactPin(source_path,artifacts3['source_package']),
            'compiled_receipt':ArtifactPin(compiled_path,artifacts3['compiled_receipt']),
            'prepared_receipt':ArtifactPin(prepared_path,artifacts3['prepared_receipt']),
            'reviewed_targets':ArtifactPin(target_path,sha256(target_path.read_bytes()).hexdigest()),
            'source_families':ArtifactPin(allocation_path,sha256(allocation_path.read_bytes()).hexdigest())}
        recipe=TrainingRecipe(**plan3['recipe']);recipe.validate()
        assert recipe.phase=='calibration' and recipe.adapt_n1 is False and len(n3.reviewed_targets['batch_order']['CALIBRATION_TRAIN'])*plan3['epochs']==plan3['expected_n3_updates']
        initial=model_fingerprint(model);core=full_fingerprint({k:v for k,v in model.state_dict().items() if k.startswith(('n1.','n2.'))})
        auth={'schema':AUTH_SCHEMA,'authorization_id':plan3['authorization_id'],'phase':'calibration','purpose':'development_construction',
            'source_class':'governed_calibration_targets','private_gradient_authorized':True,'mechanical_steps_authorized':False,
            'artifact_file_sha256':{k:v.sha256 for k,v in admitted.items()},'model_config_sha256':sha256(canonical(asdict(config))).hexdigest(),
            'model_initial_state_sha256':initial,'recipe_sha256':recipe.fingerprint(),'code_sha256':current_code_hashes(),
            'owner_run_plan_sha256':'ef7d5a78b2c01a5b013f601c1865c2033ce70b8e3c72be2d858f2bbdc0b2b4be','primary_packet_index_sha256':plan3['packet_index_sha256'],
            'acceptance_authority':False,'owner_fidelity_approved':False,'n0_approved':False}
        auth_path=output/'current-authorization.protected.json';write(auth_path,auth);admitted['authorization']=ArtifactPin(auth_path,sha256(auth_path.read_bytes()).hexdigest())
        epochs=output/'epoch-checkpoints';epochs.mkdir(mode=0o700)
        print(json.dumps({'stage':'actual_n3_training','reviewed_outcomes':20,'epochs':plan3['epochs'],'planned_optimizer_updates':plan3['expected_n3_updates']}),flush=True)
        result=calibration.run_calibration(construction,recipe,admitted,prepared=n3,epochs=plan3['epochs'],destination=output/'final-checkpoint',epoch_checkpoint_directory=epochs)
        assert result['step']==plan3['expected_n3_updates'] and full_fingerprint({k:v for k,v in model.state_dict().items() if k.startswith(('n1.','n2.'))})==core
        print(json.dumps({'stage':'actual_n3_checkpoint_retained','optimizer_updates':result['step']}),flush=True)
        response_output=output/'connected-responses.protected.jsonl'
        document_splits={r['document_id']:r['split'] for r in rows}
        with response_output.open('xb') as f:
            for row in calibration.iter_connected_responses(construction,n3,documents={name:(document_splits[name],doc) for name,doc in documents.items()}):
                f.write(canonical(row)+b'\n')
        for value in held:value.verify()
        assert sha256(helper_path.read_bytes()).hexdigest()==plan3['calibration_helper_sha256']
        assert compiler.verify_compiled(compiled,expected_source_archive_sha256=artifacts3['source_package'],expected_receipt_sha256=helper.COMPILE_SEAL)==compile_verified
        namespace._clean_code(code,REV)
        summary={'schema':'alice-personality-actual-n3-connected-construction-v1','state':'N1_N2_N3_CONNECTED_CONSTRUCTION_RETAINED_UNQUALIFIED',
            'owner':'mxrayan','original_job':577095,'n3_job':int(os.environ['SLURM_JOB_ID']),'scientific_revision':REV,
            'calibration_helper_sha256':plan3['calibration_helper_sha256'],'owner_run_plan_sha256':'ef7d5a78b2c01a5b013f601c1865c2033ce70b8e3c72be2d858f2bbdc0b2b4be','packet_index_sha256':plan3['packet_index_sha256'],
            'isolation':isolation,'n1_updates':200,'n2_updates':320,'n3_updates':result['step'],'n3_epochs':plan3['epochs'],
            'n1_n2_state_unchanged':True,'n3_checkpoint_manifest_sha256':result['manifest_sha256'],'model_state_sha256':result['model_state_sha256'],
            'response_count':len(documents),'protected_response_file_sha256':sha256(response_output.read_bytes()).hexdigest(),
            'source_geometry':[config.provider_state_count,config.provider_width],'closed_session_sha256':session['receipt_sha256'],
            'available_target_case_counts':index3['available_target_case_counts'],'calibration_cases':20,
            'owner_fidelity_targets_available':False,'measured_failure_targets_available':False,'numeric_voice_gold_created':False,
            'private_payload_exported':False,'n0_approved':False,'owner_fidelity_approved':False,'spoken_fidelity_verified':False,
            'acceptance_authority':False,'full_personality_complete':False,'qualification':'UNQUALIFIED','flora_work':False}
        summary['receipt_sha256']=sha256(canonical(summary)).hexdigest();write(output/'connected-construction-summary.json',summary)
        write(ROOT/'jobs/rayan-actual-personality-n3-20261009.json',summary)
        print(json.dumps(summary,sort_keys=True),flush=True)
code=Path(os.environ['PERSONALITY_REPO_ROOT'])
guard=code/'scripts/eipm/n1/enter_private_network_namespace.py'
assert sha256(guard.read_bytes()).hexdigest()==GUARD
namespace=module('owner_namespace_reload',guard)
try:reload(namespace)
except Exception as exc:
    import traceback
    frames=[{'module':Path(f.filename).name,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(exc.__traceback__)]
    try:
        namespace._p2_guard()
        failure_root=namespace._bounded_path(str(ROOT/'rayan-personality/private-compile'/('rayan-actual-personality-n3-'+os.environ['SLURM_JOB_ID'])),ROOT,file=False)
        with (failure_root/'failure-detail.protected.json').open('xb') as failure_stream:
            failure_stream.write(canonical({'exception_type':type(exc).__name__,'message':str(exc),'public_code_frames':frames})+b'\n')
    except Exception:pass
    print(json.dumps({'state':'N3_CONSTRUCTION_REFUSED','exception_type':type(exc).__name__,'public_code_frames':frames}),flush=True)
    raise SystemExit(2)
