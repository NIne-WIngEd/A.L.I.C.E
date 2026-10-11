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
        import stat
        output = namespace._bounded_path(str(ROOT/'rayan-personality/private-compile/rayan-actual-personality-n3-577146'),ROOT,file=False)
        assert stat.S_IMODE(output.stat().st_mode)==0o700
        summary_payload=(output/'connected-construction-summary.json').read_bytes()
        summary=json.loads(summary_payload)
        summary_seal='4a87d32227a597a85eed50b826b39ed7b390d6270eafe40e3ed14d5433acd859'
        assert sha256(canonical({k:v for k,v in summary.items() if k!='receipt_sha256'})).hexdigest()==summary['receipt_sha256']==summary_seal
        assert sha256(summary_payload).hexdigest()=='00ac2b36cf218ed694080dfd1a7ccf7d15cb4561f294762d28df432b7924a178'
        assert summary_payload==(ROOT/'jobs/rayan-actual-personality-n3-20261009.json').read_bytes()
        assert summary['owner']=='mxrayan' and summary['n3_job']==577146 and summary['original_job']==577095
        assert summary['scientific_revision']==REV and summary['source_geometry']==[49,3840]
        assert summary['state']=='N1_N2_N3_CONNECTED_CONSTRUCTION_RETAINED_UNQUALIFIED'
        assert summary['n3_updates']==160 and summary['n3_epochs']==8 and summary['n1_n2_state_unchanged'] is True
        assert all(summary[k] is False for k in ('n0_approved','acceptance_authority','owner_fidelity_approved','spoken_fidelity_verified','full_personality_complete','flora_work'))
        core_before=fingerprint({k:v for k,v in model.state_dict().items() if k.startswith(('n1.','n2.'))})
        initial_n2=fingerprint(model.state_dict())
        expected_state=model.state_dict()
        metadata={}
        for name in ('reviewed-batches.protected.json','current-authorization.protected.json','construction-allocation.protected.json'):
            path=namespace._bounded_path(str(output/name),ROOT,file=True)
            digest=sha256(path.read_bytes()).hexdigest()
            metadata[name]={'file_sha256':digest}
            value=pin(path,digest)
            if name=='current-authorization.protected.json':
                assert value['phase']=='calibration' and value['private_gradient_authorized'] is True
                assert value['model_initial_state_sha256']==initial_n2
                assert value['owner_run_plan_sha256']==summary['owner_run_plan_sha256']
        checkpoints=[]
        epoch_root=namespace._bounded_path(str(output/'epoch-checkpoints'),ROOT,file=False)
        assert {p.name for p in epoch_root.iterdir()}=={'epoch-%04d'%e for e in range(1,9)}
        for checkpoint in [*sorted(epoch_root.iterdir()),output/'final-checkpoint']:
            checkpoint=namespace._bounded_path(str(checkpoint),ROOT,file=False)
            assert {p.name for p in checkpoint.iterdir()}=={'manifest.json','resume.pt'}
            manifest_path=namespace._bounded_path(str(checkpoint/'manifest.json'),ROOT,file=True)
            digest=sha256(manifest_path.read_bytes()).hexdigest()
            manifest3=pin(manifest_path,digest)
            assert sha256(canonical({k:v for k,v in manifest3.items() if k!='manifest_sha256'})).hexdigest()==manifest3['manifest_sha256']
            assert manifest3['schema']=='alice-personality-resumable-teaching-checkpoint-v7' and manifest3['phase']=='calibration'
            assert manifest3['step']==20*manifest3['epoch'] and manifest3['batch_index']==0
            assert len(manifest3['events'])==manifest3['step']
            assert manifest3['binding']['model_initial_state_sha256']==initial_n2
            assert manifest3['qualification']=='UNQUALIFIED' and manifest3['acceptance_authority'] is False
            if checkpoint.name!='final-checkpoint':
                assert checkpoint.name=='epoch-%04d'%manifest3['epoch']
            payload_path=namespace._bounded_path(str(checkpoint/'resume.pt'),ROOT,file=True)
            assert stat.S_IMODE(payload_path.stat().st_mode)==0o600
            payload_pin=stack.enter_context(_PinnedFile(payload_path,manifest3['resume_sha256'],512*1024**2));held.append(payload_pin)
            payload_pin.stream.seek(0)
            restored=torch.load(payload_pin.stream,map_location='cpu',weights_only=True)
            assert set(restored)=={'model','objectives','optimizer','torch_rng','cuda_rng','python_rng','numpy_rng','steps','epoch','batch_index'}
            assert restored['steps']==manifest3['step'] and restored['epoch']==manifest3['epoch'] and restored['batch_index']==0
            assert set(restored['model'])==set(expected_state)
            assert all(isinstance(t,torch.Tensor) and t.shape==expected_state[k].shape and t.dtype==expected_state[k].dtype and bool(torch.isfinite(t).all()) for k,t in restored['model'].items())
            assert fingerprint(restored['model'])==manifest3['model_state_sha256']
            assert fingerprint({k:v for k,v in restored['model'].items() if k.startswith(('n1.','n2.'))})==core_before
            checkpoints.append({'name':checkpoint.name,'manifest_file_sha256':digest,'manifest_sha256':manifest3['manifest_sha256'],
                'resume_sha256':manifest3['resume_sha256'],'payload_bytes':payload_path.stat().st_size,
                'step':manifest3['step'],'epoch':manifest3['epoch'],'batch_index':0,'event_count':len(manifest3['events']),
                'model_state_sha256':manifest3['model_state_sha256'],'complete_finite_model_tensors_verified':True,'n1_n2_state_unchanged':True})
            if checkpoint.name=='final-checkpoint':
                assert manifest3['step']==160 and manifest3['epoch']==8
                assert manifest3['manifest_sha256']==summary['n3_checkpoint_manifest_sha256']
                assert manifest3['model_state_sha256']==summary['model_state_sha256']
                model.load_state_dict(restored['model'],strict=True)
                pins.append(CheckpointPin(str(checkpoint),manifest3['manifest_sha256']))
            del restored
        model.set_phase('inference');model.eval()
        assert fingerprint(model.state_dict())==summary['model_state_sha256']
        construction=ConnectedConstruction(model,checkpoints=tuple(pins));consumer=construction.consumer()
        response_path=namespace._bounded_path(str(output/'connected-responses.protected.jsonl'),ROOT,file=True)
        response_pin=stack.enter_context(_PinnedFile(response_path,summary['protected_response_file_sha256'],64*1024**2));held.append(response_pin)
        response_pin.stream.seek(0);new_responses=[json.loads(line) for line in response_pin.stream]
        assert len(new_responses)==60 and {r['document_id'] for r in new_responses}==set(documents)

        from dataclasses import asdict
        from src.alice_personality.identity.consumer import RenderedResponse
        from collections import Counter
        helper_pins={'governed_response.py': 'bc69b2fa38eb416229e51d1dc36c84dbb4cd84924141a603ed1b7deac7ebe8b2', 'conditional_path_diagnostic.py': 'b775aff5dc88684dc958aa4b50eff7b08dcae69f4806a62b3f6435d442aedc31'}
        loaded={}
        for filename,digest in helper_pins.items():
            path=namespace._bounded_path(str(ROOT/'jobs'/('rayan-20261011-'+filename)),ROOT,file=True)
            assert sha256(path.read_bytes()).hexdigest()==digest
            loaded[filename]=module('rayan_'+filename.replace('.','_'),path)
        governance=loaded['governed_response.py']
        diagnostic=loaded['conditional_path_diagnostic.py']
        recovery=pin(ROOT/'jobs/rayan-full-personality-recovery-20261010.json',
            'a4a71e7836929b2cc05c0393d72f71f20e221de15eff7dda458b75ed0ed575b9')
        assert recovery['receipt_sha256']=='3a76675c9044b5d632b59cb6933d15f6483b6dbbac797cbcd98ca7989dc1c88f'
        assert recovery['full_response_and_decision_exact_reproduction_count']==60
        job=int(os.environ['SLURM_JOB_ID'])
        new_output=ROOT/'rayan-personality/private-compile'/('rayan-governed-conditional-'+str(job))
        new_output.mkdir(mode=0o700)
        new_output=namespace._bounded_path(str(new_output),ROOT,file=False)
        assert stat.S_IMODE(new_output.stat().st_mode)==0o700
        by_id={row['document_id']:row for row in new_responses}
        measured=[]
        topics=('deadpan_teasing','teasing_close_person')
        with (new_output/'conditional-path.protected.jsonl').open('xb') as stream:
            for topic in topics:
                for suffix,donor_suffix in (('train-1','train-2'),('train-2','train-1'),('cal-new','train-1')):
                    document_id=topic+'-'+suffix
                    donor_id=topic+'-'+donor_suffix
                    document=documents[document_id]
                    produced=replay.produce(document)
                    donor_produced=replay.produce(documents[donor_id])
                    assert tuple(tuple(r for r in row if r is not None and r.identity_core_allowed) for row in produced.frame.source_records)==tuple(tuple(r for r in row if r is not None and r.identity_core_allowed) for row in donor_produced.frame.source_records)
                    measurement=diagnostic.measure_conditional_path(model,produced.frame,
                        concept_id=topic+':policy',donor_frame=donor_produced.frame)
                    original=by_id[document_id]
                    for field in ('preferences','co_valid_probabilities','contraindications'):
                        assert canonical(measurement['baseline'][field])==canonical(original['decision'][field])
                    record={'document_id':document_id,'document_sha256':document.sha256,
                        'split':original['split'],'donor_document_id':donor_id,
                        'donor_document_sha256':documents[donor_id].sha256,
                        'closed_session_sha256':session['receipt_sha256'],
                        'native_response_sha256':original['response']['receipt_sha256'],
                        'measurement':measurement,'training_eligible':False,'evaluation_eligible':False}
                    record['receipt_sha256']=sha256(canonical(record)).hexdigest()
                    stream.write(canonical(record)+b'\n');stream.flush()
                    measured.append(record)
                    print(json.dumps({'stage':'actual_conditional_path_measured','completed':len(measured),'total':6}),flush=True)
                    del produced,donor_produced
        aggregate={}
        for mode in measured[0]['measurement']['modes']:
            stats={}
            changed=0
            for field in ('preferences','co_valid_probabilities','contraindications'):
                differences=[]
                for row in measured:
                    baseline=row['measurement']['baseline'][field][0]
                    modified=row['measurement']['modes'][mode][field][0]
                    differences.extend(abs(a-b) for a,b in zip(baseline,modified))
                stats[field]={'max_absolute_delta':max(differences),'mean_absolute_delta':sum(differences)/len(differences)}
            for row in measured:
                base=row['measurement']['baseline']['preferences'][0]
                variant=row['measurement']['modes'][mode]['preferences'][0]
                changed+=int(max(range(len(base)),key=base.__getitem__)!=max(range(len(variant)),key=variant.__getitem__))
            stats['preferred_candidate_changes']=changed
            aggregate[mode]=stats
        diagnostic_summary={'schema':'alice-personality-conditional-path-summary-v1',
            'owner':'mxrayan','job':job,'scientific_revision':REV,
            'model_state_sha256':summary['model_state_sha256'],'helper_sha256':helper_pins,
            'measurement_documents':6,'original_train_documents':4,'original_calibration_documents':2,
            'measurements_authentic_responses':False,'latent_interventions_only':True,
            'source_and_candidate_authority_unchanged':True,'additional_gradient_steps':0,
            'independent_qualification':False,'n0_approved':False,'full_personality_complete':False,
            'training_eligible':False,'evaluation_eligible':False,'mode_deltas':aggregate,
            'protected_file_sha256':sha256((new_output/'conditional-path.protected.jsonl').read_bytes()).hexdigest()}
        diagnostic_summary['receipt_sha256']=sha256(canonical(diagnostic_summary)).hexdigest()
        with (ROOT/'jobs/rayan-conditional-path-20261011.json').open('xb') as stream:
            stream.write(canonical(diagnostic_summary)+b'\n')
        print(json.dumps({'stage':'conditional_path_summary_retained','receipt_sha256':diagnostic_summary['receipt_sha256']}),flush=True)
        statuses=Counter();renderings=Counter();scope_counts=Counter();splits=Counter()
        def verbatim(envelope):
            preferred=envelope['preferred_candidate_ids'][0]
            intent=next(r for r in envelope['preference_selection']['intents'] if r['candidate_id']==preferred)
            return RenderedResponse(envelope['response_text'],intent['voice_controls'],
                'rayan-verbatim-construction-renderer',None)
        with (new_output/'governed-responses.protected.jsonl').open('xb') as stream:
            for count,row in enumerate(new_responses,1):
                document=documents[row['document_id']]
                produced=replay.produce(document)
                governed=governance.respond_with_governance(consumer,document,produced,session,renderer=verbatim)
                assert canonical(governed['decision'])==canonical(row['decision'])
                assert governed['packet_sha256']==row['response']['packet_sha256']
                assert canonical(governed['preference_selection'])==canonical(row['response'])
                assert sha256(canonical({k:v for k,v in governed.items() if k!='receipt_sha256'})).hexdigest()==governed['receipt_sha256']
                assert governed['preference_selection']['model_state_sha256']==summary['model_state_sha256']
                assert all(governed[k] is False for k in ('acceptance_authority','owner_fidelity_approved','spoken_fidelity_verified'))
                record={'document_id':row['document_id'],'split':row['split'],'document_sha256':document.sha256,
                    'parent_native_response_sha256':row['response']['receipt_sha256'],'response':governed}
                record['receipt_sha256']=sha256(canonical(record)).hexdigest()
                stream.write(canonical(record)+b'\n');stream.flush()
                statuses[governed['status']]+=1;renderings[governed['rendering']['status']]+=1;splits[row['split']]+=1
                for field,value in governed['output_scope'].items():
                    if value['taught']:scope_counts[field]+=1
                del produced
                if count%10==0:
                    print(json.dumps({'stage':'actual_full_governing_response','completed':count,'total':60}),flush=True)
        for value in held:value.verify()
        namespace._clean_code(code,REV)
        assert fingerprint(model.state_dict())==summary['model_state_sha256']
        for filename,digest in helper_pins.items():
            assert sha256((ROOT/'jobs'/('rayan-20261011-'+filename)).read_bytes()).hexdigest()==digest
        result={'schema':'alice-personality-governed-conditional-continuation-v1',
            'owner':'mxrayan','job':job,'scientific_revision':REV,'isolation':isolation,
            'parent_full_recovery_sha256':recovery['receipt_sha256'],
            'model_state_sha256':summary['model_state_sha256'],'helper_sha256':helper_pins,
            'checkpoint_manifest_sha256':[p.manifest_sha256 for p in pins],
            'actual_full_native_decision_count':60,'actual_governed_response_count':60,
            'native_decision_and_preference_preserved':True,'model_state_unchanged':True,
            'response_status_counts':dict(statuses),'rendering_status_counts':dict(renderings),
            'split_counts':dict(splits),'taught_scope_document_counts':dict(scope_counts),
            'conditional_diagnostic_sha256':diagnostic_summary['receipt_sha256'],
            'protected_governed_responses_sha256':sha256((new_output/'governed-responses.protected.jsonl').read_bytes()).hexdigest(),
            'closed_session_sha256':session['receipt_sha256'],'source_geometry':[49,3840],
            'additional_gradient_steps':0,'text_realization_only':True,
            'independent_behavioral_qualification':False,'n0_approved':False,'full_personality_complete':False,
            'owner_fidelity_approved':False,'spoken_fidelity_verified':False,'private_payload_exported':False,'flora_work':False}
        result['receipt_sha256']=sha256(canonical(result)).hexdigest()
        with (ROOT/'jobs/rayan-governed-conditional-20261011.json').open('xb') as stream:stream.write(canonical(result)+b'\n')
        print(json.dumps(result,sort_keys=True),flush=True)
code=Path(os.environ['PERSONALITY_REPO_ROOT'])
guard=code/'scripts/eipm/n1/enter_private_network_namespace.py'
assert sha256(guard.read_bytes()).hexdigest()==GUARD
namespace=module('owner_namespace_governed_conditional',guard)
try:reload(namespace)
except Exception as exc:
    import traceback
    frames=[{'module':Path(f.filename).name,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(exc.__traceback__)]
    print(json.dumps({'state':'GOVERNED_CONDITIONAL_REFUSED','exception_type':type(exc).__name__,'public_code_frames':frames}),flush=True)
    raise SystemExit(2)
