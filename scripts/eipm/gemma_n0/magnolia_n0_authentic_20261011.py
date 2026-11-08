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

def verify_contrast_inputs(originals, rows, bindings, index, contrast):
    from src.alice_personality.identity.codec import CognitiveFrameDocument
    result={}
    for row in rows:
        assert set(row)=={'document_id','parent_document_id','comparison_kind','split','document'}
        name=row['document_id'];parent=row['parent_document_id']
        assert row['split']=='TRAIN' and name not in result and parent in originals
        original=originals[parent];document=CognitiveFrameDocument.from_mapping(row['document'])
        assert document.sha256==index['document_sha256'][name]
        binding=bindings[name]
        assert sha256(canonical({k:v for k,v in binding.items() if k!='binding_sha256'})).hexdigest()==binding['binding_sha256']
        if row['comparison_kind']=='original':
            assert set(binding)=={'schema','variant_id','parent_case_id','parent_document_sha256',
                'document_sha256','parent_split','independent_qualification','n0_approved','binding_sha256'}
            assert binding['schema']=='alice-personality-original-n0-contrast-control-v1'
            assert binding['variant_id']==name and binding['parent_case_id']==parent and binding['parent_split']=='TRAIN'
            assert binding['independent_qualification'] is False and binding['n0_approved'] is False
            assert document.sha256==original.sha256==binding['document_sha256']==binding['parent_document_sha256']
        else:
            assert binding['variant_id']==name and binding['comparison_kind']==row['comparison_kind']
            assert binding['parent']['case_id']==parent and binding['parent']['split']=='TRAIN'
            contrast.verify_context_contrast(original,document,binding,expected_binding_sha256=binding['binding_sha256'])
        assert canonical(document.to_mapping()['candidates'])==canonical(original.to_mapping()['candidates'])
        identity=lambda d:[s for s in d.to_mapping()['sources'] if s['record']['identity_core_allowed']]
        assert canonical(identity(document))==canonical(identity(original))
        result[name]=document
    assert set(result)==set(bindings)==set(index['document_sha256'])
    return result

def directional_report(candidate_ids, decision, target):
    regions=target['review']['candidate_regions']
    assert [r['candidate_id'] for r in regions]==candidate_ids
    errors=[]
    for number,region in enumerate(regions):
        for field,key in (('co_valid_probabilities','co_valid_region'),('contraindications','contraindication_region')):
            value=decision[field][0][number]
            assert 0<=value<=1 and region[key] in ('POSITIVE','NEGATIVE')
            passed=value>=.5 if region[key]=='POSITIVE' else value<.5
            if not passed:errors.append({'candidate_id':candidate_ids[number],'field':field,'actual':value,'expected_region':region[key]})
    return {'region_comparisons':2*len(regions),'region_errors':len(errors),'errors':errors,
        'primary_targets_provisional':True,'numeric_probability_gold':False,'independent_qualification':False}

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

        import gc
        from collections import Counter
        from src.alice_personality.identity.feature_store import FrozenFeatureStore
        from src.alice_personality.gemma_n0.preparation import verify_prepared
        from src.alice_personality.identity.consumer import RenderedResponse
        from transformers import AutoProcessor
        helper_pins={'governed_response.py': 'bc69b2fa38eb416229e51d1dc36c84dbb4cd84924141a603ed1b7deac7ebe8b2', 'n0_semantic_contrasts.py': 'c6a0e51ab687aa905f0ca4f10a1ad9b3e6c268cb6a587a093835ab220756f2b4'}
        loaded={}
        for filename,digest in helper_pins.items():
            path=namespace._bounded_path(str(ROOT/'jobs'/('rayan-20261011-'+filename)),ROOT,file=True)
            assert sha256(path.read_bytes()).hexdigest()==digest
            loaded[filename]=module('rayan_n0_'+filename.replace('.','_'),path)
        governance=loaded['governed_response.py'];contrast=loaded['n0_semantic_contrasts.py']
        source_input=namespace._bounded_path(str(ROOT/'rayan-personality/private-inputs/rayan-n0-semantic-20261011'),ROOT,file=False)
        assert stat.S_IMODE(source_input.stat().st_mode)==0o700
        contrast_index=pin(source_input/'n0-probe-index.protected.json','b7961cbb0294863dc4ab213709054bf5ccb81097e93b1113c96d5f5363fd3545')
        assert sha256(canonical({k:v for k,v in contrast_index.items() if k!='receipt_sha256'})).hexdigest()==contrast_index['receipt_sha256']
        assert contrast_index['parent_source_index_sha256']=='abdcfbda3337ddaa50bc0f6a5500debc3a3cc5b841f686bf2685bc01f30b76ce'
        assert contrast_index['original_document_file_sha256']==index['files']['target-free-documents.protected.jsonl']
        assert contrast_index['contrast_helper_sha256']==helper_pins['n0_semantic_contrasts.py']
        assert contrast_index['source_package_sha256']==artifacts['source_package']
        assert contrast_index['prepared_receipt_sha256']==artifacts['prepared_receipt']
        assert contrast_index['targets_enter_n0'] is False and contrast_index['source_and_candidate_authority_unchanged'] is True
        assert all(contrast_index[k] is False for k in ('independent_qualification','n0_approved','full_personality_complete'))
        specification=pin(source_input/'contrast-specifications.protected.json',contrast_index['primary_specifications_sha256'])
        assert sha256(canonical({k:v for k,v in specification.items() if k!='receipt_sha256'})).hexdigest()==specification['receipt_sha256']
        source_bindings=pin(source_input/'contrast-bindings.protected.json',contrast_index['files']['contrast-bindings.protected.json'])
        targets=pin(source_input/'primary-directional-targets.protected.json',contrast_index['files']['primary-directional-targets.protected.json'])
        input_path=namespace._bounded_path(str(source_input/'target-free-documents.protected.jsonl'),ROOT,file=True)
        input_pin=stack.enter_context(_PinnedFile(input_path,contrast_index['files'][input_path.name],64*1024**2));held.append(input_pin)
        input_pin.stream.seek(0);contrast_rows=[json.loads(line) for line in input_pin.stream]
        contrast_documents=verify_contrast_inputs(documents,contrast_rows,source_bindings,contrast_index,contrast)
        assert len(contrast_rows)==10==contrast_index['documents'] and len({r['parent_document_id'] for r in contrast_rows})==2
        assert set(targets)==set(contrast_documents)
        for name,target in targets.items():
            assert target['document_sha256']==contrast_documents[name].sha256
            assert all(target[k] is False for k in ('training_eligible','evaluation_eligible','independent_qualification'))
            assert target['review']['external_teacher_used'] is False and target['review']['owner_acceptance'] is False
        job=int(os.environ['SLURM_JOB_ID'])
        new_output=ROOT/'rayan-personality/private-compile'/('rayan-n0-semantic-'+str(job))
        new_output.mkdir(mode=0o700);new_output=namespace._bounded_path(str(new_output),ROOT,file=False)
        assert stat.S_IMODE(new_output.stat().st_mode)==0o700
        preparation_path=namespace._bounded_path(helper.PREPARED,ROOT,file=True)
        prepared=pin(preparation_path,artifacts['prepared_receipt'])
        preparation=verify_prepared(preparation_path)
        geometry=preparation['model_geometry']
        assert [geometry['hidden_state_count'],geometry['hidden_size']]==[49,3840]
        assert preparation['runtime']['dtype']=='bfloat16'
        processor=AutoProcessor.from_pretrained(preparation['snapshot_path'],local_files_only=True,trust_remote_code=False)
        profile=helper.token_cost_profile(list(contrast_documents.values()),processor.tokenizer,
            layers=49,width=3840,max_tokens=geometry['max_position_embeddings'])
        del processor;gc.collect()
        estimate=48*1024**3+profile['largest_padded_frame_value_bytes']*12
        helper.verify_memory_allocation(120*1024**3,estimate,os.environ['SLURM_MEM_PER_NODE'])
        disk_required=profile['state_value_bytes']+profile['serialization_overhead_budget_bytes']+1024**3
        disk=os.statvfs(new_output);assert disk.f_bavail*disk.f_frsize>=disk_required
        profile.update(estimated_live_memory_budget_bytes=estimate,declared_memory_budget_bytes=120*1024**3,
            actual_slurm_memory_bytes=int(os.environ['SLURM_MEM_PER_NODE'])*1024**2)
        fresh_store=FrozenFeatureStore.create(new_output/'feature-store',source_package_sha256=artifacts['source_package'])
        producer=FrozenFrameProducer.load(preparation_path,expected_file_sha256=artifacts['prepared_receipt'],device='cpu',
            learned_width=config.learned_width,attention_heads=config.attention_heads,
            graph_layers=config.graph_layers,readout_layers=config.readout_layers,feature_store=fresh_store)
        assert producer.config==config
        fresh_bindings={}
        for count,row in enumerate(contrast_rows,1):
            name=row['document_id'];produced=producer.produce(contrast_documents[name])
            fresh_bindings[name]=produced.binding;del produced;gc.collect()
            print(json.dumps({'stage':'authentic_new_n0_source_frame','completed':count,'total':10}),flush=True)
        fresh_session=producer.close_and_verify()
        assert fresh_session['schema']=='alice-personality-feature-session-v2'
        assert fresh_session['status']=='CLOSED_SOURCE_VERIFIED' and fresh_session['mechanical_fixture_only'] is False
        assert len(fresh_session['frame_binding_sha256'])==10
        del producer,fresh_store;gc.collect()
        for filename,value in (('feature-session.protected.json',fresh_session),('frame-bindings.protected.json',fresh_bindings),
                ('token-cost-profile.protected.json',profile)):
            with (new_output/filename).open('xb') as stream:stream.write(canonical(value)+b'\n')
        fresh_store=FrozenFeatureStore.open(new_output/'feature-store',expected_manifest_sha256=fresh_session['feature_store']['manifest_sha256'])
        fresh_replay=FrozenFrameProducer.replay(fresh_store,config,fresh_session,expected_session_sha256=fresh_session['receipt_sha256'])
        results={};statuses=Counter()
        def verbatim(envelope):
            preferred=envelope['preferred_candidate_ids'][0]
            intent=next(i for i in envelope['preference_selection']['intents'] if i['candidate_id']==preferred)
            return RenderedResponse(envelope['response_text'],dict(intent['voice_controls']),'rayan-verbatim-text-realizer-v1',None)
        with (new_output/'authentic-governed-contrasts.protected.jsonl').open('xb') as stream:
            for count,row in enumerate(contrast_rows,1):
                name=row['document_id'];document=contrast_documents[name]
                produced=fresh_replay.produce(document)
                assert produced.binding==fresh_bindings[name]
                response=governance.respond_with_governance(consumer,document,produced,fresh_session,renderer=verbatim)
                assert response['preference_selection']['model_state_sha256']==summary['model_state_sha256']
                region_report=directional_report([c.candidate_id for c in document.candidates],response['decision'],targets[name])
                lengths=[{'namespace':e['namespace'],'pointer':e['pointer'],'token_count':e['token_count']}
                         for e in produced.binding['entries']]
                record={'document_id':name,'comparison_kind':row['comparison_kind'],'parent_document_id':row['parent_document_id'],
                    'document_sha256':document.sha256,'contrast_binding':source_bindings[name],
                    'feature_binding':produced.binding,'actual_token_lengths':lengths,
                    'response':response,'primary_directional_report':region_report,
                    'training_eligible':False,'evaluation_eligible':False,'independent_qualification':False}
                record['receipt_sha256']=sha256(canonical(record)).hexdigest()
                stream.write(canonical(record)+b'\n');stream.flush();results[name]=record;statuses[response['status']]+=1
                print(json.dumps({'stage':'actual_n0_governed_contrast','completed':count,'total':10}),flush=True)
                del produced
        contrasts=[]
        for row in contrast_rows:
            if row['comparison_kind']=='original':continue
            baseline=next(r for r in contrast_rows if r['parent_document_id']==row['parent_document_id'] and r['comparison_kind']=='original')
            first=results[baseline['document_id']];second=results[row['document_id']]
            differences={field:[b-a for a,b in zip(first['response']['decision'][field][0],second['response']['decision'][field][0])]
                for field in ('co_valid_probabilities','contraindications','preferences')}
            contrasts.append({'comparison_kind':row['comparison_kind'],
                'baseline_receipt_sha256':first['receipt_sha256'],'variant_receipt_sha256':second['receipt_sha256'],
                'native_scalar_deltas':differences,
                'preferred_candidate_changed':first['response']['preferred_candidate_ids']!=second['response']['preferred_candidate_ids'],
                'acceptable_set_changed':first['response']['acceptable_candidate_ids']!=second['response']['acceptable_candidate_ids'],
                'native_packet_changed':first['response']['packet_sha256']!=second['response']['packet_sha256'],
                'token_length_confound_disclosed':True,'causal_failure_attribution':'UNRESOLVED'})
        with (new_output/'matched-comparisons.protected.json').open('xb') as stream:stream.write(canonical(contrasts)+b'\n')
        for value in held:value.verify()
        namespace._clean_code(code,REV)
        assert fingerprint(model.state_dict())==summary['model_state_sha256']
        for filename,digest in helper_pins.items():assert sha256((ROOT/'jobs'/('rayan-20261011-'+filename)).read_bytes()).hexdigest()==digest
        result={'schema':'alice-personality-authentic-n0-development-summary-v1','owner':'mxrayan','job':job,
            'scientific_revision':REV,'isolation':isolation,'model_state_sha256':summary['model_state_sha256'],
            'checkpoint_manifest_sha256':[p.manifest_sha256 for p in pins],'helper_sha256':helper_pins,
            'protected_input_index_sha256':'b7961cbb0294863dc4ab213709054bf5ccb81097e93b1113c96d5f5363fd3545',
            'source_package_sha256':artifacts['source_package'],'prepared_receipt_file_sha256':artifacts['prepared_receipt'],
            'new_closed_feature_session_sha256':fresh_session['receipt_sha256'],'source_geometry':[49,3840],
            'actual_new_source_documents':10,'matched_original_training_families':2,'actual_governed_responses':10,
            'profile':profile,'response_status_counts':dict(statuses),
            'directional_region_comparisons':sum(r['primary_directional_report']['region_comparisons'] for r in results.values()),
            'directional_region_errors':sum(r['primary_directional_report']['region_errors'] for r in results.values()),
            'matched_control_preference_changes':sum(c['preferred_candidate_changed'] for c in contrasts if c['comparison_kind']!='decisive_condition'),
            'matched_control_acceptable_set_changes':sum(c['acceptable_set_changed'] for c in contrasts if c['comparison_kind']!='decisive_condition'),
            'protected_responses_sha256':sha256((new_output/'authentic-governed-contrasts.protected.jsonl').read_bytes()).hexdigest(),
            'protected_comparisons_sha256':sha256((new_output/'matched-comparisons.protected.json').read_bytes()).hexdigest(),
            'source_and_candidate_authority_unchanged':True,'targets_enter_n0':False,'source_truncation':False,
            'upstream_gradient':False,'native_gradient':False,'state':'PREPARED_UNQUALIFIED',
            'primary_targets_provisional':True,'independent_qualification':False,'n0_approved':False,'full_personality_complete':False,
            'remaining_inherited_governing_influence':'DEFECT_REQUIRING_PRACTICAL_SUPPRESSION',
            'failure_attribution':'UNRESOLVED_ACROSS_PROVIDER_READOUT_AND_GOVERNORS',
            'owner_fidelity_approved':False,'spoken_fidelity_verified':False,'private_payload_exported':False,'flora_work':False}
        result['receipt_sha256']=sha256(canonical(result)).hexdigest()
        with (ROOT/'jobs/rayan-n0-authentic-20261011.json').open('xb') as stream:stream.write(canonical(result)+b'\n')
        print(json.dumps(result,sort_keys=True),flush=True)
code=Path(os.environ['PERSONALITY_REPO_ROOT'])
guard=code/'scripts/eipm/n1/enter_private_network_namespace.py'
assert sha256(guard.read_bytes()).hexdigest()==GUARD
namespace=module('owner_namespace_authentic_n0',guard)
try:reload(namespace)
except Exception as exc:
    import traceback
    frames=[{'module':Path(f.filename).name,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(exc.__traceback__)]
    print(json.dumps({'state':'AUTHENTIC_N0_REFUSED','exception_type':type(exc).__name__,'public_code_frames':frames}),flush=True)
    raise SystemExit(2)
