"""Owner-only saved residual measurement. No optimizer or full-native calls."""
from contextlib import ExitStack
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys
import traceback

ROOT=Path('/homes/01/mxrayan/rayan-compute')
REV='333fcd0c8239fc81a6459b13d53bbe592b33fda6'
GUARD='f5822c3657b246a5b232f42bddf93b8026bda0b6a7e573835c1cd8c90136c1e4'
NAME='rayan-residual-saved-attribution-r2-20261108'
SUMMARY_SHA='f0e6e65633982596908603928c20847a7b83543e89d0102aaff29683d78e3bd5'
LOG_SHA='15f41ec5f708e3a654a70fb5929472446a1a3f667de4a8ad469ed170e8c29582'
# Helper pin is generated only after public verification; worker and launcher
# are then sealed together. Original scientific sources are never overwritten.
HELPER_SHA='152ce596c6209776ca12f811fa742d364edfe1ea5cc312fcce6c346a34a8670c'
DEPENDENCIES={
 'source':('source.py','b8c8876da08f3f19e2a0017c81ffdd318d340ed29995f388a80f65b1427e41ee'),
 'extension':('extension.py','18d0b08b01c53856390ad8bb428215b271575b360e707537b726a9c274eab759'),
 'live':('live.py','7bde912edf49ffd1227f47d2bbf77068d6c0d79c7f65159b9e49a36522505533')}


def require(condition,message='saved residual binding differs'):
    if not condition:raise ValueError(message)


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value


def write(path,value):
    with path.open('xb') as out:out.write(canonical(value)+b'\n')
    return sha256(path.read_bytes()).hexdigest()


def seal(value):
    require(sha256(canonical({k:v for k,v in value.items() if k!='receipt_sha256'})).hexdigest()==value['receipt_sha256'])
    return value


def stats(values):
    values=list(values);require(all(math.isfinite(v) for v in values))
    return {'count':len(values),'min':min(values) if values else None,
            'mean':math.fsum(values)/len(values) if values else None,'max':max(values) if values else None}


def verify_saved_binding(binding,final,names):
    require(canonical(binding['plan'])==canonical(final['plan']) and binding['plan_sha256']==final['plan_sha256'])
    require(binding['model_config']==final['model_config'] and canonical(binding['recipe'])==canonical(final['recipe']))
    require(binding['optimizer_parameter_names']==names)


def verify_final_metadata(manifest,final,plan,expected_state_sha):
    require(manifest['state_file_sha256']==expected_state_sha and canonical(final['plan'])==canonical(plan))
    projected={key:value for key,value in final.items() if key!='arms'}
    projected['state_file_sha256']=expected_state_sha
    projected['arms']={tag:{key:value for key,value in arm.items()
        if key not in {'model_state','objective_state','optimizer_state','torch_rng','cuda_rng'}}
        for tag,arm in final['arms'].items()}
    require(canonical(projected)==canonical(manifest),'complete final manifest projection differs')


def run(namespace):
    isolation=namespace._p2_guard()  # FIRST protected operation; never re-enter.
    os.umask(0o077);sys.dont_write_bytecode=True
    code=namespace._bounded_path(os.environ['PERSONALITY_REPO_ROOT'],ROOT,file=False)
    namespace._clean_code(code,REV);sys.path.insert(0,str(code))
    import torch
    from src.alice_personality.n1.manifest_structure_audit import _PinnedFile
    from src.alice_personality.n1 import raw_inference_lineage as raw
    from src.alice_personality.identity import IdentityModel,IdentityModelConfig
    from src.alice_personality.identity.codec import fingerprint
    from src.alice_personality.teaching.cases import ReviewedTeachingCase
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with ExitStack() as stack:
        held=[]
        def pin(path,digest,limit=512*1024**2):
            path=namespace._bounded_path(str(path),ROOT,file=True)
            require(stat.S_IMODE(path.stat().st_mode)==0o600)
            value=stack.enter_context(_PinnedFile(path,digest,limit));held.append(value);return value
        def read(path,digest):return raw._json(pin(path,digest).bounded_bytes())
        def lines(path,digest):
            value=pin(path,digest);value.stream.seek(0);return [raw._json(line) for line in value.stream]
        def tensors(path,digest):
            value=pin(path,digest,4*1024**3);value.stream.seek(0)
            return torch.load(value.stream,map_location='cpu',weights_only=True)
        worker_sha=sha256(Path(__file__).read_bytes()).hexdigest();pin(Path(__file__),worker_sha)
        deps={**DEPENDENCIES,'helper':('helper.py',HELPER_SHA)};loaded={}
        for key,(suffix,digest) in deps.items():
            path=ROOT/'jobs'/(NAME+'-'+suffix);pin(path,digest);loaded[key]=module('residual_saved_'+key,path)
        source,extension,live,helper=(loaded[k] for k in ('source','extension','live','helper'))
        def custody():
            live.verify_active_namespace(namespace,isolation)
            for value in held:value.verify()
        live.verify_active_namespace(namespace,isolation)
        result=seal(read(ROOT/'jobs/rayan-conditional-residual-20261104.json',SUMMARY_SHA))
        require(result['job_id']==580417 and result['complete_source_and_checkpoint_custody_verified'])
        require(result['total_committed_experimental_updates']==80 and not result['winner_selected'])
        directory=ROOT/'rayan-personality/private-compile/rayan-conditional-residual-20261104-580417'
        plan_path=namespace._bounded_path(str(directory/'conditional-learning-plan.protected.json'),ROOT,file=True)
        require(stat.S_IMODE(plan_path.stat().st_mode)==0o600 and plan_path.stat().st_size<=16*1024**2)
        with plan_path.open('rb') as stream:plan_bytes=stream.read(16*1024**2+1)
        require(len(plan_bytes)<=16*1024**2)
        plan=raw._json(plan_bytes);require(plan_bytes==canonical(plan)+b'\n')
        require(sha256(canonical(plan)).hexdigest()==result['experiment_plan_sha256'])
        pin(plan_path,sha256(plan_bytes).hexdigest())
        manifest=read(directory/'experimental-arms/manifest.json',result['experimental_manifest_file_sha256'])
        final=tensors(directory/'experimental-arms/experimental.pt',result['experimental_state_file_sha256'])
        verify_final_metadata(manifest,final,plan,result['experimental_state_file_sha256'])
        require(final['plan_sha256']==result['experiment_plan_sha256'] and final['total_committed_updates']==80)
        stdout=pin(ROOT/'jobs'/(NAME+'-source-stdout.txt'),LOG_SHA);stdout.stream.seek(0)
        acks=[raw._json(line) for line in stdout.stream if line.startswith(b'{')]
        acks=[v for v in acks if v.get('stage')=='comparison_update_completed']
        require([v['completed_update_events'] for v in acks]==list(range(1,81)))
        for tag,offset in (('control',0),('intervention',40)):
            arm=final['arms'][tag]
            require(arm['updates']==len(arm['events'])==40 and arm['model_state_sha256']==result['arms'][tag]['model_state_sha256'])
            require(fingerprint(arm['model_state'])==arm['model_state_sha256'])
            for i,event in enumerate(arm['events']):
                require(event['arm']==tag and event['update']==i+1)
                require(sha256(canonical(event)).hexdigest()==acks[offset+i]['event_sha256'])
            # Manifest JSON sorts object keys; tensor exports preserve insertion
            # order. Compare JSON content here, exact tensor structure below.
            for key,value in manifest['arms'][tag].items():require(canonical(value)==canonical(arm[key]))
        model=IdentityModel(IdentityModelConfig(**final['model_config']))
        model.n2=extension.ConditionalJudgment(model.config);model.set_phase('inference')
        model.load_state_dict(final['arms']['intervention']['model_state'],strict=True)
        model_sha=fingerprint(model.state_dict());require(model_sha==result['arms']['intervention']['model_state_sha256'])
        names=[n for n,_ in model.named_parameters() if n.startswith('n2.')]
        transitions={};runtime=None
        for tag,offset in (('control',0),('intervention',40)):
            for first,last in ((1,2),(39,40)):
                selected=[]
                for number in (first,last):
                    count=offset+number;folder=directory/'experimental-progress'/f'group-{count:04d}'
                    sm=read(folder/'manifest.json',acks[count-1]['snapshot_manifest_sha256'])
                    saved=tensors(folder/'experimental-progress.pt',sm['state_file_sha256'])
                    expected={'arm':tag,'next_group':number,'total_committed_updates':count}
                    require(saved['cursor']==sm['cursor']==expected)
                    binding=saved['binding'];require(sha256(canonical(binding)).hexdigest()==sm['binding_sha256'])
                    verify_saved_binding(binding,final,names)
                    require(binding['runtime']['torch']==str(torch.__version__))
                    if runtime is None:
                        runtime=binding['runtime'];torch.use_deterministic_algorithms(runtime['deterministic'])
                    require(binding['runtime']==runtime and runtime['device']=='cpu' and runtime['threads']==runtime['interop_threads']==1)
                    arm=saved['arms'][tag];require(arm['updates']==number)
                    require(fingerprint(arm['model_state'])==arm['model_state_sha256'])
                    source.verify_exported_state(arm['events'],final['arms'][tag]['events'][:number])
                    if number==40:source.verify_exported_state(arm,final['arms'][tag])
                    selected.append(arm);del saved
                measured=helper.adam_transition(*selected,names)
                gradients=measured.pop('recovered_gradients')
                norm=selected[1]['events'][-1]['gradient_norm_before_clip']
                expected_norm=norm*min(1.,final['recipe']['max_grad_norm']/(norm+1e-6))
                require(math.isclose(measured['clipped_aggregate_gradient_norm'],expected_norm,rel_tol=2e-5,abs_tol=1e-7),'recovered clipped norm differs')
                measured['expected_clipped_norm']=expected_norm
                measured['branch_gradient_norm']=math.sqrt(math.fsum(float(g.square().sum()) for n,g in gradients.items() if n.startswith('n2.conditioning.')))
                measured['gate_before']=float(selected[0]['model_state']['n2.conditioning.gate'])
                measured['gate_after']=float(selected[1]['model_state']['n2.conditioning.gate'])
                transitions[f'{tag}_{first}_to_{last}']=measured
                del selected,gradients
                live.verify_active_namespace(namespace,isolation)
        print(json.dumps({'stage':'selected_saved_transitions_verified','transitions':4,'source_job_id':580417}),flush=True)
        source_plan=read(ROOT/'rayan-personality/private-inputs/rayan-cumulative-learning-r3-20261015/owner-cumulative-run.protected.json',result['source_plan_sha256'])
        ref=source_plan['packets']['broader'];folder=Path(ref['directory'])
        index=read(folder/'packet-index.protected.json',ref['index_sha256'])
        cases=[ReviewedTeachingCase.from_mapping(v) for v in lines(folder/'reviewed-teaching-cases.protected.jsonl',index['files']['reviewed-teaching-cases.protected.jsonl'])]
        require(len(cases)==len({c.case_id for c in cases}) and {c.case_id:c.sha256 for c in cases}==index['case_sha256'])
        allocation=read(folder/'construction-allocation.protected.json',index['files']['construction-allocation.protected.json'])
        arm=result['arms']['intervention']
        objectives=lines(directory/'intervention-all-n2-objectives.protected.jsonl',arm['objective_file_sha256'])
        require(len(objectives)==216 and {v['batch_sha256'] for v in objectives}==set(plan['batch_sha256']))
        objectives={v['case_sha256']:v for v in objectives if v['packet']=='broader'}
        broad=[c for c in cases if c.sha256 in objectives];require(len(broad)==len(objectives)==80)
        require(all(c.phase=='n2' and c.split=='TRAIN' and objectives[c.sha256]['case_id']==c.case_id for c in broad))
        pairs=source.source_pairs(broad,allocation)
        rows=lines(directory/'intervention-candidate-native-packets.protected.jsonl',arm['response_file_sha256'])
        require(len(rows)==236)
        natural={v['document_sha256']:v for v in rows if v['packet']=='broader'}
        disabled_rows=lines(directory/'intervention-disabled-residual-native-packets.protected.jsonl',arm['conditional_residual']['disabled_private_file_sha256'])
        disabled={v['case_sha256']:v for v in disabled_rows};require(len(natural)==len(disabled)==len(disabled_rows)==80)
        output=ROOT/'rayan-personality/private-compile'/(NAME+'-'+os.environ['SLURM_JOB_ID'])
        output.mkdir(mode=0o700,exist_ok=False)
        output=namespace._bounded_path(str(output),ROOT,file=False)
        details=[];gradients={};gate_index=None
        model.n2.conditioning.requires_grad_(True)
        branch=list(model.n2.conditioning.named_parameters());parameters=tuple(p for _,p in branch)
        gate_index=next(i for i,(n,_) in enumerate(branch) if n=='gate')
        for case in broad:
            live.verify_active_namespace(namespace,isolation)
            enabled=natural[case.document.sha256];off=disabled[case.sha256]
            require(enabled['model_state_sha256']==model_sha and enabled['frame_binding']==off['frame_binding'])
            require(enabled['arm']=='intervention' and not enabled['native_checkpoint'] and not enabled['acceptance_authority'])
            require(off['natural_input'] and off['experimental_branch_disabled'])
            with torch.no_grad():_,inference=helper.replay(model,off['decision'],enabled['decision'])
            logits,measurement=helper.replay(model,off['decision'],enabled['decision'])
            targets=helper.order_targets(case,enabled['decision'])
            loss,fields=helper.order_loss(logits,helper.tensor(enabled['decision']['candidate_mask'],boolean=True),targets,final['recipe']['loss_weights'])
            for name,value in fields.items():
                require(math.isclose(float(value.detach()),objectives[case.sha256]['field_loss'][name],rel_tol=2e-5,abs_tol=2e-6),'saved objective reproduction differs')
            derivatives=torch.autograd.grad(loss,parameters)
            vector=torch.cat([v.detach().double().flatten() for v in derivatives]);gradients[case.sha256]=vector
            details.append({'case_sha256':case.sha256,'inference_replay':inference,'grad_replay':measurement,
                'weighted_natural_order_loss':float(loss.detach()),'gate_gradient':float(derivatives[gate_index]),
                'branch_gradient_norm':float(vector.norm()),'nongate_gradient_norm':math.sqrt(math.fsum(float(v.detach().double().square().sum()) for i,v in enumerate(derivatives) if i!=gate_index))})
        pair_details=[]
        for left,right in pairs:
            a,b=gradients[left.sha256],gradients[right.sha256];den=float(a.norm()*b.norm())
            pair_details.append({'left_case_sha256':left.sha256,'right_case_sha256':right.sha256,
                'cosine':float(a.dot(b))/den if den else None,'summed_norm':float((a+b).norm()),
                'individual_norm_sum':float(a.norm()+b.norm())})
        require(fingerprint(model.state_dict())==model_sha and all(p.grad is None for p in model.parameters()))
        private_sha=write(output/'measurements.protected.json',{'contexts':details,'pairs':pair_details,'transitions':transitions})
        aggregate={k:stats(row['grad_replay'][k] for row in details) for k in details[0]['grad_replay'] if k!='grad_enabled'}
        aggregate.update({k:stats(row[k] for row in details) for k in ('gate_gradient','branch_gradient_norm','nongate_gradient_norm')})
        # Only public parameter names/aggregate metrics leave isolation, never case IDs or vectors.
        custody();namespace._clean_code(code,REV)
        summary={'schema':'alice-personality-residual-saved-attribution-v1','job_id':int(os.environ['SLURM_JOB_ID']),
            'owner':'mxrayan','source_job_id':580417,'source_summary_file_sha256':SUMMARY_SHA,
            'source_receipt_sha256':result['receipt_sha256'],'worker_sha256':worker_sha,
            'dependency_sha256':{k:v[1] for k,v in deps.items()},'isolation':isolation,'runtime':runtime,
            'consumed_file_hashes_verified_at_open_and_close':len(held),'all_80_event_hashes_verified':True,
            'selected_snapshot_tensor_bodies_reloaded':8,'selected_optimizer_transitions':transitions,
            'contexts':80,'pairs':40,'natural_context_measurements':aggregate,
            'pair_gradient_cosine':stats(v['cosine'] for v in pair_details if v['cosine'] is not None),
            'pair_cancellation_ratio':stats(v['summed_norm']/v['individual_norm_sum'] for v in pair_details if v['individual_norm_sum']),
            'aggregate_gradient_cancellation_ratio':float(torch.stack(list(gradients.values())).sum(0).norm())/sum(float(v.norm()) for v in gradients.values()) if any(bool(v.any()) for v in gradients.values()) else None,
            'private_measurements_sha256':private_sha,'new_full_native_forwards':0,'saved_branch_readout_replays':160,
            'new_private_gradient_steps':0,'new_targets_created':False,'model_state_unchanged':True,
            'historical_task_gradient_replay':False,'corrected_input_gradient_replay':False,
            'complete_objective_gradient_decomposition':False,'unique_training_cause_established':False,
            'qualification':'UNQUALIFIED','n0_approved':False,'personality_approved':False,'private_payload_exported':False}
        summary['receipt_sha256']=sha256(canonical(summary)).hexdigest();write(ROOT/'jobs'/(NAME+'.json'),summary)
        print(json.dumps({'stage':'complete_saved_residual_attribution','receipt_sha256':summary['receipt_sha256'],'new_updates':0}),flush=True)


if __name__=='__main__':
    guard=Path(os.environ['PERSONALITY_REPO_ROOT'])/'scripts/eipm/n1/enter_private_network_namespace.py'
    require(sha256(guard.read_bytes()).hexdigest()==GUARD)
    namespace=module('residual_saved_namespace',guard)
    try:run(namespace)
    except BaseException as error:
        print(json.dumps({'schema':'alice-personality-saved-residual-failure-v1','job_id':os.environ.get('SLURM_JOB_ID'),
            'error_type':type(error).__name__,'public_code_locations':[{'file':Path(f.filename).name,'line':f.lineno}
            for f in traceback.extract_tb(error.__traceback__)],'private_payload_exported':False}),flush=True)
        raise SystemExit(2) from None
