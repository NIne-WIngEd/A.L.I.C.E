"""Measure the original intervention trajectory; no optimizer or candidate export."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import traceback

ROOT=Path('/homes/01/mxrayan/rayan-compute')
NAME='rayan-conditional-gradient-attribution-20261028'
GUARD='f5822c3657b246a5b232f42bddf93b8026bda0b6a7e573835c1cd8c90136c1e4'
SUMMARY_SHA='f83ac86c2ab439350cf471718edf6d8689b8f2253e86141f50461752aec72e70'
LOG_SHA='c378363ada61702cb89a6ed48b82d5aa619fdc8cb719d93a2a7a5a776215cc50'
CONTEXT_SHA='6e449a5c02df6bd374daa9bae553e0895be62b33da323176ff8efd75aee1f1ce'
ANALYSIS_SHA='6088c6aaa99c8a7842fed04a0323ba9bb2bf0eb26502ab2fe2575bfe831040dc'


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def require(condition,message='trajectory evidence differs'):
    if not condition:raise ValueError(message)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value
def write(path,value):
    with path.open('xb') as out:out.write(canonical(value)+b'\n');out.flush();os.fsync(out.fileno())
    return sha256(path.read_bytes()).hexdigest()


def run(namespace,context_api):
    # open_context's first operation is the single namespace/P2 entry guard.
    with context_api.open_context(namespace) as ctx:
        import torch
        from src.alice_personality.identity.codec import fingerprint
        from scripts.eipm.gemma_n0 import conditional_teaching as teaching
        analysis_path=ROOT/'jobs'/(NAME+'-analysis.py');ctx['pin'](analysis_path,ANALYSIS_SHA)
        api=module('conditional_gradient_pinned_analysis',analysis_path)
        worker_sha=sha256(Path(__file__).read_bytes()).hexdigest();ctx['pin'](Path(__file__),worker_sha)
        ctx['pin'](ROOT/'jobs'/(NAME+'-context.py'),CONTEXT_SHA)
        summary=context_api.verify_seal(ctx['jpin'](ROOT/'jobs/rayan-conditional-learning-20261025.json',SUMMARY_SHA))
        require(summary['job_id']==579741 and summary['total_committed_experimental_updates']==80)
        require(summary['complete_source_and_checkpoint_custody_verified'] and not summary['winner_selected'])
        directory=ROOT/'rayan-personality/private-compile/rayan-conditional-learning-579741'
        plan_path=namespace._bounded_path(str(directory/'conditional-learning-plan.protected.json'),ROOT,file=True)
        require(stat.S_IMODE(plan_path.stat().st_mode)==0o600 and plan_path.stat().st_size<=16*1024**2)
        with plan_path.open('rb') as stream:plan_bytes=stream.read(16*1024**2+1)
        require(len(plan_bytes)<=16*1024**2)
        plan=json.loads(plan_bytes)
        require(canonical(plan)+b'\n'==plan_bytes and sha256(canonical(plan)).hexdigest()==summary['experiment_plan_sha256'])
        ctx['pin'](plan_path,sha256(plan_bytes).hexdigest());del plan_bytes
        require(plan['matched_groups']==ctx['groups'] and plan['correction_sha256']==ctx['correction_digests'])
        require(plan['model_state_sha256']==ctx['original_model_sha256'] and plan['recipe_sha256']==ctx['recipe'].fingerprint())
        require(plan['objective_state_sha256']==fingerprint(ctx['objectives'].state_dict()))
        manifest=ctx['jpin'](directory/'experimental-arms/manifest.json',summary['experimental_manifest_file_sha256'])
        require(manifest['plan_sha256']==summary['experiment_plan_sha256'] and manifest['total_committed_updates']==80)
        expected_events=manifest['arms']['intervention']['events'];require(len(expected_events)==40)
        log=ctx['pin'](ROOT/'jobs'/(NAME+'-source-stdout.txt'),LOG_SHA)
        progress=[json.loads(line) for line in log.stream if line.startswith(b'{')]
        progress=[row for row in progress if row['stage']=='comparison_update_completed']
        require([row['completed_update_events'] for row in progress]==list(range(1,81)))
        progress_pins={row['completed_update_events']:row['snapshot_manifest_sha256'] for row in progress}
        saved_source=[];public_groups=[];detail_hashes=[];broader_signal=[]
        clone=deepcopy(ctx['model']);clone.set_phase('n2',adapt_n1=False)
        objectives=deepcopy(ctx['objectives']);objectives.requires_grad_(False);objectives.eval()
        objective_sha=fingerprint(objectives.state_dict());initial_state=deepcopy(ctx['model'].state_dict())
        rng=None;prior_state=initial_state;prior_binding=None
        previous_deterministic=torch.are_deterministic_algorithms_enabled()
        previous_warn_only=torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(ctx['recipe'].deterministic)
            for group_number,group in enumerate(ctx['groups'],1):
                ctx['custody_guard']()
                require(len(group)>=5 and all(ctx['n2_lookup'][digest][0]=='broader' for digest in group[:2]))
                root=directory/'experimental-progress'/f'group-{40+group_number:04d}'
                snapshot_manifest=ctx['jpin'](root/'manifest.json',progress_pins[40+group_number])
                cursor={'arm':'intervention','next_group':group_number,'total_committed_updates':40+group_number}
                require(snapshot_manifest['schema']=='alice-personality-conditional-learning-progress-v1' and snapshot_manifest['cursor']==cursor)
                payload=ctx['pin'](root/'experimental-progress.pt',snapshot_manifest['state_file_sha256'],4*1024**3)
                state=torch.load(payload.stream,map_location='cpu',weights_only=True)
                require(state['schema']==snapshot_manifest['schema'] and state['cursor']==cursor)
                require(sha256(canonical(state['binding'])).hexdigest()==snapshot_manifest['binding_sha256'])
                binding=state['binding'];require(binding['plan']==plan and binding['plan_sha256']==summary['experiment_plan_sha256'])
                require(binding['runtime']['torch']==str(torch.__version__) and binding['runtime']['threads']==torch.get_num_threads()==1)
                require(binding['optimizer_parameter_names']==[n for n,p in clone.named_parameters() if p.requires_grad])
                if prior_binding is not None:require(binding==prior_binding)
                prior_binding=binding
                arm=state['arms']['intervention']
                require(arm['events']==expected_events[:group_number] and arm['updates']==group_number)
                require(arm['n1_n3_unchanged'] and arm['cuda_rng']==[])
                require(fingerprint(arm['model_state'])==arm['model_state_sha256'])
                require(fingerprint(arm['objective_state'])==objective_sha)
                if group_number==40:require(arm['model_state_sha256']==summary['arms']['intervention']['model_state_sha256'])
                clone.load_state_dict(prior_state,strict=True)
                if rng is None:torch.manual_seed(ctx['recipe'].seed)
                else:torch.set_rng_state(rng)
                event=expected_events[group_number-1]
                require(event['update']==group_number and event['batch_sha256']==group)
                def correction_forward(model,batch):
                    digest=batch.fingerprint();record=ctx['correction_records'].get(digest)
                    if record is None:return model(batch.frame)
                    return teaching.forward_with_correction(model,batch,record,
                        expected_record_sha256=ctx['correction_digests'][digest],apply_correction=True)
                def endpoint(count):
                    print(json.dumps({'stage':'gradient_endpoint_measured','group':group_number,
                        'endpoints_completed':count,'group_size':len(group),'new_optimizer_steps':0}),flush=True)
                measured=api.measure_group(clone,objectives,ctx['recipe'],event,arm['model_state'],
                    batch_loader=ctx['batch_loader'],validate_batch=ctx['validate_batch'],
                    correction_forward=correction_forward,on_endpoint=endpoint)
                require(torch.equal(torch.get_rng_state(),arm['torch_rng']),'replayed RNG differs')
                measured['source_snapshot_manifest_sha256']=progress_pins[40+group_number]
                measured['source_snapshot_file_sha256']=snapshot_manifest['state_file_sha256']
                measured['source_event_sha256']=sha256(canonical(event)).hexdigest()
                measured['analysis_sha256']=ANALYSIS_SHA
                path=ctx['output']/f'gradient-group-{group_number:04d}.protected.json'
                digest=write(path,measured);detail_hashes.append(digest)
                public_groups.append({key:measured[key] for key in ('group','model_state_sha256','post_state_sha256',
                    'summed_gradient_norm','original_gradient_norm','decomposition_max_abs_error',
                    'preference_vs_other_geometry','concept_vs_remaining_geometry','broader_pair_cancellation')})
                broader_signal.extend(row['preference_correction_signal'] for row in measured['endpoints'][:2])
                saved_source.append({'group':group_number,'manifest_sha256':progress_pins[40+group_number],
                    'tensor_sha256':snapshot_manifest['state_file_sha256'],'measurement_sha256':digest})
                prior_state=arm['model_state'];rng=arm['torch_rng']
                ctx['custody_guard']()
                print(json.dumps({'stage':'gradient_group_saved','groups_completed':group_number,'groups_total':40,
                    'measurement_sha256':digest,'new_optimizer_steps':0}),flush=True)
                del state,arm,measured
            require(ctx['batch_loader'].loaded==216 and len(public_groups)==40 and len(broader_signal)==80)
            require(fingerprint(ctx['model'].state_dict())==ctx['original_model_sha256'])
        finally:
            torch.use_deterministic_algorithms(previous_deterministic,warn_only=previous_warn_only)
        def statistics(values):
            values=list(values);return {'count':len(values),'min':min(values),'mean':sum(values)/len(values),'max':max(values)}
        signals={part:{field:statistics(row[part][field] for row in broader_signal) for field in next(iter(broader_signal))[part]}
                 for part in next(iter(broader_signal))}
        result={'schema':'alice-personality-conditional-gradient-attribution-v1','job_id':int(os.environ['SLURM_JOB_ID']),
            'source_job_id':579741,'owner':'mxrayan','source_summary_sha256':SUMMARY_SHA,'source_experiment_plan_sha256':summary['experiment_plan_sha256'],
            'worker_sha256':worker_sha,'context_sha256':CONTEXT_SHA,'analysis_sha256':ANALYSIS_SHA,
            'groups':public_groups,'broader_correction_signal':signals,'source_snapshot_and_measurement_pins':saved_source,
            'cases':216,'groups_measured':40,'native_forwards':432,'new_optimizer_steps':0,'native_candidate_exported':False,
            'original_losses_reproduced_exactly':True,'gradient_norms_reproduced':True,'decomposition_tolerance':{'rtol':api.RTOL,'atol':api.ATOL},
            'all_frozen_stages_unchanged':True,'private_measurement_hashes':detail_hashes,
            'local_derivatives_not_global_causal_proof':True,'private_payload_exported':False,
            'qualification':'UNQUALIFIED','n0_approved':False,'personality_approved':False}
    # Success publication follows the context's complete source/feature closure.
    result['complete_source_and_snapshot_custody_verified']=True
    result['receipt_sha256']=sha256(canonical(result)).hexdigest()
    write(ROOT/'jobs'/(NAME+'.json'),result)
    print(json.dumps({'stage':'complete_gradient_attribution','receipt_sha256':result['receipt_sha256'],'new_optimizer_steps':0}),flush=True)


if __name__=='__main__':
    try:
        guard=Path(os.environ['PERSONALITY_REPO_ROOT'])/'scripts/eipm/n1/enter_private_network_namespace.py'
        require(sha256(guard.read_bytes()).hexdigest()==GUARD)
        context_path=ROOT/'jobs'/(NAME+'-context.py');require(sha256(context_path.read_bytes()).hexdigest()==CONTEXT_SHA)
        run(module('gradient_namespace',guard),module('gradient_readonly_context',context_path))
    except BaseException as error:
        print(json.dumps({'schema':'alice-personality-gradient-attribution-failure-v1','job_id':os.environ.get('SLURM_JOB_ID'),
            'error_type':type(error).__name__,'public_code_locations':[{'file':Path(f.filename).name,'line':f.lineno} for f in traceback.extract_tb(error.__traceback__)],
            'private_payload_exported':False}),flush=True)
        raise SystemExit(2) from None
