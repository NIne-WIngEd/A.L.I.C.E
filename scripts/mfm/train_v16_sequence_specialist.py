"""Bounded balanced formation and exact-source sequence teaching.

This is a new objective and task-conditioned candidate. Parent weights are
externally pinned; Adam moments are reset explicitly. Every epoch retains all
432 original training cases. Diagnostic development is never used for gradients,
sampling, stopping or selection. Component/runtime evidence is not qualification.
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
from time import perf_counter

from cognitive_kernel.canonical import CognitiveKernelContractError, require_sha256
from cognitive_kernel.formation_sequence_specialist import FormationSequenceSpecialist, INITIALIZATION, TASKS
from . import formation_sequence_curriculum as curriculum
from . import analyze_v16_budget_failure as ancestry

source, shared, training = ancestry.source, ancestry.shared, ancestry.child.training
RUN_SCHEMA = 'mfm-v16-source-sequence-run-v1'
ARTIFACT_SCHEMA = 'mfm-v16-source-sequence-artifact-v1'
PROGRESS_SCHEMA = 'mfm-v16-source-sequence-progress-v1'
OPTIMIZER_INITIALIZATION = 'AdamW reset: new task parameters and changed objective; no inherited parent moments'


def code_binding():
    import cognitive_kernel.formation_sequence_specialist as model
    import cognitive_kernel.formation_incremental_decoder as cache
    import cognitive_kernel.formation_v1_specialist as geometry
    return {**ancestry.child.code_binding(),
        'completed_parent_verifier':shared._digest(Path(ancestry.__file__)),
        'budget_diagnostic':shared._digest(Path(ancestry.budget.__file__)),
        'original_generated_diagnostic':shared._digest(Path(ancestry.original.__file__)),
        'task_model':shared._digest(Path(model.__file__)),
        'shared_decoder_geometry':shared._digest(Path(geometry.__file__)),
        'task_curriculum':shared._digest(Path(curriculum.__file__)),
        'sequence_trainer':shared._digest(Path(__file__)),
        'incremental_decoder':shared._digest(Path(cache.__file__))}


def batch_coefficients(examples, weights, total_cases):
    ids = [e.case_id for e in examples]
    if (not ids or len(ids)!=len(set(ids)) or any(e.split!='train' for e in examples)
            or not set(ids).issubset(weights) or type(total_cases) is not int
            or total_cases<len(ids) or any(not math.isfinite(weights[i]) or weights[i]<=0 for i in ids)):
        raise CognitiveKernelContractError('sequence gradients need unique admitted train cases and global weights')
    # Uniform shuffled minibatches estimate the declared global weighted mean.
    # Dividing by weight sum within each batch would erase singleton weighting.
    return {i:total_cases*weights[i]/len(ids) for i in ids}


def stress_ids(prepared, receipt):
    rows = {r['case_id']:r for r in prepared.view['cases'] if r['split']=='train'}
    lexical = {r['case_id']:r['lexical_target_tokens'] for r in receipt['cases']}
    if set(rows)!=set(lexical):
        raise CognitiveKernelContractError('lexical stress roster differs from admitted train')
    lengths = max(lexical, key=lexical.get)
    joint = max(lexical, key=lambda key: lexical[key]*rows[key]['source_tokens'])
    augmented = tuple(k for k,r in rows.items() if r.get('source_augmented') is True)
    return list(dict.fromkeys((*prepared.view['stress_train_case_ids'],*augmented,lengths,joint)))


def payloads(prepared, example):
    import torch
    formation = source.payload(prepared,example,device=prepared.device)
    inputs, labels = curriculum.lexical_target_ids(prepared.tokenizer,example,prepared.config.max_target_tokens)
    lexical = {**formation, 'input_ids':torch.tensor([inputs],device=prepared.device),
               'labels':torch.tensor([labels],device=prepared.device)}
    return formation, lexical


def optimizer_step(model, optimizer, examples, *, prepared, weights):
    import torch
    coefficients = batch_coefficients(examples,weights,len(prepared.train))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    means = [0.,0.]
    weighted = 0.
    for example in examples:
        for index, (task,payload) in enumerate(zip(TASKS,payloads(prepared,example))):
            _, loss = model(**payload,task=task)
            if not torch.isfinite(loss):
                raise CognitiveKernelContractError('source sequence task loss is nonfinite')
            (loss*coefficients[example.case_id]/2).backward()
            value = float(loss.detach().cpu())
            means[index] += value/len(examples)
            weighted += value*coefficients[example.case_id]/2
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    if not torch.isfinite(norm) or norm.item()<=0:
        raise CognitiveKernelContractError('source sequence has no finite nonzero gradient')
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise CognitiveKernelContractError('source sequence optimizer produced nonfinite weights')
    return {'train_examples':len(examples),'formation_mean_loss':means[0],
            'source_mean_loss':means[1],'importance_weighted_batch_loss':weighted,'gradient_norm':norm.item()}


def migration_parity(model, prepared, selected):
    import torch
    old = ancestry.child.parent.FormationDecisionSpecialist(prepared.config).to(prepared.device,dtype=torch.float32).eval()
    old.load_state_dict({k:v for k,v in model.state_dict().items() if k!='task_embedding.weight'})
    model.eval()
    maximum = 0.
    with torch.inference_mode(),torch.autocast(prepared.device.type,enabled=False):
        for example in selected:
            payload = source.payload(prepared,example,device=prepared.device)
            inputs = {k:v for k,v in payload.items() if k!='labels'}
            inputs['input_ids'] = inputs['input_ids'][:,:32]
            a,b = old.next_token_logits(**inputs),model.next_token_logits(**inputs)
            maximum = max(maximum,float((a-b).abs().max().cpu()))
            if not torch.equal(a,b):
                raise CognitiveKernelContractError('zero formation task changes migrated parent logits')
    return {'case_ids':[e.case_id for e in selected],'max_abs_difference':maximum,'bit_exact':True,
            'claim':'zero formation task preserves original first32 teacher-prefix logits at migration'}


def task_strata(losses):
    """Reproduce per-stratum metrics after canonical JSON reorders case keys."""
    groups = defaultdict(lambda:[0.,0.,0])
    for case_id in sorted(losses):
        pair = losses[case_id]
        group = groups[curriculum.construction_stratum(case_id)]
        group[0]+=pair[0];group[1]+=pair[1];group[2]+=1
    return {k:{'cases':v[2],'formation_mean_loss':v[0]/v[2],'source_mean_loss':v[1]/v[2]}
            for k,v in sorted(groups.items())}


def progress(root,digest,model,prepared,weights,epoch,steps):
    import torch
    path = root/f'progress-{steps:08d}.json'
    if path.exists():
        row = shared._read_sealed(path,PROGRESS_SCHEMA)
        if row.get('run_sha256')!=digest or row.get('epoch')!=epoch or row.get('optimizer_steps')!=steps:
            raise CognitiveKernelContractError('sequence progress lineage differs')
        return row
    model.eval()
    losses = {}
    with torch.inference_mode(),torch.autocast(prepared.device.type,enabled=False):
        for e in prepared.train:
            if e.split!='train': raise CognitiveKernelContractError('sequence progress reads train only')
            pair = [float(model(**p,task=t)[1].cpu()) for t,p in zip(TASKS,payloads(prepared,e))]
            losses[e.case_id] = pair
    objective = float(curriculum.balanced_objective(losses,weights))
    row = shared._write_new(path,{'schema':PROGRESS_SCHEMA,'run_sha256':digest,'epoch':epoch,
        'optimizer_steps':steps,'train_cases':len(losses),'balanced_two_task_objective':objective,
        'strata':task_strata(losses),
        'complete_case_losses':losses,'development_read':False,'qualified_for_product':False})
    print(json.dumps({'state':'COMPLETE_TRAIN_PROGRESS','epoch':epoch,'optimizer_steps':steps,
        'balanced_two_task_objective':objective,'progress_sha256':row['record_sha256']}),flush=True)
    return row


def probe(model,optimizer,prepared,args,digest,receipt,weights):
    import torch
    cached = ancestry.child.parent.cached
    wanted = stress_ids(prepared,receipt)
    if len(wanted)>args.gradient_accumulation:
        raise CognitiveKernelContractError('new lexical geometry stress exceeds accumulation')
    selected = sorted(prepared.train,key=lambda e:wanted.index(e.case_id) if e.case_id in wanted else len(wanted))[:args.gradient_accumulation]
    migration = migration_parity(model,prepared,selected)
    results = [optimizer_step(model,optimizer,selected,prepared=prepared,weights=weights) for _ in range(2)]
    training._checkpoint(args.output_dir,specialist=model,optimizer=optimizer,run_digest=digest,
                         epoch=0,next_case=len(selected),step=2)
    checkpoint = training._verify_checkpoint(args.output_dir/'checkpoint-00000002',digest)
    expected = optimizer_step(model,optimizer,selected,prepared=prepared,weights=weights)
    expected_model,expected_optimizer = cached._cpu_tree(model.state_dict()),cached._cpu_tree(optimizer.state_dict())
    training._resume(args.output_dir/'checkpoint-00000002',args.output_dir,model,optimizer,digest)
    actual = optimizer_step(model,optimizer,selected,prepared=prepared,weights=weights)
    if any(not math.isclose(expected[k],actual[k],rel_tol=1e-5,abs_tol=1e-6)
           for k in ('formation_mean_loss','source_mean_loss','importance_weighted_batch_loss')):
        raise CognitiveKernelContractError('sequence resumed next losses differ')
    model_difference = cached.compare_tree(model.state_dict(),expected_model)
    optimizer_difference = cached.compare_tree(optimizer.state_dict(),expected_optimizer)
    model.eval()
    sample = source.payload(prepared,selected[0],device=prepared.device)
    cache_checks = {}
    for task in TASKS:
        _,ended,parity = ancestry.original.greedy_fp32(model.bind_task(task),
            sample['base_states'],sample['source_mask'],32)
        cache_checks[task] = {**parity,'eos_within_prefix_check':ended,
            'claim':'first32 actual task-bound greedy prefixes only, not complete formation'}
    return {'stress_case_ids':wanted,'accumulation_case_ids':[e.case_id for e in selected],
        'steps':results,'resumed_next_step':actual,'migration_parity':migration,
        'model_max_abs_difference':model_difference,'optimizer_max_abs_difference':optimizer_difference,
        'checkpoint_sha256':checkpoint['record_sha256'],'task_bound_actual_cache_checks':cache_checks,
        'backward_and_restart_passed':True,'optimizer_steps':3,'generated_ability_established':False}


def verify_probe(args,manifest,receipt):
    if args.probe_only:
        if args.probe_dir or args.probe_sha256: raise CognitiveKernelContractError('sequence probe cannot depend on a probe')
        return None
    if not args.probe_dir or not args.probe_sha256:
        raise CognitiveKernelContractError('new sequence targets require their actual-runtime stress/restart receipt')
    component = shared._read_sealed(args.probe_dir/'component.json',ARTIFACT_SCHEMA)
    run = shared._read_sealed(args.probe_dir/'run.json',RUN_SCHEMA)
    expected = {**manifest,'probe_only':True,'stress_probe_sha256':None}
    if (component['record_sha256']!=require_sha256(args.probe_sha256,'sequence_probe_sha256')
            or component.get('run_sha256')!=run['record_sha256']
            or {k:v for k,v in run.items() if k!='record_sha256'}!=expected
            or component.get('probe_only') is not True or component.get('qualified_for_product') is not False
            or component.get('curriculum_sha256')!=receipt['record_sha256']
            or component['result'].get('backward_and_restart_passed') is not True
            or component['result'].get('optimizer_steps')!=3
            or component['result'].get('migration_parity',{}).get('bit_exact') is not True):
        raise CognitiveKernelContractError('sequence capacity/restart receipt recipe differs')
    checkpoint = training._verify_checkpoint(args.probe_dir/'checkpoint-00000002',run['record_sha256'])
    checks = component['result'].get('task_bound_actual_cache_checks',{})
    if (checkpoint['record_sha256']!=component['result'].get('checkpoint_sha256')
            or checkpoint.get('step')!=2 or checkpoint.get('epoch')!=0
            or checkpoint.get('next_case')!=manifest['gradient_accumulation']
            or component['result'].get('stress_case_ids')!=manifest['stress_train_case_ids']
            or set(checks)!=set(TASKS) or any(c.get('argmaxes_equal') is not True for c in checks.values())):
        raise CognitiveKernelContractError('sequence capacity checkpoint/stress/cache proof differs')
    return component


def run(args):
    prepared = source.load(args)  # Original owner/network admission precedes new reads.
    import torch
    from safetensors.torch import load_file,save_file
    if (not 1<=args.epochs<=4 or not 1<=args.gradient_accumulation<=len(prepared.train)
            or args.save_every_steps<1 or args.session_seconds<1):
        raise CognitiveKernelContractError('invalid bounded sequence construction')
    old_run,old_component,old_diag,_,parent_weights,_ = ancestry.verify_child(args,prepared)
    receipt = curriculum.curriculum_receipt(prepared.train,prepared.tokenizer,prepared.config.max_target_tokens)
    weights = receipt['weights']
    manifest = {'schema':RUN_SCHEMA,'objective':curriculum.OBJECTIVE,'task_names':list(TASKS),
        'formation_objective':ancestry.child.parent.OBJECTIVE,'task_mixture':[.5,.5],
        'initialization':INITIALIZATION,'optimizer_initialization':OPTIMIZER_INITIALIZATION,
        'source_view_sha256':prepared.view['record_sha256'],'code_binding':code_binding(),
        'curriculum_sha256':receipt['record_sha256'],'corpus_sha256':args.input_sha256,
        'owner_authorization_ref':args.owner_authorization_ref,'specialist_config':prepared.config.record(),
        'specialist_dtype':'float32','device':str(prepared.device),'torch_version':torch.__version__,
        'seed':old_run['seed'],'learning_rate':old_run['learning_rate'],
        'gradient_accumulation':args.gradient_accumulation,'epochs':args.epochs,
        'save_every_steps':args.save_every_steps,'session_seconds':args.session_seconds,
        'parent_component_sha256':old_component['record_sha256'],'parent_run_sha256':old_run['record_sha256'],
        'parent_weights_sha256':old_component['result']['trained_weights_sha256'],
        'parent_diagnostic_sha256':old_diag['record_sha256'],'parent_optimizer_loaded':False,
        'seed_control_meaning':'same-seed new-task fresh decoder; historical parent has different objective',
        'stress_train_case_ids':stress_ids(prepared,receipt),'probe_only':args.probe_only,
        'stress_probe_sha256':args.probe_sha256 if not args.probe_only else None,
        'teacher_fit':True,'full_fit':False,'independent_gold':False,'final_payloads_opened':False,'qualified_for_product':False}
    capacity = verify_probe(args,manifest,receipt)
    digest = shared._record_hash(manifest)
    torch.manual_seed(old_run['seed'])
    model = FormationSequenceSpecialist(prepared.config).to(prepared.device,dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(),lr=old_run['learning_rate'])
    if args.resume_checkpoint:
        if args.probe_only or (args.output_dir/'component.json').exists() or shared._read_sealed(args.output_dir/'run.json',RUN_SCHEMA)['record_sha256']!=digest:
            raise CognitiveKernelContractError('sequence resume needs matching incomplete run')
        checkpoint = training._verify_checkpoint(args.resume_checkpoint,digest)
        saved = [int(p.name.removeprefix('checkpoint-')) for p in args.output_dir.glob('checkpoint-*') if p.is_dir()]
        if not saved or max(saved)!=checkpoint['step']:
            raise CognitiveKernelContractError('sequence must resume latest complete boundary')
        training._verify_seed_control(args.output_dir,digest,prepared.config)
        epoch_cursor,cursor,steps = training._resume(args.resume_checkpoint,args.output_dir,model,optimizer,digest)
    else:
        args.output_dir.mkdir(parents=True,mode=0o700,exist_ok=False)
        shared._write_new(args.output_dir/'run.json',manifest)
        shared._write_new(args.output_dir/'curriculum.json',{k:v for k,v in receipt.items() if k!='record_sha256'})
        seed = training._seed_control(args.output_dir,model,digest,prepared.config)
        if capacity:
            probe_run = shared._read_sealed(args.probe_dir/'run.json',RUN_SCHEMA)
            prior_seed = training._verify_seed_control(args.probe_dir,probe_run['record_sha256'],prepared.config)
            if seed['specialist_sha256']!=prior_seed['specialist_sha256']:
                raise CognitiveKernelContractError('sequence fit negative control differs from stress execution')
        model.load_formation_parent(load_file(str(parent_weights)))
        epoch_cursor=cursor=steps=0
    started = perf_counter()
    if args.probe_only:
        result = probe(model,optimizer,prepared,args,digest,receipt,weights)
    else:
        if steps==0:
            progress(args.output_dir,digest,model,prepared,weights,0,0)
        for epoch in range(epoch_cursor,args.epochs):
            order = list(prepared.train)
            random.Random(old_run['seed']+epoch).shuffle(order)
            position = cursor if epoch==epoch_cursor else 0
            if not 0<=position<=len(order) or (position!=len(order) and position%args.gradient_accumulation):
                raise CognitiveKernelContractError('sequence cursor is not an optimizer boundary')
            while position<len(order):
                measured = optimizer_step(model,optimizer,order[position:position+args.gradient_accumulation],prepared=prepared,weights=weights)
                position+=measured['train_examples'];steps+=1
                pause = perf_counter()-started>=args.session_seconds
                if steps%args.save_every_steps==0 or pause or position==len(order):
                    training._checkpoint(args.output_dir,specialist=model,optimizer=optimizer,run_digest=digest,
                                         epoch=epoch,next_case=position,step=steps)
                print(json.dumps({'optimizer_step':steps,'epoch':epoch,'train_cases_in_epoch':position,**measured}),flush=True)
                if pause and position<len(order):
                    return {'state':'CHECKPOINTED_INCOMPLETE','optimizer_steps':steps,
                        'resume_checkpoint':str(args.output_dir/f'checkpoint-{steps:08d}'),'qualified_for_product':False}
            progress(args.output_dir,digest,model,prepared,weights,epoch+1,steps)
            if perf_counter()-started>=args.session_seconds and epoch+1<args.epochs:
                return {'state':'CHECKPOINTED_INCOMPLETE','optimizer_steps':steps,
                    'resume_checkpoint':str(args.output_dir/f'checkpoint-{steps:08d}'),'qualified_for_product':False}
        path = args.output_dir/'formation-specialist.safetensors'
        save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(path))
        trained_loss = ancestry.child.parent.development_loss(model,prepared)
        model.load_state_dict(load_file(str(args.output_dir/'seed-control.safetensors')))
        seed_loss = ancestry.child.parent.development_loss(model,prepared)
        result = {'optimizer_steps':steps,'all_train_epochs_completed':True,
            'trained_weights_sha256':shared._digest(path),'diagnostic_development_teacher_forced_loss':trained_loss,
            'seeded_development_teacher_forced_loss':seed_loss,'development_has_no_gradients':True,
            'generated_ability_established':False,'independent_qualification':False}
    component = shared._write_new(args.output_dir/'component.json',{'schema':ARTIFACT_SCHEMA,
        'run_sha256':digest,'source_view_sha256':prepared.view['record_sha256'],
        'curriculum_sha256':receipt['record_sha256'],'specialist_config':prepared.config.record(),
        'probe_only':args.probe_only,'teacher_fit':True,'full_fit':False,'qualified_for_product':False,'result':result})
    return {'state':'SEQUENCE_PROBE_COMPLETED' if args.probe_only else 'SEQUENCE_LEARNING_COMPLETED_UNQUALIFIED',
        'component_sha256':component['record_sha256'],'seconds':perf_counter()-started,**result,'qualified_for_product':False}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('teacher-training-manifest','preflight-receipt','feature-bank','component-dir','source-view',
                 'parent-decision-dir','decision-component-dir','budget-diagnostic-dir','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('input-sha256','owner-authorization-ref','feature-bank-sha256','component-sha256','source-view-sha256',
                 'parent-decision-sha256','parent-checkpoint-sha256','decision-component-sha256','budget-diagnostic-sha256'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--max-new-tokens',type=int,default=2048)
    p.add_argument('--epochs',type=int,default=4);p.add_argument('--gradient-accumulation',type=int,default=16)
    p.add_argument('--save-every-steps',type=int,default=10);p.add_argument('--session-seconds',type=int,default=72000)
    p.add_argument('--probe-only',action='store_true');p.add_argument('--probe-dir',type=Path)
    p.add_argument('--probe-sha256');p.add_argument('--resume-checkpoint',type=Path)
    return p.parse_args()


def main():
    os.umask(0o077)
    os.environ.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_DISABLED':'true'})
    print(json.dumps(run(arguments()),sort_keys=True),flush=True)


if __name__=='__main__': main()
