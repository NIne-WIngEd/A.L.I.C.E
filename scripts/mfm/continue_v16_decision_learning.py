"""Bounded full-corpus learning from a separately pinned completed parent.

No architecture, target, source, learning-rate or authority change. Train-only
equal-case CE determines insufficient-progress stopping; diagnostics are not
qualification. Original parent artifacts remain immutable.
"""
import argparse
from collections import Counter
import math
import os
from pathlib import Path
import random
from time import perf_counter
from types import SimpleNamespace
import json

from cognitive_kernel.canonical import CognitiveKernelContractError, require_sha256
from . import train_v16_decision_specialist as parent
from . import analyze_v16_decision_fit as analysis

source, training, shared = parent.source, parent.training, parent.shared
RUN_SCHEMA = 'mfm-v16-decision-budget-run-v1'
ARTIFACT_SCHEMA = 'mfm-v16-decision-budget-artifact-v1'
PROGRESS_SCHEMA = 'mfm-v16-decision-budget-train-progress-v1'


def code_binding():
    return {'continuation': shared._digest(Path(__file__)),
            'train_analysis': shared._digest(Path(analysis.__file__)),
            'parent_measure': shared._digest(Path(analysis.parent_analysis.__file__)),
            **parent.code_binding()}


def should_stop(losses):
    if any(not math.isfinite(x) or x <= 0 for x in losses):
        raise CognitiveKernelContractError('complete-train stopping loss is invalid')
    return len(losses) >= 3 and all((before - after) / before < .01
        for before, after in zip(losses[-3:-1], losses[-2:]))


def verify_parent(args, prepared):
    import torch
    root=args.parent_decision_dir
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise CognitiveKernelContractError('completed parent custody is a link')
    component=shared._read_sealed(root/'component.json', parent.ARTIFACT_SCHEMA)
    run=shared._read_sealed(root/'run.json', parent.RUN_SCHEMA)
    if component['record_sha256'] != require_sha256(args.parent_decision_sha256,'parent_decision_sha256'):
        raise CognitiveKernelContractError('completed parent external component pin differs')
    expected={'code_binding':parent.code_binding(),'objective':parent.OBJECTIVE,
        'initialization':parent.INITIALIZATION,'source_view_sha256':prepared.view['record_sha256'],
        'corpus_sha256':args.input_sha256,'owner_authorization_ref':args.owner_authorization_ref,
        'specialist_config':prepared.config.record(),'specialist_dtype':'float32',
        'device':str(prepared.device),'torch_version':torch.__version__,
        'probe_only':False,'teacher_fit':True,'full_fit':False,'independent_gold':False,
        'final_payloads_opened':False,'qualified_for_product':False}
    result=component.get('result',{})
    if any(run.get(k)!=v for k,v in expected.items()) or component.get('run_sha256')!=run['record_sha256'] or component.get('source_view_sha256')!=prepared.view['record_sha256'] or component.get('specialist_config')!=prepared.config.record() or component.get('probe_only') is not False or component.get('qualified_for_product') is not False or result.get('all_train_epochs_completed') is not True:
        raise CognitiveKernelContractError('completed parent recipe/lineage differs')
    if type(run.get('epochs')) is not int or run['epochs']<1 or type(run.get('gradient_accumulation')) is not int or not 1<=run['gradient_accumulation']<=len(prepared.train):
        raise CognitiveKernelContractError('completed parent epoch/accumulation is invalid')
    steps=math.ceil(len(prepared.train)/run['gradient_accumulation'])*run['epochs']
    if result.get('optimizer_steps')!=steps:
        raise CognitiveKernelContractError('completed parent steps differ from full train roster')
    checkpoint_dir=root/f'checkpoint-{steps:08d}'
    checkpoint=training._verify_checkpoint(checkpoint_dir,run['record_sha256'])
    if checkpoint['record_sha256']!=require_sha256(args.parent_checkpoint_sha256,'parent_checkpoint_sha256') or checkpoint.get('epoch')!=run['epochs']-1 or checkpoint.get('next_case')!=len(prepared.train) or checkpoint.get('step')!=steps or result.get('optimizer_steps')!=steps or checkpoint.get('specialist_sha256')!=result.get('trained_weights_sha256') or shared._digest(root/'formation-specialist.safetensors')!=result.get('trained_weights_sha256'):
        raise CognitiveKernelContractError('parent must be exact externally pinned completed checkpoint')
    return SimpleNamespace(root=root,component=component,run=run,checkpoint=checkpoint,checkpoint_dir=checkpoint_dir)


def restore_parent(record, model, optimizer):
    epoch,cursor,step=training._resume(record.checkpoint_dir,record.root,model,optimizer,record.run['record_sha256'])
    if any(group['lr']!=record.run['learning_rate'] for group in optimizer.param_groups):
        raise CognitiveKernelContractError('parent optimizer learning rate differs from recipe')
    return epoch+1,0,step


def train_losses(model, prepared):
    import torch
    model.eval()
    case_sum=weighted_sum=0.
    tokens=cases=0
    with torch.inference_mode(),torch.autocast(prepared.device.type,enabled=False):
        for example in prepared.train:
            if example.split!='train':
                raise CognitiveKernelContractError('stopping metrics require train cases only')
            payload=source.payload(prepared,example,device=prepared.device)
            loss=float(model(**payload)[1].cpu())
            if not math.isfinite(loss):
                raise CognitiveKernelContractError('complete-train progress is nonfinite')
            n=payload['labels'].numel()
            case_sum+=loss;weighted_sum+=loss*n;tokens+=n;cases+=1
    if not cases or not tokens:
        raise CognitiveKernelContractError('complete-train progress is empty')
    return {'train_cases':cases,'target_tokens':tokens,'equal_case_mean_loss':case_sum/cases,
            'token_weighted_mean_loss':weighted_sum/tokens}


def record_progress(root,digest,model,prepared,steps,epochs,train_counts):
    path=root/f'progress-{steps:08d}.json'
    if path.exists():
        row=shared._read_sealed(path,PROGRESS_SCHEMA)
        if row.get('run_sha256')!=digest or row.get('global_optimizer_steps')!=steps or row.get('global_completed_epochs')!=epochs:
            raise CognitiveKernelContractError('existing train progress lineage differs')
        return row
    metrics=train_losses(model,prepared)
    wanted=set(prepared.view['stress_train_case_ids'])|set(source.preparation.prior.PUBLIC_SEED_PRIOR_MEMORY)
    selected=[e for e in prepared.train if e.case_id in wanted]
    fields=[]
    for example in selected:
        _,_,groups=analysis.target_groups(prepared.tokenizer,example,prepared.config.max_target_tokens)
        payload=source.payload(prepared,example,device=prepared.device)
        fields.append({'case_id':example.case_id,'split':'train',
            **analysis.measure_with_exposure(model,payload,groups,train_counts)})
    row=shared._write_new(path,{'schema':PROGRESS_SCHEMA,'run_sha256':digest,
        'global_optimizer_steps':steps,'global_completed_epochs':epochs,
        'stopping_objective':'equal-case mean of full per-case token-mean CE; development excluded',
        **metrics,'train_field_measurements':fields,'qualified_for_product':False})
    print(json.dumps({'state':'COMPLETE_TRAIN_PROGRESS','global_optimizer_steps':steps,
        'global_completed_epochs':epochs,**metrics,'progress_sha256':row['record_sha256']}),flush=True)
    return row


def read_progress(root,digest,steps):
    rows=[shared._read_sealed(p,PROGRESS_SCHEMA) for p in sorted(root.glob('progress-*.json'))]
    if any(r.get('run_sha256')!=digest or r.get('global_optimizer_steps',steps+1)>steps for r in rows):
        raise CognitiveKernelContractError('train progress is ahead of resume checkpoint or another run')
    return rows


def run(args):
    prepared=source.load(args)  # Original privacy admission precedes all new reads.
    if type(args.additional_epochs) is not int or not 1<=args.additional_epochs<=6 or args.save_every_steps<1 or args.session_seconds<1:
        raise CognitiveKernelContractError('invalid bounded continuation recipe')
    import torch
    from safetensors.torch import load_file,save_file
    record=verify_parent(args,prepared)
    base=record.run
    epochs=base['epochs']+args.additional_epochs
    manifest={'schema':RUN_SCHEMA,'source_view_sha256':prepared.view['record_sha256'],
        'code_binding':code_binding(),'objective':parent.OBJECTIVE,'initialization':parent.INITIALIZATION,
        'corpus_sha256':args.input_sha256,'owner_authorization_ref':args.owner_authorization_ref,
        'specialist_config':prepared.config.record(),'specialist_dtype':'float32',
        'device':str(prepared.device),'torch_version':torch.__version__,
        'seed':base['seed'],'learning_rate':base['learning_rate'],'gradient_accumulation':base['gradient_accumulation'],
        'parent_component_sha256':record.component['record_sha256'],'parent_run_sha256':base['record_sha256'],
        'parent_checkpoint_sha256':record.checkpoint['record_sha256'],
        'parent_optimizer_sha256':record.checkpoint['optimizer_sha256'],
        'parent_weights_sha256':record.checkpoint['specialist_sha256'],
        'parent_completed_epochs':base['epochs'],'parent_optimizer_steps':record.checkpoint['step'],
        'maximum_total_epochs':epochs,'maximum_additional_epochs':args.additional_epochs,
        'progress_every_additional_epochs':2,'insufficient_progress_intervals':2,'minimum_relative_train_improvement':.01,
        'save_every_steps':args.save_every_steps,'session_seconds':args.session_seconds,
        'seed_control_meaning':'original same-seed untrained negative control, not continuation initialization',
        'teacher_fit':True,'full_fit':False,'independent_gold':False,'final_payloads_opened':False,'qualified_for_product':False}
    digest=shared._record_hash(manifest)
    torch.manual_seed(base['seed'])
    model=parent.FormationDecisionSpecialist(prepared.config).to(prepared.device,dtype=torch.float32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=base['learning_rate'])
    if args.resume_checkpoint:
        if (args.output_dir/'component.json').exists() or shared._read_sealed(args.output_dir/'run.json',RUN_SCHEMA)['record_sha256']!=digest:
            raise CognitiveKernelContractError('continuation resume needs exact incomplete child run')
        selected=training._verify_checkpoint(args.resume_checkpoint,digest)
        saved=[int(p.name.removeprefix('checkpoint-')) for p in args.output_dir.glob('checkpoint-*') if p.is_dir()]
        if not saved or max(saved)!=selected['step']:
            raise CognitiveKernelContractError('continuation must resume latest saved optimizer boundary')
        training._verify_seed_control(args.output_dir,digest,prepared.config)
        epoch_cursor,case_cursor,steps=training._resume(args.resume_checkpoint,args.output_dir,model,optimizer,digest)
    else:
        args.output_dir.mkdir(mode=0o700,parents=True,exist_ok=False)
        shared._write_new(args.output_dir/'run.json',manifest)
        training._seed_control(args.output_dir,model,digest,prepared.config)
        seed=training._verify_seed_control(args.output_dir,digest,prepared.config)
        parent_seed=training._verify_seed_control(record.root,base['record_sha256'],prepared.config)
        if seed['specialist_sha256']!=parent_seed['specialist_sha256']:
            raise CognitiveKernelContractError('child negative control differs from original parent seed')
        epoch_cursor,case_cursor,steps=restore_parent(record,model,optimizer)
        training._checkpoint(args.output_dir,specialist=model,optimizer=optimizer,
            run_digest=digest,epoch=epoch_cursor,next_case=0,step=steps)
    counts=Counter()
    for example in prepared.train:
        _,labels=source.preparation.target_ids(prepared.tokenizer,example,prepared.config.max_target_tokens)
        counts.update(labels)
    started=perf_counter()
    history=read_progress(args.output_dir,digest,steps)
    if not history:
        if steps!=record.checkpoint['step'] or case_cursor!=0:
            raise CognitiveKernelContractError('continuation lacks original full-train baseline')
        history.append(record_progress(args.output_dir,digest,model,prepared,steps,base['epochs'],counts))
    reason='additional_epoch_cap'
    completed_epochs=base['epochs']
    for epoch in range(epoch_cursor,epochs):
        order=list(prepared.train)
        random.Random(base['seed']+epoch).shuffle(order)
        cursor=case_cursor if epoch==epoch_cursor else 0
        if not 0<=cursor<=len(order) or (cursor!=len(order) and cursor%base['gradient_accumulation']):
            raise CognitiveKernelContractError('continuation cursor is not an optimizer boundary')
        while cursor<len(order):
            measured=parent.optimizer_step(model,optimizer,order[cursor:cursor+base['gradient_accumulation']],prepared=prepared)
            cursor+=measured['train_examples'];steps+=1
            pause=perf_counter()-started>=args.session_seconds
            if steps%args.save_every_steps==0 or pause or cursor==len(order):
                training._checkpoint(args.output_dir,specialist=model,optimizer=optimizer,
                    run_digest=digest,epoch=epoch,next_case=cursor,step=steps)
            print(json.dumps({'global_optimizer_step':steps,'global_epoch':epoch,
                'train_cases_in_epoch':cursor,**measured}),flush=True)
            if pause and cursor<len(order):
                return {'state':'CHECKPOINTED_INCOMPLETE','resume_checkpoint':str(args.output_dir/f'checkpoint-{steps:08d}'),
                        'global_optimizer_steps':steps,'qualified_for_product':False}
        completed_epochs=epoch+1
        if (completed_epochs-base['epochs'])%2==0 or completed_epochs==epochs:
            row=record_progress(args.output_dir,digest,model,prepared,steps,completed_epochs,counts)
            if not history or history[-1]['record_sha256']!=row['record_sha256']: history.append(row)
            if should_stop([r['equal_case_mean_loss'] for r in history]):
                reason='insufficient_train_progress';break
        if perf_counter()-started>=args.session_seconds and completed_epochs<epochs:
            return {'state':'CHECKPOINTED_INCOMPLETE','resume_checkpoint':str(args.output_dir/f'checkpoint-{steps:08d}'),
                    'global_optimizer_steps':steps,'qualified_for_product':False}
    path=args.output_dir/'formation-specialist.safetensors'
    save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(path))
    trained_loss=parent.development_loss(model,prepared)
    model.load_state_dict(load_file(str(args.output_dir/'seed-control.safetensors')))
    seed_loss=parent.development_loss(model,prepared)
    result={'construction_schedule_completed':True,'terminal_reason':reason,
        'all_maximum_train_epochs_completed':completed_epochs==epochs,'global_completed_epochs':completed_epochs,
        'global_optimizer_steps':steps,'additional_optimizer_steps':steps-record.checkpoint['step'],
        'trained_weights_sha256':shared._digest(path),'full_train_progress_sha256':[r['record_sha256'] for r in history],
        'diagnostic_development_teacher_forced_loss':trained_loss,'seeded_development_teacher_forced_loss':seed_loss,
        'development_has_no_gradients':True,'generated_ability_established':False,'independent_qualification':False}
    receipt=shared._write_new(args.output_dir/'component.json',{'schema':ARTIFACT_SCHEMA,
        'run_sha256':digest,'source_view_sha256':prepared.view['record_sha256'],
        'specialist_config':prepared.config.record(),'teacher_fit':True,'full_fit':False,
        'qualified_for_product':False,'result':result})
    return {'state':'BOUNDED_LEARNING_COMPLETED_UNQUALIFIED','component_sha256':receipt['record_sha256'],
            'seconds':perf_counter()-started,**result,'qualified_for_product':False}


def arguments():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('teacher-training-manifest','preflight-receipt','feature-bank','component-dir','source-view','parent-decision-dir','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('input-sha256','owner-authorization-ref','feature-bank-sha256','component-sha256','source-view-sha256','parent-decision-sha256','parent-checkpoint-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--max-new-tokens',type=int,default=2048)
    parser.add_argument('--additional-epochs',type=int,default=6)
    parser.add_argument('--save-every-steps',type=int,default=10)
    parser.add_argument('--session-seconds',type=int,default=28800)
    parser.add_argument('--resume-checkpoint',type=Path)
    return parser.parse_args()


def main():
    os.umask(0o077)
    os.environ.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_DISABLED':'true'})
    print(json.dumps(run(arguments()),sort_keys=True),flush=True)


if __name__=='__main__': main()
