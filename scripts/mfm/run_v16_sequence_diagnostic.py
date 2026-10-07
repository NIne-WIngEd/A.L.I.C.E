"""Pinned, target-free generation for both new sequence candidate tasks.

Formation retains the original strict validator and fixed 29 train plus nine
development cases. Source reconstruction compares exact canonical opened-source
JSON on the actual capacity stress roster. These are paired task/control
measurements, not independent samples, semantic gold, memory authority or FINAL.
"""
import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

from cognitive_kernel.canonical import (
    CognitiveKernelContractError, canonical_json_bytes, require_sha256,
)
from cognitive_kernel.formation_sequence_specialist import (
    FormationSequenceSpecialist, INITIALIZATION, TASKS,
)
from . import train_v16_sequence_specialist as trainer

ancestry, curriculum = trainer.ancestry, trainer.curriculum
source, shared, training = trainer.source, trainer.shared, trainer.training
original = ancestry.original
SCHEMA = 'mfm-v16-source-sequence-generated-diagnostic-v1'
FORMATION_BUDGET = 2048
CONTROLS = ('trained', 'seeded-untrained')


def select_rosters(prepared, receipt):
    """Keep the historical complete scenario and actual new target extremes."""
    formation_train = ancestry.verify_roster(prepared)
    curriculum.balanced_case_weights(prepared.train)
    wanted = trainer.stress_ids(prepared, receipt)
    by_id = {e.case_id:e for e in prepared.train}
    if (not wanted or len(wanted)!=len(set(wanted)) or not set(wanted).issubset(by_id)):
        raise CognitiveKernelContractError('source generation stress roster differs from admitted train')
    return {'formation':(*formation_train,*prepared.development),
            'source_reconstruction':tuple(by_id[i] for i in wanted)}


def decode_budgets(prepared, receipt):
    longest = prepared.view['longest_target_tokens']
    lexical = receipt['maximum_lexical_target_tokens']
    if (type(longest) is not int or longest<1 or longest>FORMATION_BUDGET
            or prepared.config.max_target_tokens<FORMATION_BUDGET
            or type(lexical) is not int or not 1<=lexical<=prepared.config.max_target_tokens):
        raise CognitiveKernelContractError('sequence generation requires both complete target budgets including EOS')
    return {'formation':FORMATION_BUDGET, 'source_reconstruction':lexical}


def verify_terminal(run, result, train_cases):
    """An optimizer step alone does not establish a complete 432-case epoch."""
    epoch, accumulation = run.get('epochs'),run.get('gradient_accumulation')
    if (train_cases!=432 or type(epoch) is not int or not 1<=epoch<=4
            or type(accumulation) is not int or not 1<=accumulation<=432
            or type(run.get('save_every_steps')) is not int or run['save_every_steps']<1
            or type(run.get('session_seconds')) is not int or run['session_seconds']<1):
        raise CognitiveKernelContractError('sequence terminal construction differs from full bounded training')
    expected = math.ceil(432/accumulation)*epoch
    if (type(result.get('optimizer_steps')) is not int or result['optimizer_steps']!=expected
            or result.get('all_train_epochs_completed') is not True
            or result.get('development_has_no_gradients') is not True
            or result.get('generated_ability_established') is not False
            or result.get('independent_qualification') is not False):
        raise CognitiveKernelContractError('sequence candidate is incomplete or exceeds its diagnostic claim')
    return expected


def _verify_progress(root, run, receipt):
    steps_per_epoch = math.ceil(432/run['gradient_accumulation'])
    expected = [root/f'progress-{epoch*steps_per_epoch:08d}.json' for epoch in range(run['epochs']+1)]
    if set(root.glob('progress-*.json'))!=set(expected):
        raise CognitiveKernelContractError('sequence complete-epoch progress roster differs')
    answer=[]
    for epoch,path in enumerate(expected):
        row=shared._read_sealed(path,trainer.PROGRESS_SCHEMA)
        if (row.get('run_sha256')!=run['record_sha256'] or row.get('epoch')!=epoch
                or row.get('optimizer_steps')!=epoch*steps_per_epoch or row.get('train_cases')!=432
                or row.get('development_read') is not False or row.get('qualified_for_product') is not False):
            raise CognitiveKernelContractError('sequence complete-epoch progress binding differs')
        losses=row.get('complete_case_losses',{})
        objective=float(curriculum.balanced_objective(losses,receipt['weights']))
        if (not isinstance(row.get('balanced_two_task_objective'),(int,float))
                or not math.isclose(objective,row['balanced_two_task_objective'],rel_tol=1e-9,abs_tol=1e-9)):
            raise CognitiveKernelContractError('sequence complete-roster objective differs')
        if row.get('strata')!=trainer.task_strata(losses):
            raise CognitiveKernelContractError('sequence full formation/source strata differ')
        answer.append(row)
    return answer


def verify_candidate(args, prepared):
    """Original external parent pins are checked before reading the new child."""
    old_run,old_component,old_diag,_,_,old_progress=ancestry.verify_child(args,prepared)
    import torch
    root=args.sequence_dir
    if any(p.is_symlink() for p in (root,*root.parents)):
        raise CognitiveKernelContractError('sequence candidate directory must have direct custody')
    run=shared._read_sealed(root/'run.json',trainer.RUN_SCHEMA)
    component=shared._read_sealed(root/'component.json',trainer.ARTIFACT_SCHEMA)
    receipt=curriculum.curriculum_receipt(prepared.train,prepared.tokenizer,prepared.config.max_target_tokens)
    stored=shared._read_sealed(root/'curriculum.json',curriculum.SCHEMA)
    expected={'schema':trainer.RUN_SCHEMA,'objective':curriculum.OBJECTIVE,'task_names':list(TASKS),
        'formation_objective':ancestry.child.parent.OBJECTIVE,'task_mixture':[.5,.5],
        'initialization':INITIALIZATION,'optimizer_initialization':trainer.OPTIMIZER_INITIALIZATION,
        'source_view_sha256':prepared.view['record_sha256'],'code_binding':trainer.code_binding(),
        'curriculum_sha256':receipt['record_sha256'],'corpus_sha256':args.input_sha256,
        'owner_authorization_ref':args.owner_authorization_ref,'specialist_config':prepared.config.record(),
        'specialist_dtype':'float32','device':str(prepared.device),'torch_version':torch.__version__,
        'seed':old_run['seed'],'learning_rate':old_run['learning_rate'],
        'parent_component_sha256':old_component['record_sha256'],'parent_run_sha256':old_run['record_sha256'],
        'parent_weights_sha256':old_component['result']['trained_weights_sha256'],
        'parent_diagnostic_sha256':old_diag['record_sha256'],'parent_optimizer_loaded':False,
        'seed_control_meaning':'same-seed new-task fresh decoder; historical parent has different objective',
        'stress_train_case_ids':trainer.stress_ids(prepared,receipt),'probe_only':False,
        'teacher_fit':True,'full_fit':False,'independent_gold':False,
        'final_payloads_opened':False,'qualified_for_product':False}
    variable={'gradient_accumulation','epochs','save_every_steps','session_seconds','stress_probe_sha256','record_sha256'}
    if (component['record_sha256']!=require_sha256(args.sequence_sha256,'sequence_component_sha256')
            or set(run)!=set(expected)|variable or any(run.get(k)!=v for k,v in expected.items())
            or component.get('run_sha256')!=run['record_sha256']
            or component.get('source_view_sha256')!=prepared.view['record_sha256']
            or component.get('curriculum_sha256')!=receipt['record_sha256'] or stored!=receipt
            or component.get('specialist_config')!=prepared.config.record()
            or component.get('probe_only') is not False or component.get('teacher_fit') is not True
            or component.get('full_fit') is not False or component.get('qualified_for_product') is not False):
        raise CognitiveKernelContractError('sequence candidate objective/code/corpus/parent lineage differs')
    result=component.get('result',{})
    steps=verify_terminal(run,result,len(prepared.train))
    weights=root/'formation-specialist.safetensors'
    if shared._digest(weights)!=require_sha256(result.get('trained_weights_sha256'),'sequence_weights_sha256'):
        raise CognitiveKernelContractError('sequence trained weights differ')
    seed=training._verify_seed_control(root,run['record_sha256'],prepared.config)
    if seed['specialist_sha256']==result['trained_weights_sha256']:
        raise CognitiveKernelContractError('sequence trained weights equal the untrained negative control')
    checkpoint=training._verify_checkpoint(root/f'checkpoint-{steps:08d}',run['record_sha256'])
    if (checkpoint.get('epoch')!=run['epochs']-1 or checkpoint.get('next_case')!=432
            or checkpoint.get('step')!=steps or checkpoint.get('specialist_sha256')!=result['trained_weights_sha256']):
        raise CognitiveKernelContractError('sequence final weights are not a completed full-train boundary')
    progress=_verify_progress(root,run,receipt)
    probe_args=SimpleNamespace(probe_only=False,probe_dir=args.sequence_probe_dir,
                              probe_sha256=run['stress_probe_sha256'])
    probe=trainer.verify_probe(probe_args,{k:v for k,v in run.items() if k!='record_sha256'},receipt)
    probe_run=shared._read_sealed(args.sequence_probe_dir/'run.json',trainer.RUN_SCHEMA)
    probe_curriculum=shared._read_sealed(args.sequence_probe_dir/'curriculum.json',curriculum.SCHEMA)
    probe_seed=training._verify_seed_control(args.sequence_probe_dir,probe_run['record_sha256'],prepared.config)
    proof=probe['result'];accumulation=proof.get('accumulation_case_ids',[])
    by_id={e.case_id for e in prepared.train}
    if (probe_curriculum!=receipt or probe_seed['specialist_sha256']!=seed['specialist_sha256']
            or len(accumulation)!=run['gradient_accumulation'] or len(set(accumulation))!=len(accumulation)
            or not set(accumulation).issubset(by_id) or not set(run['stress_train_case_ids']).issubset(accumulation)
            or proof.get('migration_parity',{}).get('case_ids')!=accumulation
            or proof.get('migration_parity',{}).get('max_abs_difference')!=0
            or any(not isinstance(proof.get(k),(int,float)) or not math.isfinite(proof[k]) or proof[k]<0
                   for k in ('model_max_abs_difference','optimizer_max_abs_difference'))):
        raise CognitiveKernelContractError('sequence actual capacity/restart/seed proof differs')
    decode_budgets(prepared,receipt)
    select_rosters(prepared,receipt)
    return SimpleNamespace(run=run,component=component,curriculum=receipt,weights=weights,seed=seed,
        probe=probe,probe_run=probe_run,checkpoint=checkpoint,progress=progress,
        parent_component=old_component,parent_run=old_run,parent_diagnostic=old_diag,parent_progress=old_progress)


def verify_seed_parameters(model, state):
    """A sealed file alone does not prove that the control has the same seed."""
    import torch
    fresh=model.state_dict()
    if (set(fresh)!=set(state) or any(fresh[k].dtype!=torch.float32 or state[k].dtype!=torch.float32
            or not torch.equal(fresh[k].detach().cpu(),state[k].detach().cpu()) for k in fresh)):
        raise CognitiveKernelContractError('sequence negative control is not the actual same-seed fresh model')
    return {'actual_same_seed_parameters_equal':True,'parameter_tensors':len(fresh)}


def validate_source(raw, ended, example):
    """Exact sequence equality is separate from formation grounding/semantics."""
    expected=canonical_json_bytes(curriculum.lexical_source_record(example)).decode('utf-8')
    exact=ended and raw==expected
    return {'raw_output_text':raw,'eos_observed':ended,
        'validation_status':'exact_source_reconstruction' if exact else 'invalid',
        'validation_error':None if exact else ('no EOS inside declared complete diagnostic budget'
            if not ended else 'generated source JSON differs from exact canonical opened source bytes/order'),
        'exact_source_json_agreement':exact,'source_target_sha256':shared._record_hash(curriculum.lexical_source_record(example)),
        'memory_validator_executed':False}


def generate_case(model, prepared, example, *, task, control, weights_sha256, run_id, maximum):
    """Only source tensors enter greedy generation; comparisons follow it."""
    if task not in TASKS or control not in CONTROLS or example.split not in {'train','development'}:
        raise CognitiveKernelContractError('sequence generation task/control/split differs')
    root,export=prepared.locators[example.case_id]
    _,tensors=source.features.read_case(root,example,export,width=prepared.config.base_hidden_size)
    tokens,ended,parity=original.greedy_fp32(model.bind_task(task),tensors['states'],tensors['attention_mask'],maximum)
    raw=prepared.tokenizer.decode(tokens,skip_special_tokens=False)
    validation=(original.validate(raw,ended,example,weights_sha256,run_id) if task=='formation'
                else validate_source(raw,ended,example))
    return {'case_id':example.case_id,'split':example.split,'family':ancestry.family_name(example),
        'task':task,'control':control,'weights_sha256':weights_sha256,
        'source_view_sha256':prepared.view['record_sha256'],'context_sha256':example.context.content_digest(),
        'model_input_sha256':source.features.case_key(example),'inference_run_id':run_id,
        'max_new_tokens':maximum,'generated_token_ids':tokens,'incremental_parity':parity,**validation}


def add_measurement(summary, row):
    family=summary.setdefault(row['task'],{}).setdefault(row['family'],
        {'cases':0,'eos_outputs':0,'invalid_outputs':0})
    family['cases']+=1;family['eos_outputs']+=int(row['eos_observed'])
    family['invalid_outputs']+=int(row['validation_status']=='invalid')
    if row['task']=='formation':
        for key in ('grounded_outputs','proposals_in_grounded_outputs','exact_teacher_agreements'):
            family.setdefault(key,0)
        family['grounded_outputs']+=int(row['validation_status']=='grounded_teacher_proposal_only')
        family['proposals_in_grounded_outputs']+=row['proposal_count']
        family['exact_teacher_agreements']+=int(row['exact_teacher_decision_agreement'] is True)
    else:
        family.setdefault('exact_source_reconstructions',0)
        family['exact_source_reconstructions']+=int(row['exact_source_json_agreement'] is True)


def code_binding():
    import cognitive_kernel.formation_incremental_decoder as incremental
    return {**trainer.code_binding(),'sequence_generated_diagnostic':shared._digest(Path(__file__)),
            'original_decode_and_formation_validator':shared._digest(Path(original.__file__)),
            'incremental_decoder':shared._digest(Path(incremental.__file__))}


def run(args):
    prepared=source.load(args)  # Original owner/network admission precedes new candidate reads.
    candidate=verify_candidate(args,prepared)
    import torch
    from safetensors.torch import load_file
    torch.manual_seed(candidate.run['seed'])
    model=FormationSequenceSpecialist(prepared.config).to(prepared.device,dtype=torch.float32)
    seed_state=load_file(str(args.sequence_dir/'seed-control.safetensors'))
    seed_check=verify_seed_parameters(model,seed_state)
    rosters=select_rosters(prepared,candidate.curriculum)
    budgets=decode_budgets(prepared,candidate.curriculum)
    args.output_dir.mkdir(parents=True,mode=0o700,exist_ok=False)
    digests,summaries={},{}
    for control,path,digest in (('trained',candidate.weights,candidate.component['result']['trained_weights_sha256']),
            ('seeded-untrained',args.sequence_dir/'seed-control.safetensors',candidate.seed['specialist_sha256'])):
        model.load_state_dict(load_file(str(path)))
        summaries[control]={}
        for task in TASKS:
            logical=control+'-'+task
            file=args.output_dir/(logical+'.jsonl')
            run_id=candidate.run['record_sha256']+'-'+logical
            with file.open('x',encoding='utf-8',newline='\n') as stream:
                for example in rosters[task]:
                    row=generate_case(model,prepared,example,task=task,control=control,
                        weights_sha256=digest,run_id=run_id,maximum=budgets[task])
                    stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n');stream.flush()
                    add_measurement(summaries[control],row)
                    print(json.dumps({'state':'SEQUENCE_CASE_GENERATED','control':control,'task':task,
                        'case_id':example.case_id,'family':row['family'],'eos_observed':row['eos_observed'],
                        'validation_status':row['validation_status'],'generated_tokens':len(row['generated_token_ids'])}),flush=True)
            digests[logical]=shared._digest(file)
    return shared._write_new(args.output_dir/'diagnostic.json',{
        'schema':SCHEMA,'component_sha256':candidate.component['record_sha256'],
        'run_sha256':candidate.run['record_sha256'],'source_view_sha256':prepared.view['record_sha256'],
        'curriculum_sha256':candidate.curriculum['record_sha256'],
        'stress_probe_sha256':candidate.probe['record_sha256'],
        'stress_probe_run_sha256':candidate.probe_run['record_sha256'],
        'completed_checkpoint_sha256':candidate.checkpoint['record_sha256'],
        'complete_train_progress_sha256':[p['record_sha256'] for p in candidate.progress],
        'seed_control_sha256':candidate.seed['record_sha256'],'seed_parameters_check':seed_check,
        'historical_parent_component_sha256':candidate.parent_component['record_sha256'],
        'historical_parent_diagnostic_sha256':candidate.parent_diagnostic['record_sha256'],
        'historical_outputs_rewritten':False,'outputs_sha256':digests,'summary':summaries,
        'case_rosters':{task:[e.case_id for e in examples] for task,examples in rosters.items()},
        'code_binding':code_binding(),
        'decode':{'method':'FP32_incremental_first32_actual_greedy_reference_checked',
            'explicit_task_bound_view':True,'max_new_tokens_by_task':budgets,'device':str(prepared.device),
            'specialist_dtype':'float32','torch_version':torch.__version__,'target_visible':False,
            'matched_between_controls':True,'target_comparison_after_generation':True},
        'measurement_meaning':{'formation':'original strict grounding and exact teacher fields, not semantic entailment',
            'source_reconstruction':'exact canonical UTF-8 source JSON/order and EOS, not a memory decision',
            'sampling':'paired controls and tasks; repeated cases are not independent samples'},
        'optimizer_created':False,'independent_gold':False,'final_payloads_opened':False,
        'memory_gate_executed':False,'authority_written':False,'qualified_for_product':False,
        'downstream_usefulness_established':False})


def arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('teacher-training-manifest','preflight-receipt','feature-bank','component-dir','source-view',
            'parent-decision-dir','decision-component-dir','budget-diagnostic-dir','sequence-dir',
            'sequence-probe-dir','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('input-sha256','owner-authorization-ref','feature-bank-sha256','component-sha256',
            'source-view-sha256','parent-decision-sha256','parent-checkpoint-sha256',
            'decision-component-sha256','budget-diagnostic-sha256','sequence-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cpu')
    parser.set_defaults(max_new_tokens=FORMATION_BUDGET)
    return parser.parse_args(argv)


def main():
    os.umask(0o077)
    os.environ.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_DISABLED':'true'})
    result=run(arguments())
    print(json.dumps({'sequence_diagnostic_sha256':result['record_sha256'],
        'summary':result['summary'],'qualified_for_product':False}),flush=True)


if __name__=='__main__': main()
