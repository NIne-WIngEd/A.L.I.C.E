"""Raw generated comparison after an explicitly bounded learning continuation.

Parent outputs are reused byte-for-byte from their pinned original execution;
child and original untrained seed use the identical FP32 reference-checked
greedy path. A declared stop is not completion of the maximum epoch budget.
"""
import argparse
import json
import math
import os
from pathlib import Path

from cognitive_kernel.canonical import CognitiveKernelContractError, require_sha256
from . import continue_v16_decision_learning as child
from . import run_v16_decision_diagnostic as original

SCHEMA='mfm-v16-budget-generated-diagnostic-v1'
source,shared=child.source,child.shared


def verify_terminal(run,result,train_count,losses):
    epochs=result.get('global_completed_epochs')
    if type(epochs) is not int or not run['parent_completed_epochs']<epochs<=run['maximum_total_epochs']:
        raise CognitiveKernelContractError('bounded terminal epoch differs')
    additional=math.ceil(train_count/run['gradient_accumulation'])*(epochs-run['parent_completed_epochs'])
    maximum=epochs==run['maximum_total_epochs']
    if result.get('construction_schedule_completed') is not True or result.get('global_optimizer_steps')!=run['parent_optimizer_steps']+additional or result.get('additional_optimizer_steps')!=additional or result.get('all_maximum_train_epochs_completed') is not maximum:
        raise CognitiveKernelContractError('bounded terminal steps or maximum-completion claim differs')
    reason=result.get('terminal_reason')
    if reason=='additional_epoch_cap' and maximum:
        return
    if reason=='insufficient_train_progress' and child.should_stop(losses):
        return
    raise CognitiveKernelContractError('bounded terminal stopping reason differs')


def run(args):
    prepared=source.load(args)
    import torch
    import cognitive_kernel.formation_incremental_decoder as incremental
    from safetensors.torch import load_file
    record=child.verify_parent(args,prepared)
    component=shared._read_sealed(args.decision_component_dir/'component.json',child.ARTIFACT_SCHEMA)
    run=shared._read_sealed(args.decision_component_dir/'run.json',child.RUN_SCHEMA)
    expected={'code_binding':child.code_binding(),'source_view_sha256':prepared.view['record_sha256'],
        'parent_component_sha256':record.component['record_sha256'],'parent_run_sha256':record.run['record_sha256'],
        'parent_checkpoint_sha256':record.checkpoint['record_sha256'],
        'parent_weights_sha256':record.checkpoint['specialist_sha256'],
        'specialist_config':prepared.config.record(),'specialist_dtype':'float32',
        'torch_version':torch.__version__,'device':str(prepared.device),'qualified_for_product':False}
    if component['record_sha256']!=require_sha256(args.decision_component_sha256,'decision_component_sha256') or component.get('run_sha256')!=run['record_sha256'] or component.get('source_view_sha256')!=prepared.view['record_sha256'] or component.get('specialist_config')!=prepared.config.record() or component.get('qualified_for_product') is not False or any(run.get(k)!=v for k,v in expected.items()):
        raise CognitiveKernelContractError('bounded diagnostic child lineage differs')
    result=component['result']
    progress=child.read_progress(args.decision_component_dir,run['record_sha256'],result['global_optimizer_steps'])
    if [r['record_sha256'] for r in progress]!=result.get('full_train_progress_sha256'):
        raise CognitiveKernelContractError('bounded diagnostic progress roster differs')
    verify_terminal(run,result,len(prepared.train),[r['equal_case_mean_loss'] for r in progress])
    seed=child.training._verify_seed_control(args.decision_component_dir,run['record_sha256'],prepared.config)
    weights=args.decision_component_dir/'formation-specialist.safetensors'
    if shared._digest(weights)!=result['trained_weights_sha256'] or not prepared.view['longest_target_tokens']<=args.max_new_tokens<=prepared.config.max_target_tokens:
        raise CognitiveKernelContractError('bounded diagnostic weights or complete decode budget differ')
    parent_diag=shared._read_sealed(args.parent_diagnostic_dir/'diagnostic.json',original.SCHEMA)
    incremental_sha=shared._digest(Path(incremental.__file__))
    if parent_diag.get('incremental_decoder_sha256')!=incremental_sha:
        raise CognitiveKernelContractError('reused parent incremental decoder implementation differs')
    if parent_diag['record_sha256']!=require_sha256(args.parent_diagnostic_sha256,'parent_diagnostic_sha256') or parent_diag.get('component_sha256')!=record.component['record_sha256'] or parent_diag.get('run_sha256')!=record.run['record_sha256'] or parent_diag.get('source_view_sha256')!=prepared.view['record_sha256'] or parent_diag.get('runner_sha256')!=shared._digest(Path(original.__file__)) or parent_diag.get('decode')!={'device':str(prepared.device),'matched_between_controls':True,'max_new_tokens':args.max_new_tokens,'method':'FP32_incremental_first32_actual_greedy_reference_checked','specialist_dtype':'float32','target_visible':False,'torch_version':torch.__version__}:
        raise CognitiveKernelContractError('reused parent diagnostic settings/lineage differ')
    parent_file=args.parent_diagnostic_dir/'trained.jsonl'
    if shared._digest(parent_file)!=parent_diag['outputs_sha256']['trained']:
        raise CognitiveKernelContractError('reused parent output bytes differ')
    parent_rows=[json.loads(line,object_pairs_hook=original._unique_json) for line in parent_file.read_text(encoding='utf-8').splitlines()]
    if len(parent_rows)!=len(prepared.development) or any(
        row.get('case_id')!=example.case_id or row.get('control')!='trained' or row.get('split')!='development' or row.get('weights_sha256')!=record.checkpoint['specialist_sha256'] or row.get('model_input_sha256')!=source.features.case_key(example)
        for row,example in zip(parent_rows,prepared.development)):
        raise CognitiveKernelContractError('reused parent actual case roster differs')
    args.output_dir.mkdir(mode=0o700,parents=True,exist_ok=False)
    parent_copy=args.output_dir/'parent-fit.jsonl'
    with parent_copy.open('xb') as stream: stream.write(parent_file.read_bytes())
    model=child.parent.FormationDecisionSpecialist(prepared.config).to(prepared.device,dtype=torch.float32)
    summaries={'parent-fit':parent_diag['summary']['trained']}
    hashes={'parent-fit':shared._digest(parent_copy)}
    for control,path,digest in (
        ('trained',weights,result['trained_weights_sha256']),
        ('seeded-untrained',args.decision_component_dir/'seed-control.safetensors',seed['specialist_sha256'])):
        model.load_state_dict(load_file(str(path)))
        summary={'cases':0,'invalid_outputs':0,'grounded_outputs':0,'proposals_in_grounded_outputs':0,'exact_teacher_agreements':0}
        output=args.output_dir/(control+'.jsonl')
        with output.open('x',encoding='utf-8') as stream:
            for example in prepared.development:
                root,export=prepared.locators[example.case_id]
                _,tensors=source.features.read_case(root,example,export,width=prepared.config.base_hidden_size)
                tokens,ended,parity=original.greedy_fp32(model,tensors['states'],tensors['attention_mask'],args.max_new_tokens)
                run_id=f'mfm-budget-{run["record_sha256"][:12]}-{control}-{example.case_id}'
                row={'schema':'mfm-v16-decision-generated-row-v1','case_id':example.case_id,
                    'split':example.split,'control':control,'weights_sha256':digest,
                    'model_input_sha256':source.features.case_key(example),'inference_run_id':run_id,
                    'generated_token_ids':tokens,'incremental_parity':parity,
                    **original.validate(prepared.tokenizer.decode(tokens,skip_special_tokens=False),ended,example,digest,run_id)}
                stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n');stream.flush()
                summary['cases']+=1
                summary['invalid_outputs' if row['validation_status']=='invalid' else 'grounded_outputs']+=1
                summary['proposals_in_grounded_outputs']+=row['proposal_count']
                summary['exact_teacher_agreements']+=row['exact_teacher_decision_agreement'] is True
                print(json.dumps({'control':control,'case_id':example.case_id,'validation_status':row['validation_status'],
                    'eos_observed':ended,'generated_tokens':len(tokens),'seconds':parity['seconds']}),flush=True)
        summaries[control]=summary;hashes[control]=shared._digest(output)
    return shared._write_new(args.output_dir/'diagnostic.json',{'schema':SCHEMA,
        'component_sha256':component['record_sha256'],'run_sha256':run['record_sha256'],
        'source_view_sha256':prepared.view['record_sha256'],'runner_sha256':shared._digest(Path(__file__)),
        'original_decode_implementation_sha256':shared._digest(Path(original.__file__)),
        'incremental_decoder_sha256':incremental_sha,
        'parent_diagnostic_sha256':parent_diag['record_sha256'],
        'parent_control_reuse':'exact original trained rows and bytes; embedded control stays trained; receipt maps them to parent-fit',
        'decode':parent_diag['decode'],'outputs_sha256':hashes,'summary':summaries,
        'agreement_metric':parent_diag['agreement_metric'],'independent_gold':False,
        'qualified_for_product':False,'downstream_usefulness_established':False,
        'memory_gate_executed':False,'final_payloads_opened':False})


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('teacher-training-manifest','preflight-receipt','feature-bank','component-dir','source-view','parent-decision-dir','parent-diagnostic-dir','decision-component-dir','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('input-sha256','owner-authorization-ref','feature-bank-sha256','component-sha256','source-view-sha256','parent-decision-sha256','parent-checkpoint-sha256','parent-diagnostic-sha256','decision-component-sha256'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--max-new-tokens',type=int,default=2048)
    return p.parse_args()


def main():
    os.umask(0o077)
    os.environ.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_DISABLED':'true'})
    r=run(arguments())
    print(json.dumps({'budget_diagnostic_sha256':r['record_sha256'],'summary':r['summary'],'qualified_for_product':False}),flush=True)


if __name__=='__main__': main()
