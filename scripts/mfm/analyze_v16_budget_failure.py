"""Full-roster read-only diagnosis; preserve original producer and validator.

Family metrics expose aggregate masking. Precise scalar paths separate learned
scope labels from source/registered identities. Exact teacher divergence, JSON
faults and original strict validation remain different measurements.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from statistics import median
from time import perf_counter

from cognitive_kernel.canonical import canonical_json_bytes, CognitiveKernelContractError, require_sha256
from . import continue_v16_decision_learning as child
from . import run_v16_budget_diagnostic as budget
from . import run_v16_decision_diagnostic as original
from . import analyze_v16_decision_fit as language

source, shared = child.source, child.shared
SCHEMA = 'mfm-v16-budget-failure-analysis-v1'
LONG_SCENARIO = 'long-atlas-east-'


def family_name(example):
    if example.split == 'development': return 'development'
    if example.split != 'train': raise CognitiveKernelContractError('diagnosis cannot read another split')
    return 'longitudinal-train' if example.case_id.startswith('long-') else 'base-train'


def verify_roster(prepared):
    all_cases = (*prepared.train, *prepared.development)
    counts = Counter(family_name(e) for e in all_cases)
    if counts != {'base-train':16, 'longitudinal-train':416, 'development':9} or len({e.case_id for e in all_cases}) != 441 or any(e.split!='train' for e in prepared.train) or any(e.split!='development' for e in prepared.development):
        raise CognitiveKernelContractError('full diagnosis family/case roster differs')
    selected = tuple(e for e in prepared.train if family_name(e)=='base-train' or e.case_id.startswith(LONG_SCENARIO))
    if len(selected) != 29 or sum(e.case_id.startswith(LONG_SCENARIO) for e in selected) != 13:
        raise CognitiveKernelContractError('fixed complete train-generation scenario differs')
    return selected


def precise_scalar_spans(record):
    pieces, spans, position = [], [], 0
    def add(text, kind=None):
        nonlocal position
        pieces.append(text)
        if kind: spans.append((position, position+len(text), kind))
        position += len(text)
    def emit(value, path):
        name = '.'.join(path)
        if type(value) is dict:
            add('{')
            for index, key in enumerate(sorted(value)):
                if index: add(',')
                add(canonical_json_bytes(key).decode('utf-8'), 'key:'+'.'.join((*path,key)))
                add(':'); emit(value[key], (*path,key))
            add('}')
        elif type(value) is list:
            add('[')
            for index, item in enumerate(value):
                if index: add(',')
                emit(item, (*path,'*'))
            add(']')
        else:
            field = next((p for p in reversed(path) if p!='*'), '')
            if field in ('scope_ref','disposition_scope_ref'): kind='scope_label'
            elif field=='value_text': kind='value_text'
            elif type(value) is str and field in ('evidence_refs','ref_id'): kind='source_reference'
            elif type(value) is str and field.endswith(('_ref','_refs')): kind='other_reference'
            else: kind='value'
            add(canonical_json_bytes(value).decode('utf-8'), kind+':'+name)
    emit(record, ())
    text = ''.join(pieces)
    if text.encode('utf-8') != canonical_json_bytes(record):
        raise CognitiveKernelContractError('precise field serialization differs')
    return text, spans


def precise_target_groups(tokenizer, example, maximum):
    text, spans = precise_scalar_spans(source.preparation.decision_for_example(example))
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    inputs, labels = source.preparation.target_ids(tokenizer, example, maximum)
    if list(encoded['input_ids']) != labels[:-1] or inputs[1:] != labels[:-1]:
        raise CognitiveKernelContractError('precise field target shift differs')
    groups=[]
    for left,right in encoded['offset_mapping']:
        size,kind=max(((max(0,min(right,end)-max(left,start)),kind) for start,end,kind in spans), default=(0,'structure'))
        groups.append(kind if size else 'structure')
    return labels, [*groups,'eos']


def aggregate_measurements(rows):
    answer, seen = {}, set()
    def merge(target, key, metric):
        n, loss, accuracy = metric['tokens'], metric['mean_loss'], metric['accuracy']
        if type(n) is not int or n<=0 or not math.isfinite(loss) or not math.isfinite(accuracy) or not 0<=accuracy<=1:
            raise CognitiveKernelContractError('family measurement is invalid')
        dst=target.setdefault(key, {'tokens':0,'loss_sum':0.,'correct':0.})
        dst['tokens']+=n;dst['loss_sum']+=loss*n;dst['correct']+=accuracy*n
    for row in rows:
        if row['case_id'] in seen or row['family'] not in ('base-train','longitudinal-train','development'):
            raise CognitiveKernelContractError('family measurement case repeated or unknown')
        seen.add(row['case_id'])
        if type(row['tokens']) is not int or row['tokens']<=0 or not math.isfinite(row['mean_loss']):
            raise CognitiveKernelContractError('family measurement is invalid')
        families=[row['family']] + (['all-train'] if row['family']!='development' else [])
        for family in families:
            dst=answer.setdefault(family, {'cases':0,'tokens':0,'case_loss_sum':0.,'weighted_loss_sum':0.,'case_losses':[],'groups':{},'strata':{}})
            dst['cases']+=1;dst['tokens']+=row['tokens'];dst['case_loss_sum']+=row['mean_loss']
            dst['weighted_loss_sum']+=row['tokens']*row['mean_loss']
            dst['case_losses'].append(row['mean_loss'])
            for key,metric in row['groups'].items(): merge(dst['groups'],key,metric)
            for key,strata in row['groups_by_positive_train_corpus_frequency'].items():
                for bucket,metric in strata.items(): merge(dst['strata'],key+'|'+bucket,metric)
    for dst in answer.values():
        dst['equal_case_mean_loss']=dst.pop('case_loss_sum')/dst['cases']
        dst['token_weighted_mean_loss']=dst.pop('weighted_loss_sum')/dst['tokens']
        losses=sorted(dst.pop('case_losses'))
        dst['per_case_loss_distribution']={'minimum':losses[0],'median':median(losses),
            'p90_nearest_rank':losses[math.ceil(.9*len(losses))-1],'maximum':losses[-1]}
        for key in ('groups','strata'):
            dst[key]={field:{'tokens':v['tokens'],'mean_loss':v['loss_sum']/v['tokens'],'accuracy':v['correct']/v['tokens']} for field,v in dst[key].items()}
    return answer


def raw_fault(row, labels, groups, counts, tokenizer, eos_id):
    tokens=row['generated_token_ids']; raw=row['raw_output_text']
    if tokenizer.decode(tokens,skip_special_tokens=False)!=raw:
        raise CognitiveKernelContractError('retained tokens do not decode to original raw bytes')
    observed=[*tokens]+([eos_id] if row['eos_observed'] else [])
    first=next((i for i,(a,b) in enumerate(zip(observed,labels)) if a!=b),None)
    if first is None and len(observed)!=len(labels): first=min(len(observed),len(labels))
    status, error, at='parsed',None,None
    try: json.loads(raw,object_pairs_hook=original._unique_json)
    except json.JSONDecodeError as exc: status,error,at='malformed',exc.msg,exc.pos
    except CognitiveKernelContractError as exc: status,error='duplicate_or_invalid_json',str(exc)
    checks={p for p in (32,64,128,256,len(tokens)-1,len(tokens)) if 0<=p<len(observed)}
    error_token=None
    if at is not None:
        for i in range(len(tokens)):
            prefix=tokenizer.decode(tokens[:i+1],skip_special_tokens=False)
            if not raw.startswith(prefix): continue
            if len(prefix)>at:
                error_token=i;checks.update(p for p in (i-1,i,i+1) if 0<=p<len(observed));break
    return {'json_status':status,'json_error':error,'json_error_char':at,
        'json_error_token_index':error_token,'later_reference_checkpoints':sorted(checks),
        'first_exact_teacher_divergence_token':first,
        'exact_divergence_field_path':groups[first] if first is not None and first<len(groups) else None,
        'exact_divergence_target_positive_train_corpus_count':counts[labels[first]] if first is not None and first<len(labels) else None,
        'semantic_mismatch_established':False,'eos_observed':row['eos_observed'],
        'generated_tokens':len(tokens),'retained_validation_status':row['validation_status'],
        'retained_validation_error':row['validation_error']}


def verify_loss_remeasurement(metric,progress):
    if metric['cases']!=progress['train_cases'] or metric['tokens']!=progress['target_tokens'] or any(
            not math.isclose(metric[k],progress[k],rel_tol=1e-5,abs_tol=1e-5)
            for k in ('equal_case_mean_loss','token_weighted_mean_loss')):
        raise CognitiveKernelContractError('remeasured full-train objective differs from sealed final progress')
    return {'matched':True,'absolute_tolerance':1e-5,'relative_tolerance':1e-5}


def parity_retained(model,payload,tokens,ended,checkpoints):
    import torch
    from cognitive_kernel.formation_incremental_decoder import CachedFormationDecoder
    observed=[*tokens]+([model.config.end_token_id] if ended else [])
    checks=sorted(set(p for p in checkpoints if 0<=p<len(observed)))
    if not checks: raise CognitiveKernelContractError('later parity needs a real retained position')
    prefix=[model.config.start_token_id];maximum=0.
    model.eval()
    with torch.inference_mode(),torch.autocast(payload['base_states'].device.type,enabled=False):
        cache=CachedFormationDecoder(model,base_states=payload['base_states'],source_mask=payload['source_mask'])
        for i,expected in enumerate(observed):
            logits=cache.step(prefix[-1])
            if not torch.isfinite(logits).all() or int(logits[0].argmax())!=expected:
                raise CognitiveKernelContractError('retained greedy replay differs from original')
            if i in checks:
                reference=model.next_token_logits(base_states=payload['base_states'],source_mask=payload['source_mask'],
                    input_ids=torch.tensor([prefix],device=payload['base_states'].device))
                difference=float((reference-logits).abs().max().cpu());maximum=max(maximum,difference)
                if not torch.allclose(reference,logits,atol=1e-4,rtol=1e-4) or int(reference[0].argmax())!=expected:
                    raise CognitiveKernelContractError('later cached/reference retained prefix differs')
            prefix.append(expected)
    return {'checked_positions':checks,'max_abs_difference':maximum,'argmaxes_equal':True,'atol':1e-4,'rtol':1e-4}


def validation_stage(row,example,weights,run_id):
    stage='no_eos'
    error=None
    if row['eos_observed']:
        try:
            stage='json'
            decision=json.loads(row['raw_output_text'],object_pairs_hook=original._unique_json)
            stage='decision_codec'
            bundle=original.compile_decision(example.context,decision,artifact_sha256=weights,inference_run_id=run_id)
            stage='source_grounding'
            original.validate_formation_grounding_v16(example.context,bundle,example.opened_sources)
            stage=None
        except (ValueError,TypeError,KeyError,CognitiveKernelContractError) as exc:
            error=f'{type(exc).__name__}: {exc}'
    if (stage is None)!=(row['validation_status']=='grounded_teacher_proposal_only'):
        raise CognitiveKernelContractError('stage replay differs from original strict validity verdict')
    return {'first_failed_stage':stage,'stage_error':error,'claim_gate_executed':False}


def verify_child(args,prepared):
    import torch
    import cognitive_kernel.formation_incremental_decoder as incremental
    record=child.verify_parent(args,prepared)
    run=shared._read_sealed(args.decision_component_dir/'run.json',child.RUN_SCHEMA)
    component=shared._read_sealed(args.decision_component_dir/'component.json',child.ARTIFACT_SCHEMA)
    expected={'parent_component_sha256':record.component['record_sha256'],'parent_run_sha256':record.run['record_sha256'],
        'parent_checkpoint_sha256':record.checkpoint['record_sha256'],'parent_weights_sha256':record.checkpoint['specialist_sha256'],
        'parent_optimizer_sha256':record.checkpoint['optimizer_sha256'],'code_binding':child.code_binding(),
        'source_view_sha256':prepared.view['record_sha256'],'corpus_sha256':args.input_sha256,
        'owner_authorization_ref':args.owner_authorization_ref,'specialist_config':prepared.config.record(),
        'objective':child.parent.OBJECTIVE,'initialization':child.parent.INITIALIZATION,
        'specialist_dtype':'float32','device':str(prepared.device),'torch_version':torch.__version__}
    if component['record_sha256']!=require_sha256(args.decision_component_sha256,'child_component_sha256') or component.get('run_sha256')!=run['record_sha256'] or component.get('source_view_sha256')!=prepared.view['record_sha256'] or component.get('specialist_config')!=prepared.config.record() or component.get('qualified_for_product') is not False or any(run.get(k)!=v for k,v in expected.items()):
        raise CognitiveKernelContractError('failure analysis child lineage differs')
    result=component['result'];progress=child.read_progress(args.decision_component_dir,run['record_sha256'],result['global_optimizer_steps'])
    if [r['record_sha256'] for r in progress]!=result['full_train_progress_sha256']:
        raise CognitiveKernelContractError('failure analysis child progress roster differs')
    budget.verify_terminal(run,result,len(prepared.train),[r['equal_case_mean_loss'] for r in progress])
    weights=args.decision_component_dir/'formation-specialist.safetensors'
    if shared._digest(weights)!=result['trained_weights_sha256']:
        raise CognitiveKernelContractError('failure analysis child weights differ')
    diag=shared._read_sealed(args.budget_diagnostic_dir/'diagnostic.json',budget.SCHEMA)
    if diag['record_sha256']!=require_sha256(args.budget_diagnostic_sha256,'budget_diagnostic_sha256') or diag.get('component_sha256')!=component['record_sha256'] or diag.get('run_sha256')!=run['record_sha256'] or diag.get('source_view_sha256')!=prepared.view['record_sha256'] or diag.get('runner_sha256')!=shared._digest(Path(budget.__file__)) or diag.get('original_decode_implementation_sha256')!=shared._digest(Path(original.__file__)) or diag.get('incremental_decoder_sha256')!=shared._digest(Path(incremental.__file__)):
        raise CognitiveKernelContractError('failure analysis original diagnostic lineage differs')
    if diag['decode']!={'device':str(prepared.device),'matched_between_controls':True,'max_new_tokens':2048,
            'method':'FP32_incremental_first32_actual_greedy_reference_checked','specialist_dtype':'float32',
            'target_visible':False,'torch_version':torch.__version__} or prepared.view['longest_target_tokens']>2048:
        raise CognitiveKernelContractError('failure analysis diagnostic decoding settings differ')
    for control,digest in diag['outputs_sha256'].items():
        if shared._digest(args.budget_diagnostic_dir/(control+'.jsonl'))!=digest:
            raise CognitiveKernelContractError('failure analysis original raw output hash differs')
    retained=[json.loads(line,object_pairs_hook=original._unique_json) for line in (args.budget_diagnostic_dir/'trained.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(retained)!=len(prepared.development) or any(row.get('case_id')!=e.case_id or row.get('split')!='development' or row.get('control')!='trained' or row.get('model_input_sha256')!=source.features.case_key(e) or row.get('weights_sha256')!=result['trained_weights_sha256'] for row,e in zip(retained,prepared.development)):
        raise CognitiveKernelContractError('failure analysis original development roster differs')
    return run,component,diag,retained,weights,progress[-1]


def run(args):
    started=perf_counter()
    prepared=source.load(args)  # Original owner/network/corpus admission precedes all new reads.
    selected=verify_roster(prepared)
    manifest,component,diag,retained,weights,final_progress=verify_child(args,prepared)
    import torch
    from safetensors.torch import load_file
    model=child.parent.FormationDecisionSpecialist(prepared.config).to(prepared.device,dtype=torch.float32)
    model.load_state_dict(load_file(str(weights)))
    args.output_dir.mkdir(mode=0o700,parents=True,exist_ok=False)
    counts,targets=Counter(),{}
    for e in (*prepared.train,*prepared.development):
        labels,groups=precise_target_groups(prepared.tokenizer,e,prepared.config.max_target_tokens)
        targets[e.case_id]=(labels,groups)
        if e.split=='train': counts.update(labels)
    measured,exposures=[],{}
    for e in (*prepared.train,*prepared.development):
        payload=source.payload(prepared,e,device=prepared.device)
        labels,groups=targets[e.case_id]
        metric=language.measure_with_exposure(model,payload,groups,counts)
        measured.append({'case_id':e.case_id,'family':family_name(e),**metric})
        root,export=prepared.locators[e.case_id]
        _,tensors=source.features.read_case(root,e,export,width=prepared.config.base_hidden_size)
        ids=set(tensors['input_ids'][tensors['attention_mask'].bool()].tolist())
        language.merge_counts(exposures.setdefault(family_name(e),{}),language.exposure_counts(labels,groups,counts,ids))
        print(json.dumps({'state':'CASE_MEASURED','case_id':e.case_id,'family':family_name(e),'teacher_prefix_loss':metric['mean_loss']}),flush=True)
    family_metrics=aggregate_measurements(measured)
    remeasurement=verify_loss_remeasurement(family_metrics['all-train'],final_progress)
    # Private per-case measurements preserve all failures without expanding the public projection.
    per_case=args.output_dir/'teacher-measurements.jsonl'
    with per_case.open('x',encoding='utf-8') as stream:
        for row in measured: stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    dev_faults=[]
    for e,row in zip(prepared.development,retained):
        labels,groups=targets[e.case_id]
        fault=raw_fault(row,labels,groups,counts,prepared.tokenizer,prepared.config.end_token_id)
        payload=source.payload(prepared,e,device=prepared.device)
        parity=parity_retained(model,payload,row['generated_token_ids'],row['eos_observed'],fault['later_reference_checkpoints'])
        dev_faults.append({'case_id':e.case_id,**fault,'later_incremental_parity':parity,
            'strict_validation_stage':validation_stage(row,e,component['result']['trained_weights_sha256'],row['inference_run_id'])})
    train_faults=[]
    output=args.output_dir/'train-generated.jsonl'
    with output.open('x',encoding='utf-8') as stream:
        for e in selected:
            payload=source.payload(prepared,e,device=prepared.device)
            tokens,ended,parity=original.greedy_fp32(model,payload['base_states'],payload['source_mask'],2048)
            run_id='mfm-budget-failure-'+manifest['record_sha256'][:12]+'-'+e.case_id
            row={'case_id':e.case_id,'split':'train','control':'trained','weights_sha256':component['result']['trained_weights_sha256'],
                'model_input_sha256':source.features.case_key(e),'inference_run_id':run_id,'generated_token_ids':tokens,'incremental_parity':parity,
                **original.validate(prepared.tokenizer.decode(tokens,skip_special_tokens=False),ended,e,component['result']['trained_weights_sha256'],run_id)}
            stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n');stream.flush()
            labels,groups=targets[e.case_id]
            fault=raw_fault(row,labels,groups,counts,prepared.tokenizer,prepared.config.end_token_id)
            later=parity_retained(model,payload,tokens,ended,fault['later_reference_checkpoints'])
            train_faults.append({'case_id':e.case_id,'family':family_name(e),**fault,'later_incremental_parity':later,
                'strict_validation_stage':validation_stage(row,e,component['result']['trained_weights_sha256'],run_id)})
            print(json.dumps({'state':'TRAIN_GENERATED','case_id':e.case_id,'family':family_name(e),
                'json_status':fault['json_status'],'validation_status':row['validation_status'],'eos_observed':ended}),flush=True)
    if any(p.grad is not None for p in model.parameters()): raise CognitiveKernelContractError('read-only diagnosis acquired gradients')
    return shared._write_new(args.output_dir/'analysis.json',{'schema':SCHEMA,
        'component_sha256':component['record_sha256'],'run_sha256':manifest['record_sha256'],'diagnostic_sha256':diag['record_sha256'],
        'source_view_sha256':prepared.view['record_sha256'],'weights_sha256':component['result']['trained_weights_sha256'],
        'analyzer_sha256':shared._digest(Path(__file__)),'parent_measure_sha256':shared._digest(Path(language.parent_analysis.__file__)),
        'language_analysis_sha256':shared._digest(Path(language.__file__)),'decoder_sha256':diag['incremental_decoder_sha256'],
        'raw_development_outputs_sha256':diag['outputs_sha256'],'private_measurements_sha256':shared._digest(per_case),
        'raw_train_outputs_sha256':shared._digest(output),'completed_global_epochs':component['result']['global_completed_epochs'],
        'teacher_prefix_family_metrics':family_metrics,'exposure_by_family_and_precise_field':exposures,
        'full_train_objective_remeasurement':{'final_progress_sha256':final_progress['record_sha256'],**remeasurement},
        'train_generation_selection':'all16base plus first complete13-operation long-atlas-east scenario; fixed before output inspection',
        'train_generation_faults':train_faults,'retained_development_faults':dev_faults,
        'actual_runtime':{'torch_version':torch.__version__,'device':str(prepared.device),'dtype':'float32'},
        'frequency_basis':'positive labels in one complete admitted train corpus pass, not total gradients',
        'limits':['teacher-prefix accuracy is not free generation','exact teacher divergence is not semantic failure',
            'scope_label is author-chosen semantics, not existing source identity','token membership is not contiguous copying or entailment',
            'mixed punctuation/scalar BPE tokens use maximal-overlap grouping; structure tokens do not measure grammar independently',
            '13-operation scenario is one synthetic history, not independent gold','no optimizer, new data, grammar constraints or output repair'],
        'seconds':perf_counter()-started,'qualified_for_product':False,'independent_gold':False,
        'final_payloads_opened':False,'memory_gate_executed':False,'optimization_executed':False})


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('teacher-training-manifest','preflight-receipt','feature-bank','component-dir','source-view','parent-decision-dir','decision-component-dir','budget-diagnostic-dir','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('input-sha256','owner-authorization-ref','feature-bank-sha256','component-sha256','source-view-sha256','parent-decision-sha256','parent-checkpoint-sha256','decision-component-sha256','budget-diagnostic-sha256'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cpu')
    p.set_defaults(max_new_tokens=2048)  # Required by original source admission; identical fixed diagnostic budget.
    return p.parse_args()


def main():
    os.umask(0o077)
    os.environ.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_DISABLED':'true'})
    result=run(arguments())
    print(json.dumps({'budget_failure_analysis_sha256':result['record_sha256'],'seconds':result['seconds'],
        'qualified_for_product':False,'optimization_executed':False}),flush=True)


if __name__=='__main__': main()
