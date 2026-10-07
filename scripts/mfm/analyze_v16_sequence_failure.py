"""Read-only task/field and source-conditioning diagnosis of completed576945.

Exact author departures, strict generation validity and source interventions
are separate measurements. Foreign cases are sensitivity probes, not semantic
counterfactuals. No gradients, optimizer, output repair or FINAL access.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
from time import perf_counter

from cognitive_kernel.canonical import CognitiveKernelContractError, require_sha256
from cognitive_kernel.formation_sequence_specialist import FormationSequenceSpecialist, TASKS
from . import run_v16_sequence_diagnostic as diagnostic
from . import formation_sequence_failure_metrics as metrics

trainer, ancestry = diagnostic.trainer, diagnostic.ancestry
source, shared = diagnostic.source, diagnostic.shared
SCHEMA = 'mfm-v16-sequence-failure-analysis-v1'


def foreign_case_map(examples):
    """Target-independent cyclic same-family/operation map, fixed by case ID."""
    groups, seen = defaultdict(list), set()
    for e in examples:
        if e.case_id in seen:
            raise CognitiveKernelContractError('foreign source roster repeats a case')
        seen.add(e.case_id)
        family = ancestry.family_name(e)
        group = trainer.curriculum.construction_stratum(e.case_id) if family=='longitudinal-train' else family
        groups[group].append(e.case_id)
    result = {}
    for ids in groups.values():
        ids.sort()
        if len(ids)<2:
            raise CognitiveKernelContractError('foreign source group needs a distinct case')
        result.update({key:ids[(index+1)%len(ids)] for index,key in enumerate(ids)})
    return dict(sorted(result.items()))


def verify_remeasurement(losses, progress, curriculum):
    original = progress['complete_case_losses']
    if set(losses)!=set(original):
        raise CognitiveKernelContractError('sequence remeasurement case roster differs')
    maximum = 0.
    for key in sorted(losses):
        if len(losses[key])!=2:
            raise CognitiveKernelContractError('sequence remeasurement requires both tasks')
        for current, old in zip(losses[key],original[key]):
            if not math.isfinite(current) or not math.isclose(current,old,rel_tol=1e-5,abs_tol=1e-5):
                raise CognitiveKernelContractError('sequence per-case/task remeasurement differs')
            maximum = max(maximum,abs(current-old))
    objective = float(trainer.curriculum.balanced_objective(losses,curriculum['weights']))
    strata = trainer.task_strata(losses)
    if (not math.isclose(objective,progress['balanced_two_task_objective'],rel_tol=1e-5,abs_tol=1e-5)
            or set(strata)!=set(progress['strata']) or any(
                strata[key]['cases']!=progress['strata'][key]['cases'] or any(
                    not math.isclose(strata[key][field],progress['strata'][key][field],rel_tol=1e-5,abs_tol=1e-5)
                    for field in ('formation_mean_loss','source_mean_loss')) for key in strata)):
        raise CognitiveKernelContractError('sequence balanced objective/strata remeasurement differs')
    return {'matched':True,'maximum_case_task_loss_difference':maximum,'balanced_two_task_objective':objective,
            'strata':strata,'atol':1e-5,'rtol':1e-5,
            'tolerance_reason':'per-token FP32 projection loss versus original chunk-mean FP32 reduction'}


def fold_frequency_metrics(raw):
    """Fold two independent label-frequency bases from one decoder pass."""
    totals, task, combined = {}, {}, {}
    def add(dst,key,row):
        value = dst.setdefault(key,{'tokens':0,'loss_sum':0.,'correct':0.})
        value['tokens']+=row['tokens'];value['loss_sum']+=row['tokens']*row['mean_loss']
        value['correct']+=row['tokens']*row['accuracy']
    def finish(dst):
        return {k:{'tokens':v['tokens'],'mean_loss':v['loss_sum']/v['tokens'],
                   'accuracy':v['correct']/v['tokens']} for k,v in dst.items()}
    for key,row in raw['groups'].items():
        field, bucket, mixed = key.rsplit('|',2)
        add(totals,field,row);add(task.setdefault(field,{}),bucket,row)
        add(combined.setdefault(field,{}),mixed,row)
    return {**raw,'groups':finish(totals),
            'groups_by_positive_train_corpus_frequency':{k:finish(v) for k,v in task.items()},
            'groups_by_combined_positive_train_corpus_frequency':{k:finish(v) for k,v in combined.items()}}


def verify_retained(args, prepared, candidate):
    """Bind all four original raw files to external stdout pin and producer."""
    import torch
    root = args.sequence_diagnostic_dir
    if any(p.is_symlink() for p in (root,*root.parents)):
        raise CognitiveKernelContractError('sequence raw directory must have direct custody')
    record = shared._read_sealed(root/'diagnostic.json',diagnostic.SCHEMA)
    budgets = diagnostic.decode_budgets(prepared,candidate.curriculum)
    rosters = diagnostic.select_rosters(prepared,candidate.curriculum)
    expected = {'component_sha256':candidate.component['record_sha256'],
        'run_sha256':candidate.run['record_sha256'],'source_view_sha256':prepared.view['record_sha256'],
        'curriculum_sha256':candidate.curriculum['record_sha256'],'stress_probe_sha256':candidate.probe['record_sha256'],
        'stress_probe_run_sha256':candidate.probe_run['record_sha256'],
        'completed_checkpoint_sha256':candidate.checkpoint['record_sha256'],
        'complete_train_progress_sha256':[p['record_sha256'] for p in candidate.progress],
        'seed_control_sha256':candidate.seed['record_sha256'],
        'historical_parent_component_sha256':candidate.parent_component['record_sha256'],
        'historical_parent_diagnostic_sha256':candidate.parent_diagnostic['record_sha256'],
        'code_binding':diagnostic.code_binding(),
        'case_rosters':{task:[e.case_id for e in rows] for task,rows in rosters.items()},
        'decode':{'method':'FP32_incremental_first32_actual_greedy_reference_checked',
            'explicit_task_bound_view':True,'max_new_tokens_by_task':budgets,'device':str(prepared.device),
            'specialist_dtype':'float32','torch_version':torch.__version__,'target_visible':False,
            'matched_between_controls':True,'target_comparison_after_generation':True},
        'historical_outputs_rewritten':False,'optimizer_created':False,'independent_gold':False,
        'final_payloads_opened':False,'memory_gate_executed':False,'authority_written':False,
        'qualified_for_product':False,'downstream_usefulness_established':False}
    if (record['record_sha256']!=require_sha256(args.sequence_diagnostic_sha256,'sequence_diagnostic_sha256')
            or any(record.get(k)!=v for k,v in expected.items())
            or record.get('seed_parameters_check',{}).get('actual_same_seed_parameters_equal') is not True):
        raise CognitiveKernelContractError('sequence original diagnostic lineage/settings differ')
    logical = {control+'-'+task for control in diagnostic.CONTROLS for task in TASKS}
    if set(record['outputs_sha256'])!=logical:
        raise CognitiveKernelContractError('sequence original raw controls/tasks differ')
    retained, summary = {}, {}
    for control in diagnostic.CONTROLS:
        summary[control] = {}
        weight = (candidate.component['result']['trained_weights_sha256'] if control=='trained'
                  else candidate.seed['specialist_sha256'])
        for task in TASKS:
            key = control+'-'+task;path = root/(key+'.jsonl')
            if path.is_symlink() or shared._digest(path)!=record['outputs_sha256'][key]:
                raise CognitiveKernelContractError('sequence original raw output hash differs')
            rows = [json.loads(line,object_pairs_hook=diagnostic.original._unique_json)
                    for line in path.read_text(encoding='utf-8').splitlines()]
            if len(rows)!=len(rosters[task]):
                raise CognitiveKernelContractError('sequence original raw roster size differs')
            for row,e in zip(rows,rosters[task]):
                fields = {'case_id':e.case_id,'split':e.split,'family':ancestry.family_name(e),
                    'task':task,'control':control,'weights_sha256':weight,'source_view_sha256':prepared.view['record_sha256'],
                    'context_sha256':e.context.content_digest(),'model_input_sha256':source.features.case_key(e),
                    'inference_run_id':candidate.run['record_sha256']+'-'+key,'max_new_tokens':budgets[task]}
                if any(row.get(k)!=v for k,v in fields.items()):
                    raise CognitiveKernelContractError('sequence original raw row lineage differs')
                diagnostic.add_measurement(summary[control],row)
            retained[key] = rows
    if summary!=record['summary']:
        raise CognitiveKernelContractError('sequence original summary differs from raw rows')
    return record, retained, rosters


def source_validation_stage(row, example, fault):
    replay = diagnostic.validate_source(row['raw_output_text'],row['eos_observed'],example)
    if any(row.get(k)!=v for k,v in replay.items()):
        raise CognitiveKernelContractError('sequence source validation replay differs')
    stage = (None if replay['exact_source_json_agreement'] else 'no_eos' if not row['eos_observed']
             else 'json' if fault['json_status']!='parsed' else 'exact_source_content')
    return {'first_failed_stage':stage,'semantic_memory_test':False}


def run(args):
    started = perf_counter()
    prepared = source.load(args)  # Privacy/owner/corpus admission before new reads.
    candidate = diagnostic.verify_candidate(args,prepared)
    record, retained, rosters = verify_retained(args,prepared,candidate)
    import torch
    from safetensors.torch import load_file
    model = FormationSequenceSpecialist(prepared.config).to(prepared.device,dtype=torch.float32).eval()
    model.load_state_dict(load_file(str(candidate.weights)))
    examples = (*prepared.train,*prepared.development)
    by_id = {e.case_id:e for e in examples}
    foreign_map = foreign_case_map(examples)
    counts = {task:Counter() for task in TASKS};targets = {}
    for e in examples:
        for task in TASKS:
            labels,groups = metrics.task_targets(prepared.tokenizer,e,prepared.config.max_target_tokens,task)
            targets[(e.case_id,task)] = labels,groups
            if e.split=='train': counts[task].update(labels)
    combined = sum(counts.values(),Counter())
    args.output_dir.mkdir(parents=True,mode=0o700,exist_ok=False)
    measurements = {task:[] for task in TASKS};losses = {};token_totals = Counter()
    bucket = lambda n:'0' if n==0 else '1-9' if n<10 else '10+'
    with (args.output_dir/'teacher-measurements.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
        for e in examples:
            pair = []
            for task,payload in zip(TASKS,trainer.payloads(prepared,e)):
                labels,groups = targets[(e.case_id,task)]
                if payload['labels'].flatten().tolist()!=labels:
                    raise CognitiveKernelContractError('sequence diagnostic target tensors differ')
                joint = [field+'|'+bucket(counts[task][token])+'|'+bucket(combined[token])
                         for field,token in zip(groups,labels)]
                measured = fold_frequency_metrics(ancestry.language.parent_analysis.measure(model.bind_task(task),payload,joint))
                row = {'case_id':e.case_id,'family':ancestry.family_name(e),'task':task,**measured}
                measurements[task].append(row);pair.append(row['mean_loss'])
                if e.split=='train': token_totals[task]+=row['tokens']
                stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n');stream.flush()
            if e.split=='train': losses[e.case_id] = pair
            print(json.dumps({'state':'SEQUENCE_CASE_MEASURED','case_id':e.case_id,'tasks':list(TASKS)}),flush=True)
    remeasurement = verify_remeasurement(losses,candidate.progress[-1],candidate.curriculum)
    if token_totals['source_reconstruction']!=candidate.curriculum['lexical_target_tokens']:
        raise CognitiveKernelContractError('sequence source training token total differs')
    family = {task:ancestry.aggregate_measurements(rows) for task,rows in measurements.items()}
    combined_family = {task:ancestry.aggregate_measurements([{**row,'groups_by_positive_train_corpus_frequency':
        row['groups_by_combined_positive_train_corpus_frequency']} for row in rows]) for task,rows in measurements.items()}
    faults = []
    with (args.output_dir/'retained-failure-measurements.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
        for task in TASKS:
            for e,row in zip(rosters[task],retained['trained-'+task]):
                labels,groups = targets[(e.case_id,task)]
                payload = trainer.payloads(prepared,e)[TASKS.index(task)]
                fault = ancestry.raw_fault(row,labels,groups,counts[task],prepared.tokenizer,prepared.config.end_token_id)
                parity = ancestry.parity_retained(model.bind_task(task),payload,row['generated_token_ids'],
                    row['eos_observed'],fault['later_reference_checkpoints'])
                if task=='formation':
                    validation = ancestry.validation_stage(row,e,candidate.component['result']['trained_weights_sha256'],row['inference_run_id'])
                else:
                    validation = source_validation_stage(row,e,fault)
                position = fault['first_exact_teacher_divergence_token'];ranks = None
                foreign = foreign_map[e.case_id]
                foreign_payload = trainer.payloads(prepared,by_id[foreign])[TASKS.index(task)]
                if position is not None and position<len(labels):
                    observed = [*row['generated_token_ids']]+([prepared.config.end_token_id] if row['eos_observed'] else [])
                    alternative = observed[position] if position<len(observed) else None
                    ranks = {name:metrics.rank_measure(model.bind_task(task),changed,position,alternative)
                        for name,changed in (('original',payload),
                            ('zero',metrics.source_intervention_payload(payload)),
                            ('foreign_case',metrics.source_intervention_payload(payload,foreign_payload,'foreign_case')))}
                    if alternative is not None and ranks['original']['argmax_token_id']!=alternative:
                        raise CognitiveKernelContractError('first divergence correct-prefix argmax differs from retained greedy')
                result = {'case_id':e.case_id,'task':task,'family':ancestry.family_name(e),**fault,
                    'later_task_bound_incremental_parity':parity,'strict_validation_stage':validation,
                    'first_exact_divergence_teacher_prefix_source_interventions':ranks,
                    'foreign_case_id':foreign,'original_source_tokens':int(payload['source_mask'].bool().sum()),
                    'foreign_source_tokens':int(foreign_payload['source_mask'].bool().sum()),
                    'intervention_meaning':'fixed authored prefix; zero/foreign source sensitivity, not matched semantic counterfactual'}
                faults.append(result);stream.write(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n');stream.flush()
                print(json.dumps({'state':'SEQUENCE_RETAINED_CASE_CHECKED','task':task,'case_id':e.case_id}),flush=True)
    if any(p.grad is not None for p in model.parameters()):
        raise CognitiveKernelContractError('read-only sequence diagnosis acquired gradients')
    return shared._write_new(args.output_dir/'analysis.json',{
        'schema':SCHEMA,'component_sha256':candidate.component['record_sha256'],'run_sha256':candidate.run['record_sha256'],
        'diagnostic_sha256':record['record_sha256'],'source_view_sha256':prepared.view['record_sha256'],
        'analyzer_sha256':shared._digest(Path(__file__)),'metrics_sha256':shared._digest(Path(metrics.__file__)),
        'measurement_implementation_sha256':shared._digest(Path(ancestry.language.parent_analysis.__file__)),
        'original_diagnostic_code_binding':record['code_binding'],'full_train_remeasurement':remeasurement,
        'teacher_prefix_family_metrics':family,'combined_frequency_teacher_prefix_family_metrics':combined_family,
        'frequency_basis':'one full admitted train corpus pass per task and combined; includes EOS; dev excluded',
        'positive_train_token_types':{task:len(value) for task,value in counts.items()},
        'positive_combined_train_token_types':len(combined),'train_target_tokens':dict(token_totals),
        'foreign_map':foreign_map,'foreign_map_rule':'sorted cyclic same family; longitudinal same operation next host',
        'retained_faults':faults,'retained_raw_outputs_sha256':record['outputs_sha256'],
        'outputs_sha256':{p.name:shared._digest(p) for p in args.output_dir.glob('*.jsonl')},
        'actual_runtime':{'torch_version':torch.__version__,'device':str(prepared.device),'dtype':'float32'},
        'scope':'teacher-prefix mechanism diagnosis; exact author disagreement is not semantic failure; original strict validity unchanged',
        'optimizer_created':False,'no_optimization_or_weight_export':True,'independent_gold':False,
        'qualified_for_product':False,'final_payloads_opened':False,'seconds':perf_counter()-started})


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('teacher-training-manifest','preflight-receipt','feature-bank','component-dir','source-view',
        'parent-decision-dir','decision-component-dir','budget-diagnostic-dir','sequence-dir',
        'sequence-probe-dir','sequence-diagnostic-dir','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('input-sha256','owner-authorization-ref','feature-bank-sha256','component-sha256',
        'source-view-sha256','parent-decision-sha256','parent-checkpoint-sha256','decision-component-sha256',
        'budget-diagnostic-sha256','sequence-sha256','sequence-diagnostic-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cpu');parser.set_defaults(max_new_tokens=2048)
    return parser.parse_args(argv)


def main():
    os.umask(0o077)
    os.environ.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','WANDB_DISABLED':'true'})
    result = run(arguments())
    print(json.dumps({'sequence_failure_analysis_sha256':result['record_sha256'],
                     'seconds':result['seconds'],'qualified_for_product':False}),flush=True)


if __name__=='__main__': main()
