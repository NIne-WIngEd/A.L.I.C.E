"""Describe saved conditional-learning evidence; never alter targets or models."""
from collections import Counter, defaultdict
import math


def require(condition, message='saved conditional evidence differs'):
    if not condition:
        raise ValueError(message)


def stats(values):
    values=list(values)
    require(all(type(v) in (int,float) and math.isfinite(v) for v in values), 'nonfinite saved metric')
    return {'count':len(values),'min':min(values) if values else None,
            'mean':math.fsum(values)/len(values) if values else None,'max':max(values) if values else None}


def _target(case, field):
    selected=[t for t in case.targets if t.field==field]
    require(len(selected)<=1, 'duplicate reviewed field')
    if not selected:
        return None
    facts={tuple(f.pointers):f for f in selected[0].facts}
    require(len(facts)==len(selected[0].facts), 'duplicate reviewed pointer')
    for fact in facts.values():
        require(type(fact.available) is bool, 'invalid reviewed availability')
        if not fact.available:
            require(fact.value is None and fact.uncertainty is None, 'unknown fact carries a value')
            continue
        require(type(fact.uncertainty) in (int,float) and math.isfinite(fact.uncertainty)
                and 0<=fact.uncertainty<=1, 'invalid reviewed uncertainty')
        require(type(fact.value) in (int,float) and math.isfinite(fact.value)
                and 0<=fact.value<=1, 'invalid reviewed probability')
    return selected[0],facts


def _preference(case, decision):
    target=_target(case,'preferences')
    require(target is not None and target[0].mode=='closed_distribution','closed preference targets required')
    desired={p[0]:f.value for p,f in target[1].items() if f.available}
    require(desired and all(len(p)==1 for p in target[1]),'preference pointer geometry differs')
    ids=decision['candidate_ids'][0]
    require(len(decision['candidate_ids'])==1 and len(set(ids))==len(ids),'unique single-frame candidates required')
    for key in ('preferences','preference_logits','candidate_mask'):
        require(len(decision[key])==1 and len(decision[key][0])==len(ids),'candidate coordinate shape differs')
    probabilities=dict(zip(ids,decision['preferences'][0])); logits=dict(zip(ids,decision['preference_logits'][0]))
    stats(probabilities.values());stats(logits.values())
    allowed={p for p,m in zip(ids,decision['candidate_mask'][0]) if m}
    require(set(desired)==allowed and all(type(m) is bool for m in decision['candidate_mask'][0]),'closed target scope differs')
    expected={p for p,v in desired.items() if v==max(desired.values())}
    actual={p for p in allowed if probabilities[p]==max(probabilities[k] for k in allowed)}
    outside=allowed-expected
    margin=min(logits[p] for p in expected)-max(logits[p] for p in outside) if outside else None
    return {'correct':actual==expected,'top':sorted(actual),'target_top':sorted(expected),
            'target_set_vs_outside_margin':margin}


def transitions(cases,before,after):
    require(len({c.case_id for c in cases})==len(cases) and set(before)==set(after)=={c.case_id for c in cases},'context inventory differs')
    details=[];counts=Counter()
    for case in cases:
        a,b=(_preference(case,rows[case.case_id]) for rows in (before,after))
        counts.update(contexts=1,before_correct=int(a['correct']),after_correct=int(b['correct']),
                      repairs=int(not a['correct'] and b['correct']),harms=int(a['correct'] and not b['correct']),
                      changed_top_sets=int(a['top']!=b['top']))
        details.append({'case_sha256':case.sha256,'before':a,'after':b})
    return details,{**dict(counts),'margin_change':stats(r['after']['target_set_vs_outside_margin']-r['before']['target_set_vs_outside_margin']
        for r in details if r['before']['target_set_vs_outside_margin'] is not None), 'independent_qualification':False}


def target_compatibility(cases):
    details=[];counts=Counter(comparable_pair_facts=0,opposite_order_facts=0,tie_order_mismatches=0,
                             excluded_unavailable_or_fully_uncertain_pair_facts=0,undefined_zero_mass_pairs=0,
                             heterogeneous_preference_uncertainty_rows=0,
                             comparisons_with_uncertainty_one_preference_coordinate=0)
    for case in cases:
        preference,pairs=_target(case,'preferences'),_target(case,'candidate_pair_probabilities')
        if preference is None or pairs is None:
            continue
        require(preference[0].mode=='closed_distribution','pair comparison needs closed preferences')
        require(all(len(p)==1 for p in preference[1]),'preference pointer geometry differs')
        available=[f for f in preference[1].values() if f.available]
        # Native CE includes every available value and weights the entire row.
        row_certainty=math.fsum(1-f.uncertainty for f in available)/len(available) if available else 0.
        counts['heterogeneous_preference_uncertainty_rows']+=len({f.uncertainty for f in available})>1
        for pointers,fact in pairs[1].items():
            require(len(pointers)==2 and pointers[0]!=pointers[1],'ordered pair pointers differ')
            a,b=(preference[1].get((p,)) for p in pointers)
            if (not fact.available or fact.uncertainty==1 or row_certainty==0
                    or not all(f is not None and f.available for f in (a,b))):
                counts['excluded_unavailable_or_fully_uncertain_pair_facts']+=1;continue
            if a.value+b.value==0:
                counts['undefined_zero_mass_pairs']+=1;continue
            implied=a.value/(a.value+b.value)
            sign=lambda x:(x>0)-(x<0)
            first,second=sign(a.value-b.value),sign(fact.value-.5)
            counts['comparable_pair_facts']+=1
            counts['comparisons_with_uncertainty_one_preference_coordinate']+=a.uncertainty==1 or b.uncertainty==1
            counts['opposite_order_facts']+=first*second<0
            counts['tie_order_mismatches']+=(first==0)!=(second==0)
            details.append({'case_sha256':case.sha256,'pointers':list(pointers),
                'preference_implied_pair_probability':implied,'reviewed_pair_probability':fact.value,
                'preference_distribution_row_certainty':row_certainty,'pair_fact_certainty':1-fact.uncertainty,
                'opposite_order':first*second<0,'absolute_probability_gap':abs(implied-fact.value)})
    return details,{**dict(counts),'soft_probability_gap':stats(r['absolute_probability_gap'] for r in details),
                    'comparison_semantics':'native_distribution_row_certainty_and_pair_fact_certainty',
                    'automatic_target_changes':0,'probability_gap_alone_is_not_label_error':True}


def trajectory(events, final_rows, *, correction_batches, control):
    final={r['batch_sha256']:r for r in final_rows}
    require(len(final)==len(final_rows),'duplicate final objective')
    require([e['update'] for e in events]==list(range(1,len(events)+1)),'group cursor differs')
    seen=set(); details=[];positions=defaultdict(list)
    same=defaultdict(lambda:defaultdict(list));temporal=defaultdict(lambda:defaultdict(list))
    for event in events:
        stats([event['gradient_norm_before_clip']])
        require(event['gradient_norm_before_clip']>=0,'negative gradient norm')
        require(event['group_size']==len(event['endpoints'])==len(event['batch_sha256']),'endpoint group shape differs')
        require([e['batch_sha256'] for e in event['endpoints']]==event['batch_sha256'],'endpoint order differs')
        for endpoint in event['endpoints']:
            digest=endpoint['batch_sha256']
            require(digest in final and digest not in seen,'missing or repeated endpoint')
            seen.add(digest);after=final[digest];packet=after['packet']
            natural,replay=(endpoint['views'][k] for k in ('natural','replay'))
            require(natural['coverage']==replay['coverage']==after['available_counts'],'target coverage differs')
            require(natural['objectives'].keys()==replay['objectives'].keys()==after['field_loss'].keys(),'objective families differ')
            if control or digest not in correction_batches:
                require(natural==replay,'unchanged replay is not exact')
            current={'total_loss':natural['loss'],**natural['objectives']}
            patched={'total_loss':replay['loss'],**replay['objectives']}
            last={'total_loss':after['total_loss'],**after['field_loss']}
            stats(current.values());stats(patched.values());stats(last.values())
            positions[packet].append(event['update'])
            delta={k:patched[k]-v for k,v in current.items()}
            end={k:last[k]-v for k,v in current.items()}
            for key,value in delta.items():
                same['all'][key].append(value)
                same['corrected' if digest in correction_batches else 'uncorrected'][key].append(value)
                same[packet][key].append(value)
            for key,value in end.items():temporal[packet][key].append(value)
            details.append({'batch_sha256':digest,'packet':packet,'group':event['update'],
                            'same_state_replay_minus_natural':delta,'pre_update_to_final':end})
    require(seen==set(final) and correction_batches<=seen,'complete curriculum required')
    summarize=lambda groups:{group:{field:stats(values) for field,values in fields.items()} for group,fields in groups.items()}
    return details,{'cases':len(seen),'groups':len(events),'same_state_replay_minus_natural':summarize(same),
        'pre_update_to_final':summarize(temporal),'packet_group_positions':{k:sorted(set(v)) for k,v in positions.items()},
        'aggregate_gradient_norm_before_clip':stats(e['gradient_norm_before_clip'] for e in events),
        'gradient_conflict_established':False,'scheduling_cause_established':False,'forgetting_cause_established':False}
