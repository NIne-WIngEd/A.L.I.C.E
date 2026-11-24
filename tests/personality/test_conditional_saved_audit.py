"""Saved evidence must reveal offsetting errors without inventing causes."""
from copy import deepcopy
from types import SimpleNamespace as NS
import pytest

from scripts.eipm.gemma_n0 import conditional_saved_audit as api

def case(name, preferred=(.8,.2), pairs=()):
    fact=lambda pointers,value,available=True,uncertainty=0.:NS(pointers=pointers,value=value,available=available,uncertainty=uncertainty)
    targets=[NS(field='preferences',mode='closed_distribution',facts=[fact(('a',),preferred[0]),fact(('b',),preferred[1])])]
    if pairs:targets.append(NS(field='candidate_pair_probabilities',mode='independent',facts=[fact(*p) for p in pairs]))
    return NS(case_id=name,sha256=name,targets=targets)

def decision(values, ids=('a','b')):
    return {'candidate_ids':[list(ids)],'candidate_mask':[[True,True]],'preferences':[list(values)],'preference_logits':[list(values)]}

def test_offsetting_context_repairs_and_harms_survive_equal_aggregate():
    cases=[case('one'),case('two')]
    a={'one':decision((.9,.1)),'two':decision((.1,.9))}
    b={'one':decision((.1,.9)),'two':decision((.9,.1))}
    details,result=api.transitions(cases,a,b)
    assert result['before_correct']==result['after_correct']==1
    assert result['repairs']==result['harms']==1 and result['changed_top_sets']==2
    assert len(details)==2

def test_semantic_permutation_and_probability_ties():
    c=case('tie',(.5,.5))
    before=decision((.5,.5)); before['preference_logits']=[[12.,11.]]
    after=decision((.5,.5),('b','a')); after['preference_logits']=[[11.,12.]]
    _,result=api.transitions([c],{'tie':before},{'tie':after})
    assert result['before_correct']==result['after_correct']==1
    assert result['changed_top_sets']==0
    bad=deepcopy(after);bad['candidate_ids']=[['a','a']]
    with pytest.raises(ValueError):api.transitions([c],{'tie':before},{'tie':bad})

def test_target_order_and_soft_calibration_are_distinct():
    cases=[case('compatible',pairs=[(('a','b'),.8)]),case('soft_gap',pairs=[(('a','b'),.6)]),
           case('conflict',pairs=[(('a','b'),.2)]),case('unknown',pairs=[(('a','b'),None,False,None)]),
           case('uncertain',pairs=[(('a','b'),.2,True,1.)])]
    details,result=api.target_compatibility(cases)
    assert result['comparable_pair_facts']==3 and result['opposite_order_facts']==1
    assert result['soft_probability_gap']['max']==pytest.approx(.6)
    assert result['excluded_unavailable_or_fully_uncertain_pair_facts']==2
    assert result['automatic_target_changes']==0 and len(details)==3


def test_native_unknown_facts_stay_unknown():
    from src.alice_personality.teaching.cases import TargetFact
    unknown=TargetFact(('a','b'),None,None,False,(),())
    unknown.validate(())
    c=case('unknown',pairs=[(('a','b'),.2)])
    c.targets[1].facts=[unknown]
    details,result=api.target_compatibility([c])
    assert details==[] and result['excluded_unavailable_or_fully_uncertain_pair_facts']==1


def test_native_distribution_row_certainty_keeps_available_coordinates():
    import torch
    from src.alice_personality.teaching.contracts import ReviewedTarget
    from src.alice_personality.teaching.objectives import _distribution
    c=case('heterogeneous',pairs=[(('a','b'),.2)])
    c.targets[0].facts[0].uncertainty=1.
    target=ReviewedTarget(torch.tensor([[.8,.2]]),torch.tensor([[True,True]]),torch.tensor([[1.,0.]]))
    scores=torch.tensor([[0.,0.]],requires_grad=True)
    loss,_=_distribution(scores,target,torch.ones_like(scores,dtype=torch.bool))
    loss.backward()
    assert scores.grad[0,0]<0  # The uncertainty-1 coordinate still affects this row.
    details,result=api.target_compatibility([c])
    assert len(details)==1 and result['opposite_order_facts']==1
    assert result['heterogeneous_preference_uncertainty_rows']==1
    assert result['comparisons_with_uncertainty_one_preference_coordinate']==1
    c.targets[0].facts[1].uncertainty=1.
    assert api.target_compatibility([c])[0]==[]


def test_progress_reader_rewinds_actual_hashed_descriptor(tmp_path):
    import json
    from hashlib import sha256
    from src.alice_personality.n1.manifest_structure_audit import _PinnedFile
    from scripts.eipm.gemma_n0 import magnolia_conditional_saved_audit_20261028 as worker
    path=tmp_path/'public-progress.txt'
    records=[{'stage':'comparison_update_completed','completed_update_events':i} for i in range(1,81)]
    payload=b'public runtime preamble\n'+b''.join(json.dumps(r).encode()+b'\n' for r in records)
    path.write_bytes(payload)
    with _PinnedFile(path,sha256(payload).hexdigest(),len(payload)) as pinned:
        assert pinned.stream.tell()==len(payload)
        assert worker.progress_acknowledgments(pinned)==records

def records():
    events=[];final=[]
    for i,packet in enumerate(('voice','broader'),1):
        loss={'preferences':.8,'contraindications':.2}
        view={'loss':1.,'objectives':loss,'coverage':{'preferences':1,'contraindications':2}}
        ep={'batch_sha256':str(i),'loss':1.,'objectives':loss,'coverage':view['coverage'],'views':{'natural':deepcopy(view),'replay':deepcopy(view)}}
        events.append({'update':i,'gradient_norm_before_clip':2.,'group_size':1,'batch_sha256':[str(i)],'endpoints':[ep]})
        final.append({'batch_sha256':str(i),'packet':packet,'total_loss':1.3,'field_loss':{'preferences':1.1,'contraindications':.2},'available_counts':view['coverage']})
    return events,final

def test_exact_same_state_vs_temporal_comparison():
    events,final=records()
    details,result=api.trajectory(events,final,correction_batches={'2'},control=True)
    assert result['same_state_replay_minus_natural']['all']['total_loss']['max']==0
    assert result['pre_update_to_final']['voice']['total_loss']['mean']==pytest.approx(.3)
    assert result['packet_group_positions']['voice']==[1]
    assert result['gradient_conflict_established'] is False
    events[1]['endpoints'][0]['views']['replay']['loss']=1.1
    with pytest.raises(ValueError):api.trajectory(events,final,correction_batches={'2'},control=True)

def test_missing_duplicate_and_nonfinite_records_refused():
    events,final=records()
    for bad in (final[:1],final+[final[0]]):
        with pytest.raises(ValueError):api.trajectory(events,bad,correction_batches={'2'},control=True)
    events[0]['gradient_norm_before_clip']=float('nan')
    with pytest.raises(ValueError):api.trajectory(events,final,correction_batches={'2'},control=True)

def test_worker_enters_isolation_before_any_private_access_and_never_reenters():
    import ast
    from pathlib import Path
    from unittest.mock import patch
    from scripts.eipm.gemma_n0 import magnolia_conditional_saved_audit_20261028 as worker
    class Refused(Exception):pass
    class Namespace:
        def _p2_guard(self):raise Refused()
    with patch.object(Path,'stat',side_effect=AssertionError('before isolation')),patch.object(Path,'open',side_effect=AssertionError('before isolation')):
        with pytest.raises(Refused):worker.run(Namespace())
    tree=ast.parse(Path(worker.__file__).read_text())
    body=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='run')
    assert ast.unparse(body.body[0].value)=='namespace._p2_guard()'
    assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='_p2_guard' for n in ast.walk(tree))==1
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('backward','step','load_state_dict') for n in ast.walk(tree))

def test_actual_native_codec_can_feed_transition_analysis():
    from dataclasses import replace
    import torch
    from tests.personality.test_identity_codec import document,Tokenizer,Backbone
    from src.alice_personality.identity.codec import decision_to_mapping
    from src.alice_personality.identity.feature_producer import FrozenFrameProducer
    from src.alice_personality.identity.model import IdentityModel
    from src.alice_personality.identity.contracts import IdentityModelConfig
    torch.set_num_threads(1)
    doc=document();doc=replace(doc,candidates=tuple(replace(c,authority_allowed=True) for c in doc.candidates))
    config=IdentityModelConfig(8,3,learned_width=16,attention_heads=4)
    producer=FrozenFrameProducer(Tokenizer(),Backbone(),config,_fixture=True)
    with torch.no_grad():decoded=decision_to_mapping(IdentityModel(config)(producer.produce(doc).frame))
    ids=decoded['candidate_ids'][0]
    facts=[NS(pointers=(p,),value=1/len(ids),available=True,uncertainty=0.) for p in ids]
    native_case=NS(case_id='native',sha256='native',targets=[NS(field='preferences',mode='closed_distribution',facts=facts)])
    _,result=api.transitions([native_case],{'native':decoded},{'native':decoded})
    assert result['changed_top_sets']==result['repairs']==result['harms']==0
    producer.close_and_verify()
