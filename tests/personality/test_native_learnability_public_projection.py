from scripts.eipm.gemma_n0 import magnolia_native_learnability_20261124 as worker


def test_public_objectives_export_aggregates_only():
    rows=[{'field_loss':{'preferences':.8},'available_counts':{'preferences':1}},
          {'field_loss':{'preferences':.2},'available_counts':{'preferences':1}}]
    assert worker.aggregate_objectives(rows)=={'preferences':{'mean_loss':.5,'measured_endpoints':2,'available_facts':2}}


def test_opposing_but_zero_certainty_cannot_be_selected_as_learning_control():
    from dataclasses import replace
    import pytest
    from tests.personality.test_conditional_teaching import public_fixture
    _,_,left,_,_,_,_=public_fixture()
    right=replace(left,case_id='other',targets=tuple(replace(t,facts=tuple(replace(f,value=1-f.value,uncertainty=1.) for f in t.facts))
        if t.field=='preferences' else t for t in left.targets))
    with pytest.raises(ValueError,match='positive certainty'):
        worker.select_opposing_pair([(left,right)])
