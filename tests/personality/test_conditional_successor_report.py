"""Report mechanics only; these numbers are not evidence of learned capability."""
from scripts.eipm.gemma_n0.conditional_successor_report import reviewed_region_fit
import pytest


def packet(covalid,contra,app):
    return {'candidate_ids':[['a','b']], 'concept_ids':[['policy','mode']],
        'co_valid_probabilities':[covalid], 'contraindications':[contra],
        'conditional_concepts':{'concept_applicability':[app]}}


def targets():
    def fact(pointer,value,available=True):
        return {'pointers':[pointer],'value':value if available else None,
            'uncertainty':.2 if available else None,'available':available,
            'support_ids':[],'disconfirmation_ids':[]}
    return [
        {'field':'co_valid_probabilities','facts':[fact('a',0),fact('b',1)]},
        {'field':'contraindications','facts':[fact('a',1),fact('b',0)]},
        {'field':'concept_applicability','facts':[fact('policy',1),fact('mode',0)]}]


def test_stable_wrong_prediction_is_failure_not_invariance_success():
    result=reviewed_region_fit(packet([.9,.1],[.1,.9],[.9,.9]),targets())
    assert result['co_valid_probabilities']['errors']==2
    assert result['contraindications']['errors']==2
    assert result['concept_applicability']['errors']==1
    assert result['all_measured_regions_correct'] is False


def test_both_members_correct_measured_by_target_not_movement():
    result=reviewed_region_fit(packet([.1,.9],[.9,.1],[.9,.1]),targets())
    assert result['all_measured_regions_correct'] is True
    assert all(result[f]['comparisons']==2 for f in
        ('co_valid_probabilities','contraindications','concept_applicability'))


def test_unknown_propositions_are_excluded():
    facts=targets()
    facts[2]['facts'][0].update(value=None,uncertainty=None,available=False)
    result=reviewed_region_fit(packet([.1,.9],[.9,.1],[.1,.1]),facts)
    assert result['concept_applicability']['comparisons']==1
    assert result['all_measured_regions_correct'] is True


def test_no_known_fact_cannot_be_reported_as_correct():
    result=reviewed_region_fit(packet([.1,.9],[.9,.1],[.9,.1]),[])
    assert result['all_measured_regions_correct'] is False
    assert result['qualification'] is False


def test_fractional_gold_is_not_silently_turned_into_binary_gold():
    facts=targets()
    facts[0]['facts'][0]['value']=.2
    with pytest.raises(ValueError,match='binary regions'):
        reviewed_region_fit(packet([.1,.9],[.9,.1],[.9,.1]),facts)


def test_missing_pointer_cannot_be_counted_as_measured():
    facts=targets()
    facts[0]['facts'][0]['pointers']=['absent']
    with pytest.raises(ValueError,match='native pointer'):
        reviewed_region_fit(packet([.1,.9],[.9,.1],[.9,.1]),facts)


@pytest.mark.parametrize('ids', [[], ['ab'], [('a', 'a')], [('a', '')], [('a', 'b'), ('a', 'b')]])
def test_invalid_pointer_rows_are_still_refused(ids):
    value = packet([.1, .9], [.9, .1], [.9, .1])
    value['candidate_ids'] = ids
    with pytest.raises(ValueError, match='unique pointer IDs'):
        reviewed_region_fit(value, targets())
