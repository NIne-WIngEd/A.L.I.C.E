"""Matched architecture learning must not change teaching between arms."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import importlib
import pytest
import torch

from tests.personality import test_paired_learning_experiment as fixtures
from tests.personality.test_paired_learning_recovery import exact
from scripts.eipm.gemma_n0 import conditional_judgment_residual as architecture


@pytest.fixture
def native():
    threads=torch.get_num_threads();torch.set_num_threads(1)
    fixture=fixtures.PairedLearningExperimentTests();fixture.setUp()
    fixture.model,fixture.migration=architecture.migrate(fixture.model,seed=713)
    yield fixture
    torch.set_num_threads(threads)


def run(native,directory,**options):
    api=importlib.import_module('scripts.eipm.gemma_n0.conditional_residual_experiment')
    plan=api.make_plan(native.model,native.objectives,native.recipe,batch_sha256=native.digests,
        matched_groups=native.groups,correction_sha256={},bindings={'public_fixture':True,
        'architecture':architecture.ARCHITECTURE,'extension_sha256':sha256(open(architecture.__file__,'rb').read()).hexdigest(),
        'migration':native.migration})
    return api.run_experiment(native.model,native.objectives,native.recipe,plan=plan,
        expected_plan_sha256=api.plan_sha256(plan),batch_loader=native.batches.__getitem__,
        validate_batch=native.validate,correction_loader=lambda d:None,custody_guard=native.guard,
        progress_directory=directory,**options)


def test_full_native_objectives_and_frozen_stages_survive(native,tmp_path):
    before=deepcopy(native.model.state_dict())
    result=run(native,tmp_path/'full')
    exact(before,native.model.state_dict())
    assert result['architecture']==architecture.ARCHITECTURE
    assert result['same_corrected_replay_in_both_arms'] is True
    assert result['native_checkpoint'] is False
    for name,arm in result['arms'].items():
        assert bool(arm['model_state']['n2.conditioning.enabled'])==(name=='intervention')
        assert arm['n1_n3_unchanged'] and arm['updates']==2
        for key,value in before.items():
            if key.startswith(('n1.','n3.')):assert torch.equal(value,arm['model_state'][key])
        for event in arm['events']:
            for endpoint in event['endpoints']:
                assert set(endpoint['objectives'])==set(native.batches[endpoint['batch_sha256']].targets)
    assert result['arms']['control']['model_state']['n2.conditioning.gate'].item()==0
    assert result['arms']['intervention']['model_state']['n2.conditioning.gate'].item()!=0


def test_resume_retains_architecture_mode_optimizer_rng_and_all_updates(native,tmp_path):
    expected=run(native,tmp_path/'whole')
    class Stop(Exception):pass
    def stop(event):
        if event['arm']=='intervention' and event['update']==1:raise Stop()
    with pytest.raises(Stop):run(native,tmp_path/'partial',on_event=stop)
    snapshot=tmp_path/'partial/group-0003'
    digest=sha256((snapshot/'manifest.json').read_bytes()).hexdigest()
    resumed=run(native,tmp_path/'resumed',resume_from=snapshot,expected_resume_manifest_sha256=digest)
    exact(expected['arms'],resumed['arms'])
    assert resumed['new_updates_this_invocation']==1


def test_same_admitted_corrections_reach_both_arms_and_control_matches_original(tmp_path,native):
    from src.alice_personality.identity import IdentityModel
    from src.alice_personality.teaching import IdentityObjectives,TrainingRecipe
    from tests.personality.test_conditional_teaching import public_fixture
    from scripts.eipm.gemma_n0 import conditional_learning_experiment as old
    from scripts.eipm.gemma_n0 import conditional_teaching as teaching
    new=importlib.import_module('scripts.eipm.gemma_n0.conditional_residual_experiment')
    config,n1,n2,b1,_,binding,materialize=public_fixture()
    torch.manual_seed(61)
    original=IdentityModel(config);objectives=IdentityObjectives(config.learned_width)
    model,migration=architecture.migrate(original,seed=713)
    recipe=TrainingRecipe('n2',.003,1.,61);batches={};records={}
    for i in range(4):
        case=replace(n2,case_id=f'residual-public-{i}');batch=materialize(case);digest=batch.fingerprint();batches[digest]=batch
        records[digest]=teaching.build_correction_record(n1,case,b1,batch,frame_binding=binding,
            reviewed_n1_batches={b1.fingerprint():{'split':'TRAIN','case_ids':list(b1.case_ids),
            'source_family_ids':[list(row) for row in b1.source_family_ids]}})
    digests=list(batches);results=[]
    for api,parent,name in ((old,original,'prior'),(new,model,'extension')):
        plan=api.make_plan(parent,objectives,recipe,batch_sha256=digests,matched_groups=[digests[:2],digests[2:]],
            correction_sha256={d:teaching.record_sha256(r) for d,r in records.items()},bindings={
            'architecture':architecture.ARCHITECTURE,'extension_sha256':sha256(open(architecture.__file__,'rb').read()).hexdigest(),
            'migration':migration,'public_fixture':True})
        results.append(api.run_experiment(parent,objectives,recipe,plan=plan,expected_plan_sha256=api.plan_sha256(plan),
            batch_loader=batches.__getitem__,validate_batch=lambda b:b.fingerprint(),correction_loader=records.get,
            custody_guard=lambda:None,progress_directory=tmp_path/name))
    old_arm,new_arm=results[0]['arms']['intervention'],results[1]['arms']['control']
    for key,value in old_arm['model_state'].items():
        assert torch.equal(value,new_arm['model_state'][key]),key
    for old_event,new_event in zip(old_arm['events'],new_arm['events']):
        assert old_event['endpoints']==new_event['endpoints']
    assert results[1]['arms']['control']['events'][0]['endpoints']==results[1]['arms']['intervention']['events'][0]['endpoints']
