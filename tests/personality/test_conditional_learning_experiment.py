"""Native matched controls and durable experimental resume; no private learning claim."""
from copy import deepcopy
from hashlib import sha256
import importlib
import json

import pytest
import torch

from tests.personality import test_paired_learning_experiment as fixtures
from tests.personality.test_paired_learning_recovery import exact


@pytest.fixture
def native():
    old=torch.get_num_threads(); torch.set_num_threads(1)
    fixture=fixtures.PairedLearningExperimentTests(); fixture.setUp()
    yield fixture
    torch.set_num_threads(old)


def setup(native):
    api=importlib.import_module('scripts.eipm.gemma_n0.conditional_learning_experiment')
    plan=api.make_plan(native.model,native.objectives,native.recipe,
        batch_sha256=native.digests,matched_groups=native.groups,correction_sha256={},
        bindings={'source_class':'public_mechanical_fixture'})
    return api,plan


def run(native,directory,**options):
    api,plan=setup(native)
    return api.run_experiment(native.model,native.objectives,native.recipe,plan=plan,
        expected_plan_sha256=api.plan_sha256(plan),batch_loader=native.batches.__getitem__,
        validate_batch=native.validate,correction_loader=lambda digest:None,
        custody_guard=native.guard,progress_directory=directory,**options)


def test_identical_group_control_preserves_all_targets_and_frozen_stages(native,tmp_path):
    before=deepcopy(native.model.state_dict())
    result=run(native,tmp_path/'progress')
    exact(before,native.model.state_dict())
    a,b=[result['arms'][name] for name in ('control','intervention')]
    exact(a['model_state'],b['model_state']); exact(a['optimizer_state'],b['optimizer_state'])
    assert result['total_committed_updates']==4
    for arm in (a,b):
        assert arm['endpoint_forwards']==4
        assert len(arm['events'])==2
        for key in before:
            if key.startswith(('n1.','n3.')):assert torch.equal(before[key],arm['model_state'][key])
        for event in arm['events']:
            for endpoint in event['endpoints']:
                assert set(endpoint['views'])=={'natural','replay'}
                assert endpoint['views']['natural']==endpoint['views']['replay']
                assert set(endpoint['objectives'])==set(native.batches[endpoint['batch_sha256']].targets)
    assert result['native_checkpoint'] is False


def test_group_boundary_resume_preserves_model_optimizer_rng_and_views(native,tmp_path):
    expected=run(native,tmp_path/'whole')
    class Stop(Exception):pass
    def stop(event):
        if event['arm']=='intervention' and event['update']==1:raise Stop()
    with pytest.raises(Stop):run(native,tmp_path/'partial',on_event=stop)
    snapshot=tmp_path/'partial'/'group-0003'
    digest=sha256((snapshot/'manifest.json').read_bytes()).hexdigest()
    resumed=run(native,tmp_path/'resumed',resume_from=snapshot,expected_resume_manifest_sha256=digest)
    exact(expected['arms'],resumed['arms'])
    assert resumed['new_updates_this_invocation']==1


def test_changed_correction_plan_or_loader_refused_before_updates(native,tmp_path):
    api,plan=setup(native)
    plan['correction_sha256']={native.digests[0]:'a'*64}
    with pytest.raises(Exception,match='correction'):
        api.run_experiment(native.model,native.objectives,native.recipe,plan=plan,
            expected_plan_sha256=api.plan_sha256(plan),batch_loader=native.batches.__getitem__,
            validate_batch=native.validate,correction_loader=lambda digest:None,
            custody_guard=native.guard,progress_directory=tmp_path/'refused')
    assert not (tmp_path/'refused').exists()
