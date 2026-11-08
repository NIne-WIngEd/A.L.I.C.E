"""Recover original JSON/torch metadata semantics using actual native writers."""
import copy
import importlib
import json
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from scripts.eipm.gemma_n0 import conditional_residual_experiment as experiment
from scripts.eipm.gemma_n0.conditional_judgment_residual import migrate
from tests.personality.test_conditional_residual_experiment import run


def reader():return importlib.import_module('scripts.eipm.gemma_n0.magnolia_residual_saved_attribution_r2_20261108')


def test_actual_retained_scope_through_plan_snapshot_and_final_writers(tmp_path,monkeypatch):
    from tests.personality.test_paired_learning_experiment import PairedLearningExperimentTests
    from tests.personality.test_cumulative_bindings import ParentScopeBindingTests
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    ParentScopeBindingTests.setUpClass()
    try:
        scope_fixture=ParentScopeBindingTests();scope_fixture.setUp();scope=scope_fixture.bind()
        assert isinstance(scope['retained_parent_batches'],tuple)
        native=PairedLearningExperimentTests();native.setUp()
        native.model,native.migration=migrate(native.model,seed=713)
        make=experiment.make_plan
        def with_scope(*args,**kwargs):
            kwargs['bindings']={**kwargs['bindings'],'retained_parent_scope':scope}
            return make(*args,**kwargs)
        monkeypatch.setattr(experiment,'make_plan',with_scope)
        result=run(native,tmp_path/'progress',destination=tmp_path/'export')
        final=torch.load(tmp_path/'export/experimental.pt',weights_only=True)
        manifest=json.loads((tmp_path/'export/manifest.json').read_bytes())
        plan=json.loads(reader().canonical(result['plan']))
        digest=sha256((tmp_path/'export/experimental.pt').read_bytes()).hexdigest()
        assert final['plan']!=plan  # Actual production-shape failure, not mock data.
        reader().verify_final_metadata(manifest,final,plan,digest)
        names=[n for n,_ in native.model.named_parameters() if n.startswith('n2.')]
        for number in range(1,5):
            saved=torch.load(tmp_path/'progress'/f'group-{number:04d}'/'experimental-progress.pt',weights_only=True)
            reader().verify_saved_binding(saved['binding'],final,names)
        bad=copy.deepcopy(plan);bad['bindings']['retained_parent_scope']['retained_parent_batches'][0]='f'*64
        with pytest.raises(ValueError):reader().verify_final_metadata(manifest,final,bad,digest)
        bad=copy.deepcopy(manifest);bad['recipe']['learning_rate']*=2
        with pytest.raises(ValueError):reader().verify_final_metadata(bad,final,plan,digest)
        bad=copy.deepcopy(saved['binding']);bad['plan']['bindings']['retained_parent_scope']['retained_parent_batches'][0]='e'*64
        with pytest.raises(ValueError):reader().verify_saved_binding(bad,final,names)
        bad=copy.deepcopy(saved['binding']);bad['plan']['matched_groups'].reverse()
        with pytest.raises(ValueError):reader().verify_saved_binding(bad,final,names)
        with pytest.raises(ValueError):reader().verify_saved_binding(saved['binding'],final,list(reversed(names)))
    finally:
        ParentScopeBindingTests.doClassCleanups();torch.set_num_threads(old_threads)


def test_recovery_retains_one_owner_guard_and_original_failed_worker():
    import ast
    module=reader()
    class Refused(Exception):pass
    class Namespace:
        def _p2_guard(self):raise Refused()
    with patch.object(Path,'open',side_effect=AssertionError('private read before isolation')),patch.object(Path,'stat',side_effect=AssertionError('private stat before isolation')):
        with pytest.raises(Refused):module.run(Namespace())
    calls=[n.func.attr for n in ast.walk(ast.parse(Path(module.__file__).read_text()))
           if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert calls.count('_p2_guard')==1 and 'step' not in calls
    original=Path(module.__file__).with_name('magnolia_residual_saved_attribution_20261108.py')
    assert sha256(original.read_bytes()).hexdigest()=='34a3cfdc599ccc890b3109a0f3434bda0827f6eff4ecd0df2d695481b1f33d5b'
