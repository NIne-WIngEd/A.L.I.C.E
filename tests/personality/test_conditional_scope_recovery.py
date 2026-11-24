"""Completed native experiment recovery is read-only with full original binding."""
import ast
from copy import deepcopy
from hashlib import sha256
import importlib
import json
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from tests.personality.test_conditional_scope_experiment import execute, native
from tests.personality.test_paired_learning_recovery import exact
from scripts.eipm.gemma_n0 import conditional_scope_experiment as api

def worker():
    return importlib.import_module('scripts.eipm.gemma_n0.magnolia_conditional_scope_recovery_20261121')

def restore(native, result, root, snapshot, **overrides):
    options=dict(plan=result['plan'], expected_plan_sha256=result['plan_sha256'],
        snapshot=snapshot, manifest_sha256=sha256((snapshot/'manifest.json').read_bytes()).hexdigest(),
        batch_loader=native.batches.__getitem__, validate_batch=native.validate,
        correction_loader=lambda _:None, custody_guard=native.guard,
        progress_directory=root/'recovered-progress',destination=root/'recovered-export')
    options.update(overrides)
    return worker().restore_completed_experiment(api,native.model,native.objectives,native.recipe,**options)

def test_completed_restore_preserves_actual_states_and_never_learns(native,tmp_path):
    result=execute(native,tmp_path/'source')
    snapshot=tmp_path/'source/group-0004'
    with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('optimization forbidden')), \
         patch.object(torch.Tensor,'backward',side_effect=AssertionError('backward forbidden')):
        recovered=restore(native,result,tmp_path,snapshot)
    assert recovered['new_updates_this_invocation']==0
    assert recovered['total_committed_updates']==4
    assert recovered['materializations_this_invocation']==len(native.batches)
    exact(recovered['arms'],result['arms'])
    saved=torch.load(tmp_path/'recovered-export/experimental.pt',weights_only=True)
    exact(recovered,saved)
    for arm in result['arms']:
        a,b=deepcopy(native.model),deepcopy(native.model)
        a.load_state_dict(result['arms'][arm]['model_state']);b.load_state_dict(saved['arms'][arm]['model_state'])
        a.set_phase('inference');b.set_phase('inference')
        from src.alice_personality.identity.codec import fingerprint
        with torch.no_grad():
            for batch in native.batches.values():assert fingerprint(a(batch.frame))==fingerprint(b(batch.frame))

@pytest.mark.parametrize('bad',['incomplete','external_pin','plan_path','worker_sha'])
def test_refuses_changed_or_incomplete_recovery_before_training(native,tmp_path,bad):
    result=execute(native,tmp_path/'source')
    snapshot=tmp_path/'source'/('group-0003' if bad=='incomplete' else 'group-0004')
    options={}
    if bad=='external_pin':options['manifest_sha256']='f'*64
    if bad=='plan_path':
        plan=deepcopy(result['plan']);plan['bindings']['admission_binding']={'path':'different'}
        options.update(plan=plan,expected_plan_sha256=api.plan_sha256(plan))
    if bad=='worker_sha':
        plan=deepcopy(result['plan']);plan['bindings']['worker_sha256']='f'*64
        options.update(plan=plan,expected_plan_sha256=api.plan_sha256(plan))
    with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('optimization forbidden')):
        with pytest.raises(ValueError):restore(native,result,tmp_path,snapshot,**options)
    assert not (tmp_path/'recovered-export').exists()

def test_actual_retained_tuple_metadata_survives_completed_restore(native,tmp_path,monkeypatch):
    from tests.personality.test_cumulative_bindings import ParentScopeBindingTests
    ParentScopeBindingTests.setUpClass()
    try:
        fixture=ParentScopeBindingTests();fixture.setUp();scope=fixture.bind()
        assert isinstance(scope['retained_parent_batches'],tuple)
        make=api.make_plan
        def with_scope(*args,**kwargs):
            kwargs['bindings']={**kwargs['bindings'],'retained_parent_scope':scope,
                'admission_binding':{'artifacts':{'authorization':{'path':'/original/authorization.json','file_sha256':'a'*64}}},
                'worker_sha256':'b'*64}
            return make(*args,**kwargs)
        monkeypatch.setattr(api,'make_plan',with_scope)
        result=execute(native,tmp_path/'source')
        with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('optimization forbidden')):
            recovered=restore(native,result,tmp_path,tmp_path/'source/group-0004')
        exact(recovered['arms'],result['arms'])
        assert recovered['plan']==result['plan']
    finally:ParentScopeBindingTests.doClassCleanups()

def test_guard_precedes_all_protected_access_and_original_worker_is_unchanged():
    module=worker()
    class Refused(Exception):pass
    class Namespace:
        def _p2_guard(self):raise Refused()
    with patch.object(Path,'stat',side_effect=AssertionError('pre-isolation')),patch.object(Path,'open',side_effect=AssertionError('pre-isolation')):
        with pytest.raises(Refused):module.run(Namespace())
    original=Path(module.__file__).with_name('magnolia_conditional_scope_20261117.py')
    assert sha256(original.read_bytes()).hexdigest()=='48456f8f4a0d0c8d7c40f88045b077986069b2f5b82a1f675a54ee114f30bcca'
    # Entire assessment and closing custody remain byte-for-byte original.
    marker='        arm_results, objective_rows, broader_predictions = {}, {}, {}'
    end='        result = {'
    assert Path(module.__file__).read_text().split(marker)[1].split(end)[0]==original.read_text().split(marker)[1].split(end)[0]
    tree=ast.parse(Path(module.__file__).read_text())
    assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='_p2_guard' for n in ast.walk(tree))==1

def test_actual_worker_keeps_training_recovery_paths_separate_from_native_ancestor(tmp_path):
    from types import SimpleNamespace
    module=worker();tree=ast.parse(Path(module.__file__).read_text())
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    names={'source_output','recovery_source_output','allocation_path','authorization_path','snapshot'}
    statements=sorted((n for n in ast.walk(run) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)),key=lambda n:n.lineno)
    context={'ROOT':tmp_path,'NAME':module.NAME,'SOURCE_JOB':module.SOURCE_JOB,
        'output':tmp_path/'new-output','namespace':SimpleNamespace(_bounded_path=lambda path,*args,**kwargs:Path(path))}
    for statement in statements:
        exec(compile(ast.Module(body=[statement],type_ignores=[]),module.__file__,'exec'),context)
    expected=tmp_path/'rayan-personality/private-compile'/f'{module.NAME}-{module.SOURCE_JOB}'
    assert context['source_output'].name=='rayan-cumulative-learning-r3-577714'
    assert context['allocation_path']==expected/'construction-allocation.protected.json'
    assert context['authorization_path']==expected/'experiment-authorization.protected.json'
    assert context['snapshot']==expected/'experimental-progress/group-0080'
