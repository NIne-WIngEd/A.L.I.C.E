"""Real retained-scope metadata survives scoped native writers and both restores."""
from copy import deepcopy
from hashlib import sha256
import json
import pytest
import torch
from scripts.eipm.gemma_n0 import conditional_scope_experiment as api
from scripts.eipm.gemma_n0.magnolia_residual_saved_attribution_r2_20261108 import verify_final_metadata, verify_saved_binding
from tests.personality.test_conditional_scope_experiment import execute, native
from tests.personality.test_cumulative_bindings import ParentScopeBindingTests


def test_real_parent_scope_full_manifest_and_per_arm_parameter_maps(native, tmp_path, monkeypatch):
    ParentScopeBindingTests.setUpClass()
    try:
        fixture = ParentScopeBindingTests(); fixture.setUp(); scope = fixture.bind()
        assert isinstance(scope['retained_parent_batches'], tuple)
        original_make = api.make_plan
        def make(*args, **kwargs):
            kwargs['bindings'] = {**kwargs['bindings'], 'retained_parent_scope': scope}
            return original_make(*args, **kwargs)
        monkeypatch.setattr(api, 'make_plan', make)
        result = execute(native, tmp_path/'progress', destination=tmp_path/'export')
        saved = torch.load(tmp_path/'export/experimental.pt', weights_only=True)
        manifest = json.loads((tmp_path/'export/manifest.json').read_bytes())
        json_plan = json.loads(api.canonical(result['plan']))
        digest = sha256((tmp_path/'export/experimental.pt').read_bytes()).hexdigest()
        assert saved['plan'] != json_plan
        verify_final_metadata(manifest, saved, json_plan, digest)
        for index in range(1, 5):
            path = tmp_path/'progress'/f'group-{index:04d}'
            snapshot = torch.load(path/'experimental-progress.pt', weights_only=True)
            verify_saved_binding(snapshot['binding'], saved, result['plan']['optimizer_parameter_names'])
            resumed = execute(native, tmp_path/f'restored-{index}', resume_from=path,
                expected_resume_manifest_sha256=sha256((path/'manifest.json').read_bytes()).hexdigest())
            from tests.personality.test_paired_learning_recovery import exact
            exact(result['arms'], resumed['arms'])
        altered = deepcopy(manifest); altered['plan']['learning_scopes']['intervention'] = 'all_n2'
        with pytest.raises(ValueError): verify_final_metadata(altered, saved, json_plan, digest)
        altered = deepcopy(json_plan); altered['bindings']['retained_parent_scope']['retained_parent_batches'][0] = 'f'*64
        with pytest.raises(ValueError): verify_final_metadata(manifest, saved, altered, digest)
    finally:
        ParentScopeBindingTests.doClassCleanups()
