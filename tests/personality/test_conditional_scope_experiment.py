"""Actual native learning and recovery under two explicit trainable scopes."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import importlib
import json
import pytest
import torch

from tests.personality.test_conditional_residual_experiment import native
from tests.personality.test_paired_learning_recovery import exact
from scripts.eipm.gemma_n0 import conditional_judgment_residual as architecture
from src.alice_personality.identity.codec import canonical, fingerprint


def execute(native, directory, *, api=None, records=None, **options):
    api = api or importlib.import_module('scripts.eipm.gemma_n0.conditional_scope_experiment')
    records = records or {}
    plan = api.make_plan(native.model, native.objectives, native.recipe,
        batch_sha256=native.digests, matched_groups=native.groups,
        correction_sha256={d: api.teaching.record_sha256(r) for d, r in records.items()},
        bindings={'public_fixture': True, 'architecture': architecture.ARCHITECTURE,
            'extension_sha256': sha256(open(architecture.__file__, 'rb').read()).hexdigest(),
            'migration': native.migration})
    return api.run_experiment(native.model, native.objectives, native.recipe, plan=plan,
        expected_plan_sha256=api.plan_sha256(plan), batch_loader=native.batches.__getitem__,
        validate_batch=native.validate, correction_loader=records.get,
        custody_guard=native.guard, progress_directory=directory, **options)


def test_complete_objectives_and_exact_frozen_intervention(native, tmp_path):
    before = deepcopy(native.model.state_dict())
    result = execute(native, tmp_path/'full', destination=tmp_path/'export')
    from scripts.eipm.gemma_n0.magnolia_conditional_scope_20261117 import verify_exported_state
    restored = torch.load(tmp_path/'export/experimental.pt', weights_only=True)
    verify_exported_state(result, restored)
    manifest = json.loads((tmp_path/'export/manifest.json').read_bytes())
    assert manifest['state_file_sha256'] == sha256((tmp_path/'export/experimental.pt').read_bytes()).hexdigest()
    assert manifest['plan']['optimizer_parameter_names'] == result['plan']['optimizer_parameter_names']
    exact(before, native.model.state_dict())
    assert result['plan']['learning_scopes'] == {'control': 'all_n2', 'intervention': 'conditioning_only'}
    for name, arm in result['arms'].items():
        assert bool(arm['model_state']['n2.conditioning.enabled'])
        assert arm['frozen_scope_unchanged'] is True
        assert arm['model_state']['n2.conditioning.gate'].item() != 0
        for key, value in before.items():
            if not key.startswith('n2.') or (name == 'intervention' and not key.startswith('n2.conditioning.')):
                assert torch.equal(value, arm['model_state'][key]), key
        for event in arm['events']:
            for endpoint in event['endpoints']:
                assert set(endpoint['objectives']) == set(native.batches[endpoint['batch_sha256']].targets)
    assert any(not torch.equal(v, result['arms']['control']['model_state'][k])
               for k, v in before.items() if k.startswith('n2.') and not k.startswith('n2.conditioning.'))
    assert not result['native_checkpoint'] and not result['acceptance_authority']


@pytest.mark.parametrize('arm_name', ['control', 'intervention'])
def test_resume_is_exact_for_each_scope(native, tmp_path, arm_name):
    expected = execute(native, tmp_path/'full')
    class Stop(Exception): pass
    def stop(event):
        if event['arm'] == arm_name and event['update'] == 1: raise Stop()
    with pytest.raises(Stop): execute(native, tmp_path/'partial', on_event=stop)
    snapshot = tmp_path/'partial'/('group-0001' if arm_name == 'control' else 'group-0003')
    actual = execute(native, tmp_path/'resumed', resume_from=snapshot,
        expected_resume_manifest_sha256=sha256((snapshot/'manifest.json').read_bytes()).hexdigest())
    exact(expected['arms'], actual['arms'])


@pytest.mark.parametrize('corruption', ['scope', 'order', 'core', 'mode', 'schema'])
def test_resealed_snapshot_cannot_change_policy_or_frozen_core(native, tmp_path, corruption):
    api = importlib.import_module('scripts.eipm.gemma_n0.conditional_scope_experiment')
    execute(native, tmp_path/'full')
    snapshot = tmp_path/'full/group-0003'
    state_path = snapshot/'experimental-progress.pt'
    state = torch.load(state_path, weights_only=False)
    if corruption == 'scope': state['binding']['plan']['learning_scopes']['intervention'] = 'all_n2'
    elif corruption == 'order': state['binding']['optimizer_parameter_names']['intervention'].reverse()
    elif corruption == 'schema': state['schema'] = 'alice-personality-conditional-residual-progress-v1'
    else:
        arm = state['arms']['intervention']
        if corruption == 'mode': arm['model_state']['n2.conditioning.enabled'].fill_(False)
        else:
            key = next(k for k in arm['model_state'] if k.startswith('n2.') and not k.startswith('n2.conditioning.'))
            arm['model_state'][key].add_(.01)
        arm['model_state_sha256'] = fingerprint(arm['model_state'])
    torch.save(state, state_path)
    manifest = json.loads((snapshot/'manifest.json').read_text())
    manifest['binding_sha256'] = sha256(canonical(state['binding'])).hexdigest()
    manifest['state_file_sha256'] = sha256(state_path.read_bytes()).hexdigest()
    (snapshot/'manifest.json').write_bytes(canonical(manifest)+b'\n')
    with pytest.raises(ValueError): execute(native, tmp_path/'resumed', resume_from=snapshot,
        expected_resume_manifest_sha256=sha256((snapshot/'manifest.json').read_bytes()).hexdigest())


def test_actual_corrected_teaching_and_control_equal_prior_enabled_arm(native, tmp_path):
    from src.alice_personality.identity import IdentityModel
    from src.alice_personality.teaching import IdentityObjectives, TrainingRecipe
    from tests.personality.test_conditional_teaching import public_fixture
    from scripts.eipm.gemma_n0 import conditional_residual_experiment as prior
    api = importlib.import_module('scripts.eipm.gemma_n0.conditional_scope_experiment')
    config, n1, n2, b1, _, binding, materialize = public_fixture()
    torch.manual_seed(61)
    native.model, native.migration = architecture.migrate(IdentityModel(config), seed=713)
    native.objectives = IdentityObjectives(config.learned_width)
    native.recipe = TrainingRecipe('n2', .003, 1., 61)
    native.batches = {}; records = {}
    for i in range(4):
        case = replace(n2, case_id=f'scope-public-{i}'); batch = materialize(case)
        digest = batch.fingerprint(); native.batches[digest] = batch
        records[digest] = api.teaching.build_correction_record(n1, case, b1, batch, frame_binding=binding,
            reviewed_n1_batches={b1.fingerprint(): {'split': 'TRAIN', 'case_ids': list(b1.case_ids),
                'source_family_ids': [list(row) for row in b1.source_family_ids]}})
    native.digests = list(native.batches); native.groups = [native.digests[:2], native.digests[2:]]
    native.validate = lambda b: b.fingerprint()
    old = execute(native, tmp_path/'prior', api=prior, records=records)['arms']['intervention']
    new = execute(native, tmp_path/'scope', records=records)
    control = new['arms']['control']
    for key in ('model_state', 'optimizer_state', 'objective_state', 'torch_rng'):
        exact(old[key], control[key])
    for a, b in zip(old['events'], control['events']): assert a['endpoints'] == b['endpoints']
    for arm in new['arms'].values():
        views = arm['events'][0]['endpoints'][0]['views']
        assert views['natural'] != views['replay']
    # Frozen inherited heads remain differentiable with respect to the new route.
    model = deepcopy(native.model); api.configure_scope(model, 'intervention')
    with torch.no_grad(): model.n2.conditioning.gate.fill_(.1)
    digest = native.digests[0]
    packet = api.teaching.forward_with_correction(model, native.batches[digest], records[digest],
        expected_record_sha256=api.teaching.record_sha256(records[digest]), trainable_scope='conditioning_only')
    loss, _, _ = native.objectives(packet, native.batches[digest]); loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters()
               if n.startswith('n2.conditioning.') and n != 'n2.conditioning.gate')
    assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith('n2.conditioning.'))
    model.n2.conditional_embedding.weight.requires_grad_(True)
    with pytest.raises(ValueError):
        api.teaching.forward_with_correction(model, native.batches[digest], records[digest],
            expected_record_sha256=api.teaching.record_sha256(records[digest]), trainable_scope='conditioning_only')
