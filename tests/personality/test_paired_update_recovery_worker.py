"""Recovery wiring preserves the scientific comparison and refuses lost custody."""
import ast
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.eipm.gemma_n0 import magnolia_paired_update_comparison_20261020 as original
from scripts.eipm.gemma_n0 import magnolia_paired_update_recovery_20261021 as recovery
from tests.personality.test_paired_learning_recovery import native, run, exact, Interrupted


def functions(api):
    return {node.name: node for node in ast.parse(Path(api.__file__).read_text()).body
            if isinstance(node, ast.FunctionDef)}


def test_pairing_and_full_measurement_semantics_are_unchanged():
    before, after = functions(original), functions(recovery)
    for name in ('source_pairs', 'matched_group_geometry', 'capture_candidate',
                 'verify_exported_state', 'paired_region_comparison'):
        assert ast.dump(before[name]) == ast.dump(after[name])


def test_guard_refuses_before_any_protected_path_access():
    class Refused(Exception):
        pass
    class Namespace:
        def _p2_guard(self):
            raise Refused()
        def _bounded_path(self, *args, **kwargs):
            raise AssertionError('access before guard')
    with patch.object(recovery.Path, 'stat', side_effect=AssertionError('stat before guard')), \
            patch.object(recovery.Path, 'open', side_effect=AssertionError('open before guard')):
        with pytest.raises(Refused):
            recovery.run(Namespace())


def test_verified_failure_inventory_is_bound_without_exporting_protected_payload():
    import json
    root = Path(recovery.__file__).resolve().parents[3]
    receipt = json.loads((root / 'evaluation/eipm/gemma_n0/receipts/'
                         'magnolia_578731_failed_attempt_inventory_20261021.json').read_bytes())
    assert recovery.verify_failed_attempt_inventory(receipt) == receipt
    changed = deepcopy(receipt)
    changed['published_experimental_checkpoint_exists'] = True
    with pytest.raises(ValueError, match='seal changed'):
        recovery.verify_failed_attempt_inventory(changed)
    changed['receipt_sha256'] = recovery.sha256(recovery.canonical(
        {key: value for key, value in changed.items() if key != 'receipt_sha256'})).hexdigest()
    with pytest.raises(ValueError, match='identity changed'):
        recovery.verify_failed_attempt_inventory(changed)


def test_baseline_reuse_preserves_full_binding_and_does_not_claim_new_forward():
    source = Path(recovery.__file__).read_text()
    block = source.split('# Current saved endpoints are reusable only after exact complete rebinding.')[1]
    block = block.split('# Native validation has a fresh exact admission')[0]
    # Integration contract; native corrupt-source/model/frame refusal is exercised
    # dynamically in test_cumulative_recovery against the exact reused function.
    assert 'capture_candidate(' not in block
    assert 'packet[\'replay\'].produce(document)' in block
    for binding in ('document_sha256', 'frame_sha256', 'feature_binding_sha256', 'closed_session_sha256'):
        assert binding in block
    assert "set(expected_batches) == set(final_verified['valid'])" in block
    assert 'recover_parent_measurements(' in block
    assert 'parent_model_sha256=CURRENT_MODEL_SHA' in block
    assert 'parent_checkpoint_sha256=[value.manifest_sha256' in block
    assert "baseline_recovery['count'] == 236 and baseline_recovery['measured_case_count'] == 394" in block
    assert "'baseline_exact_native_packet_reproductions': 236" not in source
    assert "'new_baseline_document_forward_passes': 0" in source
    assert "'new_baseline_diagnostic_forward_passes': 10" in source


def test_private_experiment_progress_is_not_native_promotion_or_public_teaching():
    source = Path(recovery.__file__).read_text()
    assert "progress_directory=output/'experimental-progress'" in source
    assert "'failed_attempt_reused_training_steps': 0" in source
    assert "'native_checkpoint': False, 'resumable': False" in source
    forbidden = {'case_id', 'document_id', 'batch_sha256', 'case_sha256', 'document_sha256'}
    for call in ast.walk(ast.parse(source)):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'print':
            for value in ast.walk(call):
                if isinstance(value, ast.Dict):
                    assert not forbidden & {key.value for key in value.keys if isinstance(key, ast.Constant)}


def test_callback_publishes_external_pin_before_interruption(native, tmp_path):
    original = native.run_trial()
    pins = []
    progress = tmp_path/'interrupted'
    def stop(event):
        pins.append(recovery.committed_progress_pin(progress, event, 1))
        raise Interrupted()
    with pytest.raises(Interrupted):
        run(native, progress, on_event=stop)
    assert len(pins) == 1
    resumed = run(native, tmp_path/'resumed', resume_from=progress/'group-0001',
                  expected_resume_manifest_sha256=pins[0])
    exact(original['arms'], resumed['arms'])
    with pytest.raises(ValueError, match='cursor'):
        recovery.committed_progress_pin(progress, {'arm':'control', 'update':1}, 1)
