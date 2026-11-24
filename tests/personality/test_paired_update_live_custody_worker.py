"""Exercise the actual worker's live-custody closure with owned native pins."""
import ast
from contextlib import ExitStack
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.eipm.gemma_n0 import live_namespace_custody as live
from scripts.eipm.gemma_n0 import magnolia_paired_update_recovery_20261021 as failed
from scripts.eipm.gemma_n0 import magnolia_paired_update_live_custody_20261021 as corrected
from src.alice_personality.n1.manifest_structure_audit import _PinnedFile
from src.alice_personality.n1 import raw_inference_lineage as raw
from tests.personality.test_live_namespace_custody import fixture
from tests.personality.test_paired_update_recovery_worker import functions


def test_startup_cleanup_remains_first_and_has_one_call_site():
    run = functions(corrected)['run']
    assert ast.unparse(run.body[0].value) == 'namespace._p2_guard()'
    calls = [n for n in ast.walk(run) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == '_p2_guard']
    assert len(calls) == 1
    class Refused(Exception): pass
    class Namespace:
        def _p2_guard(self): raise Refused()
    with patch.object(Path, 'stat', side_effect=AssertionError('access before guard')), \
            patch.object(Path, 'open', side_effect=AssertionError('access before guard')):
        with pytest.raises(Refused): corrected.run(Namespace())


def compiled_custody(api, context):
    nested = next(n for n in ast.walk(functions(api)['run'])
                  if isinstance(n, ast.FunctionDef) and n.name == 'custody_guard')
    code = compile(ast.fix_missing_locations(ast.Module(body=[nested], type_ignores=[])),
                   str(api.__file__), 'exec')
    exec(code, context)
    return context['custody_guard']


def test_actual_worker_closure_keeps_pins_and_rejects_changed_input(tmp_path):
    folder = Path(corrected.__file__).parent
    namespace, isolation = fixture()
    source = (tmp_path/'public-input.json').resolve()
    source.write_bytes(b'{"public":true}\n')
    context = dict(namespace=namespace, isolation=isolation, live_api=live, raw=raw,
        os=os, sha256=sha256, Path=Path, __file__=corrected.__file__,
        worker_sha=sha256(Path(corrected.__file__).read_bytes()).hexdigest(),
        live_path=folder/'live_namespace_custody.py', LIVE_CUSTODY_SHA=corrected.LIVE_CUSTODY_SHA,
        experiment_path=folder/'paired_learning_recovery.py', EXPERIMENT_HELPER_SHA=corrected.EXPERIMENT_HELPER_SHA,
        original_experiment_path=folder/'paired_learning_experiment.py',
        ORIGINAL_EXPERIMENT_HELPER_SHA=corrected.ORIGINAL_EXPERIMENT_HELPER_SHA)
    with ExitStack() as stack:
        pin = stack.enter_context(_PinnedFile(source, sha256(source.read_bytes()).hexdigest(), 1024))
        context['held'] = [pin]
        # The unchanged failed closure tries to re-enter startup cleanup.
        with pytest.raises(AssertionError, match='startup cleanup'):
            compiled_custody(failed, context)()
        check = compiled_custody(corrected, context)
        for _ in range(3):
            check()
            pin.verify()
        assert not pin.stream.closed
    # Also exercise the existing custody failure, including ExitStack unwind.
    with pytest.raises(Exception):
        with _PinnedFile(source, sha256(source.read_bytes()).hexdigest(), 1024) as pin:
            context['held'] = [pin]
            source.write_bytes(b'{"public":false}\n')
            with pytest.raises(AssertionError): check()


def test_scientific_pairing_measurements_and_progress_are_unchanged():
    before, after = functions(failed), functions(corrected)
    for name in ('source_pairs', 'matched_group_geometry', 'capture_candidate',
                 'verify_exported_state', 'paired_region_comparison', 'committed_progress_pin'):
        assert ast.dump(before[name]) == ast.dump(after[name])
    for name in ('batch_loader', 'validate_batch', 'on_event'):
        old = next(n for n in ast.walk(before['run']) if isinstance(n, ast.FunctionDef) and n.name == name)
        new = next(n for n in ast.walk(after['run']) if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(old) == ast.dump(new)


def test_new_owner_plan_changes_only_lifecycle_and_failure_provenance():
    root = Path(corrected.__file__).resolve().parents[3]
    receipts = root/'evaluation/eipm/gemma_n0/receipts'
    old = json.loads((receipts/'paired_update_recovery_owner_plan_20261021.json').read_bytes())
    plan_bytes = (receipts/'paired_update_live_custody_owner_plan_20261021.json').read_bytes()
    new = json.loads(plan_bytes)
    assert sha256(plan_bytes).hexdigest() == corrected.EXPERIMENT_OWNER_PLAN_SHA
    allowed = {'failed_job_id', 'failure_inventory_sha256', 'parent_experiment_owner_plan_sha256',
               'live_custody_helper_sha256', 'causal_probe_sha256', 'authority_basis'}
    assert {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)} <= allowed
    assert new['live_custody_helper_sha256'] == corrected.LIVE_CUSTODY_SHA
    source = Path(corrected.__file__).read_text()
    assert 'pin(live_path, LIVE_CUSTODY_SHA)' in source
    assert "'live_custody_helper_sha256': LIVE_CUSTODY_SHA" in source
    assert "'live_custody_helper_sha256'] == LIVE_CUSTODY_SHA" in source


def test_failure_inventory_cannot_be_substituted_or_promoted_to_checkpoint():
    root = Path(corrected.__file__).resolve().parents[3]
    data = (root/'evaluation/eipm/gemma_n0/receipts/magnolia_578769_stage_inventory_20261021.json').read_bytes()
    assert sha256(data).hexdigest() == corrected.FAILURE_INVENTORY_SHA
    receipt = json.loads(data)
    assert corrected.verify_failed_attempt_inventory(receipt) == receipt
    changed = deepcopy(receipt)
    changed['progress_directory_exists'] = True
    with pytest.raises(ValueError, match='seal changed'):
        corrected.verify_failed_attempt_inventory(changed)
    changed['receipt_sha256'] = sha256(corrected.canonical(
        {k:v for k,v in changed.items() if k != 'receipt_sha256'})).hexdigest()
    with pytest.raises(ValueError, match='identity changed'):
        corrected.verify_failed_attempt_inventory(changed)


def test_public_exception_logging_preserves_chain_without_payload():
    source = Path(corrected.__file__).read_text()
    assert "'exception_chain': live_api.public_exception_chain(error)" in source
    assert 'traceback.extract_tb' not in source
    forbidden = {'case_id', 'document_id', 'batch_sha256', 'case_sha256', 'document_sha256'}
    for call in ast.walk(ast.parse(source)):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'print':
            for value in ast.walk(call):
                if isinstance(value, ast.Dict):
                    assert not forbidden & {k.value for k in value.keys if isinstance(k, ast.Constant)}


@pytest.mark.parametrize('failure', ['missing', 'changed', 'import', 'formatter'])
def test_failed_reporter_never_reveals_private_original_error(tmp_path, capsys, failure):
    jobs = tmp_path/'jobs'
    jobs.mkdir()
    path = jobs/(corrected.NAME+'-live.py')
    if failure != 'missing':
        path.write_bytes(Path(live.__file__).read_bytes() if failure != 'changed' else b'changed')
    def broken(*args): raise OSError('PRIVATE_REPORTER_PATH')
    from types import SimpleNamespace
    with patch.object(corrected, 'ROOT', tmp_path):
        if failure in ('missing', 'changed'):
            corrected.report_failure(ValueError('PRIVATE_ORIGINAL_INPUT'))
        elif failure == 'import':
            with patch.object(corrected, 'module', side_effect=broken):
                corrected.report_failure(ValueError('PRIVATE_ORIGINAL_INPUT'))
        else:
            with patch.object(corrected, 'module', return_value=SimpleNamespace(public_exception_chain=broken)):
                corrected.report_failure(ValueError('PRIVATE_ORIGINAL_INPUT'))
    captured = capsys.readouterr()
    assert not captured.err and 'PRIVATE_' not in captured.out
    assert json.loads(captured.out)['public_reporter_available'] is False
