"""R2 stages public evidence as an exact owner-only artifact, retaining recovery."""
import ast
from hashlib import sha256
import importlib
from pathlib import Path
from types import SimpleNamespace
import pytest
from tests.personality import test_conditional_scope_recovery as previous
from tests.personality.test_conditional_scope_recovery import (
    native, test_completed_restore_preserves_actual_states_and_never_learns,
    test_refuses_changed_or_incomplete_recovery_before_training,
    test_actual_retained_tuple_metadata_survives_completed_restore,
    test_guard_precedes_all_protected_access_and_original_worker_is_unchanged,
    test_actual_worker_keeps_training_recovery_paths_separate_from_native_ancestor)

@pytest.fixture(autouse=True)
def route_worker(monkeypatch):
    monkeypatch.setattr(previous,'worker',lambda:importlib.import_module(
        'scripts.eipm.gemma_n0.magnolia_conditional_scope_recovery_r2_20261121'))

def test_actual_worker_pins_prepared_public_copy_and_retains_strict_reader(tmp_path):
    # The failed original used a scheduler output path. R2 must use the
    # already-established prepared public-artifact route, with the SAME digest.
    worker=previous.worker();tree=ast.parse(Path(worker.__file__).read_text())
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    assignment=next(n for n in ast.walk(run) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='source_log' for t in n.targets))
    calls=[]
    context={'ROOT':tmp_path,'NAME':worker.NAME,'RECOVERY_NAME':worker.RECOVERY_NAME,
        'SOURCE_JOB':worker.SOURCE_JOB,'SOURCE_STDOUT_SHA':worker.SOURCE_STDOUT_SHA,
        'pin':lambda *args:calls.append(args)}
    exec(compile(ast.Module(body=[assignment],type_ignores=[]),worker.__file__,'exec'),context)
    assert calls==[(tmp_path/'jobs'/(worker.RECOVERY_NAME+'-source-stdout.txt'),worker.SOURCE_STDOUT_SHA)]
    source=Path(worker.__file__).read_text()
    assert 'assert stat.S_IMODE(path.stat().st_mode) == 0o600' in source
    failed=Path(worker.__file__).with_name('magnolia_conditional_scope_recovery_20261121.py')
    assert sha256(failed.read_bytes()).hexdigest()=='cb37c3f3732f0050a4901590ba69bda321ae2796b7d284a15eef41e72c904351'
