"""Owner isolation, live custody and committed group acknowledgments."""
import ast
from hashlib import sha256
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.eipm.gemma_n0 import magnolia_conditional_learning_20261025 as worker
from scripts.eipm.gemma_n0 import live_namespace_custody as live
from src.alice_personality.n1.manifest_structure_audit import _PinnedFile
from src.alice_personality.n1 import raw_inference_lineage as raw
from tests.personality.test_live_namespace_custody import fixture
from tests.personality.test_paired_update_live_custody_worker import compiled_custody


def test_startup_isolation_is_first_and_never_reentered():
    class Refused(Exception): pass
    class Namespace:
        def _p2_guard(self): raise Refused()
    with patch.object(Path, 'stat', side_effect=AssertionError('before isolation')), \
         patch.object(Path, 'open', side_effect=AssertionError('before isolation')):
        with pytest.raises(Refused): worker.run(Namespace())
    tree = ast.parse(Path(worker.__file__).read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    assert ast.unparse(run.body[0].value) == 'namespace._p2_guard()'
    assert sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == '_p2_guard' for n in ast.walk(tree)) == 1


def test_actual_closure_keeps_descriptors_and_checks_teaching_bytes(tmp_path):
    namespace, isolation = fixture()
    folder = Path(worker.__file__).parent
    source = tmp_path/'source.json'; source.write_bytes(b'{"public":true}\n')
    teaching = tmp_path/'teaching.py'; teaching.write_bytes((folder/'conditional_teaching.py').read_bytes())
    context = dict(namespace=namespace, isolation=isolation, live_api=live, raw=raw,
        os=os, sha256=sha256, Path=Path, __file__=worker.__file__,
        worker_sha=sha256(Path(worker.__file__).read_bytes()).hexdigest(),
        live_path=folder/'live_namespace_custody.py', LIVE_CUSTODY_SHA=worker.LIVE_CUSTODY_SHA,
        experiment_path=folder/'conditional_learning_experiment.py',
        EXPERIMENT_HELPER_SHA=sha256((folder/'conditional_learning_experiment.py').read_bytes()).hexdigest(),
        teaching_path=teaching, TEACHING_HELPER_SHA=sha256(teaching.read_bytes()).hexdigest(),
        original_experiment_path=folder/'paired_learning_experiment.py',
        ORIGINAL_EXPERIMENT_HELPER_SHA=worker.ORIGINAL_EXPERIMENT_HELPER_SHA)
    with _PinnedFile(source, sha256(source.read_bytes()).hexdigest(), 1024) as pin:
        context['held'] = [pin]
        check = compiled_custody(worker, context)
        for _ in range(3): check(); pin.verify()
        assert not pin.stream.closed
        teaching.write_bytes(b'changed')
        with pytest.raises(AssertionError): check()


def test_acknowledgment_uses_new_progress_schema_and_exact_cursor(tmp_path):
    folder=tmp_path/'group-0003'; folder.mkdir()
    event={'arm':'intervention','update':1}
    payload={'schema':'alice-personality-conditional-learning-progress-v1',
             'cursor':{'arm':'intervention','next_group':1,'total_committed_updates':3}}
    manifest=folder/'manifest.json'; manifest.write_text(json.dumps(payload))
    assert worker.committed_progress_pin(tmp_path,event,3)==sha256(manifest.read_bytes()).hexdigest()
    payload['cursor']['next_group']=2; manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError):worker.committed_progress_pin(tmp_path,event,3)
