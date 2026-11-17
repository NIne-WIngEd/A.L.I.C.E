"""Residual experiment retains original protected-read and evaluation boundaries."""
import ast
import importlib
from hashlib import sha256
import json
import os
from pathlib import Path
from unittest.mock import patch
import pytest
from scripts.eipm.gemma_n0 import live_namespace_custody as live
from src.alice_personality.n1.manifest_structure_audit import _PinnedFile
from src.alice_personality.n1 import raw_inference_lineage as raw
from tests.personality.test_live_namespace_custody import fixture
from tests.personality.test_paired_update_live_custody_worker import compiled_custody

def worker():
    return importlib.import_module('scripts.eipm.gemma_n0.magnolia_conditional_residual_20261104')

def test_single_guard_precedes_every_protected_read():
    module=worker()
    class Refused(Exception):pass
    class Namespace:
        def _p2_guard(self):raise Refused()
    with patch.object(Path,'stat',side_effect=AssertionError('pre-isolation')),patch.object(Path,'open',side_effect=AssertionError('pre-isolation')):
        with pytest.raises(Refused):module.run(Namespace())
    tree=ast.parse(Path(module.__file__).read_text())
    assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='_p2_guard' for n in ast.walk(tree))==1

def test_actual_live_closure_pins_extension_without_destroying_descriptors(tmp_path):
    module=worker();namespace,isolation=fixture();folder=Path(module.__file__).parent
    source=tmp_path/'source.json';source.write_bytes(b'{"public":true}\n')
    extension=tmp_path/'extension.py';extension.write_bytes((folder/'conditional_judgment_residual.py').read_bytes())
    context=dict(namespace=namespace,isolation=isolation,live_api=live,raw=raw,os=os,sha256=sha256,Path=Path,
        __file__=module.__file__,worker_sha=sha256(Path(module.__file__).read_bytes()).hexdigest(),
        live_path=folder/'live_namespace_custody.py',LIVE_CUSTODY_SHA=module.LIVE_CUSTODY_SHA,
        experiment_path=folder/'conditional_residual_experiment.py',EXPERIMENT_HELPER_SHA=sha256((folder/'conditional_residual_experiment.py').read_bytes()).hexdigest(),
        teaching_path=folder/'conditional_teaching.py',TEACHING_HELPER_SHA=module.TEACHING_HELPER_SHA,
        original_experiment_path=folder/'paired_learning_experiment.py',ORIGINAL_EXPERIMENT_HELPER_SHA=module.ORIGINAL_EXPERIMENT_HELPER_SHA,
        extension_path=extension,EXTENSION_SHA=sha256(extension.read_bytes()).hexdigest())
    with _PinnedFile(source,sha256(source.read_bytes()).hexdigest(),1024) as pin:
        context['held']=[pin];check=compiled_custody(module,context)
        check();pin.verify();assert not pin.stream.closed
        extension.write_bytes(b'changed')
        with pytest.raises(AssertionError):check()

def test_new_progress_schema_and_immutable_original_worker(tmp_path):
    module=worker();folder=tmp_path/'group-0003';folder.mkdir()
    event={'arm':'intervention','update':1}
    manifest=folder/'manifest.json';manifest.write_text(json.dumps({'schema':'alice-personality-conditional-residual-progress-v1',
        'cursor':{'arm':'intervention','next_group':1,'total_committed_updates':3}}))
    assert module.committed_progress_pin(tmp_path,event,3)==sha256(manifest.read_bytes()).hexdigest()
    original=Path(module.__file__).with_name('magnolia_conditional_learning_20261025.py')
    assert sha256(original.read_bytes()).hexdigest()=='dcb10bed56daa4d54605e072349b331c84921d42369f81f5a491172b70712689'

def test_offsetting_context_repair_and_harm_are_visible_without_joint_pair_change():
    from dataclasses import replace
    from tests.personality.test_conditional_teaching import public_fixture
    _,_,left,_,_,_,_=public_fixture()
    target=left.targets[0]
    right=replace(left,case_id='opposite-public-context',targets=(replace(target,
        facts=tuple(replace(f,value=1-f.value) for f in target.facts)),))
    ids=[c.candidate_id for c in left.document.candidates]
    before={left.case_id:{'candidate_ids':[ids],'preferences':[[.1,.9]]},
            right.case_id:{'candidate_ids':[ids],'preferences':[[.1,.9]]}}
    after={left.case_id:{'candidate_ids':[ids],'preferences':[[.9,.1]]},
           right.case_id:{'candidate_ids':[ids],'preferences':[[.9,.1]]}}
    result=worker().paired_region_comparison([(left,right)],before,after)['fields']['preferences']
    assert result['before_correct_top_sets']==result['after_correct_top_sets']==1
    assert result['before_both_contexts_correct']==result['after_both_contexts_correct']==0
    assert result['context_repairs']==result['context_harms']==1
    assert result['changed_top_sets']==2
    sham=worker().paired_region_comparison([(left,right)],before,before)['fields']['preferences']
    assert sham['context_repairs']==sham['context_harms']==sham['changed_top_sets']==0
