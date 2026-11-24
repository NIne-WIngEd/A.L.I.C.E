from copy import deepcopy
from hashlib import sha256
import importlib
import importlib.util
import pytest
from tests.personality.test_conditional_residual_experiment import native
from tests.personality.test_paired_learning_recovery import exact
from scripts.eipm.gemma_n0 import conditional_judgment_residual as architecture

def execute(native, path, **options):
    name='scripts.eipm.gemma_n0.native_contextual_learnability'
    assert importlib.util.find_spec(name), 'native full-path repeated learner is missing'
    api=importlib.import_module(name)
    plan=api.make_plan(native.model,native.objectives,native.recipe,batch_sha256=native.digests,
        matched_groups=native.groups,correction_sha256={},repetitions=2,measurement_updates=[2,4],
        bindings={'public_fixture':True,'architecture':architecture.ARCHITECTURE,
          'extension_sha256':sha256(open(architecture.__file__,'rb').read()).hexdigest(),'migration':native.migration})
    return api.run_experiment(native.model,native.objectives,native.recipe,plan=plan,
        expected_plan_sha256=api.plan_sha256(plan),batch_loader=native.batches.__getitem__,
        validate_batch=native.validate,correction_loader=lambda _:None,custody_guard=native.guard,
        progress_directory=path,**options)

def test_repeated_full_native_objectives_and_exact_resume(native,tmp_path):
    original=deepcopy(native.model.state_dict())
    full=execute(native,tmp_path/'full')
    assert full['total_committed_updates']==8
    for arm in full['arms'].values():
        assert arm['updates']==4 and set(arm['batch_occurrences'].values())=={2}
        for event in arm['events']:
            for endpoint in event['endpoints']:
                assert set(endpoint['objectives'])==set(native.batches[endpoint['batch_sha256']].targets)
    assert sorted(p.name for p in (tmp_path/'full').iterdir())==['group-0002','group-0004','group-0006','group-0008']
    class Stop(Exception): pass
    def stop(e):
        if e['arm']=='intervention' and e['update']==2: raise Stop()
    with pytest.raises(Stop):execute(native,tmp_path/'partial',on_event=stop)
    snapshot=tmp_path/'partial/group-0006'
    resumed=execute(native,tmp_path/'resumed',resume_from=snapshot,
        expected_resume_manifest_sha256=sha256((snapshot/'manifest.json').read_bytes()).hexdigest())
    exact(full['arms'],resumed['arms']);exact(original,native.model.state_dict())
    assert not full['acceptance_authority'] and not full['native_checkpoint']
    api=importlib.import_module('scripts.eipm.gemma_n0.native_contextual_learnability')
    for arm,offset in [('control',0),('intervention',4)]:
        path=tmp_path/'full'/f'group-{offset+4:04d}'
        kwargs=dict(plan=full['plan'],directory=path,manifest_sha256=sha256((path/'manifest.json').read_bytes()).hexdigest(),arm=arm,update=4)
        read=api.read_measurement_arm(native.model,native.objectives,native.recipe,**kwargs)
        exact(read,full['arms'][arm])
        with pytest.raises(ValueError):
            api.read_measurement_arm(native.model,native.objectives,native.recipe,**{**kwargs,'update':2})


def test_native_corrected_measurement_and_unchanged_original_training_recipe(native,tmp_path):
    from dataclasses import replace
    import torch
    from src.alice_personality.identity import IdentityModel
    from src.alice_personality.teaching import IdentityObjectives,TrainingRecipe
    from tests.personality.test_conditional_teaching import public_fixture
    from scripts.eipm.gemma_n0 import native_contextual_learnability as api,conditional_scope_experiment as prior
    from scripts.eipm.gemma_n0 import magnolia_native_learnability_20261124 as worker
    config,n1,n2,b1,_,binding,materialize=public_fixture()
    torch.manual_seed(61)
    model,migration=architecture.migrate(IdentityModel(config),seed=713)
    objectives=IdentityObjectives(config.learned_width);recipe=TrainingRecipe('n2',.0003,1.,61)
    cases=[replace(n2,case_id=f'learnability-{i}') for i in range(4)]
    # Opposing preference truth is source-defined, not chosen by model failure.
    cases[1]=replace(cases[1],targets=tuple(replace(t,facts=tuple(replace(f,value=1-f.value) for f in t.facts))
        if t.field=='preferences' else t for t in cases[1].targets))
    assert worker.select_opposing_pair([(cases[2],cases[3]),(cases[0],cases[1])])==(cases[0],cases[1])
    batches={};records={}
    for case in cases:
        batch=materialize(case);d=batch.fingerprint();batches[d]=batch
        records[d]=api.teaching.build_correction_record(n1,case,b1,batch,frame_binding=binding,
            reviewed_n1_batches={b1.fingerprint():{'split':'TRAIN','case_ids':list(b1.case_ids),
            'source_family_ids':[list(r) for r in b1.source_family_ids]}})
    digests=list(batches);results=[]
    for module,folder in ((prior,'original'),(api,'new')):
        extra={'repetitions':1,'measurement_updates':[2]} if module is api else {}
        plan=module.make_plan(model,objectives,recipe,batch_sha256=digests,matched_groups=[digests[:2],digests[2:]],
            correction_sha256={d:api.teaching.record_sha256(r) for d,r in records.items()},
            bindings={'architecture':architecture.ARCHITECTURE,'extension_sha256':sha256(open(architecture.__file__,'rb').read()).hexdigest(),
                'migration':migration},**extra)
        results.append(module.run_experiment(model,objectives,recipe,plan=plan,expected_plan_sha256=module.plan_sha256(plan),
            batch_loader=batches.__getitem__,validate_batch=lambda b:b.fingerprint(),correction_loader=records.get,
            custody_guard=lambda:None,progress_directory=tmp_path/folder))
    exact(results[0]['arms'],results[1]['arms'])
    for arm in results[1]['arms'].values():
        candidate=deepcopy(model);candidate.load_state_dict(arm['model_state']);candidate.set_phase('inference')
        before=deepcopy(candidate.state_dict());d=digests[0]
        measured=worker.measure_native_case(candidate,objectives,recipe,batches[d],records[d],api.teaching.record_sha256(records[d]),api.teaching)
        exact(before,candidate.state_dict())
        assert set(measured)=={'natural','corrected','disabled'}
        for result in measured.values():
            assert set(result['objectives']['field_loss'])==set(batches[d].targets)
        assert measured['natural']['decision']!=measured['corrected']['decision']
        assert measured['natural']['decision']!=measured['disabled']['decision']
        assert len(worker.aggregate_objectives([v['objectives'] for v in [measured['natural'],measured['corrected']]]))==len(batches[d].targets)


def test_private_guard_precedes_reads_and_public_pins_match():
    from pathlib import Path
    from unittest.mock import patch
    import json
    from scripts.eipm.gemma_n0 import magnolia_native_learnability_20261124 as worker
    class Refused(Exception):pass
    class Namespace:
        def _p2_guard(self):raise Refused()
    with patch.object(Path,'stat',side_effect=AssertionError('pre-isolation')),patch.object(Path,'open',side_effect=AssertionError('pre-isolation')):
        with pytest.raises(Refused):worker.run(Namespace())
    root=Path(__file__).resolve().parents[2]
    owner=root/'evaluation/eipm/gemma_n0/receipts/native_learnability_owner_plan_20261124.json'
    assert sha256(owner.read_bytes()).hexdigest()==worker.EXPERIMENT_OWNER_PLAN_SHA
    helper=Path(worker.__file__).with_name('native_contextual_learnability.py')
    assert sha256(helper.read_bytes()).hexdigest()==worker.EXPERIMENT_HELPER_SHA==json.loads(owner.read_bytes())['helper_sha256']
    assert sha256(Path(worker.__file__).read_bytes()).hexdigest() in Path(worker.__file__).with_suffix('.sbatch').read_text()
