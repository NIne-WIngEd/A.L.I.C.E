"""Review-only reproduction using released v103; no network or production mutation.

The scheduler rejection is explicitly synthetic. Its purpose is to show which
raw evidence v103 discards, not to assert why Magnolia rejected the real query.
"""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
from unittest import mock

PACKAGE=Path(__file__).resolve().parents[2]/'tools/qwen-origin-v103/package/ALICE_MC10D_QWEN_ORIGIN_REVISION_v1.0.3'
sys.path.insert(0,str(PACKAGE))
import contract as c
import fixture_origins
import infra_recovery
import package_revision

RELEASE='05e1e8aa3fa1233d8acb25da5e87a93b211791f5e568db850f83dca3499630f2'
commands=[]
raw_error='SYNTHETIC_PROVIDER_REJECTION: deliberately not an observed Magnolia error'
def rejected(argv,check=True):
    commands.append(argv)
    return subprocess.CompletedProcess(argv,91,'',raw_error)
with mock.patch.object(infra_recovery,'verify_source_files',return_value=infra_recovery.policy()):
    try: infra_recovery.reconcile_source(rejected)
    except c.Stop as exc: collapsed=str(exc)
    else: raise AssertionError('Failed query accepted')
assert commands==[['squeue','-h','-j','575089','-o','%A|%j|%T']]
assert collapsed=='Source queue query failed; absence is unknown' and raw_error not in collapsed
with tempfile.TemporaryDirectory() as temp:
    state_path,rules=fixture_origins.local_prepared(Path(temp)/'vault')
    before=state_path.read_bytes()
    with mock.patch.object(package_revision,'policy',return_value=rules):
        package_revision.begin_local(state_path,RELEASE)
        original_intent=(package_revision.history(state_path.parent)/'intent.json').read_bytes()
        try: package_revision.begin_local(state_path,'3'*64)
        except c.Stop as exc: replacement_stop=str(exc)
        else: raise AssertionError('Different package accepted under old intent')
        assert state_path.read_bytes()==before
        assert (package_revision.history(state_path.parent)/'intent.json').read_bytes()==original_intent
        assert replacement_stop=='immutable evidence differs: intent.json'
result={
 'schema':'alice.v103.released-boundary-review.v1',
 'scope':'Offline reproduction of released control flow; synthetic scheduler rejection and local timestamp fixture; no actual Magnolia diagnosis inferred',
 'released_package_sha256':RELEASE,
 'scheduler':{'argv':commands[0],'synthetic_exit_code':91,'synthetic_stderr':raw_error,
              'returned_stop':collapsed,'inner_error_preserved':False,'accounting_called_after_failure':False},
 'replacement_package':{'same_revision_id_different_package_sha_rejected':True,'stop':replacement_stop,
                        'original_controller_state_preserved':True,'original_intent_preserved':True},
 'expected_production_v103_local_intent_sha256':c.sha(c.canonical(package_revision.intent(RELEASE)).encode()),
 'production_local_intent_path':'C:/ALICE_Vault/tools/alice-astra/qwen-fallback-a2/revisions/'+package_revision.policy()['revision_id']+'/intent.json',
 'live_remote_calls':0,'model_jobs_submitted':0,'source_files_changed':False,
 'current_actual_raw_scheduler_exit_and_stderr_known':False}
output=Path(__file__).with_name('REPRODUCTION.json')
output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
