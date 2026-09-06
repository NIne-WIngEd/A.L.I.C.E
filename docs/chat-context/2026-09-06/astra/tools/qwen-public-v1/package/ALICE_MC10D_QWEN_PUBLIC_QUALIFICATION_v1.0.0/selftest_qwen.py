"""Offline protocol tests. Synthetic fixtures are never real Qwen qualification evidence."""
from __future__ import annotations

import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock
import zipfile
from types import SimpleNamespace

import contract as c
import controller
import evidence
import publish_context as pub
import remote_agent as remote
import worker

FAKE_PACKAGE='1'*64


def raw_response(answer, *, reason='stop', tokens=128):
    first={'model':c.TAG,'message':{'role':'assistant','thinking':'Synthetic fixture only.','content':''},'done':False}
    last={'model':c.TAG,'message':{'role':'assistant','content':c.canonical(answer)},'done':True,'done_reason':reason,'total_duration':2_000_000_000,'load_duration':1,'prompt_eval_count':256,'prompt_eval_duration':100_000_000,'eval_count':tokens,'eval_duration':1_000_000_000}
    return (c.canonical(first)+c.canonical(last)).encode()


def seal(folder):
    files=[{'path':p.relative_to(folder).as_posix(),'bytes':p.stat().st_size,'sha256':c.file_sha(p)} for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='EVIDENCE_MANIFEST.json']
    c.write(folder/'EVIDENCE_MANIFEST.json',{'schema':'alice.mc10d.qwen.result-bundle-manifest.v1','run_id':c.RUN_ID,'files':files})


@contextlib.contextmanager
def quiet():
    import io
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):yield


def simulation(root, *, wrong_task=None, truncated_task=None, slow=False, telemetry_fail=False, package_sha=FAKE_PACKAGE):
    """Execute the real worker lifecycle with fake runtime and model responses."""
    approved,tasks,policy=c.authority()
    run=root/'run';run.mkdir()
    helper=root/'bin/rayan-telemetry-push.sh';helper.parent.mkdir();helper.write_text('fixture');helper.chmod(0o700)
    manifest={'schemaVersion':2,'config':{'digest':'sha256:'+c.sha(b'config'),'size':6},'layers':[{'digest':'sha256:'+c.sha(b'model'),'size':5}]}
    manifest_raw=c.canonical(manifest).encode();digest=c.sha(manifest_raw)
    lock={'schema':'alice.mc10d.qwen.model-lock.v1','model_tag':c.TAG,'full_manifest_digest':digest,'source_url':'https://registry.ollama.ai/v2/library/qwen3.8/manifests/27b-q4_K_M','registry_header_digest':'sha256:'+digest,'resolved_at':c.now(),'manifest_sha256':digest,'approved_amendment_sha256':c.APPROVED_SHA,'inference_started':False}
    model={'name':c.TAG,'model':c.TAG,'digest':digest,'details':{'family':'qwen35','parameter_size':'27.3B','quantization_level':'Q4_K_M'},'size_vram':0,'context_length':8192}
    def runtime(run,policy,deadline):
        c.write(run/'runtime-files.json',{'files':[{'path':'bin/ollama','sha256':policy['binary_sha256'],'bytes':1}]})
        c.write(run/'runtime-decoder.json',{'backend':'system_tar_zstd','pinned_runtime_archive_verified':True})
        return run/'not-a-real-binary',{'archive_sha256':policy['archive_sha256'],'binary_sha256':policy['binary_sha256'],'archive_cache_status':'VERIFIED_CACHE','runtime_files_sha256':c.file_sha(run/'runtime-files.json'),'runtime_decoder_sha256':c.file_sha(run/'runtime-decoder.json')}
    def prepare_model(run,policy,deadline):
        c.write(run/'model-lock.json',lock)
        c.atomic_bytes(run/'model-manifest.json',manifest_raw)
        c.write(run/'model-blobs.json',{'manifest_sha256':digest,'blobs':[{'digest':x['digest'],'bytes':x['size'],'status':'VERIFIED_CACHE'} for x in [manifest['config']]+manifest['layers']]})
        return root/'no-real-model',lock
    def api(base,endpoint,body=None,timeout=60):
        if endpoint in ('/api/tags','/api/ps'):return {'models':[model]}
        if endpoint=='/api/show':return {'details':model['details'],'capabilities':['completion','thinking']}
        raise AssertionError('unexpected endpoint '+endpoint)
    calls=[]
    def stream(base,body,path,deadline):
        calls.append(body)
        if path.name=='probe-response.ndjson':
            raw=raw_response({'numbers':[1,2]},reason='length')
            if slow:
                chunks=[c.strict(x) for x in raw.splitlines()];chunks[-1]['eval_duration']=200_000_000_000
                raw=''.join(c.canonical(x) for x in chunks).encode()
        else:
            task=next(t for t in tasks if t['task_id']==path.parent.name)
            answer={**task['gold'],'rationale':'Synthetic fixture only.'}
            if task['task_id']==wrong_task:answer.update(verdict='PASS',critical_veto=False)
            raw=raw_response(answer,reason='length' if task['task_id']==truncated_task else 'stop')
        c.atomic_bytes(path,raw)
    proc=mock.Mock();proc.poll.return_value=None
    def fake_process(argv,**kwargs):
        if str(argv[0])!=str(helper):raise AssertionError('unexpected real subprocess in worker simulation')
        return subprocess.CompletedProcess(argv,1 if telemetry_fail and argv[3]=='COMPLETED' else 0)
    with mock.patch.object(c,'ROOT',root),mock.patch.object(c,'REMOTE_RUN',run),mock.patch.dict(os.environ,{'SLURM_JOB_ID':'12345','SLURM_CPUS_PER_TASK':'20','SLURM_JOB_PARTITION':'node','SLURM_MEM_PER_NODE':'49152','SLURM_JOB_GPUS':''}),mock.patch.object(worker.os,'sched_getaffinity',return_value=set(range(20)),create=True),mock.patch.object(worker,'prepare_runtime',side_effect=runtime),mock.patch.object(worker,'prepare_model',side_effect=prepare_model),mock.patch.object(worker,'start_service',return_value=(proc,'http://fixture.invalid',policy['version'])),mock.patch.object(worker,'api',side_effect=api),mock.patch.object(worker,'stream_request',side_effect=stream),mock.patch.object(worker.subprocess,'run',side_effect=fake_process),quiet():
        code=worker.work(run,package_sha)
    c.write(run/'run.json',{'run_id':c.RUN_ID,'package_sha256':package_sha,'amendment_sha256':c.APPROVED_SHA})
    c.write(run/'submission.json',{'job_id':'12345','package_sha256':package_sha})
    c.write(run/'scheduler-status.json',{'run_id':c.RUN_ID,'job_id':'12345','terminal':True,'scheduler_state':'COMPLETED' if code==0 else 'FAILED'})
    (run/'authority').mkdir()
    for p in (c.BASE/'authority').iterdir():shutil.copyfile(p,run/'authority'/p.name)
    shutil.copyfile(c.BASE/'PACKAGE_MANIFEST.json',run/'package-manifest.json')
    seal(run)
    return run,code,calls


class ContractTests(unittest.TestCase):
    def setUp(self):self.approved,self.tasks,self.policy=c.authority()
    def test_owner_approval_preserves_ungranted_authority(self):
        self.assertTrue(self.approved['owner_ratified'])
        self.assertFalse(self.approved['qualification_granted'])
        self.assertEqual(self.approved['required_independent_judge_families'],4)
        self.assertEqual(self.policy['binary_sha256'],'eb99a47aad366636488ebd9c163a9180254dffcf dfe359939f9aabc36e2399c8'.replace(' ',''))
    def test_task_requests_keep_gold_out_and_seeds_unique(self):
        seeds=[]
        for i,task in enumerate(self.tasks,1):
            body=c.request(task,i,self.approved)
            doc=c.strict(body['messages'][1]['content'])
            self.assertNotIn('gold',doc)
            self.assertEqual(set(doc['fictional_case']),{'fixed_core','target','candidate_behavior'})
            seeds.append(body['options']['seed'])
        self.assertEqual(seeds,list(range(9101,9117)))
    def test_rejects_short_model_digest(self):
        with self.assertRaises(c.Stop):c.full_digest('25b843619e94')
    def test_rejects_duplicate_json_keys_and_nonfinite(self):
        for raw in ('{"x":1,"x":2}','{"x":NaN}'):
            with self.assertRaises(c.Stop):c.strict(raw)
    def test_boolean_integer_confusion_rejected(self):
        value={**self.tasks[0]['gold'],'rationale':'fixture','critical_veto':0}
        self.assertEqual(c.classify(raw_response(value))['status'],'FORMAT_FAILED')
    def test_truncated_response_cannot_pass_even_if_json_valid(self):
        value={**self.tasks[0]['gold'],'rationale':'fixture'}
        self.assertEqual(c.classify(raw_response(value,reason='length'))['status'],'INCOMPLETE')
    def test_rationale_not_silently_truncated(self):
        value={**self.tasks[0]['gold'],'rationale':'x'*501}
        self.assertEqual(c.classify(raw_response(value))['status'],'FORMAT_FAILED')
    def test_duplicate_final_chunk_rejected(self):
        raw=raw_response({**self.tasks[0]['gold'],'rationale':'fixture'})
        self.assertEqual(c.classify(raw+raw.splitlines()[-1]+b'\n')['status'],'INCOMPLETE')
    def test_missing_finish_and_missing_metrics_rejected(self):
        raw=raw_response({**self.tasks[0]['gold'],'rationale':'fixture'})
        self.assertEqual(c.classify(raw.splitlines()[0]+b'\n')['status'],'INCOMPLETE')
        last=c.strict(raw.splitlines()[-1]);last.pop('eval_count')
        self.assertEqual(c.classify(c.canonical(last).encode())['status'],'INCOMPLETE')
    def test_hard_anchor_failure_cannot_hide_in_high_total(self):
        rows=[{'task_id':t['task_id'],'status':'COMPLETE','response':{**t['gold'],'rationale':'fixture'}} for t in self.tasks]
        next(r for r in rows if r['task_id']=='Q04_FAKE_SOURCE_HISTORY')['response'].update(verdict='PASS',critical_veto=False)
        result=c.score(rows,self.tasks)
        self.assertEqual(result['verdict_matches'],15)
        self.assertEqual(result['critical_decision_matches'],6)
        self.assertFalse(result['qualification_passed'])
    def test_secondary_fields_remain_diagnostic(self):
        rows=[{'task_id':t['task_id'],'status':'COMPLETE','response':{**t['gold'],'rationale':'fixture'}} for t in self.tasks]
        rows[1]['response']['actor_role_direction_correct']=not rows[1]['response']['actor_role_direction_correct']
        self.assertTrue(c.score(rows,self.tasks)['qualification_passed'])
    def test_duplicate_tasks_rejected(self):
        row={'task_id':self.tasks[0]['task_id'],'status':'COMPLETE','response':{**self.tasks[0]['gold'],'rationale':'fixture'}}
        with self.assertRaises(c.Stop):c.score([row,row],self.tasks)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def test_full_worker_round_trip_and_no_successor_grant(self):
        folder,code,calls=simulation(self.root)
        self.assertEqual(code,0);self.assertEqual(len(calls),17)
        summary=evidence.analyze(folder,FAKE_PACKAGE)
        self.assertTrue(summary['qualification_passed'])
        self.assertTrue(summary['calibration_only'])
        self.assertFalse(any(summary['no_authority'].values()))
    def test_semantic_failure_preserves_all_sixteen_responses(self):
        folder,code,calls=simulation(self.root,wrong_task='Q04_FAKE_SOURCE_HISTORY')
        self.assertEqual(code,76);self.assertEqual(len(calls),17)
        summary=evidence.analyze(folder,FAKE_PACKAGE)
        self.assertEqual(summary['scores']['verdict_matches'],15)
        self.assertFalse(summary['qualification_passed'])
        self.assertEqual(len(list(folder.glob('tasks/*/response.ndjson'))),16)
    def test_token_exhaustion_is_preserved_without_resampling(self):
        folder,code,calls=simulation(self.root,truncated_task='Q04_FAKE_SOURCE_HISTORY')
        summary=evidence.analyze(folder,FAKE_PACKAGE)
        self.assertEqual(code,76);self.assertEqual(len(calls),17)
        self.assertEqual(summary['scores']['tasks_complete'],15)
        self.assertFalse(summary['qualification_passed'])
    def test_slow_cpu_stops_before_any_calibration_task(self):
        folder,code,calls=simulation(self.root,slow=True)
        summary=evidence.analyze(folder,FAKE_PACKAGE)
        self.assertEqual(code,76);self.assertEqual(len(calls),1)
        self.assertEqual(summary['scores']['tasks_complete'],0)
        self.assertFalse((folder/'tasks').exists())
    def test_final_telemetry_failure_does_not_destroy_results(self):
        folder,code,calls=simulation(self.root,telemetry_fail=True)
        self.assertEqual(code,86)
        self.assertTrue(evidence.analyze(folder,FAKE_PACKAGE)['qualification_passed'])
        self.assertEqual(c.read(folder/'telemetry-publication.json')['exit_code'],1)
        self.assertFalse(evidence.analyze(folder,FAKE_PACKAGE)['final_telemetry_published'])
    def test_terminal_telemetry_can_recover_without_inference(self):
        folder,code,calls=simulation(self.root,telemetry_fail=True)
        before=(folder/'result.json').read_bytes();count=len(calls)
        state=c.read(folder/'scheduler-status.json')
        with mock.patch.object(remote.subprocess,'run',return_value=subprocess.CompletedProcess([],0)) as push:
            remote.repair_terminal_telemetry(folder,state)
        self.assertEqual(push.call_count,1)
        self.assertEqual(len(calls),count);self.assertEqual((folder/'result.json').read_bytes(),before)
        seal(folder)
        self.assertTrue(evidence.analyze(folder,FAKE_PACKAGE)['final_telemetry_published'])
    def test_mutated_request_rejected_even_with_updated_local_hashes(self):
        folder,_,_=simulation(self.root)
        td=next((folder/'tasks').iterdir())
        body=c.read(td/'request.json');body['options']['temperature']=0
        c.write(td/'request.json',body)
        for name in ('attempt.json','record.json'):
            obj=c.read(td/name);obj['request_sha256']=c.file_sha(td/'request.json');c.write(td/name,obj)
        seal(folder)
        with self.assertRaises(c.Stop):evidence.analyze(folder,FAKE_PACKAGE)
    def test_mutated_gold_or_authority_rejected(self):
        folder,_,_=simulation(self.root)
        doc=c.read(folder/'authority/tasks.json');doc['tasks'][0]['gold']['verdict']='REJECT'
        c.write(folder/'authority/tasks.json',doc);seal(folder)
        with self.assertRaises(c.Stop):evidence.analyze(folder,FAKE_PACKAGE)
    def test_forged_worker_scores_rejected(self):
        folder,_,_=simulation(self.root,wrong_task='Q04_FAKE_SOURCE_HISTORY')
        doc=c.read(folder/'result.json');doc['qualification_passed']=True
        c.write(folder/'result.json',doc);seal(folder)
        with self.assertRaises(c.Stop):evidence.analyze(folder,FAKE_PACKAGE)
    def test_raw_stream_mutation_rejected(self):
        folder,_,_=simulation(self.root)
        target=next(folder.glob('tasks/*/response.ndjson'));target.write_bytes(target.read_bytes()+b'{')
        seal(folder)
        with self.assertRaises(c.Stop):evidence.analyze(folder,FAKE_PACKAGE)
    def test_missing_finalization_cannot_grant_qualification(self):
        folder,_,_=simulation(self.root);(folder/'worker-finished.json').unlink();seal(folder)
        with self.assertRaises(c.Stop):evidence.analyze(folder,FAKE_PACKAGE)
    def test_result_zip_manifest_and_traversal(self):
        folder,_,_=simulation(self.root)
        archive=self.root/'result.zip'
        with zipfile.ZipFile(archive,'w') as z:
            for p in folder.rglob('*'):
                if p.is_file():z.write(p,p.relative_to(folder).as_posix())
        extracted=evidence.extract_verified(archive,c.file_sha(archive),self.root/'extracted')
        self.assertTrue(evidence.analyze(extracted,FAKE_PACKAGE)['qualification_passed'])
        evil=self.root/'evil.zip'
        with zipfile.ZipFile(evil,'w') as z:z.writestr('../escape','bad')
        with self.assertRaises(c.Stop):evidence.extract_verified(evil,c.file_sha(evil),self.root/'evil-out')


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def test_download_reuses_only_exact_cache(self):
        cached=self.root/'artifact';cached.write_bytes(b'good')
        with mock.patch.object(worker.urllib.request,'urlopen',side_effect=AssertionError('network must not be called')):
            self.assertEqual(worker.download('https://unused.invalid',cached,c.sha(b'good'),4,time.monotonic()+10),'VERIFIED_CACHE')
            with self.assertRaises(c.Stop):worker.download('https://unused.invalid',cached,'0'*64,4,time.monotonic()+10)
        self.assertEqual(cached.read_bytes(),b'good')
    def test_real_loopback_stream_preserves_full_bytes(self):
        _,tasks,_=c.authority();raw=raw_response({**tasks[0]['gold'],'rationale':'fixture'})
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(200);self.end_headers();self.wfile.write(raw)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            target=self.root/'raw.ndjson'
            worker.stream_request('http://127.0.0.1:'+str(server.server_port),{},target,time.monotonic()+10)
            self.assertEqual(target.read_bytes(),raw)
            self.assertEqual(c.classify(target.read_bytes())['status'],'COMPLETE')
        finally:server.shutdown();server.server_close();thread.join()
    def test_inflight_transport_failure_never_retries_task(self):
        approved,tasks,_=c.authority();calls=[]
        def fail(base,body,path,deadline):
            calls.append(body);path.write_bytes(b'');raise TimeoutError('fixture')
        with mock.patch.object(worker,'stream_request',side_effect=fail),mock.patch.object(worker,'runtime_check',return_value={}):
            result=worker.run_task(self.root,'http://fixture.invalid',tasks[0],1,approved,{'full_manifest_digest':'2'*64},'3'*64,time.monotonic()+10)
            self.assertEqual(result['status'],'INCOMPLETE')
            before=(self.root/'tasks'/tasks[0]['task_id']/'attempt.json').read_bytes()
            with self.assertRaises(c.Stop):worker.run_task(self.root,'http://fixture.invalid',tasks[0],1,approved,{'full_manifest_digest':'2'*64},'3'*64,time.monotonic()+10)
            self.assertEqual(len(calls),1)
            self.assertEqual((self.root/'tasks'/tasks[0]['task_id']/'attempt.json').read_bytes(),before)
    def test_cpu_proof_and_digest_drift_fail_closed(self):
        model={'name':c.TAG,'digest':'2'*64,'details':{'quantization_level':'Q4_K_M'},'size_vram':1,'context_length':8192}
        with mock.patch.object(worker,'api',return_value={'models':[model]}):
            with self.assertRaises(c.Stop):worker.runtime_check('x',{'full_manifest_digest':'2'*64},loaded=True)
            model['size_vram']=0;model['digest']='3'*64
            with self.assertRaises(c.Stop):worker.runtime_check('x',{'full_manifest_digest':'2'*64},loaded=True)
    def test_controller_lock_rejects_concurrent_process_and_releases(self):
        with controller.controller_lock(self.root/'lock'):
            with self.assertRaises((OSError,c.Stop)):
                with controller.controller_lock(self.root/'lock'):pass
        with controller.controller_lock(self.root/'lock'):pass
    def _submission(self,lose_response,discover_after):
        run=self.root/'run'
        helper=self.root/'bin/rayan-telemetry-push.sh';helper.parent.mkdir();helper.write_text('fixture');helper.chmod(0o700)
        (self.root/'telemetry/ledger/.git').mkdir(parents=True)
        calls=[];jobs=[]
        def command(argv,check=True):
            if argv[0]=='sbatch':
                calls.append(argv)
                if discover_after:jobs.append('34567')
                return subprocess.CompletedProcess(argv,1 if lose_response else 0,'unacknowledged' if lose_response else '34567\n','')
            raise AssertionError('unexpected scheduler command')
        with mock.patch.object(c,'ROOT',self.root),mock.patch.object(c,'REMOTE_RUN',run),mock.patch.object(remote,'submission_lock',controller.controller_lock),mock.patch.object(remote,'find_submitted',side_effect=lambda:jobs[0] if jobs else None),mock.patch.object(remote,'cmd',side_effect=command),mock.patch.object(remote.shutil,'which',return_value='/fixture/tool'):
            if lose_response:
                with self.assertRaises(c.Stop):remote.submit(run,FAKE_PACKAGE)
            else:remote.submit(run,FAKE_PACKAGE)
            if lose_response and not discover_after:
                with self.assertRaises(c.Stop):remote.submit(run,FAKE_PACKAGE)
            else:
                receipt=remote.submit(run,FAKE_PACKAGE);self.assertEqual(receipt['job_id'],'34567')
            self.assertEqual(len(calls),1)
    def test_repeated_launcher_reattaches_one_submission(self):self._submission(False,False)
    def test_lost_sbatch_ack_recovered_without_second_job(self):self._submission(True,True)
    def test_unknown_submission_never_resubmitted(self):self._submission(True,False)


class PublicationTests(unittest.TestCase):
    def test_public_payload_omits_rationales_and_is_typed(self):
        with tempfile.TemporaryDirectory() as temp:
            folder,_,_=simulation(Path(temp))
            payloads,summary=pub.public_payloads(folder,FAKE_PACKAGE)
            raw=b''.join(payloads.values())
            self.assertNotIn(b'Synthetic fixture only.',raw)
            self.assertNotIn(b'controller-diagnostics',raw)
            self.assertFalse(summary['no_authority']['four_family_binding_created'])
    @unittest.skipUnless(shutil.which('git'),'Git not installed')
    def test_isolated_git_publication_idempotent_and_preserves_concurrent_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);bare=root/'origin.git';work=root/'work';work.mkdir()
            def git(path,*args):
                cp=subprocess.run(['git','-C',str(path),*args],capture_output=True,check=True)
                return cp.stdout.decode().strip()
            subprocess.run(['git','init','--bare',str(bare)],capture_output=True,check=True)
            git(work,'init');git(work,'config','user.name','Fixture');git(work,'config','user.email','fixture@example.invalid')
            (work/'keep.txt').write_text('concurrent user edit')
            git(work,'add','keep.txt');git(work,'commit','-m','fixture');git(work,'branch','-M',pub.BRANCH)
            git(work,'remote','add','origin',str(bare));git(work,'push','origin',pub.BRANCH)
            parent=git(work,'rev-parse','HEAD')
            payload={pub.PREFIX+'/fixture/summary.json':b'{"fixture":true}\n',pub.PREFIX+'/LATEST_QWEN_PUBLIC.json':b'{"fixture":true}\n'}
            with quiet():
                first=pub.append_commit(str(bare),payload,'fixture\n')
                second=pub.append_commit(str(bare),payload,'fixture\n')
            self.assertEqual(first['commit'],second['commit'])
            self.assertEqual(second['status'],'ALREADY_PRESENT')
            self.assertEqual(git(work,'rev-parse','HEAD'),parent)
            self.assertEqual(git(bare,'show',pub.BRANCH+':keep.txt'),'concurrent user edit')
            altered={pub.PREFIX+'/fixture/summary.json':b'changed'}
            with quiet(),self.assertRaises(pub.PublishError):pub.append_commit(str(bare),altered,'fixture\n')


class ControllerTests(unittest.TestCase):
    def test_connection_loss_then_collection_then_publication_retry_reuses_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);package=root/'fixture-package.zip';package.write_bytes(b'synthetic package transfer fixture')
            package_sha=c.file_sha(package)
            source=root/'source';source.mkdir()
            folder,_,calls=simulation(source,package_sha=package_sha)
            bundle=root/'result.zip'
            with zipfile.ZipFile(bundle,'w') as z:
                for p in folder.rglob('*'):
                    if p.is_file():z.write(p,p.relative_to(folder).as_posix())
            receipt={'run_id':c.RUN_ID,'remote_path':str(c.REMOTE_RUN/'result-bundle.zip'),'zip_sha256':c.file_sha(bundle),'bytes':bundle.stat().st_size}
            events=[];fail_status=[True];fail_publish=[True]
            class FakeMagnolia:
                def __init__(self,*args):pass
                def stage(self,*args):events.append('stage')
                def action(self,action):
                    events.append(action)
                    if action=='submit':return {'run_id':c.RUN_ID,'job_id':'12345'}
                    if action=='status':
                        if fail_status[0]:
                            fail_status[0]=False
                            raise controller.TransportPending('synthetic lost connection')
                        return c.read(folder/'scheduler-status.json')
                    if action=='collect':return receipt
                    raise AssertionError('unexpected action')
                def download(self,info,path):events.append('download');shutil.copyfile(bundle,path)
            def publish(folder,package_sha,repo):
                if fail_publish[0]:
                    fail_publish[0]=False
                    raise pub.PublishError('synthetic publication outage')
                return {'commit':'f'*40},evidence.analyze(folder,package_sha)
            args=SimpleNamespace(package_zip=package,package_sha=package_sha,vault_root=root/'vault',repo_root=root/'repo',output_root=root/'downloads',ssh_key=root/'key')
            args.repo_root.mkdir();sentinel=args.repo_root/'do-not-change.txt';sentinel.write_text('user changes')
            with mock.patch.object(controller,'Magnolia',FakeMagnolia),mock.patch.object(controller,'validate_local'),mock.patch.object(pub,'publish',side_effect=publish),quiet():
                with self.assertRaises(controller.TransportPending):controller.execute(args)
                self.assertEqual(controller.execute(args),75)
                after_collection=list(events)
                self.assertEqual(controller.execute(args),0)
            self.assertEqual(events,after_collection)
            self.assertEqual(events.count('stage'),1)
            self.assertEqual(events.count('download'),1)
            self.assertEqual(len(calls),17)
            self.assertEqual(sentinel.read_text(),'user changes')
            self.assertEqual(len(list(args.output_root.glob('ALICE_QWEN_PUBLIC_RESULT_*.zip'))),1)
            state=c.read(args.vault_root/'tools/alice-astra/qwen-fallback-a1/controller-state.json')
            self.assertEqual(state['phase'],'COMPLETE')


if __name__=='__main__':
    unittest.main(verbosity=2)
