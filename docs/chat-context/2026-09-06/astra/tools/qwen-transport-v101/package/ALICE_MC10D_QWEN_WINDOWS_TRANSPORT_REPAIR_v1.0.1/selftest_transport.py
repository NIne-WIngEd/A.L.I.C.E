"""Regression tests for a Windows client talking to a POSIX server. No remote calls."""
from __future__ import annotations

import argparse
import contextlib
import io
import os
from pathlib import Path,PureWindowsPath,PurePosixPath
import shlex
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import runner as r


class TransportRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace=tempfile.TemporaryDirectory(prefix='alice-v101-tests-')
        cls.base=Path(cls.workspace.name)
        cls.workload,cls.archive=r.prepare_workload(cls.base/'workload')
        cls.c,cls.controller=r.load_workload(cls.workload)
    @classmethod
    def tearDownClass(cls):cls.workspace.cleanup()
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='alice-v101-case-')
        self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    @contextlib.contextmanager
    def windows_client(self):
        with mock.patch.object(self.c,'ROOT',PureWindowsPath(r.REMOTE_ROOT)),mock.patch.object(self.c,'REMOTE_RUN',PureWindowsPath(r.REMOTE_RUN)):
            yield
    def test_original_windows_representation_reproduces_both_bad_boundaries(self):
        with self.windows_client():
            m=self.controller.Magnolia(PureWindowsPath('C:/Users/rayns/.ssh/key'),r.WORKLOAD_SHA)
            self.assertTrue(m.directory.startswith('\\homes\\'))
            self.assertNotEqual(str(self.c.REMOTE_RUN/'result-bundle.zip'),r.REMOTE_RUN+'/result-bundle.zip')
    def test_repair_uses_posix_only_for_remote_addresses(self):
        local=PureWindowsPath('C:/Users/rayns/My Downloads/file.zip')
        with self.windows_client():
            receipt=r.adapt_remote_paths(self.c)
            self.assertIs(type(self.c.ROOT),PurePosixPath)
            self.assertEqual(str(self.c.ROOT),r.REMOTE_ROOT)
            self.assertEqual(str(self.c.REMOTE_RUN),r.REMOTE_RUN)
            self.assertEqual(str(local),r'C:\Users\rayns\My Downloads\file.zip')
            self.assertFalse(receipt['disk_source_files_modified'])
            self.assertFalse(receipt['controller_state_identity_changed'])
    def test_actual_upload_method_emits_posix_mkdir_scp_and_move(self):
        seen=[]
        with self.windows_client():
            r.adapt_remote_paths(self.c)
            key=PureWindowsPath('C:/Users/rayns/.ssh/rayan_magnolia_ed25519')
            local=PureWindowsPath('C:/Users/rayns/My Downloads/'+r.WORKLOAD_NAME+'.zip')
            m=self.controller.Magnolia(key,r.WORKLOAD_SHA)
            def execute(argv,**kwargs):
                seen.append(argv)
                return subprocess.CompletedProcess(argv,0,b'',b'')
            with mock.patch.object(m,'execute',side_effect=execute):m.stage(local)
            directory=r.REMOTE_ROOT+'/packages/'+r.RUN_ID
            self.assertEqual(shlex.split(seen[0][-1]),['mkdir','-p','--',directory])
            self.assertEqual(seen[1][0],'scp')
            self.assertEqual(seen[1][-2],str(local))
            target=seen[1][-1].split(':',1)[1]
            self.assertTrue(target.startswith(directory+'/'+r.WORKLOAD_SHA+'.zip.upload-'))
            self.assertNotIn('\\',target)
            self.assertEqual(shlex.split(seen[2][-1]),['mv','--',target,directory+'/'+r.WORKLOAD_SHA+'.zip'])
            self.assertEqual(seen[1][seen[1].index('-i')+1],str(key))
    def test_posix_server_download_receipt_was_rejected_before_repair(self):
        raw=b'fixture result';destination=self.root/'result.zip'
        receipt={'remote_path':r.REMOTE_RUN+'/result-bundle.zip','zip_sha256':self.c.sha(raw),'bytes':len(raw)}
        with self.windows_client():
            m=self.controller.Magnolia(PureWindowsPath('C:/key'),r.WORKLOAD_SHA)
            with mock.patch.object(m,'execute') as transfer:
                with self.assertRaises(self.c.Stop):m.download(receipt,destination)
                transfer.assert_not_called()
    def test_actual_download_accepts_posix_server_and_verifies_bytes(self):
        raw=b'fixture result';destination=self.root/'result.zip'
        receipt={'remote_path':r.REMOTE_RUN+'/result-bundle.zip','zip_sha256':self.c.sha(raw),'bytes':len(raw)}
        with self.windows_client():
            r.adapt_remote_paths(self.c)
            m=self.controller.Magnolia(PureWindowsPath('C:/key'),r.WORKLOAD_SHA)
            def execute(argv,**kwargs):
                self.assertEqual(argv[-2],self.controller.HOST+':'+r.REMOTE_RUN+'/result-bundle.zip')
                self.assertEqual(argv[-1],str(destination))
                destination.write_bytes(raw)
                return subprocess.CompletedProcess(argv,0,b'',b'')
            with mock.patch.object(m,'execute',side_effect=execute):m.download(receipt,destination)
            self.assertEqual(destination.read_bytes(),raw)
    def test_wrong_server_path_is_not_normalized_into_acceptance(self):
        with self.windows_client():
            r.adapt_remote_paths(self.c)
            m=self.controller.Magnolia(PureWindowsPath('C:/key'),r.WORKLOAD_SHA)
            receipt={'remote_path':'/tmp/unrelated.zip','zip_sha256':'0'*64,'bytes':0}
            with mock.patch.object(m,'execute') as transfer:
                with self.assertRaises(self.c.Stop):m.download(receipt,self.root/'result.zip')
                transfer.assert_not_called()
    def test_unexpected_root_and_run_are_refused(self):
        for root,run in [('C:/homes/01/mxrayan/rayan-compute',r.REMOTE_RUN),(r.REMOTE_ROOT,'/tmp/unrelated')]:
            module=SimpleNamespace(ROOT=PureWindowsPath(root),REMOTE_RUN=PureWindowsPath(run),RUN_ID=r.RUN_ID)
            with self.assertRaises(RuntimeError):r.adapt_remote_paths(module)
    def test_original_workload_remains_byte_exact_after_repair(self):
        before={p.relative_to(self.workload).as_posix():r.digest(p) for p in self.workload.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        with self.windows_client():r.adapt_remote_paths(self.c)
        root,_=r.prepare_workload(self.workload.parent,self.archive)
        after={p.relative_to(root).as_posix():r.digest(p) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        self.assertEqual(before,after)
        self.assertEqual(r.digest(self.archive),r.WORKLOAD_SHA)
        self.assertEqual(r.digest(root/'PACKAGE_MANIFEST.json'),r.WORKLOAD_MANIFEST_SHA)
    def test_altered_workload_zip_stops_before_extracting(self):
        bad=self.root/'bad.zip';bad.write_bytes(self.archive.read_bytes()+b'changed')
        target=self.root/'uncreated'
        with self.assertRaises(RuntimeError):r.prepare_workload(target,bad)
        self.assertFalse(target.exists())
    def test_reuses_prepared_or_submitted_state_without_identity_migration(self):
        for phase in ('PREPARED','SUBMITTED'):
            with self.subTest(phase=phase),self.windows_client():
                r.adapt_remote_paths(self.c)
                vault=self.root/phase
                statefile=vault/'tools/alice-astra/qwen-fallback-a1/controller-state.json'
                existing={'run_id':r.RUN_ID,'package_sha256':r.WORKLOAD_SHA,'phase':phase,'created_at':'2026-09-06T22:00:00Z','owner_sentinel':'preserve'}
                if phase=='SUBMITTED':existing.update(staged=True,job_id='23456')
                self.c.write(statefile,existing)
                args=argparse.Namespace(package_zip=self.archive,package_sha=r.WORKLOAD_SHA,vault_root=vault,repo_root=self.root/'repo',output_root=self.root/'downloads',ssh_key=self.root/'key')
                stages=[];actions=[];test=self
                class FakeMagnolia:
                    def __init__(self,key,package_sha,diagnostics=None):
                        test.assertEqual(package_sha,r.WORKLOAD_SHA)
                        test.assertEqual(str(test.c.ROOT),r.REMOTE_ROOT)
                    def stage(self,archive):stages.append(archive)
                    def action(self,action):
                        actions.append(action)
                        if action=='submit':return {'run_id':r.RUN_ID,'job_id':'23456'}
                        raise test.controller.TransportPending('fixture monitoring interruption')
                with mock.patch.object(self.controller,'Magnolia',FakeMagnolia),mock.patch.object(self.controller,'validate_local'),contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(self.controller.TransportPending):self.controller.execute(args)
                state=self.c.read(statefile)
                self.assertEqual(state['package_sha256'],r.WORKLOAD_SHA)
                self.assertEqual(state['run_id'],r.RUN_ID)
                self.assertEqual(state['job_id'],'23456')
                self.assertEqual(state['owner_sentinel'],'preserve')
                self.assertEqual(len(stages),1 if phase=='PREPARED' else 0)
                self.assertEqual(actions,['submit','status'])
    @unittest.skipUnless(os.name=='posix' and shutil.which('bash'),'actual POSIX shell reproduction is exercised in the Linux release build')
    def test_real_shell_before_and_after_on_distinct_client_server_paths(self):
        expected=self.root/'directory with spaces'/'packages'
        windows=PureWindowsPath(expected.as_posix())
        old=shlex.join(['mkdir','-p','--',str(windows)])
        cp=subprocess.run(['bash','-c',old],cwd=self.root,capture_output=True,timeout=10)
        self.assertEqual(cp.returncode,0)
        self.assertFalse(expected.exists())
        fixed=shlex.join(['mkdir','-p','--',str(PurePosixPath(windows.as_posix()))])
        cp=subprocess.run(['bash','-c',fixed],cwd=self.root,capture_output=True,timeout=10)
        self.assertEqual(cp.returncode,0)
        self.assertTrue(expected.is_dir())


if __name__=='__main__':unittest.main(verbosity=2)
