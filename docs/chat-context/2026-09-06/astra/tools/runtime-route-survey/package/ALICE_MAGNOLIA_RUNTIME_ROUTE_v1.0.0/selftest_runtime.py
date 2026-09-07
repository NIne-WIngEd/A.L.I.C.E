import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import zipfile

import inspect_runtime as survey
import collect_transport as transport


class RuntimeSurveyTests(unittest.TestCase):
    def test_six_query_receipts_preserve_missing_tools_and_source_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'runs'/survey.RUN_ID;run.mkdir(parents=True)
            source=run/'result.json';source.write_bytes(b'closed result\n')
            rules={'source_result_zip_sha256':'1'*64,'source_files':{'result.json':transport.sha(source.read_bytes())}}
            calls=[]
            def capture(argv, timeout):
                calls.append(argv)
                return {'argv':argv,'status':'RETURNED','exit_code':127 if len(calls)==4 else 0,'stdout':transport.stream(b''),'stderr':transport.stream(b'raw unavailable tool\xff' if len(calls)==4 else b'')}
            output=io.StringIO()
            with mock.patch.object(survey,'ROOT',root),mock.patch.object(survey,'policy',return_value=rules),mock.patch.object(survey,'verify_runtime_files',return_value=['/fixture/verified-elf']),mock.patch.object(transport,'capture',side_effect=capture),contextlib.redirect_stdout(output):survey.remote()
            report=json.loads(output.getvalue().splitlines()[-1])['report']
            self.assertEqual(len(calls),6);self.assertEqual(report['command_failures'],1)
            self.assertTrue(report['selected_source_files_unchanged']);self.assertEqual(source.read_bytes(),b'closed result\n')
            self.assertEqual([c[0] for c in calls],['uname','getconf','readelf','bash','bash','bash'])
            self.assertEqual(report['commands'][3]['stderr'],transport.stream(b'raw unavailable tool\xff'))

    def test_changed_source_stops_before_queries_and_altered_elf_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'runs'/survey.RUN_ID;(run/'runtime/bin').mkdir(parents=True)
            source=run/'result.json';source.write_bytes(b'changed')
            binary=run/'runtime/bin/ollama';binary.write_bytes(b'\x7fELFchanged')
            rules={'source_result_zip_sha256':'1'*64,'source_files':{'result.json':'2'*64},'runtime_elf_files':[{'path':'bin/ollama','bytes':11,'sha256':'3'*64}]}
            with mock.patch.object(survey,'ROOT',root),mock.patch.object(survey,'policy',return_value=rules),mock.patch.object(transport,'capture') as command,contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):survey.verify_runtime_files()
                survey.remote();command.assert_not_called()

    def test_literal_stdin_program_and_failed_ssh_preserve_local_state_and_bytes(self):
        script=survey.remote_script().decode()
        body=script.split("<<'ALICE_RUNTIME_ROUTE_PY'\n",1)[1].rsplit('\nALICE_RUNTIME_ROUTE_PY',1)[0]
        compile(body,'literal-remote-program','exec')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);state=root/'tools/alice-astra/qwen-fallback-a3/controller-state.json';state.parent.mkdir(parents=True);state.write_bytes(b'{"closed":true}\n')
            calls=[]
            def fail(argv, **kwargs):
                calls.append((argv,kwargs))
                return subprocess.CompletedProcess(argv,255,b'partial remote bytes\x00',b'actual ssh failure\xff')
            with contextlib.redirect_stdout(io.StringIO()):result,archive=survey.collect(root/'output',root,command=fail)
            self.assertTrue(result['local_states_unchanged']);self.assertEqual(result['status'],'INCOMPLETE_TRANSPORT')
            self.assertEqual(calls[0][1]['input'],script.encode());self.assertEqual(calls[0][0][-1],'bash -s')
            key=Path(r'C:\Users\rayns\.ssh\rayan_magnolia_ed25519');argv=transport.ssh_argv(key)
            self.assertEqual(argv[argv.index('-i')+1],str(key))
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.read('ssh.stderr.bin'),b'actual ssh failure\xff')
                for item in json.loads(z.read('EVIDENCE_MANIFEST.json'))['files']:
                    raw=z.read(item['path']);self.assertEqual(transport.sha(raw),item['sha256'])
            self.assertEqual(state.read_bytes(),b'{"closed":true}\n')


if __name__=='__main__':unittest.main()
