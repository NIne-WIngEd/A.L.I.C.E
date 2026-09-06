"""Synthetic offline fixtures only. No model evaluation or remote service calls."""
from __future__ import annotations

from copy import deepcopy
import json
import contextlib
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import mc10d_forensics as f
import publish_context as pub


def fixture():
    tasks, spec = f.load_authority()
    rows = []
    for task in tasks:
        answer = dict(task['gold'], rationale='SYNTHETIC SELFTEST FIXTURE; not an observed GLM response.')
        if task['task_id'] == 'Q04_FAKE_SOURCE_HISTORY':
            answer.update(verdict='PASS', critical_veto=False)
        rows.append({'task_id': task['task_id'], 'gold': task['gold'], 'response': answer,
                     'critical_task': task['gold']['critical_veto'], 'private_data_used': False,
                     'verdict_match': answer['verdict'] == task['gold']['verdict'],
                     'full_gold_match': all(answer[k] == task['gold'][k] for k in f.GOLD_FIELDS)})
    scores = f.compute_scores(rows, tasks)[0]
    result = dict(scores, artifact_id='alice.MC10D.public-fictional-judge-role-qualification-result.v2-decision-centric',
                  family='glm', tag=spec['tag'], digest=spec['digest'], qualified_profile=spec['qualified_profile'],
                  worker_script_sha256=f.WORKER_SHA, private_MC10D_candidates_used=False, hidden_MC8_used=False,
                  A_SYN_acceptance_authority=False, judge_consensus_is_truth_evidence=False)
    payloads = {
        'failure': (f.BASE / 'authority/recorded_failure_receipt.json').read_bytes(),
        'result': f.canonical(result).encode(),
        'rows': ''.join(f.canonical(row) for row in rows).encode(),
        'runtime': f.canonical(dict(spec, family='glm', role=f.ROLE)).encode(),
    }
    return payloads, tasks, spec


def change_json(payloads, key, fn):
    obj = f.strict_json(payloads[key]); fn(obj); payloads[key] = f.canonical(obj).encode()


def change_rows(payloads, fn):
    rows = [f.strict_json(line) for line in payloads['rows'].splitlines()]
    fn(rows); payloads['rows'] = ''.join(f.canonical(row) for row in rows).encode()


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.payloads, self.tasks, self.spec = fixture()

    def analyze(self):
        return f.analyze(self.payloads, self.tasks, self.spec)

    def test_hard_anchor_failure_preserved_even_with_15_verdicts_and_6_critical(self):
        report = self.analyze()
        self.assertEqual(report['recomputed']['verdict_matches'], 15)
        self.assertEqual(report['recomputed']['critical_decision_matches'], 6)
        self.assertEqual(report['failed_gates'], ['all_five_hard_anchors_correct'])
        self.assertFalse(report['qualification_or_pointwise_authority_granted'])

    def test_duplicate_missing_unknown_task_ids(self):
        for alteration in (lambda r: r.__setitem__(15, r[0]), lambda r: r.pop(), lambda r: r[0].update(task_id='Q99_OTHER')):
            with self.subTest(alteration=alteration):
                self.payloads, _, _ = fixture(); change_rows(self.payloads, alteration)
                with self.assertRaises(f.EvidenceError): self.analyze()

    def test_gold_tampering_is_not_a_new_source_of_truth(self):
        change_rows(self.payloads, lambda rows: rows[3]['gold'].update(verdict='PASS', critical_veto=False))
        with self.assertRaisesRegex(f.EvidenceError, 'embedded gold'): self.analyze()

    def test_boolean_gold_cannot_be_integer(self):
        change_rows(self.payloads, lambda rows: rows[0]['gold'].update(critical_veto=0))
        with self.assertRaisesRegex(f.EvidenceError, 'embedded gold'): self.analyze()

    def test_exact_model_runtime_and_rendered_worker(self):
        for key, field, value in [('result','family','gemma'), ('runtime','tag','different'), ('result','digest','0'*64),
                                   ('runtime','qualified_profile',{'id':'thinking_on','think':True}),
                                   ('result','worker_script_sha256','931607aa75ac8fe16d7792acfe86be37aa0ba1e17d269170c677d0f8b132551d'),
                                   ('runtime','qualified_profile',{'id':'thinking_off','think':0})]:
            with self.subTest(key=key, field=field):
                self.payloads, _, _ = fixture(); change_json(self.payloads,key,lambda x:x.update({field:value}))
                with self.assertRaises(f.EvidenceError): self.analyze()

    def test_reordered_rows_are_scored_by_frozen_id(self):
        change_rows(self.payloads, lambda rows: rows.reverse())
        self.assertEqual(self.analyze()['recomputed']['verdict_matches'],15)

    def test_forged_counter_and_pass_cannot_override_rows(self):
        for field, value in [('verdict_matches',16), ('qualification_passed',True), ('tasks',True), ('critical_tasks',8)]:
            self.payloads, _, _ = fixture(); change_json(self.payloads,'result',lambda x:x.update({field:value}))
            with self.assertRaisesRegex(f.EvidenceError,'declared result'): self.analyze()

    def test_privacy_and_authority_flags_required(self):
        for field in ('private_MC10D_candidates_used','hidden_MC8_used','A_SYN_acceptance_authority'):
            self.payloads, _, _ = fixture(); change_json(self.payloads,'result',lambda x:x.update({field:True}))
            with self.assertRaises(f.EvidenceError): self.analyze()

    def test_stale_failure_and_duplicate_json_keys(self):
        self.payloads['failure'] += b' '
        with self.assertRaisesRegex(f.EvidenceError,'failure SHA256'): self.analyze()
        self.payloads, _, _ = fixture()
        self.payloads['result'] = self.payloads['result'].replace(b'{', b'{"family":"glm",',1)
        with self.assertRaisesRegex(f.EvidenceError,'duplicate JSON key'): self.analyze()

    def test_response_schema_and_self_reported_flags(self):
        for mutation in (lambda r:r[0]['response'].update(critical_veto=0), lambda r:r[0]['response'].update(extra='untrusted'),
                         lambda r:r[0].update(verdict_match=False), lambda r:r[0].update(private_data_used=True)):
            self.payloads, _, _ = fixture(); change_rows(self.payloads, mutation)
            with self.assertRaises(f.EvidenceError): self.analyze()

    def test_rationale_length_is_a_diagnostic_not_new_semantic_gate(self):
        change_rows(self.payloads,lambda rows:rows[0]['response'].update(rationale='x'*501))
        report=self.analyze()
        self.assertEqual(len(report['schema_diagnostics']),1)
        self.assertEqual(report['failed_gates'],['all_five_hard_anchors_correct'])

    def test_threshold_boundaries_and_no_upgrading_failure_to_pass(self):
        rows=[f.strict_json(x) for x in self.payloads['rows'].splitlines()]
        # One hard-anchor failure already exists; two further noncritical errors leave 13/16.
        for index in (7,11): rows[index]['response']['verdict']='HOLD'
        scores,gates,_,_=f.compute_scores(rows,self.tasks)
        self.assertEqual(scores['verdict_matches'],13)
        self.assertFalse(gates['at_least_14_verdict_matches'])
        rows[11]['response']['verdict']='PASS'
        self.assertTrue(f.compute_scores(rows,self.tasks)[1]['at_least_14_verdict_matches'])
        # Exact gold everywhere would pass, contradicting the recorded failure.
        for row in rows:
            row['response'].update(row['gold']); row['full_gold_match']=True; row['verdict_match']=True
        self.payloads['rows']=''.join(f.canonical(row) for row in rows).encode()
        change_json(self.payloads,'result',lambda x:x.update(f.compute_scores(rows,self.tasks)[0]))
        with self.assertRaisesRegex(f.EvidenceError,'conflicts with a recomputed pass'):self.analyze()


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.vault=self.root/'vault';self.out=self.root/'out'
        self.source=self.vault/f.WORK/f.DOWNLOAD;self.source.mkdir(parents=True)
        self.payloads,_,_=fixture()
        for key,data in self.payloads.items():(self.source/f.FILES[key]).write_bytes(data)
        (self.vault/'PRIVATE_CANDIDATE_SENTINEL.jsonl').write_text('must not be read')

    def test_read_only_collection_rerun_and_bundle_contents(self):
        original=f.plain_file;read_paths=[]
        def observed(path,*args):read_paths.append(path);return original(path,*args)
        with patch.object(f,'plain_file',side_effect=observed):
            first,bundle,report=f.collect(self.vault,self.out)
            second,_,_=f.collect(self.vault,self.out)
        self.assertNotEqual(first,second)
        self.assertTrue(all(p.parent==self.source for p in read_paths))
        for key,data in self.payloads.items():self.assertEqual((self.source/f.FILES[key]).read_bytes(),data)
        with zipfile.ZipFile(bundle) as archive:
            self.assertIn('LOCAL_RESPONSE_REVIEW.md',archive.namelist())
            self.assertFalse(any('PRIVATE_CANDIDATE' in p for p in archive.namelist()))
        public=pub.public_payloads(first)
        self.assertFalse(any(b'SYNTHETIC SELFTEST FIXTURE' in value for value in public.values()))
        self.assertFalse(any(str(self.vault).encode() in value for value in public.values()))

    def test_missing_evidence_does_not_create_output(self):
        (self.source/f.FILES['rows']).unlink()
        with self.assertRaises(f.EvidenceError):f.collect(self.vault,self.out)
        self.assertFalse(self.out.exists())

    def test_output_cannot_be_inside_source_vault(self):
        with self.assertRaisesRegex(f.EvidenceError,'separate'):f.collect(self.vault,self.vault/'out')

    def test_source_change_during_collection_stops(self):
        original=f.plain_file;calls=0
        def changed(path,*args):
            nonlocal calls
            calls+=1;value=original(path,*args)
            return value+b' ' if calls==5 else value
        with patch.object(f,'plain_file',side_effect=changed):
            with self.assertRaisesRegex(f.EvidenceError,'source changed'):f.collect(self.vault,self.out)
        self.assertFalse(self.out.exists())

    def test_publication_failure_preserves_collection_and_returns_75(self):
        args=['mc10d_forensics.py','--vault-root',str(self.vault),'--output-root',str(self.out),
              '--repo-root',str(self.root/'missing-repository'),'--publish-context']
        with patch('sys.argv',args), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code=f.main()
        self.assertEqual(code,75)
        self.assertEqual(len(list(self.out.glob('ALICE_V186_FORENSICS_*.zip'))),1)
        self.assertEqual(len(list(self.out.glob('*/CONTEXT_PUBLICATION_PENDING.json'))),1)

    def test_symlink_source_refused_where_supported(self):
        p=self.source/f.FILES['rows'];moved=self.root/'moved';p.rename(moved)
        try:p.symlink_to(moved)
        except OSError:
            self.skipTest('symlink creation unavailable for this user')
        with self.assertRaisesRegex(f.EvidenceError,'linked source'):f.collect(self.vault,self.out)


@unittest.skipUnless(shutil.which('git'),'Git absent; publisher unavailable but offline collector is usable')
class GitTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.remote=self.root/'remote.git';self.seed=self.root/'seed';self.seed.mkdir()
        pub.git(self.root,'init','--bare',str(self.remote))
        pub.git(self.seed,'init','-b',pub.BRANCH)
        pub.git(self.seed,'config','user.name','Synthetic selftest')
        pub.git(self.seed,'config','user.email','selftest@example.invalid')
        (self.seed/'README.md').write_text('preserve this context\n')
        pub.git(self.seed,'add','README.md');pub.git(self.seed,'commit','-m','synthetic fixture')
        pub.git(self.seed,'remote','add','origin',str(self.remote));pub.git(self.seed,'push','origin',pub.BRANCH)
        pub.git(self.seed,'push','origin','HEAD:refs/heads/main')
        self.main=pub.text(self.seed,'rev-parse','HEAD')
        self.path=pub.PREFIX+'/synthetic-test/abc/summary.json'
        self.payload={self.path:b'{"synthetic_test_only":true}\n',pub.PREFIX+'/LATEST_V186_GLM.json':b'{"test":true}\n'}

    def test_append_idempotence_and_main_preservation(self):
        result=pub.append_commit(str(self.remote),self.payload,'synthetic offline test\n')
        self.assertEqual(result['status'],'PUBLISHED')
        again=pub.append_commit(str(self.remote),self.payload,'synthetic offline test\n')
        self.assertEqual(again['status'],'ALREADY_PRESENT')
        self.assertEqual(pub.text(self.remote,'rev-parse','refs/heads/main'),self.main)
        self.assertEqual(pub.text(self.remote,'show',pub.BRANCH+':README.md'),'preserve this context')
        with self.assertRaises(pub.PublishError):pub.append_commit(str(self.remote),{self.path:b'changed'},'must refuse')

    def test_concurrent_context_advance_is_preserved(self):
        original=pub.git;advanced=False
        def race(repo,*args,**kwargs):
            nonlocal advanced
            if args and args[0]=='push' and str(repo)!=str(self.seed) and not advanced:
                advanced=True
                (self.seed/'concurrent.txt').write_text('preserve concurrent edit\n')
                original(self.seed,'add','concurrent.txt');original(self.seed,'commit','-m','concurrent fixture')
                original(self.seed,'push','origin',pub.BRANCH)
            return original(repo,*args,**kwargs)
        with patch.object(pub,'git',side_effect=race):result=pub.append_commit(str(self.remote),self.payload,'synthetic race test\n')
        self.assertEqual(result['status'],'PUBLISHED')
        self.assertEqual(pub.text(self.remote,'show',pub.BRANCH+':concurrent.txt'),'preserve concurrent edit')


if __name__ == '__main__':
    unittest.main(verbosity=2)
