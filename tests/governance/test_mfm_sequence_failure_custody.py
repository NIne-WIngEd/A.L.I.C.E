"""Actual sealed/raw readers on public fixtures; no learned-output claims."""
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from cognitive_kernel.canonical import CognitiveKernelContractError, canonical_json_bytes
from tests.governance.test_mfm_sequence_diagnostic import public_prepared


def public_case_key(example):
    # PublicExample intentionally omits the real context serializer. Only this
    # input-identity dependency is replaced; receipts, file hashes and readers
    # remain the implementation used by the protected diagnostic.
    return sha256(example.case_id.encode('utf-8')).hexdigest()


class RetainedFixture:
    def __init__(self, module, root):
        self.module,self.root=module,root
        self.prepared,receipt=public_prepared()
        self.prepared.view['record_sha256']='0'*64
        self.candidate=SimpleNamespace(
            component={'record_sha256':'1'*64,'result':{'trained_weights_sha256':'2'*64}},
            run={'record_sha256':'3'*64},curriculum={**receipt,'record_sha256':'4'*64},
            probe={'record_sha256':'5'*64},probe_run={'record_sha256':'6'*64},
            checkpoint={'record_sha256':'7'*64},progress=[{'record_sha256':'8'*64}],
            seed={'record_sha256':'9'*64,'specialist_sha256':'a'*64},
            parent_component={'record_sha256':'b'*64},parent_diagnostic={'record_sha256':'c'*64})
        self.args=SimpleNamespace(sequence_diagnostic_dir=root)
        diagnostic=module.diagnostic
        budgets=diagnostic.decode_budgets(self.prepared,self.candidate.curriculum)
        self.rosters=diagnostic.select_rosters(self.prepared,self.candidate.curriculum)
        self.rows,self.summary,self.hashes={},{},{}
        for control in diagnostic.CONTROLS:
            self.summary[control]={}
            for task,examples in self.rosters.items():
                key=control+'-'+task
                weights='2'*64 if control=='trained' else 'a'*64
                rows=[]
                for e in examples:
                    validation=({'raw_output_text':'{}','eos_observed':True,'validation_status':'invalid',
                        'validation_error':'public fixture has no decision schema','proposal_count':0,
                        'exact_teacher_decision_agreement':None} if task=='formation' else
                        diagnostic.validate_source('{}',True,e))
                    row={'case_id':e.case_id,'split':e.split,'family':module.ancestry.family_name(e),
                        'task':task,'control':control,'weights_sha256':weights,
                        'source_view_sha256':'0'*64,'context_sha256':e.context.content_digest(),
                        'model_input_sha256':public_case_key(e),'inference_run_id':'3'*64+'-'+key,
                        'max_new_tokens':budgets[task],'generated_token_ids':[123,125],
                        'incremental_parity':{'argmaxes_equal':True},**validation}
                    rows.append(row);diagnostic.add_measurement(self.summary[control],row)
                self.rows[key]=rows;self.write_rows(key)
        self.record={'schema':diagnostic.SCHEMA,'component_sha256':'1'*64,'run_sha256':'3'*64,
            'source_view_sha256':'0'*64,'curriculum_sha256':'4'*64,'stress_probe_sha256':'5'*64,
            'stress_probe_run_sha256':'6'*64,'completed_checkpoint_sha256':'7'*64,
            'complete_train_progress_sha256':['8'*64],'seed_control_sha256':'9'*64,
            'historical_parent_component_sha256':'b'*64,'historical_parent_diagnostic_sha256':'c'*64,
            'code_binding':diagnostic.code_binding(),
            'case_rosters':{task:[e.case_id for e in examples] for task,examples in self.rosters.items()},
            'decode':{'method':'FP32_incremental_first32_actual_greedy_reference_checked',
                'explicit_task_bound_view':True,'max_new_tokens_by_task':budgets,'device':'cpu',
                'specialist_dtype':'float32','torch_version':torch.__version__,'target_visible':False,
                'matched_between_controls':True,'target_comparison_after_generation':True},
            'historical_outputs_rewritten':False,'optimizer_created':False,'independent_gold':False,
            'final_payloads_opened':False,'memory_gate_executed':False,'authority_written':False,
            'qualified_for_product':False,'downstream_usefulness_established':False,
            'seed_parameters_check':{'actual_same_seed_parameters_equal':True},
            'outputs_sha256':self.hashes,'summary':self.summary}
        sealed=module.shared._write_new(root/'diagnostic.json',self.record)
        self.args.sequence_diagnostic_sha256=sealed['record_sha256']

    def write_rows(self,key):
        path=self.root/(key+'.jsonl')
        path.write_text(''.join(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n'
                                for row in self.rows[key]),encoding='utf-8',newline='\n')
        self.hashes[key]=self.module.shared._digest(path)

    def reseal(self):
        # Each deliberate mutation is given a correct internal seal and new
        # external pin, so rejection must reach row/summary custody checks.
        sealed=self.module.shared._seal(self.record)
        (self.root/'diagnostic.json').write_bytes(canonical_json_bytes(sealed))
        self.args.sequence_diagnostic_sha256=sealed['record_sha256']

    def verify(self):
        with patch.object(self.module.source.features,'case_key',side_effect=public_case_key):
            return self.module.verify_retained(self.args,self.prepared,self.candidate)


class SequenceFailureCustodyTests(unittest.TestCase):
    def fixture(self,root):
        return RetainedFixture(import_module('scripts.mfm.analyze_v16_sequence_failure'),root)

    def test_real_sealed_reader_accepts_all_four_pinned_files_and_complete_rosters(self):
        with TemporaryDirectory() as folder:
            fixture=self.fixture(Path(folder));record,rows,rosters=fixture.verify()
            self.assertEqual(record['record_sha256'],fixture.args.sequence_diagnostic_sha256)
            self.assertEqual(set(rows),{'trained-formation','seeded-untrained-formation',
                'trained-source_reconstruction','seeded-untrained-source_reconstruction'})
            self.assertEqual(len(rows['trained-formation']),38)
            self.assertEqual(len(rows['trained-source_reconstruction']),4)
            self.assertEqual([r['case_id'] for r in rows['trained-formation']],
                             [e.case_id for e in rosters['formation']])

    def test_wrong_external_pin_rejects_an_unchanged_correctly_sealed_record(self):
        with TemporaryDirectory() as folder:
            fixture=self.fixture(Path(folder));fixture.verify()
            fixture.args.sequence_diagnostic_sha256='f'*64
            with self.assertRaisesRegex(CognitiveKernelContractError,'lineage/settings'):
                fixture.verify()

    def test_sealed_reader_rejects_changed_record_with_original_internal_digest(self):
        with TemporaryDirectory() as folder:
            fixture=self.fixture(Path(folder));fixture.verify()
            path=fixture.root/'diagnostic.json';record=json.loads(path.read_bytes())
            record['summary']['trained']['formation']['base-train']['cases']+=1
            path.write_bytes(canonical_json_bytes(record))
            with self.assertRaisesRegex(CognitiveKernelContractError,'receipt digest differs'):
                fixture.verify()

    def test_raw_byte_tamper_is_rejected_before_parsing_unchanged_json_meaning(self):
        with TemporaryDirectory() as folder:
            fixture=self.fixture(Path(folder));fixture.verify()
            path=fixture.root/'trained-source_reconstruction.jsonl'
            path.write_bytes(path.read_bytes()+b'\n')
            with self.assertRaisesRegex(CognitiveKernelContractError,'raw output hash'):
                fixture.verify()

    def test_rehashed_and_resealed_case_task_control_or_weight_swaps_are_rejected(self):
        for mutation in ('case-order','task','control','weights'):
            with self.subTest(mutation=mutation),TemporaryDirectory() as folder:
                fixture=self.fixture(Path(folder));fixture.verify()
                key='trained-formation';rows=fixture.rows[key]
                if mutation=='case-order': rows[0],rows[1]=rows[1],rows[0]
                elif mutation=='task': rows[0]['task']='source_reconstruction'
                elif mutation=='control': rows[0]['control']='seeded-untrained'
                else: rows[0]['weights_sha256']='a'*64
                fixture.write_rows(key);fixture.reseal()
                with self.assertRaisesRegex(CognitiveKernelContractError,'raw row lineage'):
                    fixture.verify()

    def test_resealed_summary_tamper_fails_against_actual_raw_measurements(self):
        with TemporaryDirectory() as folder:
            fixture=self.fixture(Path(folder));fixture.verify()
            fixture.record['summary']['trained']['formation']['base-train']['cases']+=1
            fixture.reseal()
            with self.assertRaisesRegex(CognitiveKernelContractError,'summary differs'):
                fixture.verify()

    def test_resealed_missing_control_task_and_shortened_roster_are_rejected(self):
        for mutation in ('control-task','case-count'):
            with self.subTest(mutation=mutation),TemporaryDirectory() as folder:
                fixture=self.fixture(Path(folder));fixture.verify()
                key='trained-source_reconstruction'
                if mutation=='control-task': fixture.hashes.pop(key)
                else: fixture.rows[key].pop();fixture.write_rows(key)
                fixture.reseal()
                with self.assertRaisesRegex(CognitiveKernelContractError,'controls/tasks|roster size'):
                    fixture.verify()


if __name__=='__main__': unittest.main()
