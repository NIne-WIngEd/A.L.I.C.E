"""Read-only analysis guards; public fixtures are not learned capability evidence."""
from importlib import import_module, util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cognitive_kernel.canonical import CognitiveKernelContractError
from tests.governance.test_mfm_sequence_diagnostic import public_prepared
from cognitive_kernel.canonical import canonical_json_bytes


class SequenceFailureAnalysisTests(unittest.TestCase):
    def module(self):
        name = 'scripts.mfm.analyze_v16_sequence_failure'
        self.assertIsNotNone(util.find_spec(name), 'read-only sequence analyzer missing')
        return import_module(name)

    def test_foreign_cases_preserve_family_operation_and_ignore_input_order(self):
        m = self.module()
        rows = [SimpleNamespace(case_id=i, split=s) for i,s in (
            ('v16-seed-02','train'), ('v16-seed-01','train'),
            ('long-radio-east-plan-outcome','train'), ('long-atlas-east-plan-outcome','train'),
            ('public-b','development'), ('public-a','development'))]
        expected = {'v16-seed-01':'v16-seed-02','v16-seed-02':'v16-seed-01',
            'long-atlas-east-plan-outcome':'long-radio-east-plan-outcome',
            'long-radio-east-plan-outcome':'long-atlas-east-plan-outcome',
            'public-a':'public-b','public-b':'public-a'}
        self.assertEqual(m.foreign_case_map(rows), expected)
        self.assertEqual(m.foreign_case_map(reversed(rows)), expected)
        with self.assertRaises(CognitiveKernelContractError):
            m.foreign_case_map(rows+[rows[0]])

    def test_remeasurement_checks_every_case_both_tasks_and_balanced_objective(self):
        m = self.module()
        prepared,_ = public_prepared()
        curriculum = m.trainer.curriculum.curriculum_receipt(prepared.train,prepared.tokenizer,2048)
        losses = {e.case_id:[2.,4.] for e in prepared.train}
        progress = {'complete_case_losses':losses, 'balanced_two_task_objective':3.,
                    'strata':m.trainer.task_strata(losses)}
        self.assertTrue(m.verify_remeasurement(losses,progress,curriculum)['matched'])
        changed = {k:list(v) for k,v in losses.items()}
        changed['v16-seed-01'][1] = 4.01
        with self.assertRaises(CognitiveKernelContractError):
            m.verify_remeasurement(changed,progress,curriculum)
        with self.assertRaises(CognitiveKernelContractError):
            m.verify_remeasurement(losses,{**progress,'balanced_two_task_objective':2.99},curriculum)
        with self.assertRaises(CognitiveKernelContractError):
            m.verify_remeasurement({k:v for k,v in losses.items() if k!='v16-seed-01'},progress,curriculum)

    def test_joint_frequency_metrics_do_not_merge_task_only_rare_tokens(self):
        m = self.module()
        raw = {'tokens':3,'mean_loss':2.,'next_token_accuracy':1/3,
            'groups':{'value:x|0|10+':{'tokens':1,'mean_loss':1.,'accuracy':1.},
                      'value:x|1-9|10+':{'tokens':2,'mean_loss':2.5,'accuracy':0.}}}
        result = m.fold_frequency_metrics(raw)
        self.assertEqual(result['groups']['value:x'], {'tokens':3,'mean_loss':2.,'accuracy':1/3})
        self.assertEqual(result['groups_by_positive_train_corpus_frequency']['value:x']['0']['tokens'],1)
        self.assertEqual(result['groups_by_combined_positive_train_corpus_frequency']['value:x']['10+']['tokens'],3)

    def test_original_admission_failure_prevents_new_candidate_or_output_reads(self):
        m = self.module()
        with TemporaryDirectory() as folder:
            output = Path(folder)/'must-not-exist'
            with patch.object(m.source,'load',side_effect=CognitiveKernelContractError('original admission rejected')):
                with self.assertRaisesRegex(CognitiveKernelContractError,'original admission rejected'):
                    m.run(SimpleNamespace(output_dir=output))
            self.assertFalse(output.exists())

    def test_source_stopping_failure_precedes_exact_content_comparison(self):
        m = self.module()
        self.assertTrue(callable(getattr(m,'source_validation_stage',None)), 'source failure stage helper missing')
        prepared,_ = public_prepared();e = prepared.train[0]
        raw = canonical_json_bytes(m.trainer.curriculum.lexical_source_record(e)).decode('utf-8')
        row = m.diagnostic.validate_source(raw,False,e)
        self.assertEqual(m.source_validation_stage(row,e,{'json_status':'parsed'})['first_failed_stage'],'no_eos')
        row = m.diagnostic.validate_source(raw,True,e)
        self.assertIsNone(m.source_validation_stage(row,e,{'json_status':'parsed'})['first_failed_stage'])
        row = m.diagnostic.validate_source('{',True,e)
        self.assertEqual(m.source_validation_stage(row,e,{'json_status':'malformed'})['first_failed_stage'],'json')


if __name__ == '__main__': unittest.main()
