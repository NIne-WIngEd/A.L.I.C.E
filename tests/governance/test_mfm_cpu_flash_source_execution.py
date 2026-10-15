"""CPU attention execution mechanics; no pretrained weights or capacity claims."""
from types import SimpleNamespace
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import inspect
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from scripts.mfm import export_joint_source_features as features


class CpuFlashForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def forward_case(self,mode='flash'):
        batch={'input_ids':torch.arange(8).reshape(1,8),
            'attention_mask':torch.ones(1,8,dtype=torch.int64),
            'mm_token_type_ids':torch.zeros(1,8,dtype=torch.int64)}
        q=torch.arange(32,dtype=torch.float32).reshape(1,1,8,4).div(32).to(torch.bfloat16)
        causal=torch.ones(8,8,dtype=torch.bool).tril()
        sliding=causal & (torch.arange(8)[:,None]-torch.arange(8)[None,:]<3)
        masks=(causal,torch.where(sliding,0.0,float('-inf')))
        class Model:
            def __call__(self,**kwargs):
                for key in batch:
                    if not torch.equal(batch[key],kwargs[key]):raise AssertionError('native batch changed')
                if kwargs['use_cache'] is not False or kwargs['return_dict'] is not True:
                    raise AssertionError('base feature forward semantics changed')
                result=q
                for index,mask in enumerate(masks):
                    if mode=='eager' or mode=='partial' and index==1:
                        additive=torch.where(mask,0.0,float('-inf')) if mask.dtype==torch.bool else mask
                        result=(result.float()@q.float().transpose(-1,-2)/2+additive).softmax(-1)@q.float()
                        result=result.to(torch.bfloat16)
                    elif mode=='math':
                        from torch.nn.attention import sdpa_kernel,SDPBackend
                        with sdpa_kernel(SDPBackend.MATH):
                            result=F.scaled_dot_product_attention(result,q,q,attn_mask=mask)
                    else:result=F.scaled_dot_product_attention(result,q,q,attn_mask=mask)
                return SimpleNamespace(last_hidden_state=result.transpose(1,2).reshape(1,8,4))
        base=SimpleNamespace(model=Model(),config=SimpleNamespace(text_config=SimpleNamespace(
            _attn_implementation='sdpa',num_hidden_layers=2)))
        policy={'profile':'cpu-sdpa-flash-v1','cpu_threads':1,'text_attention_layers':2}
        return base,batch,policy,q,masks

    def require_forward(self):
        self.assertTrue(hasattr(features,'_cpu_flash_forward'),
            'whole-source CPU flash execution observation is missing')

    def test_actual_cpu_flash_preserves_masks_native_fields_and_math_reference(self):
        self.require_forward();base,batch,policy,q,masks=self.forward_case()
        before={key:value.clone() for key,value in batch.items()}
        states,observed=features._cpu_flash_forward(base,batch,policy)
        from torch.nn.attention import sdpa_kernel,SDPBackend
        with sdpa_kernel(SDPBackend.MATH):
            expected=q
            for mask in masks:expected=F.scaled_dot_product_attention(expected,q,q,attn_mask=mask)
        self.assertTrue(torch.allclose(states,expected.transpose(1,2).reshape(1,8,4),atol=0.008,rtol=0.008))
        self.assertEqual(observed['cpu_flash_operator_count'],2)
        self.assertEqual(observed['math_operator_count'],0)
        self.assertGreater(observed['cpu_flash_operator_time_us'],0)
        self.assertGreaterEqual(observed['wall_seconds'],0)
        self.assertTrue(all(torch.equal(before[key],batch[key]) for key in batch))

    def test_eager_math_and_partial_execution_are_refused_and_flags_restore(self):
        self.require_forward()
        before=(torch.backends.cuda.flash_sdp_enabled(),torch.backends.cuda.math_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled())
        for mode in ('eager','math','partial'):
            base,batch,policy,_,_=self.forward_case(mode)
            with self.subTest(mode=mode),self.assertRaisesRegex(ValueError,'CPU flash'):
                features._cpu_flash_forward(base,batch,policy)
            self.assertEqual(before,(torch.backends.cuda.flash_sdp_enabled(),
                torch.backends.cuda.math_sdp_enabled(),torch.backends.cuda.mem_efficient_sdp_enabled()))

    def test_execution_policy_rejects_unknown_profile_and_non_cpu(self):
        self.assertTrue(hasattr(features,'_check_execution_profile'),
            'explicit CPU source execution policy is missing')
        self.assertIsNone(features._check_execution_profile(None,'cpu'))
        for profile,device in (('cpu-sdpa-flash-v2','cpu'),('cpu-sdpa-flash-v1','cuda:0')):
            with self.subTest(profile=profile,device=device),self.assertRaises(ValueError):
                features._check_execution_profile(profile,device)


class CpuExecutionReceiptTests(unittest.TestCase):
    def records(self):
        self.assertTrue(hasattr(features,'_make_execution_policy'),
            'sealed CPU execution policy is missing')
        self.assertTrue(hasattr(features,'_validate_execution_records'),
            'CPU execution receipt consumer is missing')
        base=SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(
            _attn_implementation='sdpa',num_hidden_layers=2)))
        with patch.object(features.platform,'system',return_value='Linux'):
            policy=features._make_execution_policy(base)
        binding={'fixture_small_width':False,'prepared_base':{},'backbone_device':'cpu',
            'runtime':{'torch_version':torch.__version__,'transformers_version':policy['transformers_version']}}
        execution={'schema':'mfm-source-cpu-sdpa-flash-forward-v1',
            'case_id':'execution-control','source_alignment_sha256':'a'*64,'source_length':8,
            'policy_sha256':policy['record_sha256'],'cpu_flash_operator_count':2,'math_operator_count':0,
            'cpu_flash_operator_time_us':1.0,'wall_seconds':0.001,'cpu_seconds':0.002,
            'process_lifetime_peak_rss_before_bytes':1000,'process_lifetime_peak_rss_after_bytes':2000}
        return {'base':binding,'execution_policy':policy},{'base':binding,'execution':execution,
            'case_id':'execution-control','alignment_sha256':'a'*64,'source_length':8}

    def test_sealed_policy_and_actual_execution_fields_are_cross_bound(self):
        roster,receipt=self.records();features._validate_execution_records(roster,receipt)
        features._validate_execution_records({'base':roster['base']},{'base':receipt['base']})
        for field,value in (('policy_sha256','0'*64),('math_operator_count',1),
            ('cpu_flash_operator_count',1),('cpu_flash_operator_time_us',-1),
            ('wall_seconds',float('nan')),('process_lifetime_peak_rss_after_bytes',999),
            ('case_id','foreign-case'),('source_alignment_sha256','b'*64),('source_length',7)):
            bad=deepcopy(receipt);bad['execution'][field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'CPU flash'):
                features._validate_execution_records(roster,bad)
        for pair in (({'base':roster['base']},receipt),(roster,{'base':receipt['base']})):
            with self.assertRaisesRegex(ValueError,'CPU flash'):
                features._validate_execution_records(*pair)
        bad=deepcopy(roster);bad['execution_policy']['text_attention_layers']=1
        with self.assertRaisesRegex(ValueError,'CPU flash'):
            features._validate_execution_records(bad,receipt)

    def test_consumer_runtime_can_differ_from_stored_cpu_producer(self):
        roster,receipt=self.records()
        with patch.object(torch,'__version__','2.11.0+cu128'), \
             patch.object(features.platform,'system',return_value='DifferentConsumer'), \
             patch.object(torch,'get_num_threads',return_value=3), \
             patch.object(features,'_process_peak_rss_bytes',side_effect=AssertionError('producer-only metric')):
            features._validate_execution_records(roster,receipt)

    def test_changed_execution_binding_refuses_before_source_or_feature_tensor_read(self):
        roster,receipt=self.records();receipt['execution']['policy_sha256']='0'*64
        with TemporaryDirectory() as temp:
            root=Path(temp);directory=root/'cases'/'execution-control';directory.mkdir(parents=True)
            (directory/'features.safetensors').write_bytes(b'never loaded')
            feature=features._write_record(directory/'receipt.json',{
                'schema':features.CASE_SCHEMA,'case_id':'execution-control','profile':'j4-p1',**receipt})
            bank=features._write_record(root/'roster.json',{
                'schema':features.SCHEMA,**roster,'cases':[{'case_id':'execution-control',
                    'receipt_sha256':feature['record_sha256']}]})
            with patch.object(features,'load_file') as tensor_read, \
                 patch.object(features,'_source_api') as source_read:
                with self.assertRaisesRegex(ValueError,'CPU flash'):
                    features.read_case(root,roster_sha256=bank['record_sha256'],case_id='execution-control',
                        expected_alignment_sha256='a'*64,expected_cache={})
                tensor_read.assert_not_called();source_read.assert_not_called()


class CpuPreparedLoaderTests(unittest.TestCase):
    def test_explicit_profile_reaches_loader_and_retains_prepared_checks(self):
        self.assertIn('execution_profile',inspect.signature(features.export_prepared_bank).parameters,
            'prepared export lacks explicit execution profile')
        from tests.governance.test_mfm_joint_source_features import (
            PreparedLoadPathTests,NativeTextFieldTests,shared)
        NativeTextFieldTests.setUpClass()
        processor=NativeTextFieldTests.processor;case=NativeTextFieldTests.case
        with TemporaryDirectory() as temp:
            root=Path(temp)
            prepared,receipt,row,runtime,base=PreparedLoadPathTests.prepared_fixture(root,processor)
            base.config.text_config._attn_implementation='sdpa'
            base.config.text_config.num_hidden_layers=48
            weight_sha=next(r['sha256'] for r in row['files'] if r['path']=='model.safetensors')
            def export(path,**kwargs):
                Path(path).mkdir()
                return {'record_sha256':'e'*64,'execution_policy':kwargs['execution_policy']}
            with patch.dict('sys.modules',{'transformers':runtime}), \
                 patch.object(shared,'_prepared_base',return_value=row) as verify, \
                 patch.object(shared,'SOURCE_WEIGHT_SHA256',weight_sha), \
                 patch.object(features,'_process_peak_rss_bytes',return_value=1000), \
                 patch.object(features.platform,'system',return_value='Linux'), \
                 patch.object(features,'_export_bank',side_effect=export) as exported:
                result=features.export_prepared_bank(root/'bank',prepared_base_dir=prepared,
                    cases=[case],j5_source=None,prepared_receipt_path=receipt,
                    expected_prepared_receipt_file_sha256=features._digest(receipt),
                    expected_prepared_receipt_sha256=row['receipt_sha256'],
                    expected_parent_source_receipt_sha256=row['parent_source_receipt_sha256'],
                    expected_cases_sha256=features.canonical_sha256([case]),device='cpu',
                    execution_profile='cpu-sdpa-flash-v1')
            self.assertEqual(verify.call_count,3)
            self.assertEqual(runtime.AutoModelForMultimodalLM.from_pretrained.call_args.kwargs['attn_implementation'],'sdpa')
            self.assertEqual(result['execution_policy']['text_attention_layers'],48)
            self.assertEqual(exported.call_args.kwargs['preflight'][0][1]['mm_token_type_ids'].dtype,torch.int64)
            self.assertFalse(exported.call_args.kwargs['fixture_small_width'])


class CpuFlashFixtureBankTests(unittest.TestCase):
    def test_actual_small_flash_forward_serializes_with_bound_execution_evidence(self):
        self.assertIn('execution_policy',inspect.signature(features._export_bank).parameters,
            'whole bank exporter lacks explicit execution policy')
        from scripts.mfm import formation_joint_input_v4 as original
        torch.set_num_threads(1)
        simple,batch,_,_,_=CpuFlashForwardTests().forward_case()
        base=torch.nn.Module();base.model=torch.nn.Module()
        base.model.forward=simple.model.__call__
        base.marker=torch.nn.Parameter(torch.zeros(1,dtype=torch.bfloat16),requires_grad=False)
        base.config=simple.config;base.config.model_type='gemma4_unified'
        base.config.text_config.hidden_size=4;base.eval()
        plain={k:v.tolist() for k,v in batch.items()}
        alignment={'input_ids':plain['input_ids'][0],'record_sha256':'a'*64,
            'cache':{'fixture':'only'},'code_binding':original.joint_code_binding_v4(),
            'processor_binding':{'fixture':'not a real processor'}}
        case={'case_id':'tiny-cpu-mechanics-only','profile':'j4-p1'}
        with TemporaryDirectory() as temp:
            root=Path(temp);weight=root/'weight.bin';weight.write_bytes(b'fixture only')
            config=root/'config.json';config.write_bytes(b'{}')
            with patch.object(features.platform,'system',return_value='Linux'), \
                 patch.object(features,'_process_peak_rss_bytes',side_effect=[1000,2000]):
                policy=features._make_execution_policy(base)
                bank=features._export_bank(root/'bank',cases=[case],processor=None,base=base,j5_source=None,
                    base_weight_path=weight,base_weight_sha256=features._digest(weight),
                    base_config_path=config,base_config_sha256=features._digest(config),
                    runtime={'torch_version':torch.__version__,'transformers_version':policy['transformers_version']},
                    fixture_small_width=True,prepared_binding=None,
                    preflight=[(original,batch,alignment,plain)],execution_policy=policy)
            receipt,tensors=features.read_case(root/'bank',roster_sha256=bank['record_sha256'],
                case_id=case['case_id'],expected_alignment_sha256=alignment['record_sha256'],
                expected_cache=alignment['cache'],allow_fixture=True)
            self.assertTrue(bank['base']['fixture_small_width'])
            self.assertEqual(tensors['states'].shape,(1,8,4))
            self.assertEqual(receipt['execution']['cpu_flash_operator_count'],2)
            self.assertEqual(receipt['execution']['policy_sha256'],bank['execution_policy']['record_sha256'])
            self.assertEqual(set(tensors),{'states','input_ids','attention_mask','mm_token_type_ids'})
            # The resource snapshots above are explicit mocked audit operands;
            # this small fixture is rejected by the production consumer default.
            with self.assertRaisesRegex(ValueError,'fixture representation'):
                features.read_case(root/'bank',roster_sha256=bank['record_sha256'],case_id=case['case_id'],
                    expected_alignment_sha256=alignment['record_sha256'],expected_cache=alignment['cache'])


if __name__=='__main__':unittest.main()
