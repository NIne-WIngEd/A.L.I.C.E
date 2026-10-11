import unittest,tempfile
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
from cognitive_kernel.canonical import canonical_json_bytes,CognitiveKernelContractError
from scripts.mfm import formation_input_content_prototype as P,formation_task_content_input as T,formation_uncertainty_input as U
from tests.governance.mfm_task_content_fixtures import TaskOriginals,seal,SCOPE,NAMESPACE,TIME,LATER


class PrototypeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='mfm-input-content-');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'content';self.f=TaskOriginals()
    def open(self):
        store=P.InputContentPrototypeStore(self.root,scope=SCOPE,namespace=NAMESPACE,store_key='fixture',created_at=TIME,
            family_bindings={'task':T.task_content_binding(),'uncertainty':U.uncertainty_input_binding()})
        self.addCleanup(store.close);return store
    def commit(self,store,family='mission',record=None,operation='mission-initial',previous=None):
        raw,parts,kw=self.f.arguments(family,record);return store.commit(family,raw,parts,operation_id=operation,
            expected_previous_content_sha256=previous,committed_at=TIME)

    def test_actual_rawbuffer_roundtrip_reopen_and_identical_replay_before_capture(self):
        store=self.open();receipt=self.commit(store);count=len(store.raw.inspect())
        with patch.object(store.raw,'capture',side_effect=AssertionError('accepted replay must not recapture')):
            self.assertEqual(self.commit(store),receipt)
        self.assertEqual(len(store.inspect_commits()),1);self.assertEqual(len(store.raw.inspect()),count)
        self.assertEqual(store.read_original_part('mission',self.f.mission.mission_id,1,'mission-title'),self.f.parts['mission-title'])
        store.close();store=self.open();self.assertEqual(store.inspect_commits()[0],receipt);store.close()

    def test_changed_operation_stale_predecessor_and_family_collision_refuse(self):
        store=self.open();self.commit(store)
        row=deepcopy(self.f.records['mission']);row['subject_ref']='other';row=seal({k:v for k,v in row.items() if k!='record_sha256'})
        with self.assertRaises(CognitiveKernelContractError):self.commit(store,record=row)
        with self.assertRaises(CognitiveKernelContractError):self.commit(store,operation='duplicate-new-operation')
        with self.assertRaises(CognitiveKernelContractError):store.read_original_part('workspace',self.f.mission.mission_id,1,'mission-title')
        self.assertEqual(len(store.inspect_commits()),1);store.close()

    def test_zero_byte_later_part_refuses_before_any_capture(self):
        store=self.open();row=deepcopy(self.f.records['result']);row['output_artifacts'][0]['artifact']=self.f.blob('empty-output',b'','application/octet-stream')
        row=seal({k:v for k,v in row.items() if k!='record_sha256'})
        with patch.object(store.raw,'capture',side_effect=AssertionError('zero byte must refuse preflight')):
            with self.assertRaises(CognitiveKernelContractError):self.commit(store,'result',row,operation='zero')
        self.assertEqual(store.inspect_commits(),());store.close()

    def test_zero_byte_refusal_reaches_store_with_complete_mixed_request(self):
        store=self.open();row=deepcopy(self.f.records['result']);_,parts,_=self.f.arguments('result')
        empty=self.f.blob('empty-output',b'','application/octet-stream');row['output_artifacts'][0]['artifact']=empty
        parts.pop('output-1');parts['empty-output']=b'';row=seal({k:v for k,v in row.items() if k!='record_sha256'})
        with patch.object(store.raw,'capture',side_effect=AssertionError('store must refuse before capture')):
            with self.assertRaises(CognitiveKernelContractError):store.commit('result',canonical_json_bytes(row),parts,
                operation_id='mixed-zero',expected_previous_content_sha256=None,committed_at=TIME)
        self.assertEqual(store.inspect_commits(),());self.assertEqual(len(store.raw.inspect()),0)

    def test_actual_second_version_and_same_string_separate_family_lookup(self):
        from cognitive_kernel import mission as G
        store=self.open();first=self.commit(store);row=deepcopy(self.f.records['mission'])
        graph=G.MissionGraphSnapshot.create(mission=self.f.mission,root_node_id=self.f.root.node_id,nodes=(self.f.root,),edges=(),
            revision=2,snapshot_at=LATER,previous_snapshot_sha256=self.f.graph.snapshot_sha256)
        row.update(content_version=2,previous_content_sha256=row['record_sha256'],updated_at=LATER,
            graph_artifact=self.f.json('graph-next',graph.metadata_record()));row=seal({k:v for k,v in row.items() if k!='record_sha256'})
        second=self.commit(store,record=row,operation='mission-next',previous=first['record_sha256'])
        self.assertEqual(second['content_store_sequence'],2);self.assertEqual(second['previous_store_entry_sha256'],first['commit_sha256'])
        self.commit(store,'attempt',operation='attempt-initial')
        outcome=deepcopy(self.f.records['outcome']);outcome.update(entity_id='attempt-1',observation_id='attempt-1');outcome=seal({k:v for k,v in outcome.items() if k!='record_sha256'})
        self.commit(store,'outcome',outcome,operation='same-id-outcome')
        self.assertEqual(store.read_original_part('attempt','attempt-1',1,'instruction'),b'Press the switch')
        self.assertEqual(store.read_original_part('outcome','attempt-1',1,'observer-report'),b'The lamp stayed dark.')
        with self.assertRaises(CognitiveKernelContractError):store.read_original_part('attempt','attempt-1',1,'observer-report')

    def test_partial_capture_is_uncommitted_and_never_auto_resumed(self):
        store=self.open();capture=store.raw.capture;calls=[]
        def interrupted(*args,**kwargs):
            calls.append(1)
            if len(calls)==2:raise OSError('public deliberate interruption')
            return capture(*args,**kwargs)
        with patch.object(store.raw,'capture',side_effect=interrupted):
            with self.assertRaises(OSError):self.commit(store)
        self.assertEqual(store.inspect_commits(),());self.assertEqual(len(store.raw.inspect()),1)
        with self.assertRaises(CognitiveKernelContractError):self.commit(store)
        store.close()

    def test_exact_commit_original_reconstructs_actual_reference_and_capture_receipts(self):
        store=self.open();receipt=self.commit(store);raw,parts,kw=self.f.arguments('mission');refs={
            'content':receipt['original_artifacts'][0]['artifact'],'commit':self.f.json('commit',receipt),'profile':self.f.json('profile',store.profile_record())}
        binding=T.task_content_binding();value=P.input_commit_original(raw,canonical_json_bytes(receipt),canonical_json_bytes(store.profile_record()),
            refs=refs,family_binding=binding,family_binding_sha256=binding['record_sha256'],scope=SCOPE,namespace=NAMESPACE)
        self.assertFalse(value['authenticated_service_claimed']);self.assertEqual(len(value['commit']['raw_buffer_receipts']),len(parts)+1)
        with self.assertRaises(CognitiveKernelContractError):P.input_commit_original(raw,canonical_json_bytes(receipt),canonical_json_bytes(store.profile_record()),
            refs={**refs,'content':kw['refs']['content-metadata']},family_binding=binding,family_binding_sha256=binding['record_sha256'],scope=SCOPE,namespace=NAMESPACE)
        bad=deepcopy(receipt);bad['raw_buffer_receipts'][0]['capture_receipt']['physical_object_created']=False
        bad=seal({k:v for k,v in bad.items() if k!='commit_sha256'},'commit_sha256')
        with self.assertRaises(CognitiveKernelContractError):P.input_commit_original(raw,canonical_json_bytes(bad),canonical_json_bytes(store.profile_record()),
            refs={**refs,'commit':self.f.json('bad-commit',bad)},family_binding=binding,family_binding_sha256=binding['record_sha256'],scope=SCOPE,namespace=NAMESPACE)
        store.close()

    def _validate_changed_receipt(self,store,receipt,mutate):
        row=deepcopy(receipt);mutate(row)
        from cognitive_kernel.canonical import canonical_sha256
        request={k:row[k] for k in ('family','entity_id','content_version','record_sha256','original_artifacts','expected_previous_content_sha256','committed_at')}
        row['request_sha256']=canonical_sha256(request);row=seal({k:v for k,v in row.items() if k!='commit_sha256'},'commit_sha256')
        raw,_,_=self.f.arguments('mission');binding=T.task_content_binding()
        return P.input_commit_original(raw,canonical_json_bytes(row),canonical_json_bytes(store.profile_record()),refs={
            'content':receipt['original_artifacts'][0]['artifact'],'commit':self.f.json('changed-commit-'+row['commit_sha256'],row),'profile':self.f.json('profile',store.profile_record())},
            family_binding=binding,family_binding_sha256=binding['record_sha256'],scope=SCOPE,namespace=NAMESPACE)

    def test_complete_raw_reference_originals_refuse_normalization_and_unknown_fields(self):
        store=self.open();receipt=self.commit(store)
        mutations=[lambda r:r.update(byte_length=str(r['byte_length'])),lambda r:r.update(byte_length=float(r['byte_length'])),
            lambda r:r.update(undeclared_original_field='unsupported'),lambda r:r['scope'].update(undeclared_original_field='unsupported'),
            lambda r:r.update(captured_at='2026-08-01T10:00:00Z')]
        for index in (0,len(receipt['raw_buffer_receipts'])-1):
            for mutation in mutations:
                with self.subTest(part=index,mutation=mutations.index(mutation)):
                    with self.assertRaises(CognitiveKernelContractError):
                        self._validate_changed_receipt(store,receipt,lambda c:mutation(c['raw_buffer_receipts'][index]['reference']))

    def test_related_commit_capture_and_artifact_originals_refuse_numeric_aliases(self):
        store=self.open();receipt=self.commit(store)
        mutations=[lambda c:c.update(content_version=1.0),lambda c:c.update(content_version=True),
            lambda c:c['original_artifacts'][-1]['artifact'].update(bytes=float(c['original_artifacts'][-1]['artifact']['bytes'])),
            lambda c:c['raw_buffer_receipts'][0]['capture_receipt']['reference'].update(unknown='unsupported'),
            lambda c:c['raw_buffer_receipts'][0]['capture_receipt'].update(physical_object_created=1)]
        for index,mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                with self.assertRaises(CognitiveKernelContractError):self._validate_changed_receipt(store,receipt,mutation)

    def test_stored_later_capture_refuses_before_any_original_payload_lookup(self):
        store=self.open();receipt=self.commit(store);raw_type=type(store.raw);store.close()
        receipt['raw_buffer_receipts'][-1]['reference']['byte_length']=str(receipt['raw_buffer_receipts'][-1]['reference']['byte_length'])
        receipt=seal({k:v for k,v in receipt.items() if k!='commit_sha256'},'commit_sha256')
        (self.root/'commits/00000001.json').write_bytes(canonical_json_bytes(receipt))
        with patch.object(raw_type,'load_opaque_payload',side_effect=AssertionError('later malformed capture must precede all payload reads')):
            with self.assertRaises(CognitiveKernelContractError):self.open()

if __name__=='__main__':unittest.main()
