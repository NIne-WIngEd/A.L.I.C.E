"""Fictional closed-feature interruption recovery through the real replay seam."""
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from src.alice_personality.identity.codec import canonical
from src.alice_personality.identity.contracts import IdentityModelConfig, IdentityError
from src.alice_personality.identity.feature_producer import FrozenFrameProducer
from src.alice_personality.identity.feature_store import FrozenFeatureStore
from tests.personality.test_identity_codec import document, Tokenizer, Backbone
from tests.personality import test_connected_construction as construction_fixture


class CompletedFeatureRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.config = IdentityModelConfig(8, 3, 16, 4)
        self.backbone = Backbone()
        self.store = FrozenFeatureStore.create(self.root/'feature-store',source_package_sha256='a'*64)
        self.producer = FrozenFrameProducer(Tokenizer(),self.backbone,self.config,_fixture=True,feature_store=self.store)
        self.doc = document()
        self.frame = self.producer.produce(self.doc)
        self.session = self.producer.close_and_verify()
        self.calls = self.backbone.calls
        entries=self.store.manifest['entries']
        self.values={
            'feature-session.protected.json':self.session,
            'frame-bindings.protected.json':{'document':self.frame.binding},
            'model-config.protected.json':asdict(self.config),
            'token-cost-profile.json':{'document_count':1,'unique_entries':len(entries),
                'unique_source_tokens':sum(row['shape'][2] for row in entries.values()),
                'source_geometry':[3,8]},
        }
        pins={}
        for name,value in self.values.items():
            path=self.root/name;path.write_bytes(canonical(value)+b'\n');pins[name]=sha256(path.read_bytes()).hexdigest()
        pins['feature-store/manifest.json']=sha256((self.root/'feature-store/manifest.json').read_bytes()).hexdigest()
        self.reference={'directory':str(self.root),'source_code_revision':'1'*40,
            'file_sha256':pins,'feature_store_manifest_sha256':self.store.manifest['manifest_sha256'],
            'closed_session_sha256':self.session['receipt_sha256']}

    def recover(self,reference=None,docs=None,source_package='a'*64):
        from scripts.eipm.gemma_n0.construct_personality import restore_completed_features
        def read(value,digest):
            path=Path(value);self.assertEqual(sha256(path.read_bytes()).hexdigest(),digest)
            return json.loads(path.read_bytes())
        return restore_completed_features(self.reference if reference is None else reference,
            documents={'document':self.doc} if docs is None else docs,
            source_package_sha256=source_package,prepared_receipt_file_sha256=self.session['prepared_receipt_file_sha256'],
            read_pinned_json=read,bounded_directory=lambda value:Path(value))

    def test_original_complete_store_and_session_replay_without_base_load_or_writes(self):
        before={p.name:p.read_bytes() for p in self.root.iterdir() if p.is_file()}
        store,config,session,bindings,profile=self.recover()
        replay=FrozenFrameProducer.replay(store,config,session,expected_session_sha256=self.reference['closed_session_sha256'])
        actual=replay.produce(self.doc)
        self.assertEqual(actual.binding,self.frame.binding)
        self.assertEqual(self.backbone.calls,self.calls)
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.root.iterdir() if p.is_file()})
        self.assertEqual(bindings['document'],actual.binding)
        self.assertEqual(profile['unique_entries'],len(store.manifest['entries']))

    def test_same_count_changed_document_cannot_relabel_old_closed_features(self):
        changed=replace(self.doc,candidates=(replace(self.doc.candidates[0],text='Never extracted.'),self.doc.candidates[1]))
        with self.assertRaises(ValueError):self.recover(docs={'document':changed})

    def test_changed_binding_or_profile_even_with_new_file_pin_fails_closed_session(self):
        for name in ('frame-bindings.protected.json','token-cost-profile.json'):
            original=canonical(self.values[name])+b'\n'
            modified=json.loads(original)
            if name.startswith('frame'):modified['document']['frame_sha256']='f'*64
            else:modified['unique_entries']+=1
            (self.root/name).write_bytes(canonical(modified)+b'\n')
            ref=json.loads(json.dumps(self.reference));ref['file_sha256'][name]=sha256((self.root/name).read_bytes()).hexdigest()
            with self.assertRaises(ValueError):self.recover(ref)
            (self.root/name).write_bytes(original)

    def test_changed_payload_is_refused_by_actual_replay_not_silently_reextracted(self):
        store,config,session,_,_=self.recover()
        payload=next((self.root/'feature-store').glob('*.pt'))
        with payload.open('ab') as stream:stream.write(b'changed')
        replay=FrozenFrameProducer.replay(store,config,session,expected_session_sha256=self.reference['closed_session_sha256'])
        with self.assertRaisesRegex(IdentityError,'payload'):replay.produce(self.doc)
        self.assertEqual(self.backbone.calls,self.calls)

    def test_unsealed_or_wrong_source_store_cannot_be_used_as_completed_extraction(self):
        with self.assertRaises(ValueError):self.recover(source_package='b'*64)
        wrong=json.loads(json.dumps(self.reference));wrong['feature_store_manifest_sha256']='f'*64
        with self.assertRaises(IdentityError):self.recover(wrong)
        (self.root/'feature-store/manifest.json').unlink()
        with self.assertRaises(FileNotFoundError):self.recover()

    def test_recovered_values_reach_actual_admitted_optimization_and_response_consumer(self):
        from scripts.eipm.gemma_n0.construct_personality import restore_completed_features
        fixture=construction_fixture.ConstructionTests();fixture.setUp()
        self.addCleanup(fixture.doCleanups);self.addCleanup(fixture.fixture.doCleanups)
        store=fixture.replay.feature_store
        values={'model-config.protected.json':asdict(fixture.config),
            'feature-session.protected.json':fixture.session,
            'frame-bindings.protected.json':{'document':fixture.produced.binding},
            'token-cost-profile.json':{'document_count':1,'unique_entries':len(store.manifest['entries']),
                'unique_source_tokens':sum(row['shape'][2] for row in store.manifest['entries'].values()),
                'source_geometry':[fixture.config.provider_state_count,fixture.config.provider_width]}}
        pins={}
        for name,value in values.items():
            path=fixture.path/name;path.write_bytes(canonical(value)+b'\n');pins[name]=sha256(path.read_bytes()).hexdigest()
        # This existing fixture calls its custody directory store; preserve it,
        # and present the actual closed manifest through a new complete copy.
        import shutil
        shutil.copytree(store.path,fixture.path/'feature-store')
        pins['feature-store/manifest.json']=sha256((fixture.path/'feature-store/manifest.json').read_bytes()).hexdigest()
        ref={'directory':str(fixture.path),'source_code_revision':'1'*40,'file_sha256':pins,
            'feature_store_manifest_sha256':store.manifest['manifest_sha256'],
            'closed_session_sha256':fixture.session['receipt_sha256']}
        def read(value,digest):
            payload=Path(value).read_bytes();self.assertEqual(sha256(payload).hexdigest(),digest);return json.loads(payload)
        recovered,config,session,_,_=restore_completed_features(ref,documents={'document':fixture.fixture.doc},
            source_package_sha256=store.manifest['source_package_sha256'],
            prepared_receipt_file_sha256=fixture.session['prepared_receipt_file_sha256'],
            read_pinned_json=read,bounded_directory=lambda value:Path(value))
        fixture.replay=FrozenFrameProducer.replay(recovered,config,session,expected_session_sha256=ref['closed_session_sha256'])
        fixture.test_complete_phases_stream_batches_and_bind_actual_checkpoint_chain()
