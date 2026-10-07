"""Preparation contracts only; synthetic encoders do not prove language learning."""
from hashlib import sha256
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


class Tokenizer:
    bos_token_id = 2
    eos_token_id = 3

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        self.last_text = text
        return [5, 6, 7]

    def decode(self, ids, **kwargs):
        return self.last_text


class Processor:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.source_ids = [2, 8, 9]

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return {'input_ids': torch.tensor([self.source_ids]),
                'attention_mask': torch.ones((1, len(self.source_ids)), dtype=torch.long)}


class ConditionalPreflightTests(unittest.TestCase):
    def interface(self):
        name = 'scripts.mfm.preflight_public_conditional_material'
        self.assertIsNotNone(importlib.util.find_spec(name), 'conditional processor census is missing')
        return importlib.import_module(name)

    def row(self):
        source = 'name[Café <|image|>], food[French]'
        target = 'The café serves French food.'
        return {'case_id': 'conditional-a', 'split': 'train', 'source_text': source,
                'target_text': target,
                'source_payload_sha256': sha256(source.encode()).hexdigest(),
                'target_payload_sha256': sha256(target.encode()).hexdigest()}

    def test_exact_escaped_source_and_complete_shifted_target(self):
        module = self.interface(); processor = Processor(); row = self.row()
        actual = module.encode_pair(processor, row, 10, 10, 16)
        rendered = processor.messages[0]['content'][0]['text']
        self.assertNotIn('<|image|>', rendered)
        self.assertEqual(json.loads(rendered.split('json_text=', 1)[1]), row['source_text'])
        self.assertEqual(json.loads(processor.tokenizer.last_text), {'value_text': row['target_text']})
        self.assertEqual(actual['source_ids'], [2, 8, 9])
        self.assertEqual(actual['decoder_inputs'], [2, 5, 6, 7])
        self.assertEqual(actual['labels'], [5, 6, 7, 3])
        self.assertFalse(processor.kwargs.get('truncation', False))

    def test_complete_length_failure_does_not_truncate(self):
        module = self.interface(); processor = Processor()
        with self.assertRaisesRegex(ValueError, 'target.*budget'):
            module.encode_pair(processor, self.row(), 10, 3, 16)
        processor.source_ids = [2] * 11
        with self.assertRaisesRegex(ValueError, 'source.*budget'):
            module.encode_pair(processor, self.row(), 10, 10, 16)

    def test_missing_or_invalid_special_and_vocabulary_ids_rejected(self):
        module = self.interface(); processor = Processor()
        processor.tokenizer.eos_token_id = None
        with self.assertRaisesRegex(ValueError, 'BOS/EOS'):
            module.encode_pair(processor, self.row(), 10, 10, 16)
        processor = Processor(); processor.source_ids = [16]
        with self.assertRaisesRegex(ValueError, 'vocabulary'):
            module.encode_pair(processor, self.row(), 10, 10, 16)

    def test_original_field_hashes_are_verified_before_encoding(self):
        module = self.interface(); processor = Processor(); row = self.row()
        row['target_text'] += ' Changed.'
        with self.assertRaisesRegex(ValueError, 'target.*digest'):
            module.encode_pair(processor, row, 10, 10, 16)
        self.assertFalse(hasattr(processor, 'messages'))

    def test_frequency_statistics_exclude_development_labels(self):
        module = self.interface()
        records = [{'case_id':'a', 'split':'train', 'source_ids':[2,4], 'decoder_inputs':[2,5], 'labels':[5,3]},
                   {'case_id':'b', 'split':'development', 'source_ids':[2,9], 'decoder_inputs':[2,11,11], 'labels':[11,11,3]}]
        result = module.token_statistics(records)
        self.assertEqual(result['rows'], 2)
        self.assertEqual(result['train_positive_label_counts'], {'3':1, '5':1})
        self.assertEqual(result['development_label_tokens'], 3)
        self.assertEqual(result['largest_target_tokens'], 3)
        self.assertEqual(result['largest_source_tokens'], 2)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            module.token_statistics(records + records[:1])

    def test_source_export_is_separate_target_free_and_deduplicated_by_encoding(self):
        module = self.interface()
        self.assertTrue(hasattr(module, 'source_inputs'), 'independent source-only census is missing')
        rows = [{'case_id':'a', 'split':'train', 'source_ids':[2,4], 'semantic_input_sha256':'a'*64,
                 'source_payload_sha256':'c'*64, 'labels':[999]},
                {'case_id':'b', 'split':'train', 'source_ids':[2,4], 'semantic_input_sha256':'a'*64,
                 'source_payload_sha256':'c'*64, 'labels':[888]},
                {'case_id':'c', 'split':'train', 'source_ids':[2,5], 'semantic_input_sha256':'a'*64,
                 'source_payload_sha256':'d'*64, 'labels':[777]}]
        actual = module.source_inputs(rows)
        self.assertEqual(len(actual), 2)
        self.assertEqual(actual, module.source_inputs(list(reversed(rows))))
        for item in actual:
            self.assertNotIn('labels', item)
            self.assertNotIn('decoder_inputs', item)
            self.assertEqual(item['attention_mask'], [1]*len(item['source_ids']))
        self.assertEqual(sorted(len(item['case_ids']) for item in actual), [1, 2])
        rows[1]['split'] = 'development'
        with self.assertRaisesRegex(ValueError, 'processed input.*split'):
            module.source_inputs(rows)

    def test_lossy_targets_and_interior_end_tokens_are_rejected(self):
        module = self.interface(); processor = Processor()
        processor.tokenizer.decode = lambda ids, **kwargs: 'different teacher text'
        with self.assertRaisesRegex(ValueError, 'round.trip'):
            module.encode_pair(processor, self.row(), 10, 10, 16)
        processor = Processor()
        processor.tokenizer.encode = lambda text, **kwargs: [5, 3, 7]
        with self.assertRaisesRegex(ValueError, 'interior.*BOS/EOS'):
            module.encode_pair(processor, self.row(), 10, 10, 16)

    def test_main_connects_complete_pairs_to_independent_source_publication(self):
        module = self.interface(); row = self.row()
        row.update(semantic_input_sha256='a'*64, semantic_reference_count=1)
        processor = Processor()
        fake_transformers = SimpleNamespace(__version__='fixture-only',
            AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: processor))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'config.json').write_text(json.dumps({'text_config':{'vocab_size':16, 'hidden_size':8}}))
            prepared = {'snapshot_path':str(root), 'receipt_sha256':'b'*64,
                        'files':[{'path':'model.safetensors', 'sha256':'c'*64}]}
            with patch.object(module, '_require_private_network_isolation'), \
                 patch.object(module, '_prepared_base', return_value=prepared), \
                 patch.object(module, 'material_rows', return_value=([row], {'receipt_sha256':'d'*64})), \
                 patch.dict(sys.modules, {'transformers':fake_transformers}):
                try:
                    module.main(['--material-dir',str(root),'--material-receipt-file-sha256','e'*64,
                        '--source-manifest',str(root/'manifest.json'),'--source-manifest-sha256','f'*64,
                        '--prepared-base-dir',str(root),'--prepared-base-receipt',str(root/'clone.json'),
                        '--output-dir',str(root/'output')])
                except KeyError as exc:
                    self.fail('main did not supply the complete source-only contract: '+str(exc))
            sources = [json.loads(line) for line in (root/'output/source-inputs.jsonl').read_text().splitlines()]
            receipt = json.loads((root/'output/receipt.json').read_text())
            self.assertEqual(sources[0]['split'], 'train')
            self.assertNotIn('labels', sources[0])
            self.assertEqual(receipt['statistics']['rows'], 1)
            self.assertFalse(receipt['model_weights_loaded'])

    def test_resealed_formation_authority_receipt_rejected_before_pairs_open(self):
        module = self.interface()
        from tests.governance.test_mfm_public_conditional_material import fixture
        from scripts.mfm.prepare_public_conditional_material import stage_material
        pin, train, readme, legal = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root/'cache'; cache.mkdir()
            for name, raw in (('train',train),('readme',readme),('legal_notice',legal)):
                (cache/pin[name]['filename']).write_bytes(raw)
            manifest_path = root/'manifest.json'
            manifest_path.write_bytes(module.canonical_json_bytes(pin))
            manifest_hash = sha256(manifest_path.read_bytes()).hexdigest()
            output = root/'material'
            stage_material(manifest_path, manifest_hash, cache, output)
            original = json.loads((output/'receipt.json').read_text())
            rows, _ = module.material_rows(output, module.digest(output/'receipt.json'), manifest_path, manifest_hash)
            self.assertEqual(len(rows), 3)
            for key, value in (('schema','unsupported'),('material_role','formation_training'),
                               ('authority',True),('qualified',True),('official_splits_used',True),
                               ('formation_training_eligible',True)):
                receipt = {k:v for k,v in original.items() if k != 'receipt_sha256'}
                receipt[key] = value
                receipt['receipt_sha256'] = sha256(module.canonical_json_bytes(receipt)).hexdigest()
                (output/'receipt.json').write_bytes(module.canonical_json_bytes(receipt))
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'conditional.*scope'):
                    module.material_rows(output, module.digest(output/'receipt.json'), manifest_path, manifest_hash)
            for name in ('README.md','CC-BY-SA-4.0-legalcode.txt','source-manifest.json'):
                receipt = {k:v for k,v in original.items() if k != 'receipt_sha256'}
                receipt['files'] = {k:dict(v) for k,v in original['files'].items()}
                old = (output/name).read_bytes(); replacement = old+b' Unbound modification.'
                (output/name).write_bytes(replacement)
                receipt['files'][name] = {'bytes':len(replacement), 'sha256':sha256(replacement).hexdigest()}
                receipt['receipt_sha256'] = sha256(module.canonical_json_bytes(receipt)).hexdigest()
                (output/'receipt.json').write_bytes(module.canonical_json_bytes(receipt))
                with self.subTest(notice=name), self.assertRaisesRegex(ValueError, 'original.*notice'):
                    module.material_rows(output, module.digest(output/'receipt.json'), manifest_path, manifest_hash)
                (output/name).write_bytes(old)


if __name__ == '__main__':
    unittest.main()
