"""Complete canonical/actual-token geometry; tiny public fixtures, no learning."""
from base64 import b64encode
from copy import deepcopy
from hashlib import sha256
import importlib
import importlib.util
import json
import unittest
from unittest.mock import patch

from tokenizers import models, processors

from cognitive_kernel.canonical import CognitiveKernelContractError, canonical_json_bytes, canonical_sha256
from cognitive_kernel.formation_learning_v16 import learning_example_v16_from_record, supervised_output_record_v16
from scripts.mfm.formation_caller_input import caller_input_record
from scripts.mfm import formation_source_alignment as source
from scripts.mfm import train_v16_formation_specialist as v16
from tests.governance.test_formation_learning_v16 import case_record
from tests.governance.test_mfm_source_alignment import Processor


def example_for_text(text='Café "A" \\ 😀\r\nCafe\u0301 banana banana', value=None):
    row = case_record()
    raw = text.encode('utf-8')
    row['sources'][0]['content_b64'] = b64encode(raw).decode('ascii')
    row['context']['base_context']['evidence'][0]['content_digest'] = sha256(raw).hexdigest()
    proposal = row['target']['proposals'][0]['proposal']
    proposal['anchors'][0]['end_byte'] = len(raw)
    proposal['value_text'] = text if value is None else value
    return learning_example_v16_from_record(row)


def reseal(record):
    record['record_sha256'] = canonical_sha256({k: v for k, v in record.items() if k != 'record_sha256'})
    return record


class TargetGeometryTests(unittest.TestCase):
    def api(self):
        name = 'scripts.mfm.formation_target_alignment'
        self.assertIsNotNone(importlib.util.find_spec(name), 'full canonical target alignment is not implemented')
        return importlib.import_module(name)

    def geometry(self, record):
        api = self.api()
        return api.canonical_field_geometry(record, expected_target_sha256=sha256(canonical_json_bytes(record)).hexdigest())

    def test_all_scalar_types_keep_distinct_lexical_and_structural_spans(self):
        record = {'a/b': {'~key': [None, True, False, 1, 1.0, -0.0, 1e20, 'null', '']},
                  'again': 'repeat', 'empty_object': {}, 'empty_array': [], 'z': ['repeat', 'repeat']}
        geometry = self.geometry(record)
        raw = canonical_json_bytes(record)
        self.assertEqual(bytes.fromhex(geometry['canonical_utf8_hex']), raw)
        nodes = {row['pointer']: row for row in geometry['nodes']}
        self.assertEqual(nodes['']['kind'], 'object')
        for index, kind, lexeme in ((0, 'null', b'null'), (1, 'boolean', b'true'),
                                    (2, 'boolean', b'false'), (3, 'integer', b'1'),
                                    (4, 'number', b'1.0'), (5, 'number', b'-0.0'),
                                    (6, 'number', b'1e+20'), (7, 'string', b'"null"'),
                                    (8, 'string', b'""')):
            row = nodes[f'/a~1b/~0key/{index}']
            self.assertEqual(row['kind'], kind)
            self.assertEqual(raw[slice(*row['canonical_utf8'])], lexeme)
        self.assertEqual(nodes['/empty_array']['kind'], 'array')
        self.assertEqual(raw[slice(*nodes['/empty_array']['canonical_utf8'])], b'[]')
        self.assertEqual(raw[slice(*nodes['/empty_object']['canonical_utf8'])], b'{}')
        self.assertEqual(nodes['/a~1b/~0key/8']['characters'], [])
        self.assertEqual([n['pointer'] for n in geometry['nodes'] if n.get('value') == 'repeat'],
                         ['/again', '/z/0', '/z/1'])
        self.assertEqual(len({tuple(nodes[p]['canonical_utf8']) for p in ('/again', '/z/0', '/z/1')}), 3)

    def test_unicode_escape_atoms_map_value_bytes_into_complete_canonical_context(self):
        value = 'é " \\ \r\n\t\b\f\x01 Cafe\u0301 😀 <|image|>'
        record = {'é/key~': value, 'value': value}
        geometry = self.geometry(record)
        raw = canonical_json_bytes(record)
        text = raw.decode('utf-8')
        node = next(n for n in geometry['nodes'] if n['pointer'] == '/é~1key~0')
        encoded_value = value.encode('utf-8')
        self.assertEqual(len(node['characters']), len(value))
        for atom in node['characters']:
            char = encoded_value[slice(*atom['value_utf8'])].decode('utf-8')
            self.assertEqual(json.loads('"' + text[slice(*atom['canonical_chars'])] + '"'), char)
            self.assertEqual(text[slice(*atom['canonical_chars'])].encode('utf-8'), raw[slice(*atom['canonical_utf8'])])
        self.assertIn(b'\\r\\n', raw)
        self.assertIn('Cafe\u0301'.encode('utf-8'), raw)
        self.assertNotIn(b'\\ud83d', raw)
        key = next(x for x in geometry['lexemes'] if x['pointer'] == '/é~1key~0' and x['role'] == 'key')
        self.assertEqual(json.loads(raw[slice(*key['canonical_utf8'])]), 'é/key~')
        self.assertLess(key['canonical_utf8'][1], node['canonical_utf8'][0])

    def test_generic_scalar_root_and_empty_key_have_unambiguous_pointers(self):
        for value, kind in ((None, 'null'), (True, 'boolean'), (2, 'integer'), (2.5, 'number'), ('', 'string')):
            with self.subTest(value=value):
                geometry = self.geometry(value)
                self.assertEqual([(n['pointer'], n['kind']) for n in geometry['nodes']], [('', kind)])
        nodes = {n['pointer']: n for n in self.geometry({'': {'/': {'~01': 'same'}}})['nodes']}
        self.assertEqual(nodes['//~1/~001']['value'], 'same')

    def test_noncanonical_material_and_external_target_pin_refuse(self):
        api = self.api()
        for record in (float('nan'), float('inf'), {1: 'key'}, {'x': ('tuple',)}, {'x': b'bytes'}, {'x': '\ud800'}):
            with self.subTest(record=repr(record)), self.assertRaises(CognitiveKernelContractError):
                api.canonical_field_geometry(record, expected_target_sha256='0' * 64)
        with self.assertRaises(CognitiveKernelContractError):
            api.canonical_field_geometry({'a': 'b'}, expected_target_sha256='0' * 64)


class ActualTargetTests(TargetGeometryTests):
    def built(self, *, text=None, value=None, merged=False, merge_pair=None, configured=False):
        api = self.api(); processor = Processor()
        if merged or merge_pair:
            pair = ('{', '"') if merged else merge_pair
            vocab = processor.tokenizer.backend_tokenizer.get_vocab()
            vocab[''.join(pair)] = max(vocab.values()) + 1
            processor.tokenizer.backend_tokenizer.model = models.BPE(vocab=vocab, merges=[pair], unk_token='<unk>')
        processor.tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
            single='<s> $A', special_tokens=[('<s>', 257)])
        processor.add_special_tokens = True
        if configured:
            processor.tokenizer.backend_tokenizer.enable_padding(length=9000, direction='left')
            processor.tokenizer.backend_tokenizer.enable_truncation(max_length=9500)
        example = example_for_text(value=value) if text is None else example_for_text(text, value=value)
        caller = caller_input_record(example.context, example.opened_sources)
        binding = source.processor_binding(processor)
        _, sidecar = source.text_source_batch_with_alignment(processor, caller,
            caller_sha256=caller['record_sha256'], processor_binding=binding, max_source_tokens=20000)
        target_sha = sha256(canonical_json_bytes(supervised_output_record_v16(example))).hexdigest()
        kwargs = {'caller_sha256': caller['record_sha256'], 'source_alignment': sidecar,
                  'source_alignment_sha256': sidecar['record_sha256'], 'processor_binding': binding,
                  'target_sha256': target_sha, 'max_source_tokens': 20000, 'max_target_tokens': 5000}
        inputs, labels, alignment = api.target_batch_with_alignment(processor, caller, example, **kwargs)
        return api, processor, caller, example, binding, sidecar, kwargs, inputs, labels, alignment

    def test_complete_original_target_and_decoder_prediction_positions_match(self):
        api, processor, caller, example, binding, _, kwargs, inputs, labels, alignment = self.built()
        with source._processor_execution_settings(processor):
            expected_inputs, expected_labels = v16._target_ids(processor.tokenizer, example, 5000)
        self.assertEqual((inputs, labels), (expected_inputs, expected_labels))
        self.assertEqual(alignment['profile'], 'output_v16')
        self.assertEqual(inputs[0], 257)
        self.assertEqual(labels[-1], 258)
        self.assertEqual(inputs[1:], labels[:-1])
        self.assertEqual(alignment['scaffolding'], {'bos': {'decoder_input_index': 0, 'token_id': 257},
            'eos': {'label_index': len(labels) - 1, 'query_index': len(labels) - 1, 'token_id': 258}})
        for j, token in enumerate(alignment['tokens']):
            self.assertEqual((token['body_index'], token['label_index'], token['query_index'], token['decoder_input_index']),
                             (j, j, j, j + 1))
            self.assertEqual(token['token_id'], labels[j])
        self.assertEqual(source.processor_binding(processor), binding)
        parsed = api.validate_target_alignment_record(alignment, target_alignment_sha256=alignment['record_sha256'])
        self.assertEqual(parsed, supervised_output_record_v16(example))
        self.assertEqual(api.verify_target_alignment(processor, caller, example, alignment,
            target_alignment_sha256=alignment['record_sha256'], **kwargs), alignment)

    def test_contextual_token_crossing_key_and_structure_retains_all_links(self):
        _, _, _, _, _, _, _, _, _, alignment = self.built(merged=True)
        token = next(t for t in alignment['tokens'] if t['canonical_chars'] == [0, 2])
        self.assertIn('open', {row['role'] for row in token['links']})
        self.assertIn('key', {row['role'] for row in token['links']})
        self.assertIn('container', {row['role'] for row in token['links']})
        self.assertTrue(any(not row['fully_contained'] for row in token['links'] if row['role'] != 'container'))
        self.assertNotIn('winning_field', token)

    def test_overlapping_unicode_offsets_are_retained_without_dropping_tokens(self):
        _, _, _, _, _, _, _, _, labels, alignment = self.built()
        offsets = [tuple(t['canonical_chars']) for t in alignment['tokens']]
        self.assertLess(len(set(offsets)), len(offsets))
        self.assertEqual(len(alignment['tokens']), len(labels) - 1)
        value = next(n for n in alignment['geometry']['nodes'] if n['pointer'].endswith('/value_text'))
        self.assertTrue(any(t['canonical_chars'][0] < value['canonical_chars'][1] and
                            t['canonical_chars'][1] > value['canonical_chars'][0] for t in alignment['tokens']))

    def test_body_special_id_is_not_silently_deleted_as_added_scaffolding(self):
        _, _, _, _, _, _, _, inputs, labels, alignment = self.built(value='Literal <s> marker.')
        self.assertIn(257, labels[:-1])
        self.assertEqual(len(alignment['tokens']), len(labels) - 1)
        marker = next(t for t in alignment['tokens'] if t['token_id'] == 257)
        self.assertGreater(marker['canonical_chars'][1], marker['canonical_chars'][0])
        self.assertEqual(inputs[marker['decoder_input_index']], 257)

    def test_explicit_source_and_target_budgets_and_unknown_adjudications_refuse(self):
        api, processor, caller, example, _, _, kwargs, _, _, _ = self.built()
        for name, value in (('max_source_tokens', 1), ('max_target_tokens', 1), ('max_source_tokens', True),
                            ('max_target_tokens', True), ('target_sha256', '0' * 64), ('caller_sha256', '0' * 64)):
            with self.subTest(name=name), self.assertRaises(CognitiveKernelContractError):
                api.target_batch_with_alignment(processor, caller, example, **{**kwargs, name: value})
        row = case_record(); row['target']['adjudications']['relationship'] = 'unknown'
        unknown = learning_example_v16_from_record(row)
        unknown_caller = caller_input_record(unknown.context, unknown.opened_sources)
        bind = source.processor_binding(processor)
        _, sc = source.text_source_batch_with_alignment(processor, unknown_caller,
            caller_sha256=unknown_caller['record_sha256'], processor_binding=bind, max_source_tokens=20000)
        with self.assertRaisesRegex(CognitiveKernelContractError, 'unknown'):
            api.target_batch_with_alignment(processor, unknown_caller, unknown,
                **{**kwargs, 'caller_sha256': unknown_caller['record_sha256'], 'source_alignment': sc,
                   'source_alignment_sha256': sc['record_sha256']})

    def test_forged_geometry_shift_and_profile_refuse_even_after_resealing(self):
        api, processor, caller, example, _, _, kwargs, _, _, alignment = self.built()
        for mutate in (lambda row: row['tokens'][0].update(label_index=1),
                       lambda row: row['geometry']['nodes'][0].update(canonical_utf8=[0, 1]),
                       lambda row: row.update(profile='local_decision_v16'),
                       lambda row: row['labels'].__setitem__(0, row['labels'][0] + 1)):
            changed = deepcopy(alignment); mutate(changed); reseal(changed)
            with self.subTest(mutate=mutate), self.assertRaises(CognitiveKernelContractError):
                api.verify_target_alignment(processor, caller, example, changed,
                    target_alignment_sha256=changed['record_sha256'], **kwargs)

    def test_runtime_execution_modes_restore_on_success_and_target_exception(self):
        api, processor, caller, example, binding, _, kwargs, _, _, alignment = self.built(configured=True)
        self.assertEqual(source.processor_binding(processor), binding)
        with self.assertRaises(CognitiveKernelContractError):
            api.target_batch_with_alignment(processor, caller, example, **{**kwargs, 'max_target_tokens': 1})
        self.assertEqual(source.processor_binding(processor), binding)
        self.assertEqual(api.verify_target_alignment(processor, caller, example, alignment,
            target_alignment_sha256=alignment['record_sha256'], **kwargs), alignment)

    def test_malformed_actual_offsets_masks_ids_and_semantic_mutation_refuse(self):
        api, processor, caller, example, binding, _, kwargs, _, _, _ = self.built()
        tokenizer_class = type(processor.tokenizer); real = tokenizer_class.__call__
        def fault(mutate):
            def observed(tokenizer, text, *args, **kw):
                encoded = real(tokenizer, text, *args, **kw)
                if text.startswith('{') and kw.get('return_offsets_mapping'):
                    mutate(encoded)
                return encoded
            return observed
        mutations = (lambda row: row['input_ids'].__setitem__(0, True),
                     lambda row: row['input_ids'].__setitem__(0, row['input_ids'][0] + 1),
                     lambda row: row['attention_mask'].__setitem__(0, 0),
                     lambda row: row['offset_mapping'].__setitem__(0, (-1, 0)),
                     lambda row: row['offset_mapping'].__setitem__(0, (1, 0)),
                     lambda row: row['offset_mapping'].__setitem__(0, (0, 999999)),
                     lambda row: row['offset_mapping'].__setitem__(0, (0.0, 1)),
                     lambda row: row['offset_mapping'].pop(),
                     lambda row: row['special_tokens_mask'].__setitem__(0, 2))
        for mutation in mutations:
            with self.subTest(mutation=mutation), patch.object(tokenizer_class, '__call__', fault(mutation)):
                with self.assertRaises(CognitiveKernelContractError):
                    api.target_batch_with_alignment(processor, caller, example, **kwargs)
            self.assertEqual(source.processor_binding(processor), binding)
        def semantic_change(row):
            processor.tokenizer.backend_tokenizer.add_tokens(['unexpected_semantic_token'])
        with patch.object(tokenizer_class, '__call__', fault(semantic_change)):
            with self.assertRaisesRegex(CognitiveKernelContractError, 'binding changed'):
                api.target_batch_with_alignment(processor, caller, example, **kwargs)

    def test_returned_geometry_and_labels_do_not_mutate_caller_or_example(self):
        api, _, caller, example, _, _, _, inputs, labels, alignment = self.built()
        before = deepcopy(caller); output = deepcopy(supervised_output_record_v16(example))
        inputs[0] = 0; labels[0] = 0
        alignment['geometry']['nodes'][0]['path'].append('bad')
        alignment['processor_binding']['tokenizer_wrapper']['all_special_ids'].append(0)
        self.assertEqual(caller, before)
        self.assertEqual(supervised_output_record_v16(example), output)

    def test_foreign_caller_refuses_before_any_teacher_target_access(self):
        api, processor, _, example, binding, _, kwargs, _, _, _ = self.built()
        other = example_for_text('Different independently frozen source.')
        foreign = caller_input_record(other.context, other.opened_sources)
        _, sc = source.text_source_batch_with_alignment(processor, foreign,
            caller_sha256=foreign['record_sha256'], processor_binding=binding, max_source_tokens=20000)
        with patch.object(api.learning, 'supervised_output_record_v16', side_effect=AssertionError('teacher target accessed')) as teacher:
            with self.assertRaisesRegex(CognitiveKernelContractError, 'independently frozen caller'):
                api.target_batch_with_alignment(processor, foreign, example, **{**kwargs,
                    'caller_sha256': foreign['record_sha256'], 'source_alignment': sc,
                    'source_alignment_sha256': sc['record_sha256']})
            teacher.assert_not_called()


if __name__ == '__main__':
    unittest.main()
