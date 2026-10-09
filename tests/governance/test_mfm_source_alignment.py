"""Connected representation controls with a real small tokenizer; no learning claim."""
from copy import deepcopy
from hashlib import sha256
import importlib
import json
import unittest

import torch
from jinja2 import Environment
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from cognitive_kernel.canonical import CognitiveKernelContractError, canonical_sha256
from cognitive_kernel.contracts import ProductHostScope
from cognitive_kernel.formation_contracts import FormationContextPacket, FormationEvidenceRef
from cognitive_kernel.formation_semantics_v16 import FormationContextV16, RegisteredSourceSensitivity
from scripts.mfm.formation_caller_input import caller_input_record


class Processor:
    def __init__(self):
        vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
        vocab.update({'<unk>': 256, '<s>': 257, '</s>': 258, '<pad>': 259})
        backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token='<unk>'))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        backend.decoder = decoders.ByteLevel()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
            bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>')
        self.mismatch = False
        self.add_special_tokens = False
        self.padding = False

    def apply_chat_template(self, messages, *, chat_template, tokenize, **kwargs):
        text = Environment().from_string(chat_template).render(messages=messages)
        if not tokenize:
            return text
        observed = kwargs.get('processor_kwargs', {})
        result = self.tokenizer(text, add_special_tokens=self.add_special_tokens,
            padding=self.padding, return_tensors='pt', return_offsets_mapping=observed.get('return_offsets_mapping', False),
            return_special_tokens_mask=observed.get('return_special_tokens_mask', False),
            return_attention_mask=True)
        if self.mismatch and not observed.get('return_offsets_mapping', False):
            result['input_ids'][0, 0] += 1
        return result


def caller(texts=('Café "A" \\ <|image|> 😀\r\nCafe\u0301 banana banana', 'Café repeated Café')):
    scope = ProductHostScope.create(product_id='alice', host_instance_id='host-a',
                                   schema_version='1.0.0', encryption_domain='private-a')
    opened = tuple((f'source-{i}', text.encode('utf-8')) for i, text in enumerate(texts))
    evidence = tuple(FormationEvidenceRef(ref_id=ref, scope=scope,
        authority_namespace_id='authority-a', content_digest=sha256(raw).hexdigest(),
        role='hypothetical_training', modality='text', subject_ref='owner') for ref, raw in opened)
    context = FormationContextV16(FormationContextPacket(scope, 'authority-a',
        tuple(ref for ref, _ in opened), evidence),
        tuple(RegisteredSourceSensitivity(ref.ref_id, 'private') for ref in evidence), ())
    return caller_input_record(context, opened)


class SourceAlignmentTests(unittest.TestCase):
    def api(self):
        return importlib.import_module('scripts.mfm.formation_source_alignment')

    def built(self, texts=None):
        api = self.api(); processor = Processor(); record = caller() if texts is None else caller(texts)
        binding = api.processor_binding(processor)
        batch, sidecar = api.text_source_batch_with_alignment(processor, record,
            caller_sha256=record['record_sha256'], processor_binding=binding, max_source_tokens=20000)
        return api, processor, record, binding, batch, sidecar

    def test_actual_batch_and_token_offsets_bind_caller_and_candidates(self):
        api, processor, record, binding, batch, sidecar = self.built()
        self.assertEqual(sidecar['schema'], 'mfm-text-source-alignment-v1')
        self.assertEqual(sidecar['input_ids'], batch['input_ids'][0].tolist())
        self.assertEqual(sidecar['attention_mask'], batch['attention_mask'][0].tolist())
        self.assertEqual(sidecar['caller_sha256'], record['record_sha256'])
        self.assertEqual(sidecar['processor_binding'], binding)
        self.assertEqual([row['ref_id'] for row in sidecar['sources']], ['source-0', 'source-1'])
        self.assertTrue(any(row['source_ref'] is None for row in sidecar['tokens']))
        self.assertTrue(any(row['source_ref'] == 'source-0' for row in sidecar['tokens']))
        self.assertEqual(api.verify_source_alignment(processor, record, sidecar,
            caller_sha256=record['record_sha256'], alignment_sha256=sidecar['record_sha256'],
            processor_binding=binding, max_source_tokens=20000), sidecar)

    def test_actual_processor_added_special_is_retained_and_ineligible(self):
        api = self.api(); processor = Processor(); record = caller(('Café banana',))
        processor.tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
            single='<s> $A', special_tokens=[('<s>', 257)])
        processor.add_special_tokens = True
        binding = api.processor_binding(processor)
        context, opened, _ = api._caller(record, record['record_sha256'])
        baseline = api.v16.source_batch_v16(processor, context, opened, 20000)
        batch, sidecar = api.text_source_batch_with_alignment(processor, record,
            caller_sha256=record['record_sha256'], processor_binding=binding, max_source_tokens=20000)
        self.assertEqual(batch['input_ids'].tolist(), baseline['input_ids'].tolist())
        self.assertEqual(sidecar['input_ids'], baseline['input_ids'][0].tolist())
        self.assertEqual(sidecar['input_ids'][0], 257)
        self.assertEqual(sidecar['tokens'][0]['rendered_chars'], [0, 0])
        self.assertFalse(sidecar['tokens'][0]['observed_source_eligible'])
        self.assertIsNone(sidecar['tokens'][0]['source_ref'])
        self.assertTrue(any(t['source_ref'] == 'source-0' for t in sidecar['tokens']))
        self.assertEqual(api.verify_source_alignment(processor, record, sidecar,
            caller_sha256=record['record_sha256'], alignment_sha256=sidecar['record_sha256'],
            processor_binding=binding, max_source_tokens=20000), sidecar)

    def test_processor_runtime_padding_truncation_preserved_for_exact_reload(self):
        for configured in (False, True):
            with self.subTest(configured=configured):
                api = self.api(); processor = Processor(); record = caller(('é banana',))
                processor.padding = True
                if configured:
                    processor.tokenizer.backend_tokenizer.enable_padding(length=8000, direction='left')
                    processor.tokenizer.backend_tokenizer.enable_truncation(max_length=9000)
                binding = api.processor_binding(processor)
                batch, sidecar = api.text_source_batch_with_alignment(processor, record,
                    caller_sha256=record['record_sha256'], processor_binding=binding, max_source_tokens=20000)
                self.assertEqual(api.processor_binding(processor), binding)
                self.assertEqual(api.verify_source_alignment(processor, record, sidecar,
                    caller_sha256=record['record_sha256'], alignment_sha256=sidecar['record_sha256'],
                    processor_binding=binding, max_source_tokens=20000), sidecar)

    def test_unicode_escape_atoms_have_exact_original_byte_boundaries(self):
        api, _, record, _, _, sidecar = self.built()
        raw = __import__('base64').b64decode(record['opened_sources'][0]['payload_base64'])
        row = sidecar['sources'][0]
        for atom in row['characters']:
            a, b = atom['original_utf8']; x, y = atom['rendered_chars']
            value = raw[a:b].decode('utf-8')
            self.assertEqual(json.loads('"' + sidecar['rendered_text'][x:y] + '"'), value)
        emoji = next(atom for atom in row['characters'] if raw[slice(*atom['original_utf8'])] == '😀'.encode())
        self.assertEqual(emoji['original_utf8'][1] - emoji['original_utf8'][0], 4)
        self.assertEqual(emoji['rendered_chars'][1] - emoji['rendered_chars'][0], 12)
        self.assertNotIn('<|image|>', sidecar['rendered_text'])
        self.assertEqual(sidecar['rendered_text'].count('\\u003c'), 1)

    def test_all_exact_repeated_and_overlapping_occurrences_survive(self):
        api, _, record, _, _, sidecar = self.built(('banana banana', 'banana'))
        result = api.literal_alignment(record, sidecar, 'ana',
            caller_sha256=record['record_sha256'], alignment_sha256=sidecar['record_sha256'])
        self.assertEqual([(row['source_ref'], row['original_utf8']) for row in result['occurrences']],
            [('source-0', [1, 4]), ('source-0', [3, 6]), ('source-0', [8, 11]),
             ('source-0', [10, 13]), ('source-1', [1, 4]), ('source-1', [3, 6])])
        self.assertTrue(all(row['complete_eligible_token_coverage'] for row in result['occurrences']))
        self.assertEqual(json.loads(bytes.fromhex(result['canonical_literal_utf8_hex']).decode()), 'ana')

    def test_canonical_output_differs_from_source_escape_and_preserves_nfd(self):
        api, _, record, _, _, sidecar = self.built()
        literal = 'Café "A" \\ <|image|> 😀\r\n'
        result = api.literal_alignment(record, sidecar, literal,
            caller_sha256=record['record_sha256'], alignment_sha256=sidecar['record_sha256'])
        self.assertEqual(len(result['occurrences']), 1)
        canonical = bytes.fromhex(result['canonical_literal_utf8_hex'])
        self.assertIn('é'.encode(), canonical)
        self.assertIn('😀'.encode(), canonical)
        self.assertEqual(json.loads(canonical), literal)
        nfd = api.literal_alignment(record, sidecar, 'Cafe\u0301',
            caller_sha256=record['record_sha256'], alignment_sha256=sidecar['record_sha256'])
        self.assertEqual(len(nfd['occurrences']), 1)

    def test_context_mentions_and_paraphrases_never_become_observed_literals(self):
        api, _, record, _, _, sidecar = self.built(('just evidence',))
        for literal in ('authority-a', 'source-0', 'Interpret authorized', 'an authored paraphrase'):
            self.assertEqual(api.literal_alignment(record, sidecar, literal,
                caller_sha256=record['record_sha256'], alignment_sha256=sidecar['record_sha256'])['occurrences'], [])

    def test_external_caller_processor_and_sidecar_pins_refuse_replacement(self):
        api, processor, record, binding, _, sidecar = self.built()
        with self.assertRaises(CognitiveKernelContractError):
            api.text_source_batch_with_alignment(processor, record, caller_sha256='0'*64,
                processor_binding=binding, max_source_tokens=20000)
        changed = deepcopy(binding); changed['tokenizer_backend_sha256'] = '0'*64
        with self.assertRaises(CognitiveKernelContractError):
            api.text_source_batch_with_alignment(processor, record, caller_sha256=record['record_sha256'],
                processor_binding=changed, max_source_tokens=20000)
        altered = deepcopy(sidecar); altered['tokens'][0]['source_ref'] = 'source-0'
        altered['record_sha256'] = canonical_sha256({k:v for k,v in altered.items() if k!='record_sha256'})
        with self.assertRaises(CognitiveKernelContractError):
            api.verify_source_alignment(processor, record, altered,
                caller_sha256=record['record_sha256'], alignment_sha256=altered['record_sha256'],
                processor_binding=binding, max_source_tokens=20000)

    def test_processor_ids_must_equal_exact_rendered_tokenization(self):
        api, processor, record, binding, _, _ = self.built()
        processor.mismatch = True
        with self.assertRaisesRegex(CognitiveKernelContractError, 'IDs'):
            api.text_source_batch_with_alignment(processor, record,
                caller_sha256=record['record_sha256'], processor_binding=binding, max_source_tokens=20000)

    def test_observed_processor_geometry_refuses_missing_malformed_or_changed_fields(self):
        for fault in ('missing-offsets', 'offset-shape', 'negative-offset', 'special-type', 'changed-mask'):
            with self.subTest(fault=fault):
                api = self.api(); processor = Processor(); record = caller(('é',))
                processor.padding = True
                binding = api.processor_binding(processor); original = processor.apply_chat_template
                def observed(*args, **kwargs):
                    result = original(*args, **kwargs)
                    if kwargs.get('processor_kwargs', {}).get('return_offsets_mapping', False):
                        if fault == 'missing-offsets': result.pop('offset_mapping')
                        elif fault == 'offset-shape': result['offset_mapping'] = result['offset_mapping'][0]
                        elif fault == 'negative-offset': result['offset_mapping'][0, 0, 0] = -1
                        elif fault == 'special-type': result['special_tokens_mask'] = result['special_tokens_mask'].float()
                        elif fault == 'changed-mask': result['attention_mask'][0, 0] = 0
                    return result
                processor.apply_chat_template = observed
                with self.assertRaises(CognitiveKernelContractError):
                    api.text_source_batch_with_alignment(processor, record,
                        caller_sha256=record['record_sha256'], processor_binding=binding, max_source_tokens=20000)
                self.assertEqual(api.processor_binding(processor), binding)

    def test_processing_semantic_tokenizer_change_is_not_restored_or_admitted(self):
        api = self.api(); processor = Processor(); record = caller(('é',))
        processor.padding = True
        binding = api.processor_binding(processor); original = processor.apply_chat_template
        def mutated(*args, **kwargs):
            result = original(*args, **kwargs)
            if kwargs.get('processor_kwargs', {}).get('return_offsets_mapping', False):
                processor.tokenizer.add_tokens(['public-semantic-change'])
            return result
        processor.apply_chat_template = mutated
        with self.assertRaisesRegex(CognitiveKernelContractError, 'binding changed'):
            api.text_source_batch_with_alignment(processor, record,
                caller_sha256=record['record_sha256'], processor_binding=binding, max_source_tokens=20000)
        self.assertIsNone(processor.tokenizer.backend_tokenizer.padding)
        self.assertNotEqual(api.processor_binding(processor), binding)

    def test_native_and_overflow_refuse_before_claiming_alignment(self):
        api, processor, record, binding, _, _ = self.built()
        native = deepcopy(record)
        native['context']['base_context']['evidence'][0]['modality'] = 'image'
        # Reconstructing a valid native input is the caller's responsibility.
        from cognitive_kernel.formation_semantics_v16 import context_v16_from_record
        context = context_v16_from_record(native['context'])
        opened = tuple((row['ref_id'], __import__('base64').b64decode(row['payload_base64'])) for row in record['opened_sources'])
        native = caller_input_record(context, opened)
        with self.assertRaisesRegex(CognitiveKernelContractError, 'native'):
            api.text_source_batch_with_alignment(processor, native,
                caller_sha256=native['record_sha256'], processor_binding=binding, max_source_tokens=20000)
        with self.assertRaises(CognitiveKernelContractError):
            api.text_source_batch_with_alignment(processor, record,
                caller_sha256=record['record_sha256'], processor_binding=binding, max_source_tokens=5)

    def test_empty_literal_refuses_and_original_occurrence_is_not_normalized(self):
        api, _, record, _, _, sidecar = self.built(('Cafe\u0301',))
        with self.assertRaises(CognitiveKernelContractError):
            api.literal_alignment(record, sidecar, '', caller_sha256=record['record_sha256'],
                                  alignment_sha256=sidecar['record_sha256'])
        self.assertEqual(api.literal_alignment(record, sidecar, 'Café', caller_sha256=record['record_sha256'],
            alignment_sha256=sidecar['record_sha256'])['occurrences'], [])


if __name__ == '__main__':
    unittest.main()
