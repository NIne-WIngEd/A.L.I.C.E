"""Complete target-free processing with a real first-party tiny tokenizer."""
from copy import deepcopy
import importlib
import json
import unittest
from unittest.mock import patch

from tokenizers import processors
from cognitive_kernel.canonical import canonical_sha256, CognitiveKernelContractError
from tests.governance import test_mfm_caller_meaning as meanings
from tests.governance import test_mfm_source_alignment as sources
from scripts.mfm import formation_source_alignment as source


def source_identity():
    return meanings.sealed({'schema': 'mfm-joint-source-identity-v1',
        'repository': 'fictional/public-text-source', 'revision': 'fictional-revision-1',
        'source_receipt_sha256': '3' * 64, 'source_frozen': True, 'precision': 'bf16'})


def companion(value=None):
    from scripts.mfm import formation_caller_meaning as api
    caller, snapshot, entries, _ = meanings.inputs(value)
    policy = api.normative_view_policy()
    record = api.caller_meaning_record(caller, caller_sha256=caller['record_sha256'],
        snapshot=snapshot, snapshot_sha256=snapshot['record_sha256'], candidate_meanings=entries,
        view_policy=policy, view_policy_sha256=policy['record_sha256'])
    return caller, record


class JointInputTests(unittest.TestCase):
    def api(self):
        return importlib.import_module('scripts.mfm.formation_joint_input')

    def build(self, *, value=None, processor=None):
        api = self.api(); caller, meaning = companion(value)
        processor = processor or sources.Processor(); identity = source_identity()
        kwargs = {'caller_sha256': caller['record_sha256'], 'meaning_sha256': meaning['record_sha256'],
            'processor_binding': source.processor_binding(processor), 'source_identity': identity,
            'source_identity_sha256': identity['record_sha256'], 'max_source_tokens': 50000}
        batch, sidecar = api.joint_source_batch_with_alignment(processor, caller, meaning, **kwargs)
        return api, processor, caller, meaning, kwargs, batch, sidecar

    def test_whole_processor_ids_masks_and_target_free_reload(self):
        with patch('cognitive_kernel.formation_learning_v16.supervised_output_record_v16',
                   side_effect=AssertionError('teacher inaccessible')):
            api, p, caller, meaning, kwargs, batch, row = self.build()
            self.assertEqual(api.verify_joint_alignment(p, caller, meaning, row,
                alignment_sha256=row['record_sha256'], **kwargs), row)
        self.assertEqual(row['input_ids'], batch['input_ids'][0].tolist())
        self.assertEqual(row['attention_mask'], batch['attention_mask'][0].tolist())
        self.assertEqual([b['section'] for b in row['blocks']], ['task_context_snapshot', 'observation',
            *(['candidate_meaning'] * 10), 'candidate_index'])
        self.assertFalse(row['teacher_accessed']); self.assertFalse(row['learning_qualified'])
        self.assertFalse(row['native_locator_alignment']); self.assertFalse(row['authority_granted'])

    def test_original_utf8_and_nested_escapes_join_exact_character_atoms(self):
        _, _, caller, _, _, _, row = self.build()
        from base64 import b64decode
        raw = b64decode(caller['opened_sources'][0]['payload_base64'])
        original = row['sources'][0]
        self.assertEqual(original['original_bytes'], len(raw))
        for atom in original['characters']:
            rendered = row['rendered_text'][slice(*atom['rendered_chars'])]
            decoded = json.loads('"' + json.loads('"' + rendered + '"') + '"')
            self.assertEqual(decoded.encode(), raw[slice(*atom['original_utf8'])])
        self.assertNotIn('<|image|>', row['rendered_text'])
        self.assertTrue(any(not t['full_character_boundaries'] and t['observed_source_eligible'] for t in row['tokens']))

    def test_roles_candidates_index_and_source_authority_remain_separate(self):
        _, _, _, meaning, _, _, row = self.build()
        self.assertEqual(len(row['candidates']), 10)
        self.assertEqual([(c['ref_id'], c['kind']) for c in row['candidates'][1:3]],
                         [('owner', 'source_subject'), ('owner', 'entity')])
        for candidate in row['candidates']:
            self.assertTrue(candidate['meaning_token_indices']); self.assertTrue(candidate['index_token_indices'])
            self.assertEqual(candidate['eligible_paths'], meaning['candidate_inventory'][candidate['slot']]['eligible_paths'])
            self.assertGreater(min(candidate['index_token_indices']), max(candidate['meaning_token_indices']))
        observation = row['sources'][0]['rendered_content_chars']
        for token in row['tokens']:
            if token['observed_source_eligible']:
                self.assertTrue(observation[0] <= token['rendered_chars'][0] < token['rendered_chars'][1] <= observation[1])
                self.assertEqual(token['source_ref'], 'source-0')
        self.assertTrue(any(t['candidate_slots'] and not t['observed_source_eligible'] for t in row['tokens']))

    def test_all_scalar_container_key_and_structure_overlaps_survive(self):
        _, _, _, _, _, _, row = self.build()
        fields = [f for f in row['fields'] if f['section'] == 'candidate_meaning' and f['candidate_slot'] == 6]
        self.assertTrue({'object', 'array', 'string', 'boolean', 'integer', 'number', 'null'} <=
                        {f.get('kind') for f in fields})
        self.assertTrue({'key', 'value', 'open', 'close', 'comma', 'colon', 'container'} <= {f['role'] for f in fields})
        self.assertTrue(any('/literal~1~0' in f['pointer'] for f in fields))
        self.assertTrue(any(len(t['field_indices']) > 2 for t in row['tokens']))
        for index, field in enumerate(row['fields']):
            self.assertTrue(field['supporting_token_indices'])
            for token_index in field['supporting_token_indices']:
                self.assertIn(index, row['tokens'][token_index]['field_indices'])

    def test_supplied_content_changes_view_but_never_invents_a_body(self):
        _, _, _, _, _, _, first = self.build()
        _, _, _, meaning, _, _, second = self.build(value={'literal/~': 'changed', 'count': 3,
            'weight': 0.75, 'flag': False, 'missing': None, 'list': ['ana', 'ana']})
        self.assertNotEqual(first['input_ids'], second['input_ids'])
        self.assertNotEqual(first['cache']['processed_view_sha256'], second['cache']['processed_view_sha256'])
        self.assertEqual(meaning['candidate_meanings'][7]['availability'], 'metadata_only')
        self.assertIsNone(meaning['candidate_meanings'][7]['body_base64'])
        self.assertIn('"content":null', json.loads(second['blocks'][9]['json_string']))

    def test_supplied_body_file_bytes_join_every_original_canonical_field(self):
        from base64 import b64decode
        _, _, _, meaning, _, _, row = self.build()
        matched = 0
        for field in row['fields']:
            if field['body_file_sha256'] is None: continue
            entry = meaning['candidate_meanings'][field['candidate_slot']]
            raw = b64decode(entry['body_base64'])
            self.assertEqual(field['body_file_sha256'], entry['body_sha256'])
            rendered = row['rendered_text'][slice(*field['rendered_chars'])]
            self.assertEqual(json.loads('"' + rendered + '"').encode(), raw[slice(*field['body_original_utf8'])])
            matched += 1
        self.assertGreater(matched, 100)

    def test_explicit_content_absence_changes_view_without_target_or_new_prior(self):
        from scripts.mfm import formation_caller_meaning as M
        api, p, caller, meaning, kwargs, _, old = self.build()
        entries = deepcopy(meaning['candidate_meanings']); entries[7]['availability'] = 'unavailable'
        policy = M.normative_view_policy()
        changed = M.caller_meaning_record(caller, caller_sha256=caller['record_sha256'],
            snapshot=meaning['snapshot'], snapshot_sha256=meaning['snapshot']['record_sha256'],
            candidate_meanings=entries, view_policy=policy, view_policy_sha256=policy['record_sha256'])
        with patch('scripts.mfm.v16_prior_memory_inputs.local_uncertainty_allocations',
                   side_effect=AssertionError('output declarations inaccessible')), patch(
                   'cognitive_kernel.formation_learning_v16.supervised_output_record_v16',
                   side_effect=AssertionError('teacher inaccessible')):
            _, current = api.joint_source_batch_with_alignment(p, caller, changed,
                **{**kwargs, 'meaning_sha256': changed['record_sha256']})
        self.assertNotEqual(current['input_ids'], old['input_ids'])
        self.assertEqual(current['candidate_inventory_sha256'], old['candidate_inventory_sha256'])
        self.assertNotIn('v16-seed-18-interview-date-unresolved', current['rendered_text'])
        self.assertIsNone(changed['candidate_meanings'][7]['body_base64'])

    def test_registered_candidate_order_and_rename_preserve_typed_roles(self):
        from dataclasses import replace
        from base64 import b64decode
        from scripts.mfm import formation_caller_input as C, formation_caller_meaning as M
        api, p, caller, meaning, kwargs, _, _ = self.build()
        context, opened, registry = C.caller_input_from_record(caller)
        rows = {(c['ref_id'], c['kind'], c['origin']): e for c, e in
                zip(meaning['candidate_inventory'], meaning['candidate_meanings'])}
        variants = [replace(context, targets=tuple(reversed(context.targets))),
                    replace(context, targets=tuple(replace(t, ref_id='renamed-scene') if t.kind == 'scene' else t
                                                   for t in context.targets))]
        for version in variants:
            new = C.caller_input_record(version, opened, registry); entries = []
            for slot, candidate in enumerate(C.selection_candidates(version, opened, registry)):
                key = (candidate['ref_id'], candidate['kind'], candidate['origin'])
                old_key = ('scene-1', 'scene', 'context_target') if key[0] == 'renamed-scene' else key
                entry = deepcopy(rows[old_key]); entry['slot'] = slot
                if key[0] == 'renamed-scene':
                    body = json.loads(b64decode(entry['body_base64'])); body['ref_id'] = 'renamed-scene'
                    entry = meanings.body_entry(slot, 'registered-description-v1', body)
                entries.append(entry)
            policy = M.normative_view_policy()
            record = M.caller_meaning_record(new, caller_sha256=new['record_sha256'],
                snapshot=meaning['snapshot'], snapshot_sha256=meaning['snapshot']['record_sha256'],
                candidate_meanings=entries, view_policy=policy, view_policy_sha256=policy['record_sha256'])
            _, row = api.joint_source_batch_with_alignment(p, new, record,
                **{**kwargs, 'caller_sha256': new['record_sha256'], 'meaning_sha256': record['record_sha256']})
            self.assertEqual([(r['ref_id'], r['kind']) for r in row['candidates']],
                             [(r['ref_id'], r['kind']) for r in record['candidate_inventory']])
            subject = next(r for r in row['candidates'] if r['origin'] == 'source_subject')
            self.assertNotIn('episodes[].participant_refs', subject['eligible_paths'])

    def test_special_tokens_are_retained_without_feature_or_source_links(self):
        p = sources.Processor(); p.add_special_tokens = True
        p.tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
            single='<s> $A', special_tokens=[('<s>', 257)])
        _, _, _, _, _, _, row = self.build(processor=p)
        self.assertEqual(row['input_ids'][0], 257)
        self.assertEqual(row['tokens'][0]['field_indices'], [])
        self.assertEqual(row['tokens'][0]['candidate_slots'], [])
        self.assertFalse(row['tokens'][0]['observed_source_eligible'])

    def test_budget_identity_processor_and_resealed_sidecar_refuse(self):
        api, p, caller, meaning, kwargs, _, row = self.build()
        for fault in ('budget', 'identity', 'processor', 'field', 'ancestry'):
            changed = deepcopy(kwargs); altered = deepcopy(row)
            if fault == 'budget': changed['max_source_tokens'] = len(row['input_ids']) - 1
            elif fault == 'identity': changed['source_identity_sha256'] = '0' * 64
            elif fault == 'processor': changed['processor_binding']['tokenizer_backend_sha256'] = '0' * 64
            elif fault == 'field': altered['fields'][0]['supporting_token_indices'].pop()
            else: altered['cache']['provenance_sha256'] = '0' * 64
            altered = meanings.sealed({k: v for k, v in altered.items() if k != 'record_sha256'})
            with self.subTest(fault=fault), self.assertRaises(CognitiveKernelContractError):
                api.verify_joint_alignment(p, caller, meaning, altered,
                    alignment_sha256=altered['record_sha256'], **changed)

    def test_execution_modes_restored_on_success_and_offset_failure(self):
        for failure in (False, True):
            p = sources.Processor(); p.padding = True
            p.tokenizer.backend_tokenizer.enable_padding(length=40000, direction='left')
            p.tokenizer.backend_tokenizer.enable_truncation(max_length=40001)
            before = source.processor_binding(p)
            if failure:
                original = p.apply_chat_template
                def bad(*args, **kwargs):
                    result = original(*args, **kwargs)
                    if kwargs.get('processor_kwargs', {}).get('return_offsets_mapping'):
                        result['offset_mapping'][0, 0, 0] = -1
                    return result
                p.apply_chat_template = bad
                with self.assertRaises(CognitiveKernelContractError): self.build(processor=p)
            else: self.build(processor=p)
            self.assertEqual(source.processor_binding(p), before)

    def test_input_ancestry_validates_before_processor_and_native_refuses(self):
        api = self.api(); p = sources.Processor(); caller, meaning = companion()
        identity = source_identity(); kwargs = {'caller_sha256': caller['record_sha256'],
            'meaning_sha256': meaning['record_sha256'], 'processor_binding': source.processor_binding(p),
            'source_identity': identity, 'source_identity_sha256': identity['record_sha256'], 'max_source_tokens': 50000}
        with patch.object(p, 'apply_chat_template', side_effect=AssertionError('processor must not run')):
            with self.assertRaises(CognitiveKernelContractError):
                api.joint_source_batch_with_alignment(p, caller, meaning, **{**kwargs, 'caller_sha256': '0' * 64})
        from scripts.mfm import formation_caller_input as C, formation_caller_meaning as M
        from cognitive_kernel.formation_semantics_v16 import context_v16_from_record
        from base64 import b64decode
        native = deepcopy(caller); native['context']['base_context']['evidence'][0]['modality'] = 'image'
        context = context_v16_from_record(native['context'])
        opened = tuple((r['ref_id'], b64decode(r['payload_base64'])) for r in caller['opened_sources'])
        native = C.caller_input_record(context, opened, caller['registry_snapshot'])
        policy = M.normative_view_policy()
        new_meaning = M.caller_meaning_record(native, caller_sha256=native['record_sha256'],
            snapshot=meaning['snapshot'], snapshot_sha256=meaning['snapshot']['record_sha256'],
            candidate_meanings=meaning['candidate_meanings'], view_policy=policy, view_policy_sha256=policy['record_sha256'])
        with self.assertRaisesRegex(CognitiveKernelContractError, 'native'):
            api.joint_source_batch_with_alignment(p, native, new_meaning,
                **{**kwargs, 'caller_sha256': native['record_sha256'], 'meaning_sha256': new_meaning['record_sha256']})


if __name__ == '__main__': unittest.main()
