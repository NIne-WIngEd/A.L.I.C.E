"""Reader custody and objective contracts, not evidence of model learning."""
import copy
import importlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from cognitive_kernel.canonical import canonical_json_bytes
from scripts.mfm import export_public_conditional_features as features
from scripts.mfm import preflight_public_conditional_material as census_code
from scripts.mfm import train_v1_formation_specialist as shared
import test_mfm_public_conditional_features as feature_fixtures


class PublicConditionalTrainingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = feature_fixtures.PublicConditionalFeatureTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.directory = self.fixture.census_dir
        self.bank_dir = self.fixture.output
        self.pairs = [self.pair('a', [2, 4], 'f' * 64, [3]),
                      self.pair('b', [2, 5], 'f' * 64, [4, 5, 6]),
                      self.pair('c', [2, 6], '1' * 64, [7, 8]),
                      self.pair('d', [2, 7], '0' * 63 + 'a', [9], 'development')]
        self.census = self.fixture.census
        self.census['max_target_tokens'] = 8
        self.write_census()
        self.export_bank()

    def pair(self, case, ids, group, body, split='train'):
        return dict(case_id=case, source_ids=ids, split=split,
                    semantic_input_sha256=group, source_payload_sha256=case * 64,
                    semantic_reference_count=2 if group == 'f' * 64 else 1,
                    decoder_inputs=[2, *body], labels=[*body, 1])

    def write_census(self, *, sources=None, statistics=None, encoded_bytes=None):
        path = self.directory / 'encoded-pairs.jsonl'
        path.write_bytes(encoded_bytes if encoded_bytes is not None else
                         b''.join(canonical_json_bytes(p) + b'\n' for p in self.pairs))
        self.census['encoded_pairs_sha256'] = shared._digest(path)
        self.census['statistics'] = statistics or census_code.token_statistics(self.pairs)
        self.fixture.sources = sources if sources is not None else census_code.source_inputs(self.pairs)
        self.census['unique_source_inputs'] = len(self.fixture.sources)
        self.fixture.write_census()
        self.pin = self.fixture.external_pin

    def export_bank(self):
        census, sources, export = self.fixture.binding(features)
        features.run_export(self.bank_dir, census, sources, export, feature_fixtures.TinyBase(), stop_after_seconds=60)
        self.bank_pin = features._read_sealed(self.bank_dir / 'bank.json', features.BANK_SCHEMA)['record_sha256']

    def module(self):
        name = 'scripts.mfm.public_conditional_training'
        self.assertIsNotNone(importlib.util.find_spec(name), 'public training reader is missing')
        return importlib.import_module(name)

    def load(self):
        return self.module().load_public_training(self.directory, self.pin, self.bank_dir, self.bank_pin)

    def test_independent_exact_source_join_and_detached_payload(self):
        prepared = self.load()
        self.assertEqual(len(prepared.train), 3)
        self.assertEqual(len(prepared.development), 1)
        self.assertTrue(hasattr(prepared, 'pair_by_case'), 'complete indexed case mapping is missing')
        self.assertEqual(set(prepared.pair_by_case), {'a', 'b', 'c', 'd'})
        self.assertEqual(len(prepared.source_by_key), 4)
        payload = self.module().payload(prepared, prepared.train[1], device='cpu')
        self.assertEqual(payload['input_ids'].tolist(), [[2, 4, 5, 6]])
        self.assertEqual(payload['labels'].tolist(), [[4, 5, 6, 1]])
        self.assertEqual(payload['base_states'][0, :, 0].tolist(), [6, 15])
        self.assertFalse(payload['base_states'].requires_grad)
        self.assertEqual(payload['source_mask'].tolist(), [[1, 1]])
        forged = dict(prepared.train[1], source_ids=[2, 4])
        with self.assertRaisesRegex(ValueError, 'pair.*membership'):
            self.module().payload(prepared, forged, device='cpu')

    def test_resealed_pair_join_attacks_reach_roster_guard(self):
        for field, value in [('source_ids', [2, 5]), ('case_id', 'other'),
                             ('semantic_input_sha256', '2' * 64),
                             ('source_payload_sha256', 'e' * 64)]:
            with self.subTest(field=field):
                prior = copy.deepcopy(self.pairs)
                self.pairs[0][field] = value
                # Keep the independently exported source roster unchanged.
                self.write_census(sources=self.fixture.sources)
                with self.assertRaisesRegex(ValueError, 'roster|ancestry|reference'):
                    self.load()
                self.pairs = prior

    def test_resealed_invalid_sequences_and_whitelist(self):
        mutations = [('decoder_inputs', [2, 6]), ('decoder_inputs', [3, 3]),
                     ('labels', [3, 2]), ('labels', [2, 1]),
                     ('labels', [16, 1]), ('labels', [True, 1]), ('target_text', 'secret')]
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                prior = copy.deepcopy(self.pairs)
                self.pairs[0][field] = value
                self.write_census(sources=self.fixture.sources,
                                  statistics=self.census['statistics'])
                with self.assertRaisesRegex(ValueError, 'sequence|token|fields'):
                    self.load()
                self.pairs = prior

    def test_duplicate_case_and_json_keys(self):
        self.pairs[1]['case_id'] = 'a'
        self.write_census(sources=self.fixture.sources, statistics=self.census['statistics'])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.load()
        raw = canonical_json_bytes(self.pairs[0]).decode().replace('"case_id":"a"', '"case_id":"a","case_id":"a"')
        self.write_census(sources=self.fixture.sources, statistics=self.census['statistics'],
                          encoded_bytes=(raw + '\n').encode())
        with self.assertRaisesRegex(ValueError, 'duplicate JSON'):
            self.load()

    def test_reference_count_statistics_and_source_coverage(self):
        self.pairs[0]['semantic_reference_count'] = 1
        self.write_census(sources=self.fixture.sources)
        with self.assertRaisesRegex(ValueError, 'reference'):
            self.load()
        self.pairs[0]['semantic_reference_count'] = 2
        wrong = census_code.token_statistics(self.pairs)
        wrong['largest_target_tokens'] = 7
        self.write_census(statistics=wrong)
        with self.assertRaisesRegex(ValueError, 'statistics'):
            self.load()
        self.pairs = self.pairs[:-1]
        self.write_census(sources=self.fixture.sources, statistics=self.census['statistics'])
        with self.assertRaisesRegex(ValueError, 'statistics|roster'):
            self.load()

    def test_mixed_semantic_or_raw_hash_split_rejected(self):
        module = self.module()
        for field in ('semantic_input_sha256', 'source_payload_sha256'):
            pairs = copy.deepcopy(self.pairs)
            pairs[-1][field] = pairs[0][field]
            with self.assertRaisesRegex(ValueError, 'split'):
                module.case_weights(pairs, split='train')

    def test_external_pins_and_encoded_file_bytes(self):
        module = self.module()
        with self.assertRaisesRegex(ValueError, 'external.*census'):
            module.load_public_training(self.directory, '0' * 64, self.bank_dir, self.bank_pin)
        with self.assertRaisesRegex(ValueError, 'external.*bank'):
            module.load_public_training(self.directory, self.pin, self.bank_dir, '0' * 64)
        with (self.directory / 'encoded-pairs.jsonl').open('ab') as stream:
            stream.write(b'\n')
        with self.assertRaisesRegex(ValueError, 'pair.*digest'):
            self.load()

    def test_missing_bank_and_partial_progress_are_rejected(self):
        original_bank = features._read_sealed(self.bank_dir / 'bank.json', features.BANK_SCHEMA)
        (self.bank_dir / 'bank.json').unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.load()
        progress = features._read_sealed(self.bank_dir / 'progress.json', features.PROGRESS_SCHEMA)
        progress.update(complete=False, completed=1,
                        source_chain_sha256=features._source_chain(original_bank['sources'][:1],
                                                                  original_bank['export_sha256']))
        partial_progress = features._write_record(self.bank_dir / 'progress.json',
                              {k: v for k, v in progress.items() if k != 'record_sha256'}, replace=True)
        body = {k: v for k, v in original_bank.items() if k != 'record_sha256'}
        body.update(complete=False, sources=original_bank['sources'][:1],
                    progress_sha256=partial_progress['record_sha256'])
        partial_bank = features._write_record(self.bank_dir / 'bank.json', body)
        self.bank_pin = partial_bank['record_sha256']
        with self.assertRaisesRegex(ValueError, 'complete.*bank'):
            self.load()

    def test_resealed_false_source_only_export_flags(self):
        export = features._read_sealed(self.bank_dir / 'export.json', features.EXPORT_SCHEMA)
        for field, value in [('target_visible_to_backbone', True), ('base_frozen', False),
                             ('language_head_used', True), ('optimizer_created', True)]:
            with self.subTest(field=field):
                body = {k: v for k, v in export.items() if k != 'record_sha256'}
                body[field] = value
                features._write_record(self.bank_dir / 'export.json', body, replace=True)
                with self.assertRaisesRegex(ValueError, 'source-only|export.*binding'):
                    self.load()

    def test_payload_rechecks_tensor_custody(self):
        prepared = self.load()
        source = prepared.sources[0]
        path = self.bank_dir / 'sources' / source['source_key'] / 'features.safetensors'
        path.write_bytes(b'changed')
        pair = next(p for p in prepared.pairs if p['case_id'] in source['case_ids'])
        with self.assertRaisesRegex(ValueError, 'tensor.*digest'):
            self.module().payload(prepared, pair, device='cpu')

    def test_bank_replacement_between_reads_retains_external_pin(self):
        original = self.load().bank
        path = self.bank_dir / 'bank.json'
        original_bytes = path.read_bytes()
        body = {k: v for k, v in original.items() if k != 'record_sha256'}
        body['replacement_marker'] = 'complete independently valid replacement'
        replacement = features._write_record(path, body, replace=True)
        # Both real records independently satisfy every unchanged bank guard.
        self.bank_pin = replacement['record_sha256']
        self.assertEqual(self.load().bank, replacement)
        self.assertNotEqual(original['record_sha256'], replacement['record_sha256'])
        self.bank_pin = original['record_sha256']
        path.write_bytes(original_bytes)
        verify = features.verify_bank
        verified = []

        def replace_before_verification(*args, **kwargs):
            features._write_record(path, body, replace=True)
            record = verify(*args, **kwargs)
            verified.append(record)
            return record

        with patch.object(features, 'verify_bank', replace_before_verification):
            with self.assertRaisesRegex(ValueError, 'external.*bank'):
                self.load()
        self.assertEqual(verified, [replacement])

    def test_same_pinned_bank_replacement_between_reads_is_valid(self):
        original = self.load().bank
        path = self.bank_dir / 'bank.json'
        body = {k: v for k, v in original.items() if k != 'record_sha256'}
        verify = features.verify_bank

        def replace_identical_before_verification(*args, **kwargs):
            features._write_record(path, body, replace=True)
            return verify(*args, **kwargs)

        with patch.object(features, 'verify_bank', replace_identical_before_verification):
            prepared = self.load()
        self.assertEqual(prepared.bank, original)
        self.assertEqual(prepared.bank['record_sha256'], self.bank_pin)

    def test_epoch_order_uses_seed_plus_epoch_on_encoded_file_order(self):
        module = self.module()
        prepared = self.load()
        self.assertEqual([p['case_id'] for p in prepared.train], ['a', 'b', 'c'])
        for epoch, expected in [(0, ['c', 'b', 'a']), (2, ['b', 'c', 'a'])]:
            with self.subTest(epoch=epoch):
                self.assertEqual([p['case_id'] for p in module.epoch_order(prepared, seed=123, epoch=epoch)],
                                 expected)
        # A second valid full census changes only its immutable encoded order.
        self.pairs = [self.pairs[i] for i in (1, 2, 0, 3)]
        self.directory = self.fixture.census_dir = self.fixture.root / 'reordered-census'
        self.directory.mkdir()
        self.bank_dir = self.fixture.output = self.fixture.root / 'reordered-bank'
        self.write_census()
        self.export_bank()
        reordered = self.load()
        self.assertEqual([p['case_id'] for p in reordered.train], ['b', 'c', 'a'])
        self.assertEqual([p['case_id'] for p in module.epoch_order(reordered, seed=123, epoch=0)],
                         ['a', 'c', 'b'])

    def test_group_reference_objective_and_importance_gradients(self):
        module = self.module()
        weights = module.case_weights(self.pairs, split='train')
        self.assertEqual(weights, {'a': .25, 'b': .25, 'c': .5})
        x = torch.tensor(2., requires_grad=True)
        means = {'a': (x * torch.tensor([1., 3.])).mean(),
                 'b': (x * torch.tensor([2., 4., 6., 8.])).mean(),
                 'c': (x * torch.tensor([3., 6., 9.])).mean()}
        direct = ((means['a'] + means['b']) / 2 + means['c']) / 2
        full = module.batch_coefficients(self.pairs[:3], weights, total_train_cases=3)
        loss = sum(full[k] * means[k] for k in full)
        self.assertEqual(loss.item(), direct.item())
        self.assertEqual(torch.autograd.grad(loss, x, retain_graph=True)[0].item(),
                         torch.autograd.grad(direct, x, retain_graph=True)[0].item())
        singleton_losses = []
        for pair in self.pairs[:3]:
            coeff = module.batch_coefficients([pair], weights, total_train_cases=3)
            self.assertEqual(coeff[pair['case_id']], 3 * weights[pair['case_id']])
            singleton_losses.append(coeff[pair['case_id']] * means[pair['case_id']])
        average = sum(singleton_losses) / 3
        self.assertEqual(torch.autograd.grad(average, x, retain_graph=True)[0].item(),
                         torch.autograd.grad(direct, x, retain_graph=True)[0].item())
        partial = module.batch_coefficients(self.pairs[:2], weights, total_train_cases=3)
        self.assertEqual(partial, {'a': .375, 'b': .375})
        partial_loss = sum(partial[k] * means[k] for k in partial)
        direct_partial = 3 / 2 * (.25 * means['a'] + .25 * means['b'])
        self.assertEqual(torch.autograd.grad(partial_loss, x, retain_graph=True)[0].item(),
                         torch.autograd.grad(direct_partial, x)[0].item())

    def test_batch_and_weight_roster_rejections(self):
        module = self.module()
        weights = module.case_weights(self.pairs, split='train')
        for batch, total in [([], 3), ([self.pairs[0]] * 2, 3),
                             ([self.pairs[-1]], 3), ([self.pairs[0]], 2)]:
            with self.assertRaises(ValueError):
                module.batch_coefficients(batch, weights, total_train_cases=total)
        for mutation in ('duplicate', 'count', 'split'):
            pairs = copy.deepcopy(self.pairs)
            if mutation == 'duplicate':
                pairs.append(pairs[0])
            elif mutation == 'count':
                pairs[0]['semantic_reference_count'] = 1
            else:
                pairs[0]['split'] = 'FINAL'
            with self.assertRaises(ValueError):
                module.case_weights(pairs, split='train')

    def test_epoch_order_is_complete_repeatable_and_train_only(self):
        prepared = self.load()
        module = self.module()
        order = module.epoch_order(prepared, seed=123, epoch=0)
        self.assertEqual(order, module.epoch_order(prepared, seed=123, epoch=0))
        self.assertEqual({p['case_id'] for p in order}, {'a', 'b', 'c'})
        self.assertEqual(len(order), 3)
        for seed, epoch in [(True, 0), (1, -1)]:
            with self.assertRaises(ValueError):
                module.epoch_order(prepared, seed=seed, epoch=epoch)


if __name__ == '__main__':
    unittest.main()
