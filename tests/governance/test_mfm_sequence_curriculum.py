"""Public construction fixtures; these do not prove learned memory competence."""
from copy import deepcopy
from hashlib import sha256
from importlib import import_module, util
import json
from types import SimpleNamespace
import unittest

from cognitive_kernel.canonical import CognitiveKernelContractError


class CharacterTokenizer:
    bos_token_id, eos_token_id = 1, 2

    def encode(self, text, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError('auxiliary target must not add automatic specials')
        return [ord(char) + 3 for char in text]

    def decode(self, ids, *, skip_special_tokens):
        if skip_special_tokens:
            raise AssertionError('roundtrip must preserve literal content')
        return ''.join(chr(value - 3) for value in ids)


class SourceOnlyExample:
    """Expose admitted source fields while making teacher access observable."""
    def __init__(self, case_id='v16-seed-01', split='train', sources=None):
        self.case_id, self.split = case_id, split
        sources = sources if sources is not None else (
            ('source-first', 'text', b'Keep both spaces.  \r\n'),
            ('source-second', 'structured', '{"note":"été"}'.encode('utf-8')),
            ('source-third', 'code', b'print("hello")\n'),
        )
        self.opened_sources = tuple((ref, raw) for ref, _, raw in sources)
        evidence = tuple(SimpleNamespace(ref_id=ref, modality=mode,
                                          content_digest=sha256(raw).hexdigest())
                         for ref, mode, raw in sources)
        self.context = SimpleNamespace(base=SimpleNamespace(evidence=evidence))

    @property
    def target(self):
        raise AssertionError('source curriculum must not inspect teacher output')

    def validate(self):
        raise AssertionError('target-reading example validation belongs to admission')


def original_roster():
    # Independent public inventory of the admitted construction, not product constants.
    base = [f'v16-seed-{i:02}' for i in (*range(1, 13), 19, 20, 21)]
    base += ['role-source-person-revocation']
    crafts = ('atlas', 'ceramics', 'orchard', 'theatre', 'harbor', 'library', 'garden', 'radio')
    sites = ('east', 'west', 'north', 'south')
    operations = ('norm-agreed', 'norm-unsettled', 'episode-mission', 'plan-outcome',
                  'procedure-revision', 'source-person', 'revocation-deletion',
                  'record-correction', 'owner-deletion', 'sensitive-boundary',
                  'assistant-self', 'hypothetical-code', 'conflicting-outside')
    return tuple(SourceOnlyExample(case_id) for case_id in (
        *base, *(f'long-{craft}-{site}-{operation}'
                 for craft in crafts for site in sites for operation in operations)))


class SequenceCurriculumTests(unittest.TestCase):
    def module(self):
        name = 'scripts.mfm.formation_sequence_curriculum'
        self.assertIsNotNone(util.find_spec(name), 'sequence curriculum helper is missing')
        return import_module(name)

    def test_lexical_target_uses_exact_source_bytes_and_original_order(self):
        module = self.module()
        record = module.lexical_source_record(SourceOnlyExample())
        self.assertEqual(record, {'sources': [
            {'ref_id': 'source-first', 'text': 'Keep both spaces.  \r\n'},
            {'ref_id': 'source-second', 'text': '{"note":"été"}'},
            {'ref_id': 'source-third', 'text': 'print("hello")\n'},
        ]})

    def test_lexical_serializer_preserves_unicode_and_literal_special_looking_text(self):
        module = self.module()
        example = SourceOnlyExample(sources=(('source-a', 'text',
            'é\u0301 雪 <not-an-eos>\n\x00 "quoted"'.encode('utf-8')),))
        inputs, labels = module.lexical_target_ids(CharacterTokenizer(), example, 4096)
        self.assertEqual(inputs[0], 1)
        self.assertEqual(labels[-1], 2)
        self.assertEqual(inputs[1:], labels[:-1])
        decoded = CharacterTokenizer().decode(labels[:-1], skip_special_tokens=False)
        self.assertEqual(json.loads(decoded)['sources'][0]['text'],
                         'é\u0301 雪 <not-an-eos>\n\x00 "quoted"')
        self.assertIn('雪', decoded)

    def test_lexical_token_budget_rejects_overflow_instead_of_truncating(self):
        module = self.module()
        example = SourceOnlyExample()
        inputs, labels = module.lexical_target_ids(CharacterTokenizer(), example, 4096)
        self.assertEqual(module.lexical_target_ids(CharacterTokenizer(), example, len(labels)),
                         (inputs, labels))
        with self.assertRaisesRegex(CognitiveKernelContractError, 'never truncate'):
            module.lexical_target_ids(CharacterTokenizer(), example, len(labels) - 1)

    def test_lexical_tokenization_rejects_changed_roundtrip(self):
        module = self.module()
        class ChangedTokenizer(CharacterTokenizer):
            def decode(self, ids, *, skip_special_tokens):
                return super().decode(ids, skip_special_tokens=skip_special_tokens).replace('été', 'ete')
        with self.assertRaisesRegex(CognitiveKernelContractError, 'round-trip'):
            module.lexical_target_ids(ChangedTokenizer(), SourceOnlyExample(), 4096)

    def test_literal_eos_must_not_become_an_interior_termination_token(self):
        module = self.module()
        class LiteralEosTokenizer(CharacterTokenizer):
            def encode(self, text, *, add_special_tokens):
                before, after = text.split('<eos>')
                return super().encode(before, add_special_tokens=add_special_tokens) + [2] + \
                    super().encode(after, add_special_tokens=add_special_tokens)
        example = SourceOnlyExample(sources=(('source-a', 'text', b'Literal <eos> text'),))
        with self.assertRaisesRegex(CognitiveKernelContractError, 'interior EOS'):
            module.lexical_target_ids(LiteralEosTokenizer(), example, 4096)

    def test_lexical_path_rejects_native_binary_and_invalid_utf8(self):
        module = self.module()
        for modality, raw in (('image', b'not-text'), ('audio', b'not-text'),
                              ('video', b'not-text'), ('sensor', b'not-text'),
                              ('text', b'\xff')):
            with self.subTest(modality=modality, raw=raw):
                with self.assertRaises(CognitiveKernelContractError):
                    module.lexical_source_record(SourceOnlyExample(sources=(('source-a', modality, raw),)))

    def test_lexical_path_excludes_final_and_unknown_splits(self):
        module = self.module()
        for split in ('final', 'FINAL', 'unlisted'):
            with self.assertRaises(CognitiveKernelContractError):
                module.lexical_source_record(SourceOnlyExample(split=split))
        self.assertEqual(module.lexical_source_record(SourceOnlyExample(split='development')),
                         module.lexical_source_record(SourceOnlyExample()))

    def test_source_only_helper_rejects_missing_reordered_or_changed_bytes(self):
        module = self.module()
        original = SourceOnlyExample()
        variants = []
        missing = deepcopy(original); missing.opened_sources = missing.opened_sources[:-1]; variants.append(missing)
        reordered = deepcopy(original); reordered.opened_sources = tuple(reversed(reordered.opened_sources)); variants.append(reordered)
        changed = deepcopy(original); changed.opened_sources = (('source-first', b'Changed'), *changed.opened_sources[1:]); variants.append(changed)
        duplicated = deepcopy(original); duplicated.opened_sources = (*duplicated.opened_sources, duplicated.opened_sources[0]); variants.append(duplicated)
        for example in variants:
            with self.assertRaises(CognitiveKernelContractError): module.lexical_source_record(example)

    def test_construction_strata_parse_complete_operation_suffix_without_dash_splitting(self):
        module = self.module()
        self.assertEqual(module.construction_stratum('v16-seed-21'), 'diverse:v16-seed-21')
        self.assertEqual(module.construction_stratum('long-ceramics-north-revocation-deletion'),
                         'longitudinal:revocation-deletion')
        for case_id in ('long-unknown-east-norm-agreed', 'long-atlas-east-norm',
                        'long-atlas-east-norm-agreed-extra', 'v16-seed-18', 'LONG-atlas-east-norm-agreed'):
            with self.assertRaises(CognitiveKernelContractError): module.construction_stratum(case_id)

    def test_full_roster_has_29_equal_strata_and_unit_total_case_weight(self):
        module = self.module()
        balance = module.balanced_case_weights(original_roster())
        self.assertEqual(len(balance['weights']), 432)
        self.assertEqual(len(balance['group_counts']), 29)
        self.assertEqual(sorted(balance['group_counts'].values()), [1] * 16 + [32] * 13)
        self.assertAlmostEqual(sum(balance['weights'].values()), 1.)
        self.assertAlmostEqual(balance['weights']['v16-seed-01'], 1 / 29)
        self.assertAlmostEqual(balance['weights']['long-atlas-east-plan-outcome'], 1 / (29 * 32))
        self.assertEqual(balance['roster_sha256'],
                         module.balanced_case_weights(tuple(reversed(original_roster())))['roster_sha256'])

    def test_roster_rejects_missing_duplicate_unknown_and_development_rows(self):
        module = self.module(); roster = original_roster()
        for rows in (roster[:-1], (*roster[:-1], roster[0]),
                     (*roster[:-1], SourceOnlyExample('unlisted-case')),
                     (*roster[:-1], SourceOnlyExample(roster[-1].case_id, 'development'))):
            with self.assertRaises(CognitiveKernelContractError): module.balanced_case_weights(rows)

    def test_balanced_objective_averages_both_tasks_with_construction_weights(self):
        module = self.module(); balance = module.balanced_case_weights(original_roster())
        losses = {example.case_id: ((4., 2.) if example.case_id.startswith('v16-') or
                   example.case_id == 'role-source-person-revocation' else (0., 2.))
                  for example in original_roster()}
        # Sixteen diverse strata have task mean 3; thirteen operation strata have task mean 1.
        self.assertAlmostEqual(module.balanced_objective(losses, balance['weights']), 61 / 29)
        for bad in ({**losses, 'v16-seed-01': (float('nan'), 2.)},
                    {**losses, 'v16-seed-01': (-1., 2.)},
                    {k: v for k, v in losses.items() if k != 'v16-seed-01'}):
            with self.assertRaises(CognitiveKernelContractError): module.balanced_objective(bad, balance['weights'])
        with self.assertRaises(CognitiveKernelContractError):
            module.balanced_objective(losses, {k: v * 2 for k, v in balance['weights'].items()})

    def test_balanced_objective_preserves_actual_scalar_tensor_gradients(self):
        import torch
        module = self.module(); balance = module.balanced_case_weights(original_roster())
        formation = torch.tensor(4., dtype=torch.float64, requires_grad=True)
        lexical = torch.tensor(2., dtype=torch.float64, requires_grad=True)
        losses = {case_id: (formation, lexical) for case_id in balance['weights']}
        objective = module.balanced_objective(losses, balance['weights'])
        objective.backward()
        self.assertAlmostEqual(float(objective.detach()), 3.)
        self.assertAlmostEqual(float(formation.grad), .5)
        self.assertAlmostEqual(float(lexical.grad), .5)

    def test_receipt_binds_complete_source_roster_and_exact_lexical_token_frequencies(self):
        module = self.module(); roster = original_roster()
        receipt = module.curriculum_receipt(roster, CharacterTokenizer(), 4096)
        self.assertEqual(receipt['train_cases'], 432)
        self.assertEqual(receipt['construction_strata'], 29)
        # The tokenizer is one code point per token; the source record is hand-verifiable JSON.
        text = '{"sources":[{"ref_id":"source-first","text":"Keep both spaces.  \\r\\n"},' \
               '{"ref_id":"source-second","text":"{\\"note\\":\\"été\\"}"},' \
               '{"ref_id":"source-third","text":"print(\\"hello\\")\\n"}]}'
        self.assertEqual(receipt['lexical_target_tokens'], 432 * (len(text) + 1))
        self.assertEqual(receipt['positive_token_frequency'][str(2)], 432)
        self.assertEqual(receipt['positive_token_frequency'][str(ord('é') + 3)], 864)
        self.assertEqual(sum(receipt['positive_token_frequency'].values()), receipt['lexical_target_tokens'])
        reordered = module.curriculum_receipt(tuple(reversed(roster)), CharacterTokenizer(), 4096)
        self.assertEqual(receipt['record_sha256'], reordered['record_sha256'])
        changed = list(roster); changed[0] = SourceOnlyExample(sources=(('different-source', 'text', b'Different'),))
        changed_receipt = module.curriculum_receipt(tuple(changed), CharacterTokenizer(), 4096)
        self.assertNotEqual(receipt['record_sha256'], changed_receipt['record_sha256'])
        self.assertFalse(receipt['teacher_output_used'])
        self.assertFalse(receipt['independent_gold'])


if __name__ == '__main__':
    unittest.main()
