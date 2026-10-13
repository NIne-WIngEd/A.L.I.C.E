"""Complete native text source geometry controls for the J6 construction profile."""
import unittest

import torch
from transformers.feature_extraction_utils import BatchFeature


class JointV6NativeFieldsTests(unittest.TestCase):
    def test_native_batchfeature_container_keeps_complete_tensor_contract(self):
        from scripts.mfm.formation_joint_input_v6 import validate_native_batch_pair
        fields = {'input_ids': torch.tensor([[2, 17, 1]], dtype=torch.long),
            'attention_mask': torch.ones((1, 3), dtype=torch.long),
            'mm_token_type_ids': torch.zeros((1, 3), dtype=torch.long)}
        default = BatchFeature(data=fields)
        offset = BatchFeature(data={**{k: v.clone() for k, v in fields.items()},
            'offset_mapping': torch.tensor([[[0, 0], [1, 3], [0, 0]]]),
            'special_tokens_mask': torch.tensor([[1, 0, 1]])})
        self.assertEqual(validate_native_batch_pair(default, offset,
            proposed_max_source_tokens=3)['source_tokens'], 3)
        offset['mm_token_type_ids'][0, 1] = 1
        with self.assertRaises(ValueError):
            validate_native_batch_pair(default, offset,
                proposed_max_source_tokens=3)

    def test_saved_native_fields_keep_integral_text_values_and_complete_lengths(self):
        from scripts.mfm.formation_joint_input_v6 import _validate_saved_native_fields
        fields = {'input_ids': [2, 17, 1],
            'attention_mask': [1, 1, 1],
            'mm_token_type_ids': [0, 0, 0]}
        batch = {key: [value.copy()] for key, value in fields.items()}
        self.assertEqual(_validate_saved_native_fields(fields, batch,
            source_tokens=3, proposed_max_source_tokens=3), 3)
        for key, index, bad in (
                ('input_ids', 1, -1), ('input_ids', 1, 17.0),
                ('attention_mask', 1, 0), ('attention_mask', 1, True),
                ('mm_token_type_ids', 1, 1), ('mm_token_type_ids', 1, 0.0)):
            altered = {name: values.copy() for name, values in fields.items()}
            altered[key][index] = bad
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                _validate_saved_native_fields(altered, batch,
                    source_tokens=3, proposed_max_source_tokens=3)
            altered_batch = {name: [values.copy()] for name, values in fields.items()}
            altered_batch[key][0][index] = bad
            with self.subTest(key=key, bad=bad, place='batch'), self.assertRaises(ValueError):
                _validate_saved_native_fields(fields, altered_batch,
                    source_tokens=3, proposed_max_source_tokens=3)
        with self.assertRaises(ValueError):
            _validate_saved_native_fields(fields, batch,
                source_tokens=3.0, proposed_max_source_tokens=3)
        with self.assertRaises(ValueError):
            _validate_saved_native_fields(fields, batch,
                source_tokens=3, proposed_max_source_tokens=True)
        short = {**batch, 'mm_token_type_ids': [[0, 0]]}
        with self.assertRaises(ValueError):
            _validate_saved_native_fields(fields, short,
                source_tokens=3, proposed_max_source_tokens=3)

    def test_default_offset_must_match_all_three_complete_native_fields(self):
        from scripts.mfm.formation_joint_input_v6 import validate_native_batch_pair
        default = {
            'input_ids': torch.tensor([[2, 17, 1]], dtype=torch.long),
            'attention_mask': torch.ones((1, 3), dtype=torch.long),
            'mm_token_type_ids': torch.tensor([[0, 0, 0]], dtype=torch.long)}
        offset = {**{key: value.clone() for key, value in default.items()},
            'offset_mapping': torch.tensor([[[0, 0], [1, 3], [0, 0]]]),
            'special_tokens_mask': torch.tensor([[1, 0, 1]])}
        result = validate_native_batch_pair(default, offset,
                                            proposed_max_source_tokens=3)
        self.assertEqual(result['source_tokens'], 3)
        self.assertEqual(result['native_fields']['mm_token_type_ids'], [0, 0, 0])
        offset['mm_token_type_ids'][0, 1] = 1
        with self.assertRaisesRegex(ValueError, 'default.*offset.*native'):
            validate_native_batch_pair(default, offset,
                                       proposed_max_source_tokens=3)
        offset['mm_token_type_ids'][0, 1] = 0
        default['mm_token_type_ids'][0, 1] = 1
        offset['mm_token_type_ids'][0, 1] = 1
        with self.assertRaisesRegex(ValueError, 'native text field values'):
            validate_native_batch_pair(default, offset,
                                       proposed_max_source_tokens=3)
        default['mm_token_type_ids'][0, 1] = 0
        offset['mm_token_type_ids'][0, 1] = 0
        default['input_ids'][0, 1] = -1
        offset['input_ids'][0, 1] = -1
        with self.assertRaisesRegex(ValueError, 'native text field values'):
            validate_native_batch_pair(default, offset,
                                       proposed_max_source_tokens=3)
        default['input_ids'][0, 1] = 17
        offset['input_ids'][0, 1] = 17
        with self.assertRaisesRegex(ValueError, 'complete source budget'):
            validate_native_batch_pair(default, offset,
                                       proposed_max_source_tokens=2)


if __name__ == '__main__':
    unittest.main()
