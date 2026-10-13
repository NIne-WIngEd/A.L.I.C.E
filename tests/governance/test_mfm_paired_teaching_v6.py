"""J6 source-before-target material mechanics; no fit or authority claim."""
import importlib.util
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class PairedV6ContractTests(unittest.TestCase):
    def test_versioned_input_and_target_contract_present(self):
        from scripts.mfm import formation_paired_teaching_v6 as pair
        self.assertEqual(pair.INPUT, 'mfm-paired-full-source-j6-input-v1')
        self.assertTrue(callable(pair.freeze_input))
        self.assertTrue(callable(pair.validate_input))
        self.assertTrue(callable(pair.convert_original_target))
        self.assertTrue(callable(pair.require_complete_labels))

    def test_versioned_target_geometry_contract_present(self):
        from scripts.mfm import formation_joint_target_alignment_v6 as target
        self.assertEqual(target.PAIR,
            'mfm-paired-full-source-j6-joint-target-geometry-v1')
        self.assertTrue(callable(target.align_target))
        self.assertTrue(callable(target.validate_pair))

    def test_actual_initial_cut_source_before_incomplete_author_target(self):
        import torch
        torch.set_num_threads(1)
        from scripts.mfm import formation_paired_teaching_v6 as pair
        from scripts.mfm import formation_joint_input_v6 as v6
        import scripts.mfm as mfm

        root = Path(__file__).resolve().parents[4]
        base = root / 'work/mfm-first-party-full-source-successor-plan-20261013'
        entry_path = root / ('work/mfm-first-party-longitudinal-corpus-design-20261011/'
                            'new-world-source-observation-plan-001/loader-construction-002/entry.py')
        spec = importlib.util.spec_from_file_location('original_saved005_entry', entry_path)
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        mfm.__path__ = [str(entry.TREE / 'scripts/mfm'), *mfm.__path__]
        from scripts.mfm import formation_joint_input_v5 as j5

        source_dir = base / 'j6-source-binding-migration-001/actual-001'
        view_dir = base / 'j6-authoring-view-002/actual-001'
        author_dir = base / 'j6-authoring-view-002/revalidated-assertions-001'
        source_receipt = json.loads((source_dir / 'receipt.json').read_bytes())
        view_receipt = json.loads((view_dir / 'receipt.json').read_bytes())
        author_receipt = json.loads((author_dir / 'receipt.json').read_bytes())
        sref = source_receipt['migrated_source_files'][0]
        vref = view_receipt['views'][0]
        aref = author_receipt['cases'][0]
        source_raw = (source_dir / sref['path']).read_bytes()
        migration_raw = (source_dir / 'receipt.json').read_bytes()
        frozen = pair.freeze_input(j5, v6, lambda: source_raw,
            lambda: migration_raw, expected_source_file_sha256=sref['file_sha256'],
            split_row={'case_id': 'j6-reading-room-initial-state',
                       'split': 'train', 'lineage': {'world': 'same-train-world'}})
        self.assertEqual(frozen['source_tokens'], 22475)
        self.assertEqual(set(frozen['native_batch']), set(pair.NATIVE))
        self.assertFalse(frozen['training_authorized'])
        opens = []

        def material(ref):
            def read():
                opens.append(ref['path'])
                return (author_dir / ref['path']).read_bytes()
            return read

        with TemporaryDirectory() as temp:
            path = Path(temp) / 'paired-source.json'
            saved_receipt = pair.publish_input(path, j5, v6, frozen,
                expected_source_file_sha256=sref['file_sha256'],
                migration_reader=lambda: migration_raw)
            target = pair.convert_saved_target(path, saved_receipt,
                expected_receipt_sha256=saved_receipt['record_sha256'],
                expected_source_file_sha256=sref['file_sha256'],
                j5=j5, v6=v6, migration_reader=lambda: migration_raw,
                view_reader=lambda: (view_dir / vref['path']).read_bytes(),
                expected_view_file_sha256=vref['file_sha256'],
                draft_reader=material(aref['draft']),
                expected_draft_file_sha256=aref['draft']['file_sha256'],
                support_reader=material(aref['support']),
                expected_support_file_sha256=aref['support']['file_sha256'],
                annotation_reader=material(aref['annotation']),
                expected_annotation_file_sha256=aref['annotation']['file_sha256'])
            contents = path.read_bytes()
            path.write_bytes(contents[:-2] + b'x\n')
            target_reads = []
            with self.assertRaisesRegex(Exception, 'saved input FILE'):
                pair.convert_saved_target(path, saved_receipt,
                    expected_receipt_sha256=saved_receipt['record_sha256'],
                    expected_source_file_sha256=sref['file_sha256'],
                    j5=j5, v6=v6, migration_reader=lambda: migration_raw,
                    view_reader=lambda: target_reads.append('view'),
                    expected_view_file_sha256=vref['file_sha256'],
                    draft_reader=lambda: target_reads.append('draft'),
                    expected_draft_file_sha256=aref['draft']['file_sha256'],
                    support_reader=lambda: target_reads.append('support'),
                    expected_support_file_sha256=aref['support']['file_sha256'],
                    annotation_reader=lambda: target_reads.append('annotation'),
                    expected_annotation_file_sha256=aref['annotation']['file_sha256'])
            self.assertEqual(target_reads, [])
        self.assertEqual(len(opens), 3)
        self.assertEqual(target['state'], 'incomplete')
        self.assertTrue(target['unknown_dimensions'])
        self.assertTrue(target['unresolved_selection_fields'])
        self.assertFalse(target['whole_copy_alternatives_authored'])
        original = json.loads(bytes.fromhex(target['original_target_hex']))
        self.assertEqual(set(original), {'schema', 'context', 'source_ids',
            'proposals', 'dispositions', 'adjudications'})
        with self.assertRaisesRegex(Exception, 'incomplete J6 author choices'):
            pair.require_complete_labels(frozen, target)
        # A resealed packet cannot promote the retained author's unknowns and
        # missing ordinal/copy annotations into invented complete labels.
        from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256
        invented = deepcopy(target)
        invented_draft = json.loads(bytes.fromhex(invented['raw_author_draft_hex']))
        invented_original = json.loads(bytes.fromhex(invented['original_target_hex']))
        statuses = {name: ('negative' if state == 'unknown' else state)
                    for name, state in invented_original['adjudications'].items()}
        invented_draft['target']['adjudications'] = statuses
        invented_original['adjudications'] = statuses
        for prefix, value in (('raw_author_draft', invented_draft),
                              ('original_target', invented_original)):
            raw = canonical_json_bytes(value) + b'\n'
            invented[prefix + '_hex'] = raw.hex()
            invented[prefix + '_file_sha256'] = __import__('hashlib').sha256(raw).hexdigest()
        invented['unknown_dimensions'] = []
        invented['unresolved_selection_fields'] = []
        invented['whole_copy_alternatives_authored'] = True
        invented['state'] = 'complete'
        invented['record_sha256'] = canonical_sha256({k: v for k, v in
            invented.items() if k != 'record_sha256'})
        with self.assertRaisesRegex(Exception, 'author annotation'):
            pair.require_complete_labels(frozen, invented)
        from scripts.mfm import formation_joint_target_alignment_v6 as target_v6
        from scripts.mfm import formation_target_alignment as canonical_target
        from cognitive_kernel import formation_learning_v16 as learning
        draft = json.loads(bytes.fromhex(target['raw_author_draft_hex']))
        output = learning.learning_example_v16_from_record(draft).target.output_record()
        geometry = canonical_target.canonical_field_geometry(output,
            expected_target_sha256=canonical_sha256(output))
        with self.assertRaisesRegex(Exception, 'ordinal missing'):
            target_v6.field_rows(geometry, frozen, [])
        # Explicitly chosen fixture ordinals and one whole literal alternative
        # exercise the geometry math; this is not a fit label for the six cuts.
        semantic = (
            '/dispositions/0/action', '/dispositions/0/scope_ref',
            '/dispositions/1/action', '/dispositions/1/scope_ref',
            '/proposals/0/episode', '/proposals/0/proposal/confidence',
            '/proposals/0/proposal/disposition_scope_ref',
            '/proposals/0/proposal/domain',
            '/proposals/0/proposal/epistemic_status',
            '/proposals/0/proposal/kind',
            '/proposals/0/proposal/temporal_granularity',
            '/proposals/0/proposal/valid_from',
            '/proposals/0/proposal/valid_to',
            '/proposals/0/proposal/value_text',
            '/proposals/0/sensitivity_hint')
        ordinal = {
            '/dispositions/0/evidence_refs/0': 0,
            '/dispositions/1/evidence_refs/0': 0,
            '/proposals/0/mission_target_refs/0': 3,
            '/proposals/0/proposal/anchors/0/ref_id': 0,
            '/proposals/0/proposal/evidence_refs/0': 0,
            '/proposals/0/proposal/subject_ref': 1}
        authored = ([{'pointer': pointer, 'mode': 'semantic_authoring'}
            for pointer in semantic] +
            [{'pointer': pointer, 'mode': 'external_selection',
              'candidate_ordinal': selected}
             for pointer, selected in ordinal.items()])
        rows = target_v6.field_rows(geometry, frozen, authored)
        self.assertEqual(sum(row['annotation_status'] == 'authored'
            for row in rows), 21)
        self.assertEqual({row['pointer']: row['selected_candidate_ordinal']
            for row in rows if 'selected_candidate_ordinal' in row}, ordinal)
        from tests.governance.test_mfm_source_alignment import Processor
        from scripts.mfm import formation_target_alignment as target_api
        toy_processor = Processor()
        text = bytes.fromhex(geometry['canonical_utf8_hex']).decode('utf-8')
        observed = toy_processor.tokenizer(text, add_special_tokens=False,
            padding=False, truncation=False, return_offsets_mapping=True,
            return_attention_mask=True, return_special_tokens_mask=True)
        tokens = target_api._token_rows(geometry, observed['input_ids'],
            observed['attention_mask'], observed['offset_mapping'],
            observed['special_tokens_mask'])
        literal = b'every Tuesday'
        target_value = draft['target']['proposals'][0]['proposal']['value_text'].encode()
        target_start = target_value.index(literal)
        source_data = __import__('base64').b64decode(
            frozen['source']['caller']['opened_sources'][0]['payload_base64'])
        source_start = source_data.index(literal)
        copy_rows = target_v6.literal_rows(geometry, tokens, frozen, [{
            'pointer': '/proposals/0/proposal/value_text',
            'target_value_utf8': [target_start, target_start + len(literal)],
            'source_occurrences': [{'source_ref': 'charter-001',
                'original_utf8': [source_start, source_start + len(literal)]}]}])
        self.assertEqual(len(copy_rows), 1)
        self.assertEqual(copy_rows[0]['selected_occurrences'], [{
            'source_ref': 'charter-001',
            'original_utf8': [source_start, source_start + len(literal)]}])
        with self.assertRaisesRegex(Exception, 'whole authored copy alternatives'):
            target_v6.literal_rows(geometry, tokens, frozen, [{
                'pointer': '/proposals/0/proposal/value_text',
                'target_value_utf8': [target_start, target_start + len(literal)],
                'source_occurrences': None}])

        mutated = deepcopy(frozen)
        mutated['source_cache']['processed_view_sha256'] = '0' * 64
        mutated['record_sha256'] = canonical_sha256({k: v for k, v in
            mutated.items() if k != 'record_sha256'})
        target_reads = []

        def forbidden():
            target_reads.append('opened')
            raise AssertionError('target read before source validation')

        with self.assertRaisesRegex(Exception, 'whole J6 paired source'):
            pair.convert_original_target(j5, v6, mutated,
                expected_input_sha256=mutated['record_sha256'],
                expected_source_file_sha256=sref['file_sha256'],
                migration_reader=lambda: migration_raw,
                view_reader=forbidden, expected_view_file_sha256=vref['file_sha256'],
                draft_reader=forbidden,
                expected_draft_file_sha256=aref['draft']['file_sha256'],
                support_reader=forbidden,
                expected_support_file_sha256=aref['support']['file_sha256'],
                annotation_reader=forbidden,
                expected_annotation_file_sha256=aref['annotation']['file_sha256'])
        self.assertEqual(target_reads, [])

        bad_source = bytearray(source_raw)
        bad_source[-2] = ord('x')
        with self.assertRaises(Exception):
            pair.freeze_input(j5, v6, lambda: bytes(bad_source),
                lambda: migration_raw,
                expected_source_file_sha256=sref['file_sha256'],
                split_row=frozen['split_row'])


if __name__ == '__main__':
    unittest.main()
