"""Public fictional membership mechanics; no private or learned-voice claim."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import importlib
import unittest

from scripts.eipm.gemma_n0.construct_personality import canonical, verify_reference_families
from src.alice_personality.identity.codec import SourceEntry
from src.alice_personality.identity.contracts import SourceRecord
from tests.personality.test_identity_codec import document

try:
    adapter = importlib.import_module('scripts.eipm.gemma_n0.connected_voice_references')
except ModuleNotFoundError as error:
    if error.name != 'scripts.eipm.gemma_n0.connected_voice_references':
        raise
    adapter = None


FAMILY = 'fictional-source-family'
OTHER_FAMILY = 'fictional-unrelated-family'
E0 = 'opaque-source-id'
EINF = 'fictional-inference-id'
VOICE = 'fictional-delivery-proposal'
FLAGS = ('acceptance_authority', 'private_gradient_authorized',
         'historical_authority_granted', 'training_authorized', 'source_admission')
ORIGINAL_FORBIDDEN = ('Alice_lived_memory', 'autobiographical_recall_allowed',
    'historical_Elaina_truth', 'generator_has_acceptance_authority', 'model_training_authority')


def body_hash(body):
    return sha256(canonical(body)).hexdigest()


def fictional_inputs():
    """Hand-authored public bodies obey the two existing builder contracts."""
    families = [
        {'family_id': FAMILY, 'split': 'train', 'record_ids': [E0, EINF]},
        {'family_id': OTHER_FAMILY, 'split': 'train', 'record_ids': ['unrelated-e0']},
    ]
    originals = {}
    for rid, kind, family in ((E0, 'E0', FAMILY), (EINF, 'EINF', FAMILY),
                              ('unrelated-e0', 'E0', OTHER_FAMILY)):
        body = {'unit_id' if kind == 'E0' else 'curated_id': rid,
            'provenance_class': 'E0' if kind == 'E0' else 'E-INF',
            'claim_subject_partitions': ['source_person:elaina'],
            'loss_mask': {'conditional_identity': True, 'excluded_from_identity_loss': False}}
        originals[rid] = {'draft_workitem': {'record_id': rid, 'source_kind': kind,
            'source_family_id': family, 'original_source_family_id': family,
            'original_split': 'train', 'source_declarations': {
                'historical_truth_allowed': kind == 'E0', 'runtime_behavioral_prior_allowed': False}},
            'original_source_record': body, 'original_source_record_sha256': body_hash(body),
            'source_kind_for_future_document': kind}
    body = {'proposal_id': VOICE, 'provenance_class': 'A-SYN',
        **{flag: False for flag in ORIGINAL_FORBIDDEN},
        'owner_final_review_required': True, 'runtime_behavioral_prior_allowed': True,
        'identity_supporting_E0_unit_ids': [E0], 'supporting_curated_EINF_ids': [EINF],
        'support_mode': 'fictional-supported-policy', 'n0_voice_competency': 'administrative-fixture',
        'voice_expression_policy': {'delivery': 'Use a fictional calm delivery.'}}
    work = {'schema': 'alice-personality-source-linked-curriculum-workitem-v1',
        'state': 'SOURCE_LINKED_DRAFT', 'record_id': VOICE, 'source_kind': 'ASYN_VOICE',
        'evidence_category': 'A-SYN', 'source_family_id': FAMILY,
        'original_source_family_id': None, 'original_split': 'train',
        'voice_source_record_sha256': body_hash(body),
        'evidence_links': {'identity_supporting_e0_ids': [E0], 'supporting_curated_einf_ids': [EINF]},
        'source_declarations': {'training_authority': False, 'historical_truth_allowed': False,
            'alice_lived_memory': False, 'autobiographical_recall_allowed': False,
            'owner_final_review_required': True, 'runtime_behavioral_prior_allowed': True,
            'support_mode': 'fictional-supported-policy', 'source_coverage_bin': 'administrative-fixture'},
        'target_review_complete': False, **{flag: False for flag in FLAGS}}
    originals[VOICE] = {'schema': 'alice-personality-source-complete-case-authoring-item-v1',
        'state': 'CASE_AUTHORING_REQUIRED', 'draft_workitem': work,
        'original_source_record': body, 'original_source_record_sha256': body_hash(body),
        'source_kind_for_future_document': 'ASYN_TARGETED', **{flag: False for flag in FLAGS}}
    return families, originals


def fictional_document():
    doc = document()
    source = SourceRecord(VOICE, FAMILY, 'ASYN_TARGETED', 'A-SYN',
                          True, True, False, False, False, True)
    doc = replace(doc, sources=(*doc.sources,
        SourceEntry('A fictional calm delivery policy, with no historical authority.', source)))
    doc.validate()
    return doc


def protocol_for(doc, originals):
    return {'source_body_sha256': {
        entry.record.record_id: originals[entry.record.record_id]['original_source_record_sha256']
        for entry in doc.sources}, 'source_render_sha256': {
        entry.record.record_id: sha256(entry.text.encode()).hexdigest() for entry in doc.sources}}


ALLOCATION = {'reference_family_ids': [FAMILY], 'family_splits': {FAMILY: 'TRAIN'}}


class OriginalNativeControls(unittest.TestCase):
    def test_original_checker_refuses_voice_that_was_never_a_compiler_member(self):
        families, originals = fictional_inputs()
        doc = fictional_document()
        with self.assertRaisesRegex(ValueError, 'connected personality construction refused'):
            verify_reference_families(doc, ALLOCATION, families, originals, protocol_for(doc, originals))

    def test_original_e0_control_passes_without_a_connected_view(self):
        families, originals = fictional_inputs()
        doc = document()
        doc.validate()
        verify_reference_families(doc, ALLOCATION, families, originals, protocol_for(doc, originals))


class ConnectedVoiceReferencesTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(adapter, 'The verified connected voice membership view is missing')
        self.families, self.originals = fictional_inputs()

    def connect(self, *, expected_voice_count=1):
        return adapter.connected_voice_reference_families(self.families, self.originals,
            expected_voice_count=expected_voice_count)

    def rebind(self, rid=VOICE):
        """Only test perturbations intentionally re-pin a fictional body's hash."""
        row = self.originals[rid]
        row['original_source_record_sha256'] = body_hash(row['original_source_record'])
        if rid == VOICE:
            row['draft_workitem']['voice_source_record_sha256'] = row['original_source_record_sha256']

    def test_verified_view_adds_only_voice_and_preserves_all_original_memberships(self):
        before_families, before_originals = deepcopy(self.families), deepcopy(self.originals)
        connected = self.connect()
        self.assertEqual(connected, [
            {'family_id': FAMILY, 'split': 'train', 'record_ids': [E0, EINF, VOICE]},
            {'family_id': OTHER_FAMILY, 'split': 'train', 'record_ids': ['unrelated-e0']},
        ])
        self.assertEqual(self.families, before_families)
        self.assertEqual(self.originals, before_originals)
        connected[0]['record_ids'].append('caller-local-change')
        self.assertEqual(self.families, before_families)

    def test_connected_view_passes_unchanged_native_checker_with_a_real_document(self):
        doc = fictional_document()
        verify_reference_families(doc, ALLOCATION, self.connect(), self.originals,
            protocol_for(doc, self.originals))
        doc = document()
        verify_reference_families(doc, ALLOCATION, self.connect(), self.originals,
            protocol_for(doc, self.originals))

    def test_native_checks_still_refuse_altered_body_render_permission_consumer_and_split(self):
        connected = self.connect()
        doc = fictional_document()
        protocol = protocol_for(doc, self.originals)
        bad_protocol = deepcopy(protocol)
        bad_protocol['source_render_sha256'][VOICE] = '0' * 64
        bad_body = deepcopy(protocol)
        bad_body['source_body_sha256'][VOICE] = '0' * 64
        for altered in (bad_protocol, bad_body):
            with self.subTest(protocol=altered), self.assertRaises(ValueError):
                verify_reference_families(doc, ALLOCATION, connected, self.originals, altered)
        for changes in ({'runtime_behavioral_prior_allowed': False}, {'source_kind': 'ASYN_BASE'},
                        {'source_family_id': OTHER_FAMILY}, {'alice_lived_memory': True},
                        {'autobiographical_recall_allowed': True}):
            bad = replace(doc, sources=(doc.sources[0], replace(doc.sources[1],
                record=replace(doc.sources[1].record, **changes))))
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                verify_reference_families(bad, ALLOCATION, connected, self.originals, protocol)
        self.originals[VOICE]['draft_workitem']['original_split'] = 'test'
        with self.assertRaises(ValueError):
            verify_reference_families(doc, ALLOCATION, connected, self.originals, protocol)

    def test_default_count_requires_the_complete_96_voice_rows(self):
        with self.assertRaises(ValueError):
            adapter.connected_voice_reference_families(self.families, self.originals)
        for count in (0, -1, True, '1', 2):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.connect(expected_voice_count=count)

    def test_complete_shared_generator_view_retains_original_ids_at_the_default_count(self):
        template = self.originals.pop(VOICE)
        voice_ids = [f'fictional-delivery-{index:03d}' for index in range(96)]
        for rid in reversed(voice_ids):
            row = deepcopy(template)
            row['draft_workitem']['record_id'] = rid
            row['original_source_record']['proposal_id'] = rid
            row['original_source_record_sha256'] = body_hash(row['original_source_record'])
            row['draft_workitem']['voice_source_record_sha256'] = row['original_source_record_sha256']
            self.originals[rid] = row
        connected = adapter.connected_voice_reference_families(self.families, self.originals)
        self.assertEqual(connected[0]['record_ids'], [E0, EINF, *voice_ids])
        self.assertEqual(connected[1], self.families[1])

    def test_incomplete_or_duplicate_original_compiler_membership_is_refused(self):
        for mutation in ('duplicate_family', 'duplicate_member', 'cross_family_member', 'missing_context'):
            self.families, self.originals = fictional_inputs()
            if mutation == 'duplicate_family':
                self.families.append(deepcopy(self.families[0]))
            elif mutation == 'duplicate_member':
                self.families[0]['record_ids'].append(E0)
            elif mutation == 'cross_family_member':
                self.families[1]['record_ids'].append(E0)
            else:
                del self.originals[EINF]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.connect()

    def test_voice_subtype_declared_consumer_id_and_draft_state_cannot_be_relabelled(self):
        for container, key, value in (
            ('work', 'source_kind', 'ASYN_TARGETED'), ('row', 'source_kind_for_future_document', 'ASYN_BASE'),
            ('work', 'record_id', 'alias'), ('body', 'proposal_id', 'alias'),
            ('work', 'state', 'ACCEPTED'), ('work', 'original_source_family_id', FAMILY),
            ('work', 'original_split', 'dev'), ('work', 'evidence_category', 'E0')):
            self.families, self.originals = fictional_inputs()
            row = self.originals[VOICE]
            target = {'work': row['draft_workitem'], 'row': row, 'body': row['original_source_record']}[container]
            target[key] = value
            if container == 'body':
                self.rebind()
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.connect()

    def test_body_and_both_independent_hash_bindings_must_agree(self):
        for mutation in ('body', 'authoring_hash', 'voice_hash'):
            self.families, self.originals = fictional_inputs()
            row = self.originals[VOICE]
            if mutation == 'body':
                row['original_source_record']['voice_expression_policy']['delivery'] = 'Altered fictional policy.'
            elif mutation == 'authoring_hash':
                row['original_source_record_sha256'] = '0' * 64
            else:
                row['draft_workitem']['voice_source_record_sha256'] = '0' * 64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.connect()

    def test_literal_direct_and_inference_supports_must_match_the_original_body(self):
        for field, value in (('identity_supporting_e0_ids', []),
                             ('identity_supporting_e0_ids', ['unrelated-e0']),
                             ('supporting_curated_einf_ids', []),
                             ('supporting_curated_einf_ids', [E0])):
            self.families, self.originals = fictional_inputs()
            self.originals[VOICE]['draft_workitem']['evidence_links'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.connect()

    def test_direct_support_is_required_and_support_duplicates_or_missing_ids_are_refused(self):
        for direct, inference in (([], []), ([E0, E0], [EINF]), ([E0], [EINF, EINF]),
                                  (['missing-e0'], [EINF]), ([E0], ['missing-einf'])):
            self.families, self.originals = fictional_inputs()
            row = self.originals[VOICE]
            row['original_source_record']['identity_supporting_E0_unit_ids'] = direct
            row['original_source_record']['supporting_curated_EINF_ids'] = inference
            row['draft_workitem']['evidence_links']['identity_supporting_e0_ids'] = direct
            row['draft_workitem']['evidence_links']['supporting_curated_einf_ids'] = inference
            self.rebind()
            with self.subTest(direct=direct, inference=inference), self.assertRaises(ValueError):
                self.connect()

    def test_support_cannot_migrate_family_split_or_consumer_kind(self):
        for rid, key, value, container in (
            (E0, 'original_source_family_id', OTHER_FAMILY, 'work'),
            (E0, 'source_family_id', OTHER_FAMILY, 'work'),
            (E0, 'original_split', 'test', 'work'),
            (E0, 'source_kind', 'EINF', 'work'),
            (EINF, 'source_kind_for_future_document', 'E0', 'row')):
            self.families, self.originals = fictional_inputs()
            row = self.originals[rid]
            (row['draft_workitem'] if container == 'work' else row)[key] = value
            with self.subTest(rid=rid, key=key), self.assertRaises(ValueError):
                self.connect()
        self.families, self.originals = fictional_inputs()
        self.families[0]['split'] = 'dev'
        with self.assertRaises(ValueError):
            self.connect()

    def test_cross_family_support_and_shared_generator_family_divergence_are_refused(self):
        row = self.originals[VOICE]
        row['original_source_record']['identity_supporting_E0_unit_ids'].append('unrelated-e0')
        row['draft_workitem']['evidence_links']['identity_supporting_e0_ids'].append('unrelated-e0')
        self.rebind()
        with self.assertRaises(ValueError):
            self.connect()
        self.families, self.originals = fictional_inputs()
        second = deepcopy(self.originals[VOICE])
        second_id = 'fictional-second-voice'
        second['draft_workitem']['record_id'] = second_id
        second['draft_workitem']['source_family_id'] = OTHER_FAMILY
        second['original_source_record']['proposal_id'] = second_id
        second['original_source_record']['identity_supporting_E0_unit_ids'] = ['unrelated-e0']
        second['original_source_record']['supporting_curated_EINF_ids'] = []
        second['draft_workitem']['evidence_links'] = {
            'identity_supporting_e0_ids': ['unrelated-e0'], 'supporting_curated_einf_ids': []}
        self.originals[second_id] = second
        self.rebind(second_id)
        second['draft_workitem']['voice_source_record_sha256'] = second['original_source_record_sha256']
        with self.assertRaises(ValueError):
            self.connect(expected_voice_count=2)

    def test_existing_voice_membership_and_duplicate_identity_are_refused(self):
        self.families[0]['record_ids'].append(VOICE)
        with self.assertRaises(ValueError):
            self.connect()
        self.families, self.originals = fictional_inputs()
        self.originals['alias'] = deepcopy(self.originals[VOICE])
        with self.assertRaises(ValueError):
            self.connect(expected_voice_count=2)

    def test_original_provenance_and_permissions_cannot_grant_new_authority(self):
        for key, value in (('provenance_class', 'E0'),
            *((key, True) for key in ORIGINAL_FORBIDDEN), ('owner_final_review_required', False),
            ('runtime_behavioral_prior_allowed', 1)):
            self.families, self.originals = fictional_inputs()
            self.originals[VOICE]['original_source_record'][key] = value
            self.rebind()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.connect()

    def test_exact_known_source_declarations_preserve_original_permission_values(self):
        for key, value in (('training_authority', True), ('historical_truth_allowed', True),
            ('alice_lived_memory', True), ('autobiographical_recall_allowed', True),
            ('owner_final_review_required', False), ('runtime_behavioral_prior_allowed', False),
            ('runtime_behavioral_prior_allowed', 1), ('support_mode', 'altered-admin'),
            ('source_coverage_bin', 'altered-admin'), ('invented_authority', False)):
            self.families, self.originals = fictional_inputs()
            self.originals[VOICE]['draft_workitem']['source_declarations'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.connect()

    def test_existing_authoring_and_workitem_authority_flags_remain_false(self):
        for location in ('row', 'work'):
            for flag in FLAGS:
                self.families, self.originals = fictional_inputs()
                row = self.originals[VOICE]
                (row if location == 'row' else row['draft_workitem'])[flag] = True
                with self.subTest(location=location, flag=flag), self.assertRaises(ValueError):
                    self.connect()

    def test_empty_optional_inference_and_false_runtime_prior_are_preserved(self):
        row = self.originals[VOICE]
        row['original_source_record']['supporting_curated_EINF_ids'] = []
        row['draft_workitem']['evidence_links']['supporting_curated_einf_ids'] = []
        row['original_source_record']['runtime_behavioral_prior_allowed'] = False
        row['draft_workitem']['source_declarations']['runtime_behavioral_prior_allowed'] = False
        self.rebind()
        connected = self.connect()
        self.assertEqual(connected[0]['record_ids'], [E0, EINF, VOICE])
        self.assertIs(row['draft_workitem']['source_declarations']['runtime_behavioral_prior_allowed'], False)


if __name__ == '__main__':
    unittest.main()
