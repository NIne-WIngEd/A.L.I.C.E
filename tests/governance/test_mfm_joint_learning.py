"""Tiny original-wire mechanics; fixtures never qualify a trained MFM."""
import copy
from dataclasses import replace
import json
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256
from cognitive_kernel import formation_original_copy_decision as copies
from cognitive_kernel.formation_v1_specialist import SpecialistConfig
from scripts.mfm import formation_original_copy_target_alignment as target
from tests.governance.test_mfm_original_copy_specialist import OriginalCopyAssemblyTests


class JointLearningTests(unittest.TestCase):
    _fixtures = OriginalCopyAssemblyTests._fixtures
    setUpClass = classmethod(OriginalCopyAssemblyTests.setUpClass.__func__)
    record = OriginalCopyAssemblyTests.record

    def setUp(self):
        torch.set_num_threads(1)
        OriginalCopyAssemblyTests.setUp(self)
        from cognitive_kernel.formation_joint_learning import FormationJointLearning
        self.model = FormationJointLearning(self.model.config)
        self.args = OriginalCopyAssemblyTests.args(self)

    def packet(self, *, no_copy=False, no_actions=False):
        decision, _ = self.record(no_copy=no_copy)
        if no_actions:
            decision = {'schema':copies.SCHEMA, 'proposals':[], 'dispositions':[]}
        inventory = copies.bind_original_copy_inventory(self.caller, self.joint,
            caller_sha256=self.caller['record_sha256'], joint_alignment_sha256=self.joint['record_sha256'],
            expected_binding=self.binding)
        reference_leaves = copies.reference_requests(decision, inventory)
        copy_leaves = copies.copy_requests(decision, inventory)
        alignment = target._align(self.processor, decision, reference_leaves, copy_leaves,
            inventory['reference_inventory']['binding'], copies.copy_binding(inventory),
            max_target_tokens=self.model.config.max_target_tokens, mode='teacher_forced')
        ordinals = [{'pointer': row['pointer'], 'candidate_ordinal': row['eligible_candidate_indices'][0]}
                    for row in reference_leaves]
        tuples = {'binding': copies.copy_binding(inventory), 'rows': [
            {'pointer': row['pointer'], 'alternatives': [{'source_slot': 0, 'start_byte': 0, 'end_byte': 1}]}
            for row in copy_leaves]}
        view = {'schema':target.VIEW_SCHEMA, 'decision':copy.deepcopy(decision),
            'binding':copy.deepcopy(inventory['reference_inventory']['binding']),
            'copy_binding':copies.copy_binding(inventory),
            'source_profile':{'input_schema':'mfm-paired-supplied-input-v1',
                'joint_schema':inventory['joint_schema'],
                'pair_schema':'mfm-paired-supplied-joint-target-geometry-v1'},
            'processor_binding':copy.deepcopy(inventory['reference_inventory']['processor_binding']),
            'original_pair_sha256':'1'*64, 'original_frozen_input_sha256':'2'*64,
            'original_target_sha256':'3'*64, 'reference_leaves':copy.deepcopy(reference_leaves),
            'authored_ordinals':copy.deepcopy(ordinals), 'copy_leaves':copy.deepcopy(copy_leaves),
            'authored_copy_tuples':copy.deepcopy(tuples), 'training_authorized':False,
            'learning_qualified':False, 'authority_granted':False}
        view['record_sha256'] = canonical_sha256(view)
        return dict(processor=self.processor, base_states=self.states, source_mask=self.mask,
            inventory=inventory, decision=decision, alignment=alignment,
            alignment_sha256=alignment['record_sha256'],
            target_view=view, expected_view_sha256=view['record_sha256'],
            input_ids=torch.tensor([alignment['decoder_inputs']]), labels=torch.tensor([alignment['labels']]),
            authored_ordinals=ordinals, authored_copy_tuples=tuples)

    def test_single_decoder_original_lexical_ce_and_owned_gradients(self):
        from unittest.mock import patch
        packet = self.packet()
        source = packet['base_states'].detach().clone().requires_grad_()
        packet['base_states'] = source
        with patch.object(self.model, '_decode', wraps=self.model._decode) as decoder:
            result = self.model.joint_loss(**packet)
            self.assertEqual(decoder.call_count, 1)
        self.assertEqual(result['token_count'], len(packet['alignment']['labels']))
        self.assertGreater(result['reference_request_count'], 0)
        self.assertEqual(result['copy_request_count'], 1)
        hidden = result['hidden']
        logits = torch.nn.functional.linear(hidden['hidden'], self.model.token_embedding.weight, self.model.output_bias)
        expected = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
            packet['labels'].reshape(-1))
        torch.testing.assert_close(result['lexical_loss'], expected)
        result['total_loss'].backward()
        self.assertIsNone(source.grad)
        for name in ('token_embedding.weight', 'decoder.layers.0.self_attn.in_proj_weight',
                     'reference_query.weight', 'copy_marker_query.weight', 'copy_boundary.0.weight'):
            gradient = dict(self.model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)

    def test_resealed_omitted_markers_and_partial_shift_refuse(self):
        packet = self.packet()
        bad = copy.deepcopy(packet)
        bad['alignment']['requests'] = []
        bad['alignment']['record_sha256'] = canonical_sha256({k:v for k,v in bad['alignment'].items() if k != 'record_sha256'})
        bad['alignment_sha256'] = bad['alignment']['record_sha256']
        with self.assertRaises(ValueError): self.model.joint_loss(**bad)
        bad = dict(packet, labels=packet['labels'].clone())
        bad['labels'][0, -1] = 3
        with self.assertRaises(ValueError): self.model.joint_loss(**bad)

    def test_same_wire_rejects_substituted_eligible_choices(self):
        packet = self.packet()
        alternate = copy.deepcopy(packet)
        row = alternate['authored_ordinals'][0]
        eligible = alternate['alignment']['requests'][0]['eligible_candidate_indices']
        self.assertGreater(len(eligible), 1)
        row['candidate_ordinal'] = next(value for value in eligible if value != row['candidate_ordinal'])
        with self.assertRaises(ValueError): self.model.joint_loss(**alternate)
        alternate = copy.deepcopy(packet)
        boundaries = [b['byte_offset'] for b in alternate['inventory']['sources'][0]['boundaries']]
        alternative = alternate['authored_copy_tuples']['rows'][0]['alternatives'][0]
        alternative['end_byte'] = next(value for value in boundaries if value > alternative['end_byte'])
        with self.assertRaises(ValueError): self.model.joint_loss(**alternate)

    def test_model_tokenizer_geometry_must_match_serving_contract(self):
        packet = self.packet()
        for config in (replace(self.model.config, end_token_id=3),
                       replace(self.model.config, vocabulary_size=self.model.config.vocabulary_size+1)):
            wrong = type(self.model)(config)
            with self.assertRaises(ValueError): wrong.joint_loss(**packet)

    def test_resealed_alignment_rejects_bool_integer_type_substitution(self):
        packet = self.packet()
        altered = copy.deepcopy(packet)
        altered['alignment']['learning_qualified'] = 0
        altered['alignment']['record_sha256'] = canonical_sha256({
            key:value for key,value in altered['alignment'].items() if key != 'record_sha256'})
        altered['alignment_sha256'] = altered['alignment']['record_sha256']
        with self.assertRaises(ValueError): self.model.joint_loss(**altered)

    def test_target_decision_type_must_match_pinned_view(self):
        packet = self.packet()
        altered = copy.deepcopy(packet)
        altered['decision']['proposals'][0]['proposal']['anchors'][0]['start_byte'] = False
        with self.assertRaises(ValueError): self.model.joint_loss(**altered)

    def test_coherently_resealed_processor_numeric_alias_refuses_before_decode(self):
        from unittest.mock import patch
        packet = self.packet()
        changed = copy.deepcopy(packet)
        reference = changed['inventory']['reference_inventory']
        reference['processor_binding']['tokenizer_wrapper']['bos_token_id'] = 2.0
        inventory = changed['inventory']
        inventory['record_sha256'] = canonical_sha256({k:v for k,v in inventory.items() if k != 'record_sha256'})
        binding = copies.copy_binding(inventory)
        view = changed['target_view']
        view['processor_binding'] = copy.deepcopy(reference['processor_binding'])
        view['copy_binding'] = binding
        view['authored_copy_tuples']['binding'] = binding
        view['record_sha256'] = canonical_sha256({k:v for k,v in view.items() if k != 'record_sha256'})
        changed['expected_view_sha256'] = view['record_sha256']
        changed['authored_copy_tuples']['binding'] = binding
        alignment = target._align(self.processor, changed['decision'], copies.reference_requests(changed['decision'], inventory),
            copies.copy_requests(changed['decision'], inventory), reference['binding'], binding,
            max_target_tokens=self.model.config.max_target_tokens, mode='teacher_forced')
        changed['alignment'] = alignment
        changed['alignment_sha256'] = alignment['record_sha256']
        with patch.object(self.model, '_decode', wraps=self.model._decode) as decoder:
            with self.assertRaises(ValueError): self.model.joint_loss(**changed)
            decoder.assert_not_called()

    def test_reload_rejects_manifest_type_aliases_and_invalid_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = [root / name for name in ('weights.safetensors','config.json','manifest.json')]
            provenance = {'parent':'tiny'}
            source_pins = {'source_version':1}
            record = self.model.export_component(*paths, component_id='tiny-control', provenance=provenance,
                provenance_sha256=canonical_sha256(provenance), source_pins=source_pins,
                source_pins_sha256=canonical_sha256(source_pins), evidence_scope='untrained_mechanics')
            model = type(self.model)(self.model.config)
            base = dict(weight_sha256=sha256(paths[0].read_bytes()).hexdigest(),
                config_sha256=sha256(paths[1].read_bytes()).hexdigest(),
                expected_provenance_sha256=canonical_sha256(provenance), expected_source_pins=source_pins,
                expected_evidence_scope='untrained_mechanics')
            changed_path = root / 'changed-manifest.json'
            variants = []
            shape = copy.deepcopy(record); shape['tensor_inventory']['copy_boundary.0.bias']['shape'][0] = 8.0
            variants.append(('shape_float',shape,{}))
            count = copy.deepcopy(record); count['weight_bytes'] = float(count['weight_bytes'])
            variants.append(('weight_bytes_float',count,{}))
            for value in ('NOT A VALID ID!', None):
                row = copy.deepcopy(record); row['component_id'] = value
                variants.append(('component_id',row,{}))
            row = copy.deepcopy(record); row['source_pins'] = {'source_version':1.0}
            variants.append(('source_alias',row,{}))
            for value in ({}, []):
                row = copy.deepcopy(record); row['source_pins'] = value
                row['source_pins_sha256'] = canonical_sha256(value)
                variants.append(('source_container',row,{'expected_source_pins':value}))
                row = copy.deepcopy(record); row['provenance'] = value
                row['provenance_sha256'] = canonical_sha256(value)
                variants.append(('provenance_container',row,{'expected_provenance_sha256':canonical_sha256(value)}))
            from unittest.mock import patch
            for name, variant, changes in variants:
                changed_path.write_bytes(canonical_json_bytes(variant))
                with self.subTest(name=name), patch.object(model,'load_state_dict',wraps=model.load_state_dict) as load:
                    with self.assertRaises(ValueError):
                        model.load_component(paths[0],paths[1],changed_path,
                            **dict(base,manifest_sha256=sha256(changed_path.read_bytes()).hexdigest(),**changes))
                    load.assert_not_called()

    def test_no_copy_preserves_none_and_zero_count(self):
        packet = self.packet(no_copy=True)
        result = self.model.joint_loss(**packet)
        self.assertIsNone(result['copy_loss'])
        self.assertEqual(result['copy_request_count'], 0)

    def test_no_reference_or_copy_keeps_only_whole_wire_lexical_term(self):
        result = self.model.joint_loss(**self.packet(no_actions=True))
        self.assertEqual(result['reference_request_count'], 0)
        self.assertEqual(result['copy_request_count'], 0)
        self.assertIsNone(result['reference_loss'])
        self.assertIsNone(result['copy_loss'])
        torch.testing.assert_close(result['total_loss'], result['lexical_loss'])

    def test_lexical_gradient_matches_original_chunked_forward(self):
        packet = self.packet()
        sibling = type(self.model)(self.model.config)
        sibling.load_state_dict(self.model.state_dict(), strict=True)
        original = sibling(base_states=packet['base_states'], source_mask=packet['source_mask'],
            input_ids=packet['input_ids'], labels=packet['labels'])[1]
        original.backward()
        result = self.model.joint_loss(**packet)
        result['lexical_loss'].backward()
        torch.testing.assert_close(result['lexical_loss'], original, atol=0, rtol=0)
        for name in ('token_embedding.weight', 'output_bias', 'decoder.layers.0.self_attn.in_proj_weight',
                     'source_projection.weight'):
            torch.testing.assert_close(dict(self.model.named_parameters())[name].grad,
                dict(sibling.named_parameters())[name].grad, atol=1e-6, rtol=1e-6)

    def test_whole_component_roundtrip_and_atomic_refusals(self):
        with torch.no_grad():
            self.model.reference_query.weight.zero_(); self.model.reference_query.bias.zero_()
            self.model.reference_path.weight.zero_()
            for layer in (self.model.copy_source_query, self.model.copy_start_query, self.model.copy_end_query):
                layer.weight.zero_(); layer.bias.zero_()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = [root / name for name in ('weights.safetensors', 'config.json', 'manifest.json')]
            provenance = {'original_parent': 'tiny-untrained-control'}
            source_pins = {'source_file_sha256': 'a'*64}
            record = self.model.export_component(*paths, component_id='tiny-control',
                provenance=provenance, provenance_sha256=canonical_sha256(provenance),
                source_pins=source_pins, source_pins_sha256=canonical_sha256(source_pins),
                evidence_scope='untrained_mechanics')
            from cognitive_kernel import formation_semantics_v16 as grounder
            grounder_key = 'src/cognitive_kernel/formation_semantics_v16.py'
            self.assertEqual(record['interface_pins'][grounder_key],
                sha256(Path(grounder.__file__).read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError): self.model.export_component(*paths,
                component_id='tiny-control', provenance=provenance,
                provenance_sha256=canonical_sha256(provenance), source_pins=source_pins,
                source_pins_sha256=canonical_sha256(source_pins), evidence_scope='untrained_mechanics')
            fresh = type(self.model)(self.model.config)
            expected = dict(weight_sha256=sha256(paths[0].read_bytes()).hexdigest(),
                config_sha256=sha256(paths[1].read_bytes()).hexdigest(),
                manifest_sha256=sha256(paths[2].read_bytes()).hexdigest(),
                expected_provenance_sha256=canonical_sha256(provenance), expected_source_pins=source_pins,
                expected_evidence_scope='untrained_mechanics')
            fresh.load_component(*paths, **expected)
            for name, value in self.model.state_dict().items():
                torch.testing.assert_close(fresh.state_dict()[name], value, atol=0, rtol=0)
            before = {k:v.clone() for k,v in fresh.state_dict().items()}
            from unittest.mock import patch
            from cognitive_kernel import formation_joint_learning as joint
            original_file_hash = joint._file_sha256
            def changed_grounder(path):
                if Path(path).name == 'formation_semantics_v16.py':
                    return 'f'*64
                return original_file_hash(path)
            with patch.object(joint, '_file_sha256', side_effect=changed_grounder):
                with self.assertRaises(ValueError): fresh.load_component(*paths, **expected)
            with self.assertRaises(ValueError): fresh.load_component(*paths, **dict(expected, expected_source_pins={'other': 'b'*64}))
            with self.assertRaises(ValueError): fresh.load_component(*paths, **dict(expected, config_sha256='f'*64))
            typed_config = dict(self.model.config.record())
            typed_config['width'] = float(typed_config['width'])
            typed_config['end_token_id'] = True
            changed_config_path = root / 'changed-config.json'
            changed_config_path.write_bytes(canonical_json_bytes(typed_config))
            changed_config_manifest = copy.deepcopy(record)
            changed_config_manifest['config_file_sha256'] = sha256(changed_config_path.read_bytes()).hexdigest()
            changed_config_manifest_path = root / 'changed-config-manifest.json'
            changed_config_manifest_path.write_bytes(canonical_json_bytes(changed_config_manifest))
            with self.assertRaises(ValueError):
                fresh.load_component(paths[0], changed_config_path, changed_config_manifest_path,
                    **dict(expected, config_sha256=changed_config_manifest['config_file_sha256'],
                        manifest_sha256=sha256(changed_config_manifest_path.read_bytes()).hexdigest()))
            altered = root / 'altered.safetensors'
            altered_manifest = root / 'altered-manifest.json'
            for mutation in ('extra', 'missing', 'shape', 'nan', 'dtype'):
                state = {k:v.clone() for k,v in self.model.state_dict().items()}
                if mutation == 'extra': state['foreign'] = torch.zeros(1)
                if mutation == 'missing': state.pop('copy_end_query.weight')
                if mutation == 'shape': state['output_bias'] = state['output_bias'][:-1]
                if mutation == 'nan': state['copy_end_query.weight'][0, 0] = float('nan')
                if mutation == 'dtype': state['copy_end_query.weight'] = state['copy_end_query.weight'].bfloat16()
                save_file(state, str(altered))
                changed_record = copy.deepcopy(record)
                changed_record['weight_file_sha256'] = sha256(altered.read_bytes()).hexdigest()
                changed_record['weight_bytes'] = altered.stat().st_size
                changed_record['tensor_inventory'] = {
                    key:{'shape':list(value.shape),'dtype':'F32' if value.dtype == torch.float32 else 'BF16'}
                    for key,value in sorted(state.items())}
                altered_manifest.write_bytes(canonical_json_bytes(changed_record))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    fresh.load_component(altered, paths[1], altered_manifest,
                        **dict(expected, weight_sha256=changed_record['weight_file_sha256'],
                            manifest_sha256=sha256(altered_manifest.read_bytes()).hexdigest()))
            for name, value in before.items():
                torch.testing.assert_close(fresh.state_dict()[name], value, atol=0, rtol=0)
            self.model = fresh
            decision, raw = self.record()
            ids = self._fixtures.stream(self, raw)
            generated = self.model.generate_original_copy_decision(**self.args, max_new_tokens=len(ids)+1)
            self.assertEqual(generated['validation_status'], 'grounded_proposal_only', generated['validation_error'])
            self.assertEqual(generated['generated_token_ids'], ids)
            self.assertIsNotNone(generated['bundle'])


if __name__ == '__main__': unittest.main()
