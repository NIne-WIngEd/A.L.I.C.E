"""Direct original-packet lowering and owned joint fitting.

Original paired records are geometry evidence, never training permission. The
P2/J5 supplied reader is permanently unauthenticated; no flag is rewritten.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from importlib import util as import_util
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import tempfile

import torch
from safetensors.torch import load_file, save_file

from cognitive_kernel.canonical import (canonical_json_bytes, canonical_sha256,
    require_identifier, require_sha256, CognitiveKernelContractError)
from cognitive_kernel import formation_original_copy_decision as copies
from cognitive_kernel.formation_dataset_admission import admit_formation_corpus
from scripts.mfm import formation_paired_teaching as j4_paired
from scripts.mfm import formation_joint_input_v4 as j4_input
from scripts.mfm import formation_joint_target_alignment as j4_target
from scripts.mfm import formation_original_copy_target_alignment as mixed
from scripts.mfm import formation_source_alignment as source


PROGRAM_SCHEMA = 'mfm-native-joint-fit-program-v1'
J6_PROGRAM_SCHEMA = 'mfm-native-joint-j6-fit-program-v2'
RECIPE_SCHEMA = 'mfm-native-joint-fit-recipe-v1'
CHECKPOINT_SCHEMA = 'mfm-native-joint-owned-checkpoint-v1'
PAIR_AUTHORING_SCHEMA = 'mfm-native-joint-pair-authoring-v1'
P2_SUCCESSOR_AUTHORING_SCHEMA = 'mfm-native-joint-p2-successor-authoring-v1'
J6_SUCCESSOR_AUTHORING_SCHEMA = 'mfm-native-joint-j6-model-author-conversion-v1'


def _need(condition, message):
    if not condition:
        raise CognitiveKernelContractError(message)


def _private_original_modules(refs):
    """Load only the exact three original J5 modules declared by the program."""
    names = {'source': 'formation_joint_input_v5',
             'paired': 'formation_paired_teaching_v2',
             'target': 'formation_joint_target_alignment_v2'}
    _need(type(refs) is dict and set(refs) == set(names),
          'complete private original module FileRefs required')
    paths = {}
    for role, name in names.items():
        ref = refs[role]
        _need(type(ref) is dict and set(ref) == {'path', 'sha256'} and
              type(ref['path']) is str and ref['path'] and
              Path(ref['path']).is_absolute(),
              'private original module FILE reference differs')
        require_sha256(ref['sha256'], 'private original module FILE')
        path = Path(ref['path']).resolve(strict=True)
        _need(path.is_file() and path.name == name + '.py' and
              _file_sha256(path) == ref['sha256'],
              'private original module physical FILE differs')
        paths[role] = path
    _need(len(set(paths.values())) == 3,
          'private original module FILE identities overlap')
    modules = {}
    for role, name in names.items():
        qualified = 'scripts.mfm.' + name
        existing = sys.modules.get(qualified)
        if existing is not None:
            _need(Path(existing.__file__).resolve(strict=True) == paths[role],
                  'private original module was already imported from other bytes')
            module = existing
        else:
            spec = import_util.spec_from_file_location(qualified, paths[role])
            _need(spec is not None and spec.loader is not None,
                  'private original module cannot be loaded')
            module = import_util.module_from_spec(spec)
            sys.modules[qualified] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                del sys.modules[qualified]
                raise
        _need(_file_sha256(module.__file__) == refs[role]['sha256'],
              'private original imported FILE differs')
        modules[role] = module
    return modules


def lower_original_case(*, profile, frozen, accepted, pair, expected, processor,
                        max_target_tokens, private_paired=None, private_target=None):
    """Construct one complete mixed-wire target from the original pair.

    This is target lowering only; a production fit must separately admit exact
    source/target FileRefs and join a frozen feature-bank readback.
    """
    if profile == 'j4-p1':
        _need(private_paired is None and private_target is None,
              'J4 may not use a private target reader')
        paired, original = j4_paired, j4_target
    else:
        _need(profile == 'j5-p2' and private_paired is not None and
              private_target is not None and
              private_paired.__name__ == 'scripts.mfm.formation_paired_teaching_v2' and
              private_target.__name__ == 'scripts.mfm.formation_joint_target_alignment_v2',
              'explicit original J5 paired and target modules required')
        paired, original = private_paired, private_target
    _need(type(max_target_tokens) is int and max_target_tokens > 0,
          'explicit whole mixed-wire target limit required')
    _need(type(frozen) is dict and type(accepted) is dict and type(pair) is dict,
          'complete original records required')
    paired.validate_input(frozen, expected_freeze_sha256=frozen['record_sha256'],
                          expected=expected)
    _need(frozen['split_row']['split'] in ('train', 'development') and
          frozen['training_authorized'] is False and
          frozen['current_service_authenticated'] is False and
          accepted.get('training_authorized') is False and
          accepted.get('author_authenticated') is False and
          accepted.get('state') == 'complete',
          'original supplied profile is not a complete target')
    original.validate_pair(pair, frozen, accepted, expected=expected,
                           expected_pair_sha256=pair['record_sha256'])
    example = paired.admitted(accepted, frozen, accepted['record_sha256'])
    _need(type(pair.get('fields')) is list and all(
        row['annotation_status'] == 'authored'
        for row in pair['fields'] if row['responsibility'] == 'semantic_authoring'),
        'complete semantic authoring annotations required')
    _need(canonical_sha256(source.processor_binding(processor)) ==
          canonical_sha256(expected['processor_binding']),
          'actual target processor differs from original source')
    joint = frozen['joint']; caller = frozen['caller']
    episode_proof = frozen['meaning']['episode_snapshot']['historical_prefix_proof']
    task_proof = frozen['meaning']['task_snapshot']['historical_prefix_proof']
    world_sha256 = episode_proof['world']['world_sha256']
    cut_sha256 = episode_proof['cut']['record_sha256']
    _need(task_proof['world']['world_sha256'] == world_sha256 and
          task_proof['cut']['record_sha256'] == cut_sha256,
          'actual original episode/task world-cut proof differs')
    require_sha256(world_sha256, 'original world proof')
    require_sha256(cut_sha256, 'original cut proof')
    binding = {'caller_sha256': caller['record_sha256'],
               'candidate_inventory_sha256': joint['candidate_inventory_sha256'],
               **joint['cache']}
    inventory = copies.bind_original_copy_inventory(caller, joint,
        caller_sha256=caller['record_sha256'],
        joint_alignment_sha256=joint['record_sha256'], expected_binding=binding)
    view = mixed.original_copy_decision_view(example, pair, frozen,
        pair_sha256=pair['record_sha256'],
        frozen_input_sha256=frozen['record_sha256'], inventory=inventory)
    _need(all(type(row['alternatives']) is list and bool(row['alternatives'])
              for row in view['authored_copy_tuples']['rows']),
          'each copy marker needs complete authored tuple alternatives')
    _need(len(view['authored_ordinals']) == len(view['reference_leaves']) and
          len(view['authored_copy_tuples']['rows']) == len(view['copy_leaves']),
          'every mixed-wire marker needs an authored choice')
    alignment = mixed.align_original_copy_decision_target(processor, view,
        view_sha256=view['record_sha256'], max_target_tokens=max_target_tokens)
    _need(canonical_sha256(source.processor_binding(processor)) ==
          canonical_sha256(expected['processor_binding']),
          'target processor changed during lowering')
    return {'profile': profile, 'case_id': frozen['split_row']['case_id'],
            'split_row': deepcopy(frozen['split_row']),
            'original_frozen_sha256': frozen['record_sha256'],
            'original_target_file_sha256': accepted['target_file_sha256'],
            'original_pair_sha256': pair['record_sha256'],
            'world_sha256': world_sha256, 'cut_sha256': cut_sha256,
            'source_alignment_sha256': joint['record_sha256'],
            'source_cache': deepcopy(joint['cache']),
            'source_projection': deepcopy(frozen['batch']),
            'inventory': inventory, 'target_view': view, 'alignment': alignment,
            'decision': deepcopy(view['decision']),
            'authored_ordinals': deepcopy(view['authored_ordinals']),
            'authored_copy_tuples': deepcopy(view['authored_copy_tuples'])}


def lower_j6_original_case(*, frozen, accepted, pair, source_descriptor,
                           migration_reader, j5_source, processor,
                           max_target_tokens, material15_file_sha256):
    """Lower one complete J6 original through the existing copy consumer."""
    from scripts.mfm import formation_joint_input_v6 as j6
    from scripts.mfm import formation_paired_teaching_v6 as j6_paired
    from scripts.mfm import formation_joint_target_alignment_v6 as j6_target

    require_sha256(material15_file_sha256, 'physical J6 material15 FILE')
    _need(type(max_target_tokens) is int and max_target_tokens > 0 and
          type(source_descriptor) is dict and
          type(frozen) is dict and type(accepted) is dict and type(pair) is dict,
          'complete J6 original lowering operands required')
    _need(type(pair.get('max_target_tokens')) is int and
          max_target_tokens == pair['max_target_tokens'],
          'J6 original and mixed-wire target budget must be identical')
    _need(source_descriptor['profile'] == 'j6-source-only' and
          source_descriptor['case_id'] == frozen['split_row']['case_id'] and
          source_descriptor['source_file_sha256'] == frozen['source_file_sha256'] and
          source_descriptor['source_record_sha256'] == frozen['source']['record_sha256'] and
          source_descriptor['j5_file_sha256'] == _file_sha256(j5_source.__file__) and
          source_descriptor['proposed_max_source_tokens'] ==
              frozen['proposed_max_source_tokens'],
          'material-derived J6 source descriptor differs')
    j6_paired.validate_input(j5_source, j6, frozen,
        expected_input_sha256=frozen['record_sha256'],
        expected_source_file_sha256=source_descriptor['source_file_sha256'],
        migration_reader=migration_reader)
    _need(frozen['split_row']['split'] == 'train' and
          frozen['training_authorized'] is False and
          frozen['current_service_authenticated'] is False and
          frozen['capacity_admitted'] is False and
          accepted['training_authorized'] is False and
          accepted['authenticated_service'] is False and
          accepted['independent_gold'] is False,
          'J6 construction records cannot claim authority or capacity')
    example, _ = j6_paired.require_complete_labels(frozen, accepted)
    j6_target.validate_pair(pair, frozen, accepted,
        expected_pair_sha256=pair['record_sha256'])
    actual_processor = source.processor_binding(processor)
    _need(canonical_sha256(actual_processor) ==
          canonical_sha256(frozen['processor_binding']) and
          canonical_sha256(pair['processor_binding']) ==
          canonical_sha256(actual_processor),
          'J6 actual processor/source/target binding differs')
    source_freeze = frozen['source']
    joint = source_freeze['alignment']
    caller = source_freeze['caller']
    _need(joint['record_sha256'] == frozen['source']['alignment']['record_sha256'] and
          canonical_sha256(frozen['source_cache']) ==
              canonical_sha256(joint['cache']) and
          canonical_sha256(frozen['native_batch']) ==
              canonical_sha256(source_freeze['native_batch']) and
          frozen['world_sha256'] == source_freeze['world_sha256'] and
          frozen['cut_sha256'] == source_freeze['cut_sha256'],
          'whole J6 source/native/world/cut differs')
    world_id = frozen['split_row']['lineage']['world']
    require_identifier(world_id, 'J6 connected world ID')
    binding = {'caller_sha256': caller['record_sha256'],
               'candidate_inventory_sha256': joint['candidate_inventory_sha256'],
               **joint['cache']}
    inventory = copies.bind_original_copy_inventory(caller, joint,
        caller_sha256=caller['record_sha256'],
        joint_alignment_sha256=joint['record_sha256'],
        expected_binding=binding)
    view = mixed.original_copy_decision_view(example, pair, frozen,
        pair_sha256=pair['record_sha256'],
        frozen_input_sha256=frozen['record_sha256'], inventory=inventory)
    _need(len(view['authored_ordinals']) == len(view['reference_leaves']) and
          len(view['authored_copy_tuples']['rows']) == len(view['copy_leaves']) and
          all(type(row['alternatives']) is list and row['alternatives']
              for row in view['authored_copy_tuples']['rows']),
          'complete J6 authored ordinal/copy alternatives required')
    alignment = mixed.align_original_copy_decision_target(processor, view,
        view_sha256=view['record_sha256'],
        max_target_tokens=max_target_tokens)
    _need(canonical_sha256(source.processor_binding(processor)) ==
          canonical_sha256(actual_processor),
          'J6 target processor changed during mixed-wire lowering')
    return {'profile': 'j6-firstparty-successor',
            'case_id': frozen['split_row']['case_id'],
            'split_row': deepcopy(frozen['split_row']),
            'original_frozen_sha256': frozen['record_sha256'],
            'original_target_file_sha256': accepted['original_target_file_sha256'],
            'original_pair_sha256': pair['record_sha256'],
            'world_sha256': frozen['world_sha256'],
            'mass_world_id': world_id, 'cut_sha256': frozen['cut_sha256'],
            'material15_file_sha256': material15_file_sha256,
            'source_file_sha256': frozen['source_file_sha256'],
            'source_alignment_sha256': joint['record_sha256'],
            'source_cache': deepcopy(joint['cache']),
            'source_projection': deepcopy(frozen['native_batch']),
            'source_descriptor': deepcopy(source_descriptor),
            'inventory': inventory, 'target_view': view,
            'alignment': alignment, 'decision': deepcopy(view['decision']),
            'authored_ordinals': deepcopy(view['authored_ordinals']),
            'authored_copy_tuples': deepcopy(view['authored_copy_tuples'])}


def join_frozen_source(lowered, *, feature_root, feature_roster_sha256,
                       j5_source=None, allow_fixture=False):
    """Read back the entire saved native batch and detached source states."""
    from scripts.mfm import export_joint_source_features as features
    receipt, tensors = features.read_case(feature_root,
        roster_sha256=feature_roster_sha256, case_id=lowered['case_id'],
        expected_alignment_sha256=lowered['source_alignment_sha256'],
        expected_cache=lowered['source_cache'], j5_source=j5_source,
        allow_fixture=allow_fixture,
        expected_j6_source=lowered.get('source_descriptor'))
    j6 = lowered['profile'] == 'j6-firstparty-successor'
    fields = ('input_ids', 'attention_mask', 'mm_token_type_ids') if j6 else (
        'input_ids', 'attention_mask')
    _need(receipt['profile'] == ('j6-source-only' if j6 else lowered['profile']) and
          canonical_sha256({key: receipt['frozen_batch'][key] for key in fields}) ==
          canonical_sha256(lowered['source_projection']) and
          torch.equal(tensors['attention_mask'], torch.tensor(
              lowered['inventory']['source_geometry']['attention_mask']).unsqueeze(0)),
          'native source readback differs from original projected IDs/mask')
    return {'lowered': lowered, 'feature_receipt': receipt,
            'source_states': tensors['states'].detach(),
            'source_mask': tensors['attention_mask'].detach(),
            'native_batch': {key: value.detach() for key, value in tensors.items()
                             if key != 'states'}}


def joint_operands(joined, *, processor, device):
    """Exact lowerer output for the complete joint objective."""
    row = joined['lowered']; alignment = row['alignment']
    return dict(processor=processor,
        base_states=joined['source_states'].to(device).detach(),
        source_mask=joined['source_mask'].to(device).detach(),
        inventory=row['inventory'], decision=row['decision'], alignment=alignment,
        alignment_sha256=alignment['record_sha256'],
        input_ids=torch.tensor([alignment['decoder_inputs']], dtype=torch.long,
                               device=device),
        labels=torch.tensor([alignment['labels']], dtype=torch.long, device=device),
        authored_ordinals=row['authored_ordinals'],
        authored_copy_tuples=row['authored_copy_tuples'],
        target_view=row['target_view'],
        expected_view_sha256=row['target_view']['record_sha256'])


def preflight_admitted_cases(*, admission_manifest_path, admission_manifest_sha256,
                             owner_authorization_ref, cases):
    """Join every train original to independently admitted physical originals.

    P2's original supplied records remain fixed false; only a separately
    admitted physical original plus its independently pinned pair-authoring
    conversion can establish the genuine successor material path.
    """
    admission = admit_formation_corpus(admission_manifest_path,
        expected_sha256=admission_manifest_sha256, teacher_training=True,
        owner_authorization_ref=owner_authorization_ref)
    _need(type(cases) is list and len(cases) == len(admission.train),
          'every admitted train case requires one complete joint packet')
    manifest_raw = Path(admission_manifest_path).read_bytes()
    _need(sha256(manifest_raw).hexdigest() == admission_manifest_sha256,
          'admitted material manifest changed during joint preflight')
    def unique(pairs):
        value = {}
        for key, entry in pairs:
            _need(key not in value, 'duplicate admitted material key')
            value[key] = entry
        return value
    manifest = json.loads(manifest_raw.decode('utf-8'), object_pairs_hook=unique)
    _need(type(manifest) is dict and type(manifest.get('cases')) is list,
          'admitted case author provenance missing')
    source_rows = {row['case_id']: row for row in manifest['cases']}
    admitted_by_id = {row.case_id: row for row in admission.train}
    seen = set()
    for row in cases:
        _need(type(row) is dict and set(row) ==
              {'frozen', 'accepted', 'pair', 'joined'},
              'actual original/source case handoff differs')
        frozen, accepted, pair, joined = (row['frozen'], row['accepted'],
                                         row['pair'], row['joined'])
        lowered = joined['lowered']
        case_id = lowered['case_id']
        _need(case_id in admitted_by_id and case_id not in seen and
              case_id == frozen['split_row']['case_id'] and
              lowered['original_frozen_sha256'] == frozen['record_sha256'] and
              lowered['original_target_file_sha256'] == accepted[
                  'original_target_file_sha256' if lowered['profile'] ==
                  'j6-firstparty-successor' else 'target_file_sha256'] and
              frozen['split_row']['split'] == 'train',
              'admitted train case and original lowerer differ')
        seen.add(case_id)
        _need(lowered['profile'] in ('j4-p1', 'j5-p2',
                                      'j6-firstparty-successor'),
              'unknown original source profile')
        caller = (frozen['source']['caller'] if lowered['profile'] ==
                  'j6-firstparty-successor' else frozen['caller'])
        context, opened, _ = j4_input.caller_api.caller_input_from_record(caller)
        expected_sources = sorted(sha256(raw).hexdigest() for _, raw in opened)
        admitted = admitted_by_id[case_id]
        _need(expected_sources == sorted(ref.sha256 for ref in admitted.source_payloads)
              and admitted.target_payload.sha256 == lowered['original_target_file_sha256'],
              'independently admitted physical source/target bytes differ')
        _need(joined['feature_receipt']['case_id'] == case_id and
              (lowered['profile'] != 'j6-firstparty-successor' or
               joined['feature_receipt']['profile'] == 'j6-source-only') and
              joined['feature_receipt']['base']['fixture_small_width'] is False and
              joined['feature_receipt']['target_visible_to_backbone'] is False,
              'production frozen source bank/target isolation differs')
        admitted_row = source_rows[case_id]
        _need(type(admitted_row) is dict and
              admitted_row['author_id'] == admitted_row['target_provenance']['producer_id'] and
              admitted_row['host_family'] == context.base.scope.host_instance_id,
              'admitted original host/author and conversion producer differ')
        conversion_ref = admitted_row['target_provenance']['conversion']
        conversion = _record_ref(admission.root, conversion_ref)
        expected_conversion = {
            'schema': (PAIR_AUTHORING_SCHEMA if lowered['profile'] == 'j4-p1'
                       else P2_SUCCESSOR_AUTHORING_SCHEMA),
            'case_id': case_id,
            'author_id': admitted_row['author_id'],
            'original_frozen_sha256': frozen['record_sha256'],
            'original_target_file_sha256': lowered['original_target_file_sha256'],
            'original_pair_sha256': pair['record_sha256'],
            'source_alignment_sha256': lowered['source_alignment_sha256'],
            'source_file_sha256s': expected_sources,
            'target_provenance_output_sha256':
                admitted_row['target_provenance']['output']['sha256'],
            'annotations_sha256': canonical_sha256(pair['annotations']) if
                lowered['profile'] != 'j6-firstparty-successor' else None,
            'literal_declarations_sha256':
                canonical_sha256(pair['literal_declarations']) if
                lowered['profile'] != 'j6-firstparty-successor' else None,
            'authored_ordinals_sha256':
                canonical_sha256(lowered['authored_ordinals']),
            'authored_copy_tuples_sha256':
                canonical_sha256(lowered['authored_copy_tuples']),
            'world_sha256': lowered['world_sha256'],
            'cut_sha256': lowered['cut_sha256'],
            'variant_key': lowered['variant_key']}
        if lowered['profile'] == 'j5-p2':
            expected_conversion['original_geometry_profile'] = 'j5-p2'
        if lowered['profile'] == 'j6-firstparty-successor':
            _need(manifest['schema'] == 'mfm-v16-owner-teacher-corpus-v3' and
                  admitted_row['target_origin'] == 'owner-authorized-model-author' and
                  admitted_row['target_provenance']['input']['sha256'] ==
                      accepted['projected_view_file_sha256'] and
                  admitted_row['target_provenance']['output']['sha256'] ==
                      accepted['raw_author_draft_file_sha256'] and
                  admitted_row['author_id'] == accepted['author_id'] and
                  admitted_row['target_provenance']['producer_version'] ==
                      accepted['author_producer_version'] and
                  frozen['source_file_sha256'] ==
                      lowered['source_file_sha256'] and
                  canonical_sha256(frozen['native_batch']) ==
                      canonical_sha256(joined['feature_receipt']['frozen_batch']),
                  'J6 model-author provenance/source-native admission differs')
            expected_conversion.update({
                'schema': J6_SUCCESSOR_AUTHORING_SCHEMA,
                'material15_file_sha256': lowered['material15_file_sha256'],
                'source_file_sha256': lowered['source_file_sha256'],
                'source_record_sha256': frozen['source']['record_sha256'],
                'source_native_sha256': canonical_sha256(frozen['native_batch']),
                'target_provenance_input_sha256':
                    admitted_row['target_provenance']['input']['sha256'],
                'raw_author_draft_file_sha256':
                    accepted['raw_author_draft_file_sha256'],
                'support_file_sha256': accepted['support_file_sha256'],
                'annotation_file_sha256': accepted['annotation_file_sha256'],
                'choice_file_sha256': pair['choice_file_sha256'],
                'choice_record_sha256': pair['choice_packet']['record_sha256'],
                'connected_world_id': lowered['mass_world_id']})
            expected_conversion['annotations_sha256'] = canonical_sha256(
                pair['choice_packet']['field_annotations'])
            expected_conversion['literal_declarations_sha256'] = canonical_sha256(
                pair['choice_packet']['literal_declarations'])
        _need(canonical_sha256(conversion) ==
              canonical_sha256(expected_conversion),
              'admitted physical author conversion does not bind complete pair/choices')
    _need(seen == set(admitted_by_id), 'incomplete admitted train roster')
    j6_worlds = {row['joined']['lowered']['mass_world_id'] for row in cases
                 if row['joined']['lowered']['profile'] == 'j6-firstparty-successor'}
    _need(not j6_worlds or
          (len(j6_worlds) == 1 and all(row['joined']['lowered']['profile'] ==
              'j6-firstparty-successor' for row in cases)),
          'J6 admitted originals must share one connected train world')
    gradient_paths = tuple(path for row in admission.train
                           for path in (*(ref.path for ref in row.source_payloads),
                                        row.target_payload.path))
    development_paths = tuple(path for row in admission.development
                              for path in (*(ref.path for ref in row.source_payloads),
                                           row.target_payload.path))
    audit = admission.audit_handoff(gradient_paths=gradient_paths,
                                    development_paths=development_paths)
    return admission, audit


def _file_sha256(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json_file(path, expected_sha256):
    require_sha256(expected_sha256, 'frozen JSON FILE')
    raw = Path(path).read_bytes()
    _need(sha256(raw).hexdigest() == expected_sha256,
          'frozen JSON FILE bytes differ')
    def unique(pairs):
        value = {}
        for key, entry in pairs:
            _need(key not in value, 'duplicate frozen JSON key')
            value[key] = entry
        return value
    row = json.loads(raw.decode('utf-8'), object_pairs_hook=unique)
    _need(type(row) is dict and canonical_json_bytes(row) == raw,
          'frozen JSON is not a canonical object')
    return row


def _record_ref(root, ref):
    _need(type(ref) is dict and set(ref) == {'path', 'sha256'} and
          type(ref['path']) is str and ref['path'] and '\\' not in ref['path'] and
          not Path(ref['path']).is_absolute() and
          all(part not in ('', '.', '..') for part in ref['path'].split('/')),
          'original record FileRef path differs')
    target = (root / ref['path']).resolve(strict=True)
    _need(target.is_relative_to(root) and target.is_file(),
          'original record FileRef leaves its material root')
    return _json_file(target, ref['sha256'])


def _material_bytes(root, ref):
    """Read one exact source-material FileRef within the physical root."""
    _need(type(ref) is dict and set(ref) in (
              {'path', 'sha256'},
              {'path', 'bytes', 'file_sha256', 'record_sha256'}) and
          type(ref.get('path')) is str and
          ref['path'] and '\\' not in ref['path'] and
          not Path(ref['path']).is_absolute() and
          all(part not in ('', '.', '..') for part in ref['path'].split('/')),
          'J6 material FileRef path differs')
    pin = ref.get('file_sha256', ref.get('sha256'))
    require_sha256(pin, 'J6 material FILE')
    root = Path(root).resolve(strict=True)
    path = (root / ref['path']).resolve(strict=True)
    _need(path.is_relative_to(root) and path.is_file(),
          'J6 material FileRef leaves physical root')
    raw = path.read_bytes()
    _need(sha256(raw).hexdigest() == pin and
          ('bytes' not in ref or type(ref['bytes']) is int and
           ref['bytes'] == len(raw)), 'J6 material physical FILE differs')
    return path, raw


def _material_json(root, ref):
    path, raw = _material_bytes(root, ref)
    def unique(pairs):
        row = {}
        for key, value in pairs:
            _need(key not in row, 'duplicate J6 material JSON key')
            row[key] = value
        return row
    record = json.loads(raw.decode('utf-8'), object_pairs_hook=unique)
    _need(type(record) is dict and
          raw in (canonical_json_bytes(record), canonical_json_bytes(record) + b'\n') and
          ('record_sha256' not in ref or
           record.get('record_sha256') == ref['record_sha256']),
          'J6 material JSON/record identity differs')
    return path, record


def read_j6_material15(material_root, manifest_ref, *, j5_source):
    """Read source and all original material refs; never open author labels."""
    from scripts.mfm import formation_joint_input_v6 as j6
    from scripts.mfm import formation_paired_teaching_v6 as j6_paired

    root = Path(material_root).resolve(strict=True)
    _, case = _material_json(root, manifest_ref)
    _need(case.get('schema') == 'mfm-j6-material15-source-manifest-v1' and
          case.get('record_sha256') == canonical_sha256({k: v for k, v in
              case.items() if k != 'record_sha256'}) and
          case.get('training_authorized') is False and
          case.get('capacity_admitted') is False and
          case.get('author_material_opened') is False,
          'complete J6 source material seal/status differs')
    roles = ('world', 'cut', 'request', 'verification_boundary',
             'delivery_boundary', 'view', 'verification_receipt', 'bundle',
             'caller', 'meaning', 'joint', 'batch', 'expected',
             'split_roster', 'code_expectations')
    _need(type(case.get('metadata')) is dict and
          set(case['metadata']) == set(roles),
          'complete fifteen J6 metadata roles required')
    data = {role: _material_json(root, case['metadata'][role])[1]
            for role in roles}
    for role in roles[10:]:
        _need(data[role].get('record_sha256') == canonical_sha256({k: v for k, v in
              data[role].items() if k != 'record_sha256'}),
              'current J6 metadata seal differs')
    bundle = data['bundle']
    _need(canonical_sha256(data['cut']) == canonical_sha256(
              data['view']['historical_prefix_proof']['cut']) and
          data['world']['world_id'] == data['cut']['world_id'] and
          canonical_sha256(bundle['cut']) == canonical_sha256(data['cut']) ==
              canonical_sha256(data['request']['cut']) ==
              canonical_sha256(data['view']['cut']) and
          bundle['replay_sha256'] == data['view']['replay_sha256'] and
          bundle['input_request_sha256'] == data['request']['record_sha256'] and
          data['request']['read_boundary'] == data['delivery_boundary'] and
          bundle['delivery_read_receipt']['boundary_sha256'] ==
              data['delivery_boundary']['record_sha256'] and
          data['verification_receipt']['boundary_sha256'] ==
              data['verification_boundary']['record_sha256'] and
          data['verification_receipt']['cut_sha256'] == data['cut']['record_sha256'] and
          data['verification_receipt']['world_sha256'] ==
              data['world']['world_sha256'] and
          data['verification_receipt']['replay_sha256'] ==
              data['view']['replay_sha256'] and
          canonical_sha256(bundle['verification_receipt']) ==
              canonical_sha256(data['verification_receipt']) and
          canonical_sha256(bundle['verification_recheck_reads']) ==
              canonical_sha256(data['verification_receipt']['reads']),
          'J6 causal cut/request/view/boundary joins differ')
    original_proof = bundle['original_artifact_pins']
    original_delivery = bundle['delivery_read_receipt']['reads']
    verification_reads = data['verification_receipt']['reads']
    _need(type(original_proof) is list and type(original_delivery) is list and
          type(verification_reads) is list and
          type(case.get('verification_artifact_ids')) is list and
          type(case.get('ordered_proof_delivery_union')) is list and
          type(case.get('delivery')) is list and
          canonical_sha256(case['verification_artifact_ids']) ==
              canonical_sha256([row['artifact_id'] for row in original_proof]) and
          canonical_sha256([row['artifact'] for row in verification_reads]) ==
              canonical_sha256(original_proof),
          'J6 proof verification roster differs')
    expected_union = list(original_proof)
    by_id = {row['artifact_id']: row for row in original_proof}
    _need(len(by_id) == len(original_proof), 'J6 proof artifact duplicated')
    for call in original_delivery:
        artifact = call['artifact']
        aid = artifact['artifact_id']
        if aid in by_id:
            _need(canonical_sha256(by_id[aid]) == canonical_sha256(artifact),
                  'J6 delivery/proof artifact identity differs')
        else:
            expected_union.append(artifact)
            by_id[aid] = artifact
    _need(len(case['ordered_proof_delivery_union']) == len(expected_union) and
          canonical_sha256([row['artifact'] for row in
              case['ordered_proof_delivery_union']]) ==
              canonical_sha256(expected_union),
          'J6 proof/delivery ordered union differs')
    _need(len(case['delivery']) == len(original_delivery) and
          canonical_sha256([{key: row[key] for key in ('artifact', 'family', 'key')}
              for row in case['delivery']]) ==
          canonical_sha256([{key: call[key] for key in ('artifact', 'family', 'key')}
              for call in original_delivery]),
          'J6 delivery call roster differs')
    proof = {}
    for row in case['ordered_proof_delivery_union']:
        artifact, ref = row['artifact'], row['original']
        aid = artifact['artifact_id']
        _need(aid not in proof, 'J6 proof artifact duplicated')
        _, raw = _material_bytes(root, ref)
        _need(type(artifact['bytes']) is int and len(raw) == artifact['bytes'] and
              sha256(raw).hexdigest() == artifact['file_sha256'],
              'J6 proof original body differs')
        proof[aid] = ref
    for call in verification_reads:
        artifact = call['artifact']
        _need(artifact['artifact_id'] in proof and
              type(call['body_bytes']) is int and
              call['body_bytes'] == artifact['bytes'] and
              call['body_sha256'] == artifact['file_sha256'] and
              call['pre_boundary_sha256'] ==
                  data['verification_boundary']['record_sha256'] and
              call['post_boundary_sha256'] ==
                  data['verification_boundary']['record_sha256'],
              'J6 proof verification body/boundary differs')
    for row, call in zip(case['delivery'], original_delivery):
        artifact, ref = row['artifact'], row['original']
        _need(canonical_sha256(proof[artifact['artifact_id']]) ==
              canonical_sha256(ref),
              'J6 delivery/proof original identity differs')
        _, raw = _material_bytes(root, ref)
        _need(type(call['body_bytes']) is int and
              len(raw) == artifact['bytes'] == call['body_bytes'] and
              sha256(raw).hexdigest() == artifact['file_sha256'] ==
                  call['body_sha256'] and
              call['pre_boundary_sha256'] ==
                  data['delivery_boundary']['record_sha256'] and
              call['post_boundary_sha256'] ==
                  data['delivery_boundary']['record_sha256'],
              'J6 delivery original body/boundary differs')
    source_path, frozen = _material_json(root, case['source_freeze'])
    _need(data['caller'] == frozen['caller'] and
          data['meaning'] == frozen['meaning'] and
          data['bundle']['caller'] == frozen['caller'] and
          data['bundle']['meaning'] == frozen['meaning'] and
          data['joint'] == frozen['alignment'] and
          data['batch']['fields'] == frozen['native_batch'] and
          data['expected']['source_file_sha256'] ==
              case['source_freeze']['file_sha256'] and
          canonical_sha256(data['expected']['source_cache']) ==
              canonical_sha256(frozen['alignment']['cache']) and
          canonical_sha256(data['expected']['processor_binding']) ==
              canonical_sha256(frozen['alignment']['processor_binding']) and
          data['expected']['source_tokens'] ==
              frozen['alignment']['source_tokens'] and
          data['expected']['proposed_max_source_tokens'] ==
              frozen['proposed_max_source_tokens'],
          'J6 fifteen metadata/source joins differ')
    code = data['code_expectations']
    _need(code['current_j6_source_file_sha256'] == _file_sha256(j6.__file__) and
          code['paired_source_file_sha256'] == _file_sha256(j6_paired.__file__) and
          code['original_j5_source_file_sha256'] == _file_sha256(j5_source.__file__) and
          case['source_migration_receipt_file_sha256'] ==
              j6_paired.SOURCE_MIGRATION_FILE,
          'actual J6/J5 source interface differs')
    j6.validate_frozen_source(j5_source, frozen,
        j5_file_sha256=_file_sha256(j5_source.__file__),
        proposed_max_source_tokens=frozen['proposed_max_source_tokens'])
    split = case['split_identity']
    roster = data['split_roster']
    _need(split['case_id'] == case['case_id'] and
          split['cut_id'] == case['cut_id'] == frozen['cut_id'] and
          split['cut_sha256'] == frozen['cut_sha256'] and
          split['prefix_world_sha256'] == frozen['world_sha256'] and
          split['source_file_sha256'] == case['source_freeze']['file_sha256'] and
          split['source_record_sha256'] == frozen['record_sha256'] and
          split['world_id'] == roster['world_id'] and
          split['split'] == 'train' and
          roster['one_train_ancestry_component'] is True and
          roster['development_cases'] == [] and
          any(row['case_id'] == case['case_id'] and
              row['cut_id'] == case['cut_id'] for row in roster['cases']),
          'J6 connected train world/cut/source identity differs')
    descriptor = {'case_id': case['case_id'], 'profile': 'j6-source-only',
        'source_file': str(source_path),
        'source_file_sha256': case['source_freeze']['file_sha256'],
        'source_record_sha256': frozen['record_sha256'],
        'j5_file_sha256': _file_sha256(j5_source.__file__),
        'proposed_max_source_tokens': frozen['proposed_max_source_tokens']}
    return case, descriptor


def _expected_original(frozen, *, j5_source=None):
    joint = frozen['joint']
    binding = (j4_input.joint_code_binding_v4() if j5_source is None else
               j5_source.joint_code_binding_v5())
    return {'caller_sha256': frozen['caller']['record_sha256'],
        'meaning_sha256': frozen['meaning']['record_sha256'],
        'joint_sha256': joint['record_sha256'],
        'batch_sha256': canonical_sha256(frozen['batch']),
        'joint_code_binding': binding,
        'processor_binding': joint['processor_binding'],
        'source_identity': joint['source_identity'],
        'max_source_tokens': joint['max_source_tokens']}


def _case_masses(rows):
    worlds = {}
    for row in rows:
        lower = row['joined']['lowered']
        world, cut, variant = (lower.get('mass_world_id', lower['world_sha256']),
                               lower['cut_sha256'],
                               lower['variant_key'])
        if 'mass_world_id' in lower:
            require_identifier(world, 'admitted connected world ID')
            require_sha256(lower['world_sha256'], 'original prefix world proof')
        else:
            require_sha256(world, 'admitted original world proof')
        require_sha256(cut, 'admitted original cut proof')
        _need(type(variant) is str and
              require_identifier(variant, 'authored variant key') == variant,
              'explicit admitted authored variant key required')
        group = worlds.setdefault(world, {}).setdefault(cut, {})
        _need(variant not in group,
              'duplicate authored variant key within original cut')
        group[variant] = lower['case_id']
    _need(worlds and len({row['joined']['lowered']['case_id'] for row in rows}) == len(rows),
          'joint fit needs unique cases')
    return {case_id: 1.0 / (len(worlds) * len(cuts) * len(variants))
        for cuts in worlds.values() for variants in cuts.values()
        for case_id in variants.values()}


def _fit_update(model, processor, all_rows, window, optimizer, *, device,
                feature_root=None, feature_roster_sha256=None, j5_source=None):
    """One completed AdamW boundary over an explicit frozen case window."""
    masses = _case_masses(all_rows)
    identities = {id(row) for row in all_rows}
    _need(type(window) is list and window and
          all(id(row) in identities for row in window),
          'optimizer window differs from complete admitted roster')
    mass_scale = len(all_rows) / len(window)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    observations = []
    for row in window:
        if feature_root is None:
            # Isolated tiny unit mechanics only. The production entry always
            # provides a frozen physical bank and reopens one case at a time.
            joined = row['joined']
        else:
            descriptor = row['joined']
            joined = join_frozen_source(descriptor['lowered'],
                feature_root=feature_root,
                feature_roster_sha256=feature_roster_sha256,
                j5_source=j5_source if descriptor['lowered']['profile'] != 'j4-p1'
                    else None,
                allow_fixture=False)
            _need(canonical_sha256(joined['feature_receipt']) ==
                  canonical_sha256(descriptor['feature_receipt']),
                  'actual consumed source case differs from preflight receipt')
        operands = joint_operands(joined, processor=processor, device=device)
        result = model.joint_loss(**operands)
        loss = result['total_loss']
        _need(loss.ndim == 0 and torch.isfinite(loss).item() and
              result['token_count'] > 0,
              'complete joint case loss is invalid')
        case_id = joined['lowered']['case_id']
        coefficient = masses[case_id] * mass_scale
        (loss * coefficient).backward()
        observations.append({'case_id': case_id, 'mass': masses[case_id],
            'global_mass_coefficient': coefficient,
            'lexical_loss': float(result['lexical_loss'].detach()),
            'reference_loss': None if result['reference_loss'] is None else
                float(result['reference_loss'].detach()),
            'copy_loss': None if result['copy_loss'] is None else
                float(result['copy_loss'].detach()),
            'total_loss': float(loss.detach()),
            'token_count': result['token_count'],
            'reference_request_count': result['reference_request_count'],
            'copy_request_count': result['copy_request_count']})
        del operands, result, loss, joined
    _need(all(parameter.grad is None or torch.isfinite(parameter.grad).all().item()
              for parameter in model.parameters()), 'joint owned gradients are nonfinite')
    optimizer.step()
    _need(all(torch.isfinite(parameter).all().item()
              for parameter in model.parameters()), 'joint owned update is nonfinite')
    return observations


def _fit_epoch(model, processor, rows, optimizer, *, device,
               feature_root=None, feature_roster_sha256=None, j5_source=None):
    """Tiny one-window mechanics compatibility; production uses _fit_update."""
    return _fit_update(model, processor, rows, rows, optimizer,
        device=device, feature_root=feature_root,
        feature_roster_sha256=feature_roster_sha256, j5_source=j5_source)


def _owned_parameter_groups(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    _need(len(names) == len(tuple(model.parameters())),
          'owned model parameters have ambiguous identities')
    groups = []
    seen = set()
    for group in optimizer.param_groups:
        roster = []
        for parameter in group['params']:
            identity = id(parameter)
            _need(identity in names and identity not in seen,
                  'optimizer owns missing or duplicate model parameter')
            seen.add(identity)
            roster.append(names[identity])
        groups.append(roster)
    _need(seen == set(names), 'optimizer omits owned model parameter')
    return groups


def _owned_optimizer_steps(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return {names[id(parameter)]: int(value['step'].item())
            for parameter, value in optimizer.state.items()}


def _save_checkpoint(directory, *, model, optimizer, update_index,
                     consumed_cases, case_order, recipe_sha256,
                     program_sha256, admission_sha256, feature_roster_sha256,
                     history):
    destination = Path(directory)
    _need(not destination.exists() and destination.parent.is_dir(),
          'checkpoint destination must be new')
    temporary = Path(tempfile.mkdtemp(prefix='.joint-checkpoint-', dir=destination.parent))
    try:
        weights = temporary / 'weights.safetensors'
        state = {name: tensor.detach().cpu().contiguous()
                 for name, tensor in model.state_dict().items()}
        _need(all(t.dtype == torch.float32 and torch.isfinite(t).all().item()
                  for t in state.values()), 'whole checkpoint state differs')
        save_file(state, str(weights))
        trainer = temporary / 'trainer.pt'
        torch.save({'optimizer': optimizer.state_dict(),
                    'cpu_rng': torch.random.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    'update_index': update_index}, trainer)
        _need(type(update_index) is int and update_index >= 1 and
              type(consumed_cases) is int and consumed_cases >= update_index and
              type(case_order) is list and case_order and
              type(history) is list and len(history) == update_index,
              'completed optimizer cursor/history differs')
        core = {'schema': CHECKPOINT_SCHEMA,
            'update_index': update_index,
            'consumed_cases': consumed_cases,
            'case_order_sha256': canonical_sha256(case_order),
            'next_case_index': consumed_cases % len(case_order),
            'completed_passes': consumed_cases // len(case_order),
            'weight_file_sha256': _file_sha256(weights),
            'trainer_file_sha256': _file_sha256(trainer),
            'recipe_sha256': recipe_sha256, 'program_sha256': program_sha256,
            'admission_sha256': admission_sha256,
            'feature_roster_sha256': feature_roster_sha256,
            'owned_parameter_groups': _owned_parameter_groups(model, optimizer),
            'owned_optimizer_steps': _owned_optimizer_steps(model, optimizer),
            'history': history}
        (temporary / 'checkpoint.json').write_bytes(canonical_json_bytes(core))
        os.rename(temporary, destination)
        return core
    finally:
        if temporary.exists():
            for path in temporary.iterdir():
                path.unlink()
            temporary.rmdir()


def _restore_checkpoint(directory, *, model, optimizer, recipe_sha256,
                        program_sha256, admission_sha256, feature_roster_sha256,
                        expected_checkpoint_file_sha256, case_order,
                        accumulation_cases):
    directory = Path(directory).resolve(strict=True)
    _need({p.name for p in directory.iterdir()} ==
          {'weights.safetensors', 'trainer.pt', 'checkpoint.json'},
          'complete checkpoint file roster differs')
    raw = (directory / 'checkpoint.json').read_bytes()
    require_sha256(expected_checkpoint_file_sha256, 'external checkpoint FILE')
    _need(sha256(raw).hexdigest() == expected_checkpoint_file_sha256,
          'restart checkpoint differs from external FILE pin')
    record = json.loads(raw.decode('utf-8'))
    _need(canonical_json_bytes(record) == raw and type(record) is dict and
          set(record) == {'schema', 'update_index', 'consumed_cases',
                          'case_order_sha256', 'next_case_index',
                          'completed_passes', 'weight_file_sha256',
                          'trainer_file_sha256', 'recipe_sha256',
                          'program_sha256', 'admission_sha256',
                          'feature_roster_sha256', 'owned_parameter_groups',
                          'owned_optimizer_steps', 'history'} and
          record['schema'] == CHECKPOINT_SCHEMA and
          type(record['update_index']) is int and
          record['update_index'] >= 1 and
          type(accumulation_cases) is int and accumulation_cases > 0 and
          type(case_order) is list and case_order and
          record['case_order_sha256'] == canonical_sha256(case_order) and
          type(record['consumed_cases']) is int and
          record['consumed_cases'] ==
          record['update_index'] * accumulation_cases and
          type(record['next_case_index']) is int and
          record['next_case_index'] == record['consumed_cases'] % len(case_order) and
          type(record['completed_passes']) is int and
          record['completed_passes'] == record['consumed_cases'] // len(case_order) and
          (record['recipe_sha256'], record['program_sha256'],
           record['admission_sha256'], record['feature_roster_sha256']) ==
          (recipe_sha256, program_sha256, admission_sha256,
           feature_roster_sha256) and
          _file_sha256(directory / 'weights.safetensors') == record['weight_file_sha256'] and
          _file_sha256(directory / 'trainer.pt') == record['trainer_file_sha256'],
          'checkpoint immutable recipe/material/weights differ')
    _need(record['owned_parameter_groups'] ==
          _owned_parameter_groups(model, optimizer),
          'restart owned parameter names/order/groups differ')
    _need(type(record['history']) is list and
          len(record['history']) == record['update_index'] and
          all(type(item) is dict and set(item) ==
              {'update_index', 'next_case_index', 'completed_passes', 'cases'} and
              type(item['update_index']) is int and
              item['update_index'] == index and
              type(item['next_case_index']) is int and
              item['next_case_index'] ==
              (index * accumulation_cases) % len(case_order) and
              type(item['completed_passes']) is int and
              item['completed_passes'] ==
              (index * accumulation_cases) // len(case_order) and
              type(item['cases']) is list and
              len(item['cases']) == accumulation_cases and
              [case['case_id'] for case in item['cases']] ==
              [case_order[((index - 1) * accumulation_cases + offset) % len(case_order)]
               for offset in range(accumulation_cases)]
              for index, item in enumerate(record['history'], start=1)),
          'restart complete optimizer-boundary history differs')
    state = load_file(str(directory / 'weights.safetensors'), device='cpu')
    current = model.state_dict()
    _need(set(state) == set(current) and all(
        state[name].shape == current[name].shape and
        state[name].dtype == torch.float32 and torch.isfinite(state[name]).all().item()
        for name in current), 'checkpoint whole model state differs')
    trainer = torch.load(directory / 'trainer.pt', map_location='cpu',
                         weights_only=True)
    _need(type(trainer) is dict and set(trainer) ==
          {'optimizer', 'cpu_rng', 'cuda_rng', 'update_index'} and
          type(trainer['update_index']) is int and
          trainer['update_index'] == record['update_index'] and
          type(trainer['cpu_rng']) is torch.Tensor and
          trainer['cpu_rng'].device.type == 'cpu' and
          trainer['cpu_rng'].dtype == torch.uint8 and
          trainer['cpu_rng'].shape == torch.random.get_rng_state().shape and
          type(trainer['cuda_rng']) is list and
          len(trainer['cuda_rng']) ==
          (torch.cuda.device_count() if torch.cuda.is_available() else 0) and
          all(type(value) is torch.Tensor and value.dtype == torch.uint8 and
              value.device.type == 'cpu' and
              value.shape == current_rng.shape
              for value, current_rng in zip(trainer['cuda_rng'],
                  torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])),
          'checkpoint optimizer/cursor/RNG differs')
    # Native generators parse opaque engine bytes without touching live RNG.
    # Shape and dtype alone do not establish a valid restorable state.
    try:
        torch.Generator(device='cpu').set_state(trainer['cpu_rng'])
        for index, value in enumerate(trainer['cuda_rng']):
            torch.Generator(device=f'cuda:{index}').set_state(value)
    except RuntimeError as exc:
        raise CognitiveKernelContractError(
            'checkpoint native RNG contents cannot be restored') from exc
    loaded_optimizer = trainer['optimizer']
    expected_optimizer = optimizer.state_dict()
    _need(type(loaded_optimizer) is dict and
          set(loaded_optimizer) == {'state', 'param_groups'} and
          type(loaded_optimizer['state']) is dict and
          type(loaded_optimizer['param_groups']) is list and
          len(loaded_optimizer['param_groups']) == len(expected_optimizer['param_groups']) and
          all(type(saved) is dict and set(saved) == set(current) and
              saved['params'] == current['params'] and
              canonical_sha256({k:v for k,v in saved.items() if k != 'params'}) ==
              canonical_sha256({k:v for k,v in current.items() if k != 'params'})
              for saved, current in zip(loaded_optimizer['param_groups'],
                                        expected_optimizer['param_groups'])),
          'restart owned optimizer groups differ from immutable schedule')
    parameters = [parameter for group in optimizer.param_groups
                  for parameter in group['params']]
    names = [name for group in _owned_parameter_groups(model, optimizer)
             for name in group]
    _need(type(record['owned_optimizer_steps']) is dict and
          all(type(name) is str and name in names and
              type(step) is int and 1 <= step <= record['update_index']
              for name, step in record['owned_optimizer_steps'].items()) and
          {names[index]: int(value['step'].item())
           for index, value in loaded_optimizer['state'].items()
           if type(index) is int and 0 <= index < len(names) and
              type(value) is dict and type(value.get('step')) is torch.Tensor and
              value['step'].numel() == 1 and torch.isfinite(value['step']).all().item()} ==
          record['owned_optimizer_steps'],
          'restart actual named parameter moment/step roster differs')
    _need(all(type(index) is int and 0 <= index < len(parameters) and
              type(value) is dict and set(value) ==
              {'step', 'exp_avg', 'exp_avg_sq'} and
              type(value['step']) is torch.Tensor and
              value['step'].numel() == 1 and
              value['step'].dtype in (torch.float32, torch.float64) and
              1 <= float(value['step'].item()) <= record['update_index'] and
              float(value['step'].item()).is_integer() and
              torch.isfinite(value['step']).all().item() and
              all(type(value[key]) is torch.Tensor and
                  value[key].shape == parameters[index].shape and
                  value[key].dtype == torch.float32 and
                  torch.isfinite(value[key]).all().item()
                  for key in ('exp_avg', 'exp_avg_sq'))
              for index, value in loaded_optimizer['state'].items()),
          'restart complete owned optimizer moments differ')
    model.load_state_dict(state, strict=True)
    optimizer.load_state_dict(trainer['optimizer'])
    torch.random.set_rng_state(trainer['cpu_rng'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(trainer['cuda_rng'])
    return record


def _recipe(record, *, program_sha256, admission_sha256,
            feature_roster_sha256, seed_manifest_sha256):
    _need(type(record) is dict and set(record) == {
        'schema', 'program_sha256', 'admission_sha256',
        'feature_roster_sha256', 'seed_manifest_sha256', 'case_order',
        'update_budget', 'accumulation_cases',
        'learning_rate', 'weight_decay', 'adam_betas', 'adam_eps',
        'device', 'seed', 'checkpoint_every_updates', 'precision',
        'term_weights', 'mass_policy', 'update_policy', 'runtime',
        'trainer_file_sha256', 'generation_max_new_tokens'} and
        record['schema'] == RECIPE_SCHEMA and
        (record['program_sha256'], record['admission_sha256'],
         record['feature_roster_sha256'], record['seed_manifest_sha256']) ==
        (program_sha256, admission_sha256,
         feature_roster_sha256, seed_manifest_sha256) and
        canonical_sha256(record['term_weights']) ==
        canonical_sha256({'lexical': 1, 'reference': 1, 'copy': 1}) and
        record['mass_policy'] == 'equal-world-equal-cut-equal-variant' and
        record['update_policy'] == 'frozen-order-case-window-admitted-mass' and
        record['precision'] == 'owned-fp32-source-bf16' and
        type(record['case_order']) is list and record['case_order'] and
        len(set(record['case_order'])) == len(record['case_order']) and
        all(type(value) is str and require_identifier(value, 'case_id') == value
            for value in record['case_order']) and
        type(record['update_budget']) is int and record['update_budget'] > 0 and
        type(record['accumulation_cases']) is int and
        0 < record['accumulation_cases'] <= len(record['case_order']) and
        type(record['seed']) is int and record['seed'] >= 0 and
        type(record['checkpoint_every_updates']) is int and
        record['checkpoint_every_updates'] > 0 and
        type(record['device']) is str and record['device'] and
        type(record['learning_rate']) in (int, float) and
        0 < record['learning_rate'] < float('inf') and
        type(record['weight_decay']) in (int, float) and
        0 <= record['weight_decay'] < float('inf') and
        type(record['adam_betas']) is list and len(record['adam_betas']) == 2 and
        all(type(value) in (int, float) and 0 < value < 1
            for value in record['adam_betas']) and
        type(record['adam_eps']) in (int, float) and
        0 < record['adam_eps'] < float('inf') and
        type(record['generation_max_new_tokens']) is int and
        record['generation_max_new_tokens'] > 0 and
        type(record['runtime']) is dict and
        set(record['runtime']) == {'torch_version', 'transformers_version'} and
        record['runtime']['torch_version'] == torch.__version__ and
        record['runtime']['transformers_version'] == version('transformers') and
        record['trainer_file_sha256'] == _file_sha256(__file__),
        'immutable actual joint objective/schedule recipe differs')


def fit_joint_from_program(*, program_path, program_file_sha256, material_root,
                           admission_manifest_path, admission_manifest_sha256,
                           owner_authorization_ref, feature_root,
                           feature_roster_sha256, prepared_base_dir,
                           prepared_receipt_path,
                           prepared_receipt_file_sha256,
                           prepared_receipt_sha256,
                           parent_source_receipt_sha256,
                           seed_weight_path, seed_config_path, seed_manifest_path,
                           seed_weight_sha256, seed_config_sha256,
                           seed_manifest_sha256, seed_provenance_sha256,
                           seed_source_pins, recipe_path, recipe_file_sha256,
                           output_dir, resume_checkpoint=None,
                           resume_checkpoint_file_sha256=None):
    """Fit the new joint stage from exact original FileRefs and frozen source.

    No candidate is qualified here. The current supplied P2 corpus still fails
    its own false authority/incomplete/overflow guards; J5 requires a separate
    physically admitted author-conversion successor packet.
    """
    from transformers import AutoProcessor
    from cognitive_kernel.formation_v1_specialist import SpecialistConfig
    from cognitive_kernel.formation_joint_learning import FormationJointLearning
    from cognitive_kernel import formation_joint_learning as joint_source
    from scripts.mfm import export_joint_source_features as features
    from scripts.mfm import train_v1_formation_specialist as shared

    material_root = Path(material_root).resolve(strict=True)
    feature_root = Path(feature_root).resolve(strict=True)
    prepared = Path(prepared_base_dir).resolve(strict=True)
    receipt_path = Path(prepared_receipt_path).resolve(strict=True)
    output = Path(output_dir).resolve(strict=False)
    _need(material_root.is_dir() and feature_root.is_dir() and
          output.parent.is_dir() and output != material_root and
          material_root not in output.parents and output != feature_root and
          feature_root not in output.parents and output != prepared and
          prepared not in output.parents,
          'joint fit output must be outside immutable material/source roots')
    program = _json_file(program_path, program_file_sha256)
    j6_program = program.get('schema') == J6_PROGRAM_SCHEMA
    _need(set(program) in ({'schema', 'cases'},
                           {'schema', 'cases', 'private_original_modules'},
                           {'schema', 'cases', 'private_original_modules',
                            'source_migration'}) and
          program['schema'] in (PROGRAM_SCHEMA, J6_PROGRAM_SCHEMA) and
          type(program['cases']) is list and program['cases'],
          'complete original training program differs')
    common_case_keys = {'case_id', 'profile', 'frozen', 'accepted', 'pair',
                        'max_target_tokens', 'variant_key'}
    _need(all(type(row) is dict and set(row) ==
              (common_case_keys | {'material15'} if j6_program else common_case_keys) and
              type(row['case_id']) is str and
              require_identifier(row['case_id'], 'case_id') == row['case_id'] and
              type(row['variant_key']) is str and
              require_identifier(row['variant_key'], 'variant_key') ==
              row['variant_key'] and
              type(row['max_target_tokens']) is int and
              row['max_target_tokens'] > 0 for row in program['cases']),
          'original train case FileRef program differs')
    _need(all(row['profile'] == 'j6-firstparty-successor'
              for row in program['cases']) if j6_program else
          all(row['profile'] in ('j4-p1', 'j5-p2-successor')
              for row in program['cases']),
          'supplied P2 is not independently admitted successor material')
    _need(not j6_program or (len(program['cases']) == 6 and
          len({row['case_id'] for row in program['cases']}) == 6),
          'complete six-cut J6 program required')
    uses_j5 = any(row['profile'] in ('j5-p2-successor', 'j6-firstparty-successor')
                  for row in program['cases'])
    _need(('private_original_modules' in program) == uses_j5,
          'J5 successor requires exact original module FILE roster')
    _need(('source_migration' in program) == j6_program,
          'J6 requires one pinned six-cut source migration FILE')
    private = (_private_original_modules(program['private_original_modules'])
               if uses_j5 else None)
    j5_source = None if private is None else private['source']
    migration_raw = None
    if j6_program:
        from scripts.mfm import formation_paired_teaching_v6 as j6_paired
        _, migration_raw = _material_bytes(material_root,
            program['source_migration'])
        _need(sha256(migration_raw).hexdigest() ==
              j6_paired.SOURCE_MIGRATION_FILE,
              'J6 actual six-cut source migration differs')
        # Historical false construction flags cannot substitute for a new
        # physically admitted owner-authorized model-author corpus.
        admitted_early = admit_formation_corpus(admission_manifest_path,
            expected_sha256=admission_manifest_sha256, teacher_training=True,
            owner_authorization_ref=owner_authorization_ref)
        admission_raw = Path(admission_manifest_path).read_bytes()
        _need(sha256(admission_raw).hexdigest() == admission_manifest_sha256,
              'J6 admitted corpus changed after source-rights verification')
        admitted_record = json.loads(admission_raw.decode('utf-8'))
        train_rows = [item for item in admitted_record['cases']
                      if item['split'] == 'train']
        _need(admitted_record['schema'] ==
              'mfm-v16-owner-teacher-corpus-v3' and
              [item['case_id'] for item in train_rows] ==
              [item['case_id'] for item in program['cases']] ==
              [item.case_id for item in admitted_early.train] and
              all(item['target_origin'] == 'owner-authorized-model-author'
                  for item in train_rows),
              'J6 requires exact admitted v3 model-author train originals')
    recipe = _json_file(recipe_path, recipe_file_sha256)
    _recipe(recipe, program_sha256=program_file_sha256,
            admission_sha256=admission_manifest_sha256,
            feature_roster_sha256=feature_roster_sha256,
            seed_manifest_sha256=seed_manifest_sha256)
    _need(recipe['case_order'] == [row['case_id'] for row in program['cases']],
          'actual case order differs from immutable fit recipe')
    # Corpus authority precedes processor and any feature tensor/model load.
    # Constructed J6 records are fixed false and never grant this admission.
    if not j6_program:
        admit_formation_corpus(admission_manifest_path,
            expected_sha256=admission_manifest_sha256, teacher_training=True,
            owner_authorization_ref=owner_authorization_ref)
    roster = features._read_record(feature_root / 'roster.json',
                                   features.SCHEMA, feature_roster_sha256)
    first = features._verified_prepared(None, receipt_path, prepared,
        prepared_receipt_file_sha256, prepared_receipt_sha256,
        parent_source_receipt_sha256)
    files = {entry['path']: entry for entry in first['files']}
    _need(first['schema'] == shared.CLONE_SCHEMA and
          files['model.safetensors']['sha256'] == shared.SOURCE_WEIGHT_SHA256 and
          roster['base']['fixture_small_width'] is False and
          canonical_sha256(roster['base']['prepared_base']) ==
          canonical_sha256({
              'receipt_sha256': first['receipt_sha256'],
              'receipt_file_sha256': prepared_receipt_file_sha256,
              'parent_source_receipt_sha256': parent_source_receipt_sha256,
              'source_weight_sha256': shared.SOURCE_WEIGHT_SHA256,
              'foundation_commit': shared.FOUNDATION_COMMIT,
              'foundation_verifier_sha256': shared.FOUNDATION_VERIFIER_SHA256,
              'foundation_inventory_sha256': shared.FOUNDATION_INVENTORY_SHA256,
              'prepared_files': first['files'],
              'prepared_schema': first['schema']}),
          'actual local processor/source bank preparation differs')
    processor = AutoProcessor.from_pretrained(str(prepared),
        local_files_only=True, trust_remote_code=False)
    active_interfaces = joint_source._interface_pins()
    feature_exporter_sha256 = _file_sha256(features.__file__)
    cases = []
    j6_materials = []
    for row in program['cases']:
        if j6_program:
            material15, source_descriptor = read_j6_material15(material_root,
                row['material15'], j5_source=j5_source)
            j6_materials.append(material15)
            _need(material15['case_id'] == row['case_id'] and
                  material15['split_identity']['variant_key'] == row['variant_key'],
                  'J6 physical material/case/variant differs')
            frozen = _material_json(material_root, row['frozen'])[1]
            _need(frozen['source_file_sha256'] ==
                  source_descriptor['source_file_sha256'] and
                  frozen['source']['record_sha256'] ==
                  source_descriptor['source_record_sha256'] and
                  frozen['split_row']['lineage']['world'] ==
                  material15['split_identity']['world_id'] and
                  frozen['cut_sha256'] ==
                  material15['split_identity']['cut_sha256'],
                  'J6 material15/source/paired real-world join differs')
            accepted = _material_json(material_root, row['accepted'])[1]
            pair = _material_json(material_root, row['pair'])[1]
            lowered = lower_j6_original_case(frozen=frozen,
                accepted=accepted, pair=pair,
                source_descriptor=source_descriptor,
                migration_reader=lambda: migration_raw,
                j5_source=j5_source, processor=processor,
                max_target_tokens=row['max_target_tokens'],
                material15_file_sha256=row['material15']['sha256'])
        else:
            frozen = _record_ref(material_root, row['frozen'])
            accepted = _record_ref(material_root, row['accepted'])
            pair = _record_ref(material_root, row['pair'])
            original_profile = ('j4-p1' if row['profile'] == 'j4-p1' else 'j5-p2')
            expected = _expected_original(frozen,
                j5_source=j5_source if original_profile == 'j5-p2' else None)
            lowered = lower_original_case(profile=original_profile, frozen=frozen,
                accepted=accepted, pair=pair, expected=expected,
                processor=processor, max_target_tokens=row['max_target_tokens'],
                private_paired=private['paired'] if original_profile == 'j5-p2' else None,
                private_target=private['target'] if original_profile == 'j5-p2' else None)
        lowered['variant_key'] = row['variant_key']
        _need(lowered['case_id'] == row['case_id'],
              'original case ID differs from frozen roster')
        joined = join_frozen_source(lowered, feature_root=feature_root,
            feature_roster_sha256=feature_roster_sha256,
            j5_source=j5_source if lowered['profile'] != 'j4-p1' else None,
            allow_fixture=False)
        _need(canonical_sha256(joined['feature_receipt']['code']['processor_binding']) ==
              canonical_sha256(source.processor_binding(processor)),
              'actual loaded processor differs from frozen feature bank')
        _need(joined['source_states'].dtype == torch.bfloat16 and
              joined['source_states'].shape[-1] == roster['base']['hidden_width'],
              'preflight whole source precision/width differs')
        descriptor = {'lowered': lowered,
                      'feature_receipt': deepcopy(joined['feature_receipt'])}
        cases.append({'frozen': frozen, 'accepted': accepted,
                      'pair': pair, 'joined': descriptor})
        del joined
    if j6_program:
        _need(len({item['metadata']['split_roster']['file_sha256']
                   for item in j6_materials}) == 1 and
              len({item['metadata']['code_expectations']['file_sha256']
                   for item in j6_materials}) == 1 and
              len({item['split_identity']['world_id']
                   for item in j6_materials}) == 1 and
              len({item['split_identity']['cut_sha256']
                   for item in j6_materials}) == 6 and
              [item['case_id'] for item in j6_materials] == recipe['case_order'],
              'J6 six-cut complete material train ancestry/roster differs')
    admission, audit = preflight_admitted_cases(
        admission_manifest_path=admission_manifest_path,
        admission_manifest_sha256=admission_manifest_sha256,
        owner_authorization_ref=owner_authorization_ref, cases=cases)
    features._verified_prepared(first, receipt_path, prepared,
        prepared_receipt_file_sha256, prepared_receipt_sha256,
        parent_source_receipt_sha256)
    if j6_program:
        _, current_migration = _material_bytes(material_root,
            program['source_migration'])
        _need(current_migration == migration_raw,
              'J6 six-cut source migration changed during owned fit')
    config = _json_file(seed_config_path, seed_config_sha256)
    model = FormationJointLearning(SpecialistConfig(**config))
    _need(model.config.base_hidden_size == roster['base']['hidden_width'] and
          model.config.max_target_tokens >= max(
              len(row['joined']['lowered']['alignment']['labels']) for row in cases) and
          recipe['generation_max_new_tokens'] <= model.config.max_target_tokens,
          'seed/source/whole-target geometry differs')
    model.load_component(seed_weight_path, seed_config_path,
        seed_manifest_path, weight_sha256=seed_weight_sha256,
        config_sha256=seed_config_sha256,
        manifest_sha256=seed_manifest_sha256,
        expected_provenance_sha256=seed_provenance_sha256,
        expected_source_pins=seed_source_pins,
        expected_evidence_scope='untrained_mechanics')
    device = torch.device(recipe['device'])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
        lr=recipe['learning_rate'], weight_decay=recipe['weight_decay'],
        betas=tuple(recipe['adam_betas']), eps=recipe['adam_eps'])
    history = []
    start_update = 0
    consumed_cases = 0
    if resume_checkpoint is not None:
        checkpoint = Path(resume_checkpoint).resolve(strict=True)
        _need(checkpoint.parent == output and output.is_dir(),
              'restart checkpoint must be inside this fit output')
        state = _restore_checkpoint(checkpoint, model=model,
            optimizer=optimizer, recipe_sha256=recipe_file_sha256,
            program_sha256=program_file_sha256,
            admission_sha256=admission_manifest_sha256,
            feature_roster_sha256=feature_roster_sha256,
            expected_checkpoint_file_sha256=resume_checkpoint_file_sha256,
            case_order=recipe['case_order'],
            accumulation_cases=recipe['accumulation_cases'])
        start_update = state['update_index']
        consumed_cases = state['consumed_cases']
        history = deepcopy(state['history'])
        _need(start_update < recipe['update_budget'],
              'restart has no remaining owned updates')
    else:
        _need(resume_checkpoint_file_sha256 is None and not output.exists(),
              'new fit output must be absent and not claim restart')
        output.mkdir()
        torch.manual_seed(recipe['seed'])
    for update_index in range(start_update + 1, recipe['update_budget'] + 1):
        window = [cases[(consumed_cases + offset) % len(cases)]
                  for offset in range(recipe['accumulation_cases'])]
        observations = _fit_update(model, processor, cases, window, optimizer,
            device=device, feature_root=feature_root,
            feature_roster_sha256=feature_roster_sha256, j5_source=j5_source)
        consumed_cases += recipe['accumulation_cases']
        history.append({'update_index': update_index,
            'next_case_index': consumed_cases % len(cases),
            'completed_passes': consumed_cases // len(cases),
            'cases': observations})
        if (update_index % recipe['checkpoint_every_updates'] == 0 or
                update_index == recipe['update_budget']):
            _save_checkpoint(output / f'checkpoint-update-{update_index:06d}',
                model=model, optimizer=optimizer, update_index=update_index,
                consumed_cases=consumed_cases, case_order=recipe['case_order'],
                recipe_sha256=recipe_file_sha256,
                program_sha256=program_file_sha256,
                admission_sha256=admission_manifest_sha256,
                feature_roster_sha256=feature_roster_sha256,
                history=history)
    features._verified_prepared(first, receipt_path, prepared,
        prepared_receipt_file_sha256, prepared_receipt_sha256,
        parent_source_receipt_sha256)
    if j6_program:
        _, current_migration = _material_bytes(material_root,
            program['source_migration'])
        _need(current_migration == migration_raw,
              'J6 six-cut source migration changed during owned fit')
    _need(canonical_sha256(_json_file(program_path, program_file_sha256)) ==
          canonical_sha256(program) and
          _file_sha256(seed_weight_path) == seed_weight_sha256 and
          _file_sha256(seed_config_path) == seed_config_sha256 and
          _file_sha256(seed_manifest_path) == seed_manifest_sha256,
          'original program or zero-update seed changed during owned fit')
    for authored, original_case in zip(program['cases'], cases):
        reader = (_material_json if j6_program else
                  lambda root, ref: (None, _record_ref(root, ref)))
        _need(all(canonical_sha256(reader(material_root, authored[key])[1]) ==
                  canonical_sha256(original_case[value])
                  for key, value in (('frozen', 'frozen'),
                                     ('accepted', 'accepted'), ('pair', 'pair'))),
              'physical original frozen/target/pair packet changed during fit')
        if j6_program:
            material15, descriptor = read_j6_material15(material_root,
                authored['material15'], j5_source=j5_source)
            _need(material15['case_id'] == original_case['joined']['lowered']['case_id'] and
                  canonical_sha256(descriptor) == canonical_sha256(
                      original_case['joined']['lowered']['source_descriptor']),
                  'physical J6 fifteen-ref material changed during fit')
    final_admission = admit_formation_corpus(admission_manifest_path,
        expected_sha256=admission_manifest_sha256, teacher_training=True,
        owner_authorization_ref=owner_authorization_ref)
    _need([row.case_id for row in final_admission.train] ==
          [row.case_id for row in admission.train],
          'current admitted training originals changed during owned fit')
    _need(_file_sha256(__file__) == recipe['trainer_file_sha256'] and
          _file_sha256(features.__file__) == feature_exporter_sha256 and
          canonical_sha256(joint_source._interface_pins()) ==
          canonical_sha256(active_interfaces) and
          (private is None or all(_file_sha256(private[role].__file__) ==
              program['private_original_modules'][role]['sha256']
              for role in ('source', 'paired', 'target'))) and
          all(row['joined']['feature_receipt']['code']['exporter_sha256'] ==
              feature_exporter_sha256 for row in cases),
          'active trainer/source/consumer interface changed during owned fit')
    for row in cases:
        lowered = row['joined']['lowered']
        receipt, tensors = features.read_case(feature_root,
            roster_sha256=feature_roster_sha256,
            case_id=lowered['case_id'],
            expected_alignment_sha256=lowered['source_alignment_sha256'],
            expected_cache=lowered['source_cache'],
            j5_source=j5_source if lowered['profile'] != 'j4-p1' else None,
            expected_j6_source=lowered.get('source_descriptor'),
            allow_fixture=False)
        _need(canonical_sha256(receipt) ==
              canonical_sha256(row['joined']['feature_receipt']),
              'physical frozen source bank changed during owned fit')
        del tensors
    provenance = {'stage': 'native-joint-owned-fit',
        'seed_manifest_sha256': seed_manifest_sha256,
        'program_file_sha256': program_file_sha256,
        'admission_manifest_sha256': admission_manifest_sha256,
        'feature_roster_sha256': feature_roster_sha256,
        'recipe_file_sha256': recipe_file_sha256,
        'trainer_file_sha256': recipe['trainer_file_sha256'],
        'private_original_modules': deepcopy(program.get('private_original_modules')),
        'runtime': deepcopy(recipe['runtime']),
        'prepared_receipt_file_sha256': prepared_receipt_file_sha256,
        'prepared_receipt_sha256': prepared_receipt_sha256,
        'completed_updates': recipe['update_budget'],
        'consumed_cases': consumed_cases,
        'qualification': 'none'}
    trained_source_pins = {**deepcopy(seed_source_pins),
        'native_joint_trainer_file_sha256': recipe['trainer_file_sha256'],
        'native_joint_trainer_runtime': deepcopy(recipe['runtime']),
        'native_joint_private_original_modules':
            deepcopy(program.get('private_original_modules', {}))}
    component_paths = tuple(output / name for name in
        ('component.safetensors', 'config.json', 'component.json'))
    model.export_component(*component_paths,
        component_id='native-joint-unqualified', provenance=provenance,
        provenance_sha256=canonical_sha256(provenance),
        source_pins=trained_source_pins,
        source_pins_sha256=canonical_sha256(trained_source_pins),
        evidence_scope='unqualified_training')
    reloaded = FormationJointLearning(model.config)
    reloaded.load_component(*component_paths,
        weight_sha256=_file_sha256(component_paths[0]),
        config_sha256=_file_sha256(component_paths[1]),
        manifest_sha256=_file_sha256(component_paths[2]),
        expected_provenance_sha256=canonical_sha256(provenance),
        expected_source_pins=trained_source_pins,
        expected_evidence_scope='unqualified_training')
    first_case = cases[0]
    first_lower = first_case['joined']['lowered']
    reloaded.to(device).eval()
    first_source = join_frozen_source(first_lower, feature_root=feature_root,
        feature_roster_sha256=feature_roster_sha256,
        j5_source=j5_source if first_lower['profile'] != 'j4-p1' else None,
        allow_fixture=False)
    consumer = reloaded.generate_original_copy_decision(
        processor=processor,
        base_states=first_source['source_states'].to(device).detach(),
        source_mask=first_source['source_mask'].to(device).detach(),
        caller=(first_case['frozen']['source']['caller'] if j6_program else
                first_case['frozen']['caller']),
        caller_sha256=(first_case['frozen']['source']['caller']['record_sha256']
                       if j6_program else
                       first_case['frozen']['caller']['record_sha256']),
        joint_alignment=(first_case['frozen']['source']['alignment'] if j6_program
                         else first_case['frozen']['joint']),
        joint_alignment_sha256=first_lower['source_alignment_sha256'],
        expected_binding=copies.copy_binding(first_lower['inventory']),
        artifact_sha256=_file_sha256(component_paths[0]),
        inference_run_id='native-joint-fit-readback',
        max_new_tokens=recipe['generation_max_new_tokens'])
    bundle = consumer['bundle']
    consumer_record = {key: value for key, value in consumer.items()
                       if key != 'bundle'}
    consumer_record['bundle'] = None if bundle is None else {
        'output_record': bundle.output_record(),
        'context_v16_digest': bundle.context_v16_digest,
        'base_metadata_record': bundle.base.metadata_record(),
        'content_digest': bundle.content_digest()}
    (output / 'generated-consumer.json').write_bytes(
        canonical_json_bytes(consumer_record))
    return {'component_paths': tuple(str(path) for path in component_paths),
            'history': history, 'admission_audit': audit,
            'corpus_id': admission.corpus_id,
            'generated_consumer': consumer_record,
            'evidence_scope': 'unqualified_training'}
