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
BROADER_PROGRAM_SCHEMA = 'mfm-native-joint-broader-fit-program-v3'
FULL_SOURCE_PROFILES = ('j6-firstparty-successor','broader-firstparty-successor')
SOURCE_FEATURE_PROFILES = {'j6-firstparty-successor':'j6-source-only',
                           'broader-firstparty-successor':'broader-source-only'}
BROADER_SUCCESSOR_AUTHORING_SCHEMA = 'mfm-native-joint-broader-model-author-conversion-v1'
RECIPE_SCHEMA = 'mfm-native-joint-fit-recipe-v1'
METADATA_RECIPE_SCHEMA = 'mfm-native-joint-fit-recipe-v2'
SUCCESSOR_RECIPE_SCHEMA = 'mfm-native-joint-fit-recipe-v3'
SUCCESSOR_CONTRACT_SCHEMA = 'mfm-native-joint-successor-contract-v1'
DEFERRED_GENERATION_POLICY = 'defer-all-case-native-assessment-v1'
METADATA_CONSUMER_PROFILE = 'frozen-bank-tokenizer-metadata-v1'
METADATA_VERIFIER_FILE_SHA256 = '7848c5f2b12db7d08fc1ff41df5680d9b7d25b4101e6dad54e6eb4c558e0c7d5'
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
    return _lower_full_source_case(frozen,accepted,pair,source_descriptor,processor,
        max_target_tokens,material15_file_sha256,profile='j6-firstparty-successor',example=example)


def _lower_full_source_case(frozen,accepted,pair,source_descriptor,processor,
                            max_target_tokens,material15_file_sha256,*,profile,example):
    actual_processor = source.processor_binding(processor)
    _need(canonical_sha256(actual_processor) ==
          canonical_sha256(frozen['processor_binding']) and
          canonical_sha256(pair['processor_binding']) ==
          canonical_sha256(actual_processor),
          'J6 actual processor/source/target binding differs')
    inventory,view,joint,world_id=_full_source_view(frozen,example,pair)
    alignment = mixed.align_original_copy_decision_target(processor, view,
        view_sha256=view['record_sha256'],
        max_target_tokens=max_target_tokens)
    _need(canonical_sha256(source.processor_binding(processor)) ==
          canonical_sha256(actual_processor),
          'J6 target processor changed during mixed-wire lowering')
    return {'profile': profile,
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


def _full_source_view(frozen,example,pair):
    """Pure shared caller/reference/whole-copy lowering, before learned loads."""
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
    return inventory,view,joint,world_id


def lower_broader_original_case(*,frozen,accepted,pair,source_descriptor,
        observation_reader,native_reader,j5_source,processor,max_target_tokens,
        material15_file_sha256):
    """Validate the distinct observed profile, then reuse the original math."""
    from scripts.mfm import formation_broader_paired_teaching as broad
    from scripts.mfm import formation_broader_target_alignment as target
    from scripts.mfm import formation_joint_input_v6 as j6
    require_sha256(material15_file_sha256,'broader physical material15 FILE')
    _need(source_descriptor['profile']=='broader-source-only' and
      source_descriptor['case_id']==frozen['split_row']['case_id'] and
      source_descriptor['source_record_sha256']==frozen['source']['record_sha256'] and
      source_descriptor['observation_file_sha256']==frozen['source_observation_file_sha256'] and
      source_descriptor['observation_record_sha256']==frozen['source_observation_record_sha256'] and
      source_descriptor['whole_world_file_sha256']==frozen['whole_world_file_sha256'] and
      source_descriptor['whole_world_sha256']==frozen['whole_world_sha256'] and
      type(max_target_tokens) is int and max_target_tokens==pair['max_target_tokens'] and
      frozen['split_row']['split']=='train','broader physical source/material/target budget differs')
    broad.validate_input(j5_source,j6,frozen,expected_input_sha256=frozen['record_sha256'],
      expected_source_file_sha256=source_descriptor['source_file_sha256'],
      observation_reader=observation_reader,native_reader=native_reader)
    example,_=broad.require_complete_labels(frozen,accepted)
    target.validate_pair(pair,frozen,accepted,expected_pair_sha256=pair['record_sha256'])
    lowered=_lower_full_source_case(frozen,accepted,pair,source_descriptor,processor,
      max_target_tokens,material15_file_sha256,profile='broader-firstparty-successor',example=example)
    lowered.update(source_observation_file_sha256=frozen['source_observation_file_sha256'],
      source_observation_record_sha256=frozen['source_observation_record_sha256'],
      whole_world_sha256=frozen['whole_world_sha256'],whole_world_file_sha256=frozen['whole_world_file_sha256'])
    return lowered


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
    j6 = lowered['profile'] in FULL_SOURCE_PROFILES
    fields = ('input_ids', 'attention_mask', 'mm_token_type_ids') if j6 else (
        'input_ids', 'attention_mask')
    _need(receipt['profile'] == (SOURCE_FEATURE_PROFILES[lowered['profile']] if j6 else lowered['profile']) and
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


def preflight_admitted_cases(*,admission_manifest_path,admission_manifest_sha256,
                             owner_authorization_ref,cases):
    """Exact author admission plus independent real frozen feature checks."""
    admission,audit=preflight_admitted_author_cases(admission_manifest_path=admission_manifest_path,
      admission_manifest_sha256=admission_manifest_sha256,
      owner_authorization_ref=owner_authorization_ref,cases=cases)
    for row in cases:
        frozen=row['frozen'];joined=row['joined'];lowered=joined['lowered'];case_id=lowered['case_id']
        _need(joined['feature_receipt']['case_id'] == case_id and
              (lowered['profile'] not in FULL_SOURCE_PROFILES or
               joined['feature_receipt']['profile'] == SOURCE_FEATURE_PROFILES[lowered['profile']]) and
              joined['feature_receipt']['base']['fixture_small_width'] is False and
              joined['feature_receipt']['target_visible_to_backbone'] is False,
              'production frozen source bank/target isolation differs')
        if lowered['profile'] in FULL_SOURCE_PROFILES:
            _need(canonical_sha256(frozen['native_batch'])==
                  canonical_sha256(joined['feature_receipt']['frozen_batch']),
                  'real frozen native source bank differs from admitted author source')
    return admission,audit


def preflight_admitted_author_cases(*, admission_manifest_path, admission_manifest_sha256,
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
                  'original_target_file_sha256' if lowered['profile'] in
                  FULL_SOURCE_PROFILES else 'target_file_sha256'] and
              frozen['split_row']['split'] == 'train',
              'admitted train case and original lowerer differ')
        seen.add(case_id)
        _need(lowered['profile'] in ('j4-p1', 'j5-p2',
                                      'j6-firstparty-successor','broader-firstparty-successor'),
              'unknown original source profile')
        caller = (frozen['source']['caller'] if lowered['profile'] in
                  FULL_SOURCE_PROFILES else frozen['caller'])
        context, opened, _ = j4_input.caller_api.caller_input_from_record(caller)
        expected_sources = sorted(sha256(raw).hexdigest() for _, raw in opened)
        admitted = admitted_by_id[case_id]
        _need(expected_sources == sorted(ref.sha256 for ref in admitted.source_payloads)
              and admitted.target_payload.sha256 == lowered['original_target_file_sha256'],
              'independently admitted physical source/target bytes differ')
        admitted_row = source_rows[case_id]
        _need(type(admitted_row) is dict and
              admitted_row['author_id'] == admitted_row['target_provenance']['producer_id'] and
              admitted_row['host_family'] == context.base.scope.host_instance_id,
              'admitted original host/author and conversion producer differ')
        if lowered['profile'] in FULL_SOURCE_PROFILES:
            _validate_model_author_originals(accepted,lowered['profile'])
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
                lowered['profile'] not in FULL_SOURCE_PROFILES else None,
            'literal_declarations_sha256':
                canonical_sha256(pair['literal_declarations']) if
                lowered['profile'] not in FULL_SOURCE_PROFILES else None,
            'authored_ordinals_sha256':
                canonical_sha256(lowered['authored_ordinals']),
            'authored_copy_tuples_sha256':
                canonical_sha256(lowered['authored_copy_tuples']),
            'world_sha256': lowered['world_sha256'],
            'cut_sha256': lowered['cut_sha256'],
            'variant_key': lowered['variant_key']}
        if lowered['profile'] == 'j5-p2':
            expected_conversion['original_geometry_profile'] = 'j5-p2'
        if lowered['profile'] in FULL_SOURCE_PROFILES:
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
                      lowered['source_file_sha256'],
                  'J6 model-author provenance/source-native admission differs')
            expected_conversion.update({
                'schema': (BROADER_SUCCESSOR_AUTHORING_SCHEMA if lowered['profile']=='broader-firstparty-successor' else J6_SUCCESSOR_AUTHORING_SCHEMA),
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
        if lowered['profile']=='broader-firstparty-successor':
            expected_conversion.update(source_observation_file_sha256=lowered['source_observation_file_sha256'],
              source_observation_record_sha256=lowered['source_observation_record_sha256'],
              whole_world_sha256=lowered['whole_world_sha256'],whole_world_file_sha256=lowered['whole_world_file_sha256'],
              native_target_observation_file_sha256=accepted['native_geometry_file_sha256'])
        _need(canonical_sha256(conversion) ==
              canonical_sha256(expected_conversion),
              'admitted physical author conversion does not bind complete pair/choices')
    _need(seen == set(admitted_by_id), 'incomplete admitted train roster')
    j6_worlds = {row['joined']['lowered']['mass_world_id'] for row in cases
                 if row['joined']['lowered']['profile'] in FULL_SOURCE_PROFILES}
    _need(not j6_worlds or
          (len(j6_worlds) == 1 and all(row['joined']['lowered']['profile'] in
              FULL_SOURCE_PROFILES for row in cases)),
          'J6 admitted originals must share one connected train world')
    audit = _audit_admitted_payloads(admission)
    return admission, audit


def _validate_model_author_originals(accepted,profile):
    """Bind training provenance to embedded original bytes, never declarations."""
    from scripts.mfm import formation_paired_teaching_v6 as original
    _need(profile in FULL_SOURCE_PROFILES,'physical model-author profile differs')
    draft_raw=bytes.fromhex(accepted['raw_author_draft_hex'])
    support_raw=bytes.fromhex(accepted['raw_support_hex'])
    annotation_raw=bytes.fromhex(accepted['raw_annotation_hex'])
    _need(sha256(draft_raw).hexdigest()==accepted['raw_author_draft_file_sha256'] and
          sha256(support_raw).hexdigest()==accepted['support_file_sha256'] and
          sha256(annotation_raw).hexdigest()==accepted['annotation_file_sha256'],
          'physical model-author draft/support/annotation FILE differs')
    draft=original.original_json(draft_raw)
    support=original.original_json(support_raw)
    annotation=original.original_json(annotation_raw)
    view_pin=accepted['projected_view_file_sha256']
    if profile=='broader-firstparty-successor':
        descriptors=accepted['original_material_descriptors']
        raws=accepted['original_material_hex']
        _need(draft_raw==bytes.fromhex(raws['case']) and
              support_raw==bytes.fromhex(raws['support']) and
              all(sha256(bytes.fromhex(raws[key])).hexdigest()==descriptors[key]['file_sha256']
                  for key in ('case','support','source_view')) and
              accepted['raw_author_draft_file_sha256']==descriptors['case']['file_sha256'] and
              accepted['support_file_sha256']==descriptors['support']['file_sha256'] and
              view_pin==descriptors['source_view']['file_sha256'] and
              draft['lineage']['current_source_view_file_sha256']==view_pin and
              support['source_view_file_sha256']==view_pin and
              accepted['author_id']==draft['lineage']['target_author'] and
              accepted['author_producer_version']==draft['lineage']['producer_family'],
              'physical model-author broader input/output/support/producer differs')
    else:
        _need(draft['lineage']['current_cut_view_file_sha256']==view_pin and
              support['current_cut_view_file_sha256']==view_pin and
              annotation['view_file_sha256']==view_pin and
              annotation['draft_file_sha256']==accepted['raw_author_draft_file_sha256'] and
              accepted['author_id']==draft['status']['author_id'] and
              accepted['author_producer_version']==draft['status']['producer_version'],
              'physical model-author J6 input/output/support/producer differs')


def _audit_admitted_payloads(admission):
    """Audit each physical original once; chronological case mass is separate."""
    def paths(cases):
        return tuple(dict.fromkeys(path for row in cases
            for path in (*(ref.path for ref in row.source_payloads),
                         row.target_payload.path)))
    return admission.audit_handoff(gradient_paths=paths(admission.train),
                                    development_paths=paths(admission.development))


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


def read_j6_material15(material_root,manifest_ref,*,j5_source):
    return _read_full_source_material15(material_root,manifest_ref,j5_source=j5_source,broader=False)


def read_broader_material15(material_root,manifest_ref,*,j5_source):
    return _read_full_source_material15(material_root,manifest_ref,j5_source=j5_source,broader=True)


def validate_broader_material_roster(root,connected,program_cases,materials,actual_cases):
    """Join every material/native prefix to the exact connected whole world/cut."""
    from scripts.mfm import formation_broader_paired_teaching as broad
    result=broad.validate_connected_roster(root,connected,
      expected_roster_sha256=connected['record_sha256'],program_cases=program_cases)
    _need(len(materials)==len(actual_cases)==9,'broader complete nine material/source joins required')
    for index,(roster,material,actual) in enumerate(zip(connected['cases'],materials,actual_cases,strict=True)):
        _,world=_material_json(root,material['metadata']['world'])
        _,cut=_material_json(root,material['metadata']['cut'])
        _,expected_world=broad.read_json_ref(root,connected['legacy_world' if index<6 else 'successor_world'])
        _,expected_cut=broad.read_json_ref(root,roster['cut'])
        _,view=_material_json(root,material['metadata']['view'])
        frozen=actual['frozen'];source=frozen['source']
        proof=source['meaning']['episode_snapshot']['historical_prefix_proof']
        _need(world==expected_world and cut==expected_cut and
          source['meaning']['task_snapshot']['historical_prefix_proof']==proof and
          view['historical_prefix_proof']==proof and proof['cut']==cut and
          source['cut_sha256']==cut['record_sha256']==material['split_identity']['cut_sha256'] and
          material['case_id']==roster['case_id']==frozen['split_row']['case_id'] and
          actual['joined']['lowered']['profile']==roster['profile'] and
          material['split_identity']['world_id']==result['world_id'],
          'broader material/native source differs from exact connected original world/cut')
        rows=world['branches'][0]['records'][:cut['sequence']]
        prefix={k:deepcopy(v) for k,v in world.items() if k not in ('branches','world_sha256')}
        prefix['branches']=[{'branch_id':cut['branch_id'],'parent_cut':None,'records':rows}]
        prefix=broad.world_replay.C.seal(prefix,'world_sha256')
        _need(proof['world']==prefix and source['world_sha256']==prefix['world_sha256'],
          'broader material source proof is not its exact selected original prefix')
        if index>=6:
            _need(frozen['whole_world_sha256']==world['world_sha256'],
                  'broader paired whole-world ancestry differs')
    for subset in (materials[:6],materials[6:]):
        _need(len({m['metadata']['split_roster']['file_sha256'] for m in subset})==1 and
              len({m['metadata']['code_expectations']['file_sha256'] for m in subset})==1,
              'broader legacy/new metadata subsets require their exact separate rosters/code expectations')
    return result


def _bracket_broader_program(root,program,connected,observation_raw):
    _,current=_material_json(root,program['connected_roster'])
    _,raw=_material_bytes(root,program['source_observation'])
    _need(current==connected and raw==observation_raw,
          'broader exact connected roster/observation changed during owned fit')


def preflight_broader_author_metadata(*,material_root,program,connected_record,
        j5_source,migration_raw,observation_raw,
        admission_manifest_path,admission_manifest_sha256,owner_authorization_ref):
    """Validate exact nine-case native author conversion before heavy readers.

    No feature receipt is synthesized or read; real feature checks remain
    in their original post-load path. No processor, states, tokenizer, model
    or weights are loaded. Existing corpus authority is rechecked.
    """
    from scripts.mfm import formation_joint_input_v6 as j6
    from scripts.mfm import formation_paired_teaching_v6 as legacy
    from scripts.mfm import formation_joint_target_alignment_v6 as legacy_target
    from scripts.mfm import formation_broader_paired_teaching as broad
    from scripts.mfm import formation_broader_target_alignment as broad_target
    root=Path(material_root).resolve(strict=True)
    cases=[];materials=[]
    for row in program['cases']:
        newer=row['profile']=='broader-firstparty-successor'
        reader=read_broader_material15 if newer else read_j6_material15
        material,descriptor=reader(root,row['material15'],j5_source=j5_source)
        materials.append(material)
        frozen=_material_json(root,row['frozen'])[1]
        _need(material['case_id']==row['case_id']==frozen['split_row']['case_id'] and
          material['split_identity']['variant_key']==row['variant_key'] and
          frozen['source_file_sha256']==descriptor['source_file_sha256'] and
          frozen['source']['record_sha256']==descriptor['source_record_sha256'] and
          frozen['cut_sha256']==material['split_identity']['cut_sha256'] and
          frozen['split_row']['lineage']['world']==material['split_identity']['world_id'],
          'broader early physical source/material/case/variant differs')
        if newer:
            _need(descriptor['observation_file_sha256']==sha256(observation_raw).hexdigest(),
                  'broader early exact program observation differs')
            native_root=Path(descriptor['observation_file']).parent
            broad.validate_input(j5_source,j6,frozen,expected_input_sha256=frozen['record_sha256'],
              expected_source_file_sha256=descriptor['source_file_sha256'],
              observation_reader=lambda:observation_raw,native_reader=lambda r:(native_root/r['path']).read_bytes())
        else:
            legacy.validate_input(j5_source,j6,frozen,expected_input_sha256=frozen['record_sha256'],
              expected_source_file_sha256=descriptor['source_file_sha256'],migration_reader=lambda:migration_raw)
        # No teacher FileRef is opened until full source/material admission.
        accepted=_material_json(root,row['accepted'])[1];pair=_material_json(root,row['pair'])[1]
        example,_=(broad if newer else legacy).require_complete_labels(frozen,accepted)
        (broad_target if newer else legacy_target).validate_pair(pair,frozen,accepted,
          expected_pair_sha256=pair['record_sha256'])
        _need(type(row['max_target_tokens']) is int and row['max_target_tokens']==pair['max_target_tokens'],
              'broader early original target budget differs')
        inventory,view,joint,world_id=_full_source_view(frozen,example,pair)
        lowered={'profile':row['profile'],'case_id':row['case_id'],'variant_key':row['variant_key'],
          'original_frozen_sha256':frozen['record_sha256'],'original_target_file_sha256':accepted['original_target_file_sha256'],
          'original_pair_sha256':pair['record_sha256'],'world_sha256':frozen['world_sha256'],
          'cut_sha256':frozen['cut_sha256'],'mass_world_id':world_id,
          'material15_file_sha256':row['material15'].get('sha256',row['material15'].get('file_sha256')),
          'source_file_sha256':frozen['source_file_sha256'],'source_alignment_sha256':joint['record_sha256'],
          'source_cache':deepcopy(joint['cache']),'source_descriptor':descriptor,
          'authored_ordinals':view['authored_ordinals'],'authored_copy_tuples':view['authored_copy_tuples']}
        if newer:
            lowered.update(source_observation_file_sha256=frozen['source_observation_file_sha256'],
              source_observation_record_sha256=frozen['source_observation_record_sha256'],
              whole_world_sha256=frozen['whole_world_sha256'],whole_world_file_sha256=frozen['whole_world_file_sha256'])
        cases.append({'frozen':frozen,'accepted':accepted,'pair':pair,
          'joined':{'lowered':lowered}})
    validate_broader_material_roster(root,connected_record,program['cases'],materials,cases)
    preflight_admitted_author_cases(admission_manifest_path=admission_manifest_path,
      admission_manifest_sha256=admission_manifest_sha256,
      owner_authorization_ref=owner_authorization_ref,cases=cases)
    return cases,materials


def _read_full_source_material15(material_root, manifest_ref, *, j5_source,broader):
    """Read source and all original material refs; never open author labels."""
    from scripts.mfm import formation_joint_input_v6 as j6
    from scripts.mfm import formation_paired_teaching_v6 as j6_paired

    root = Path(material_root).resolve(strict=True)
    _, case = _material_json(root, manifest_ref)
    _need(case.get('schema') == ('mfm-broader-material15-source-manifest-v1' if broader else
                                'mfm-j6-material15-source-manifest-v1') and
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
    observation_path=observation_record=None
    if broader:
        from scripts.mfm import formation_broader_paired_teaching as broad
        from scripts.mfm import formation_broader_target_alignment as broad_target
        observation_path,observation_raw=_material_bytes(root,case['source_observation'])
        observation_pin=sha256(observation_raw).hexdigest()
        observation_record=broad.observation(observation_raw,observation_pin)
        entries=[r for r in observation_record['source_files'] if r['cut_id']==case['cut_id']]
        _need(len(entries)==1 and entries[0]['file_sha256']==case['source_freeze']['file_sha256'] and
              entries[0]['record_sha256']==frozen['record_sha256'] and
              entries[0]['source_tokens']==frozen['alignment']['source_tokens'] and
              observation_record['proposed_max_source_tokens']==frozen['proposed_max_source_tokens'],
              'broader exact observation/source roster differs')
        def native_reader(ref):
            path=(observation_path.parent/ref['path']).resolve(strict=True)
            _need(path.is_relative_to(observation_path.parent) and '\\' not in ref['path'] and
                  not Path(ref['path']).is_absolute() and all(p not in ('','.','..') for p in ref['path'].split('/')),
                  'broader observed native path differs')
            return path.read_bytes()
        broad._native_originals(frozen,entries[0],native_reader)
        world_path,world_raw=_material_bytes(root,case['metadata']['world'])
        broad.whole_world_prefix(world_raw,sha256(world_raw).hexdigest(),frozen)
        _need(code['current_j6_source_file_sha256']==_file_sha256(j6.__file__) and
              code['paired_source_file_sha256']==_file_sha256(broad.__file__) and
              code['target_geometry_file_sha256']==_file_sha256(broad_target.__file__) and
              code['original_j5_source_file_sha256']==_file_sha256(j5_source.__file__) and
              code['source_observation_file_sha256']==observation_pin,
              'broader actual source/target/observation code interface differs')
    else:
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
    descriptor = {'case_id': case['case_id'], 'profile': 'broader-source-only' if broader else 'j6-source-only',
        'source_file': str(source_path),
        'source_file_sha256': case['source_freeze']['file_sha256'],
        'source_record_sha256': frozen['record_sha256'],
        'j5_file_sha256': _file_sha256(j5_source.__file__),
        'proposed_max_source_tokens': frozen['proposed_max_source_tokens']}
    if broader:
        descriptor.update(observation_file=str(observation_path),
          observation_file_sha256=observation_pin,observation_record_sha256=observation_record['record_sha256'],
          whole_world_file=str(world_path),whole_world_file_sha256=sha256(world_raw).hexdigest(),
          whole_world_sha256=data['world']['world_sha256'])
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


def _consumer_recipe(record, profile):
    """Keep the relocated processor role explicit in immutable fit/resume input."""
    _need(type(record) is dict and profile in (None, METADATA_CONSUMER_PROFILE),
          'explicit prepared consumer profile differs')
    if profile is None:
        _need(record.get('schema') in (RECIPE_SCHEMA, SUCCESSOR_RECIPE_SCHEMA) and
              'prepared_consumer' not in record,
              'complete-clone default requires original recipe v1')
        return None
    custody = record.get('prepared_consumer')
    _need(record.get('schema') in (METADATA_RECIPE_SCHEMA, SUCCESSOR_RECIPE_SCHEMA) and
          type(custody) is dict and set(custody) == {
              'profile', 'metadata_verifier_file_sha256', 'asset_roster_sha256'} and
          custody['profile'] == profile and
          custody['metadata_verifier_file_sha256'] == METADATA_VERIFIER_FILE_SHA256,
          'explicit metadata consumer recipe or verifier differs')
    require_sha256(custody['asset_roster_sha256'], 'processor asset roster')
    return custody


def _verified_fit_prepared(prepared, receipt_path, root,
                          expected_receipt_file_sha256, expected_receipt_sha256,
                          expected_parent_source_sha256, *, roster, recipe,
                          prepared_consumer_profile=None, prior_metadata=None):
    """Verify the selected consumer custody without relabelling producer proof."""
    from scripts.mfm import export_joint_source_features as features
    from scripts.mfm import train_v1_formation_specialist as shared

    custody = _consumer_recipe(recipe, prepared_consumer_profile)
    if custody is None:
        _need(prior_metadata is None, 'complete-clone default cannot inherit metadata custody')
        return features._verified_prepared(prepared, receipt_path, root,
            expected_receipt_file_sha256, expected_receipt_sha256,
            expected_parent_source_sha256), None
    verifier_path = Path(__file__).with_name('train_public_conditional_foundation.py')
    _need(_file_sha256(verifier_path) == custody['metadata_verifier_file_sha256'],
          'published tokenizer metadata verifier FILE differs')
    from scripts.mfm import train_public_conditional_foundation as public

    receipt_path = Path(receipt_path)
    public._safe_root(receipt_path.parent)
    receipt_path = public._safe_child(receipt_path.parent, receipt_path.name)
    for value in (expected_receipt_file_sha256, expected_receipt_sha256,
                  expected_parent_source_sha256):
        require_sha256(value, 'metadata original clone pin')
    _need(_file_sha256(receipt_path) == expected_receipt_file_sha256,
          'metadata original receipt physical FILE differs')
    row = json.loads(receipt_path.read_bytes())
    _need(type(row) is dict and row.get('schema') == shared.CLONE_SCHEMA and
          row.get('role') == 'mfm' and row.get('qualification') in (None, 'unqualified') and
          row.get('receipt_sha256') == expected_receipt_sha256 and
          canonical_sha256({k: v for k, v in row.items() if k != 'receipt_sha256'}) ==
              expected_receipt_sha256 and
          row.get('parent_source_receipt_sha256') == expected_parent_source_sha256 and
          type(row.get('snapshot_path')) is str and bool(row['snapshot_path']),
          'metadata original MFM receipt or source ancestry differs')
    files = row.get('files')
    names = {*public.METADATA_NAMES, 'model.safetensors'}
    _need(type(files) is list and len(files) == len(names) and
          all(type(item) is dict and set(item) == {'path', 'size', 'sha256'} and
              type(item['path']) is str and item['path'] in names and
              type(item['size']) is int and item['size'] > 0 for item in files) and
          {item['path'] for item in files} == names,
          'metadata original complete prepared FILE roster differs')
    for item in files:
        require_sha256(item['sha256'], 'original prepared FILE')
    by_name = {item['path']: item for item in files}
    base = roster.get('base') if type(roster) is dict else None
    binding = {'receipt_sha256': expected_receipt_sha256,
        'receipt_file_sha256': expected_receipt_file_sha256,
        'parent_source_receipt_sha256': expected_parent_source_sha256,
        'source_weight_sha256': shared.SOURCE_WEIGHT_SHA256,
        'foundation_commit': shared.FOUNDATION_COMMIT,
        'foundation_verifier_sha256': shared.FOUNDATION_VERIFIER_SHA256,
        'foundation_inventory_sha256': shared.FOUNDATION_INVENTORY_SHA256,
        'prepared_files': files, 'prepared_schema': row['schema']}
    _need(by_name['model.safetensors']['sha256'] == shared.SOURCE_WEIGHT_SHA256 and
          type(base) is dict and base.get('fixture_small_width') is False and
          base.get('hidden_width') == 3840 and
          base.get('representation') == 'base.model.last_hidden_state' and
          base.get('weight_file_sha256') == by_name['model.safetensors']['sha256'] and
          base.get('config_file_sha256') == by_name['config.json']['sha256'] and
          canonical_sha256(base.get('prepared_base')) == canonical_sha256(binding),
          'frozen bank producer and metadata clone ancestry differ')
    metadata = public.verify_tokenizer_metadata(root, receipt_path,
        expected_receipt_sha256,
        {'prepared_clone_receipt_sha256': expected_receipt_sha256,
         'prepared_base_weight_sha256': shared.SOURCE_WEIGHT_SHA256})
    _need(metadata['asset_roster_sha256'] == custody['asset_roster_sha256'],
          'consumer processor asset roster differs from immutable recipe')
    evidence = {'schema': 'mfm-native-joint-tokenizer-metadata-custody-v1',
        'profile': prepared_consumer_profile,
        'metadata_verifier_file_sha256': custody['metadata_verifier_file_sha256'],
        'receipt_file_sha256': expected_receipt_file_sha256,
        'receipt_sha256': expected_receipt_sha256,
        'parent_source_receipt_sha256': expected_parent_source_sha256,
        'feature_roster_sha256': recipe['feature_roster_sha256'],
        'producer_prepared_base_sha256': canonical_sha256(binding),
        'asset_roster_sha256': metadata['asset_roster_sha256'],
        'assets': metadata['assets'], 'base_weights_opened': False,
        'complete_clone_weight_verification_performed_here': False,
        'meaning': metadata['meaning']}
    evidence['record_sha256'] = canonical_sha256(evidence)
    _need((prepared is None) == (prior_metadata is None) and
          (prepared is None or
           (canonical_sha256(row) == canonical_sha256(prepared) and
            canonical_sha256(evidence) == canonical_sha256(prior_metadata))),
          'consumer metadata custody changed across owned-fit boundary')
    return row, evidence


def _bind_processor_custody(metadata, processor):
    """Preserve sealed metadata and seal its complete actual processor binding."""
    _need(type(metadata) is dict and
          metadata.get('schema') == 'mfm-native-joint-tokenizer-metadata-custody-v1' and
          metadata.get('profile') == METADATA_CONSUMER_PROFILE and
          metadata.get('base_weights_opened') is False and
          metadata.get('complete_clone_weight_verification_performed_here') is False and
          metadata.get('record_sha256') == canonical_sha256(
              {k: v for k, v in metadata.items() if k != 'record_sha256'}),
          'verified inner processor metadata evidence differs')
    record = {'schema': 'mfm-native-joint-processor-custody-v1',
              'metadata': deepcopy(metadata),
              'processor_binding': deepcopy(source.processor_binding(processor))}
    record['record_sha256'] = canonical_sha256(record)
    return record


def _recipe(record, *, program_sha256, admission_sha256,
            feature_roster_sha256, seed_manifest_sha256,
            prepared_consumer_profile=None,
            successor_contract_file_sha256=None, expected_trainer_file_sha256=None):
    custody = _consumer_recipe(record, prepared_consumer_profile)
    successor = successor_contract_file_sha256 is not None
    if successor:
        require_sha256(successor_contract_file_sha256, 'successor contract FILE')
        _need(record.get('successor_contract_file_sha256') == successor_contract_file_sha256 and
              record.get('generation_policy') == DEFERRED_GENERATION_POLICY,
              'explicit successor recipe/assessment policy differs')
    _need(type(record) is dict and set(record) == {
        'schema', 'program_sha256', 'admission_sha256',
        'feature_roster_sha256', 'seed_manifest_sha256', 'case_order',
        'update_budget', 'accumulation_cases',
        'learning_rate', 'weight_decay', 'adam_betas', 'adam_eps',
        'device', 'seed', 'checkpoint_every_updates', 'precision',
        'term_weights', 'mass_policy', 'update_policy', 'runtime',
        'trainer_file_sha256', 'generation_max_new_tokens'} |
        ({'prepared_consumer'} if custody is not None else set()) |
        ({'successor_contract_file_sha256', 'generation_policy'} if successor else set()) and
        record['schema'] == (SUCCESSOR_RECIPE_SCHEMA if successor else
            METADATA_RECIPE_SCHEMA if custody is not None else RECIPE_SCHEMA) and
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
        record['trainer_file_sha256'] == (expected_trainer_file_sha256
            if expected_trainer_file_sha256 is not None else _file_sha256(__file__)),
        'immutable actual joint objective/schedule recipe differs')


def _read_only_mount(path):
    """Production successors consume a read-only filesystem, without a write probe."""
    return (hasattr(os, 'statvfs') and
            bool(os.statvfs(path).f_flag & getattr(os, 'ST_RDONLY', 1)))


def _successor_parent_path(value, output):
    path = Path(value)
    _need(path.is_absolute() and not any(p.is_symlink() or
          getattr(p, 'is_junction', lambda: False)() for p in (path, *path.parents)),
          'successor parent path alias refused')
    path = path.resolve(strict=True)
    _need(not path.is_relative_to(output) and not output.is_relative_to(path) and
          _read_only_mount(path), 'successor parent must be external and read-only')
    return path


def _bracket_successor(successor):
    record = successor['record']
    _need(_file_sha256(successor['path']) == successor['file_sha256'] and
          _file_sha256(record['parent_recipe']['path']) == record['parent_recipe']['sha256'] and
          all(_file_sha256(successor['checkpoint'] / name) == pin
              for name, pin in record['parent_checkpoint']['files'].items()),
          'successor parent checkpoint FILE or recipe changed')


def _successor_contract(path, expected_file_sha256, *, recipe, output,
                        prepared_consumer_profile):
    """Bind an immutable exhausted parent; no tensors or live state are loaded."""
    record = _json_file(path, expected_file_sha256)
    _need(set(record) == {'schema', 'parent_recipe', 'parent_checkpoint',
          'parent_trainer_file_sha256', 'parent_update_index', 'parent_consumed_cases',
          'parent_history_sha256', 'assessment', 'record_sha256'} and
          record['schema'] == SUCCESSOR_CONTRACT_SCHEMA and
          record['record_sha256'] == canonical_sha256(
              {k: v for k, v in record.items() if k != 'record_sha256'}),
          'sealed successor contract differs')
    _recipe(recipe, program_sha256=recipe['program_sha256'],
        admission_sha256=recipe['admission_sha256'], feature_roster_sha256=recipe['feature_roster_sha256'],
        seed_manifest_sha256=recipe['seed_manifest_sha256'],
        prepared_consumer_profile=prepared_consumer_profile,
        successor_contract_file_sha256=expected_file_sha256)
    parent_ref, checkpoint_ref = record['parent_recipe'], record['parent_checkpoint']
    _need(type(parent_ref) is dict and set(parent_ref) == {'path', 'sha256'} and
          type(checkpoint_ref) is dict and set(checkpoint_ref) == {'path', 'files'} and
          type(checkpoint_ref['files']) is dict and set(checkpoint_ref['files']) ==
              {'checkpoint.json', 'weights.safetensors', 'trainer.pt'},
          'successor complete parent references differ')
    for pin in (parent_ref['sha256'], record['parent_trainer_file_sha256'],
                record['parent_history_sha256'], *checkpoint_ref['files'].values()):
        require_sha256(pin, 'successor parent FILE/history')
    output = Path(output).resolve(strict=False)
    parent_path = _successor_parent_path(parent_ref['path'], output)
    checkpoint = _successor_parent_path(checkpoint_ref['path'], output)
    _need(parent_path.is_file() and checkpoint.is_dir() and
          {p.name for p in checkpoint.iterdir()} == set(checkpoint_ref['files']) and
          all((checkpoint / name).is_file() and not (checkpoint / name).is_symlink()
              for name in checkpoint_ref['files']), 'successor parent file roster differs')
    parent = _json_file(parent_path, parent_ref['sha256'])
    _recipe(parent, program_sha256=recipe['program_sha256'],
        admission_sha256=recipe['admission_sha256'], feature_roster_sha256=recipe['feature_roster_sha256'],
        seed_manifest_sha256=recipe['seed_manifest_sha256'],
        prepared_consumer_profile=prepared_consumer_profile,
        successor_contract_file_sha256=(parent.get('successor_contract_file_sha256')
            if parent.get('schema') == SUCCESSOR_RECIPE_SCHEMA else None),
        expected_trainer_file_sha256=record['parent_trainer_file_sha256'])
    mutable = {'schema', 'trainer_file_sha256', 'update_budget', 'checkpoint_every_updates',
               'successor_contract_file_sha256', 'generation_policy'}
    _need(canonical_sha256({k:v for k,v in parent.items() if k not in mutable}) ==
          canonical_sha256({k:v for k,v in recipe.items() if k not in mutable}) and
          recipe['accumulation_cases'] == len(recipe['case_order']),
          'successor must preserve the complete parent objective/optimizer/material schedule')
    successor = {'record': record, 'file_sha256': expected_file_sha256,
        'path': Path(path), 'checkpoint': checkpoint, 'parent_recipe': parent}
    _bracket_successor(successor)
    metadata = _json_file(checkpoint / 'checkpoint.json', checkpoint_ref['files']['checkpoint.json'])
    _need(metadata.get('schema') == CHECKPOINT_SCHEMA and
          (metadata['recipe_sha256'], metadata['program_sha256'],
           metadata['admission_sha256'], metadata['feature_roster_sha256']) ==
              (parent_ref['sha256'], recipe['program_sha256'],
               recipe['admission_sha256'], recipe['feature_roster_sha256']) and
          metadata['case_order_sha256'] == canonical_sha256(recipe['case_order']) and
          metadata['weight_file_sha256'] == checkpoint_ref['files']['weights.safetensors'] and
          metadata['trainer_file_sha256'] == checkpoint_ref['files']['trainer.pt'] and
          type(record['parent_update_index']) is int and
          record['parent_update_index'] == metadata['update_index'] == parent['update_budget'] and
          type(record['parent_consumed_cases']) is int and
          record['parent_consumed_cases'] == metadata['consumed_cases'] ==
              record['parent_update_index'] * recipe['accumulation_cases'] and
          metadata['next_case_index'] == 0 and
          recipe['update_budget'] > record['parent_update_index'] and
          canonical_sha256(metadata['history']) == record['parent_history_sha256'] and
          len(metadata['history']) == record['parent_update_index'],
          'successor exhausted parent cursor/history differs')
    assessment = record['assessment']
    _need(type(assessment) is dict and set(assessment) == {'policy', 'case_order_sha256',
          'feature_roster_sha256', 'seed_manifest_sha256', 'interface_pins_sha256', 'max_new_tokens'} and
          assessment['policy'] == 'separate-immutable-all-case-native-v1' and
          assessment['case_order_sha256'] == canonical_sha256(recipe['case_order']) and
          assessment['feature_roster_sha256'] == recipe['feature_roster_sha256'] and
          assessment['seed_manifest_sha256'] == recipe['seed_manifest_sha256'] and
          type(assessment['max_new_tokens']) is int and
          assessment['max_new_tokens'] == recipe['generation_max_new_tokens'],
          'successor immutable complete native assessment requirement differs')
    require_sha256(assessment['interface_pins_sha256'], 'assessment native interfaces')
    successor['metadata'] = metadata
    return successor


def _restore_successor(successor, *, model, optimizer, recipe, recipe_sha256,
                       output, resume_checkpoint=None, resume_checkpoint_file_sha256=None):
    """Validate the parent prefix before invoking the unchanged strict restorer."""
    _bracket_successor(successor)
    output = Path(output).resolve(strict=False)
    if resume_checkpoint is None:
        _need(not output.exists() and resume_checkpoint_file_sha256 is None,
              'new successor output must be absent')
        checkpoint = successor['checkpoint']
        expected = successor['record']['parent_checkpoint']['files']['checkpoint.json']
        restore_recipe = successor['record']['parent_recipe']['sha256']
        metadata = successor['metadata']
    else:
        checkpoint = Path(resume_checkpoint).resolve(strict=True)
        _need(output.is_dir() and checkpoint.parent == output,
              'successor restart checkpoint must be inside its output')
        expected = resume_checkpoint_file_sha256
        metadata = _json_file(checkpoint / 'checkpoint.json', expected)
        restore_recipe = recipe_sha256
    prefix = successor['record']['parent_update_index']
    _need(type(metadata.get('history')) is list and
          canonical_sha256(metadata['history'][:prefix]) == successor['record']['parent_history_sha256'],
          'successor parent history prefix differs')
    _need(type(metadata.get('update_index')) is int and
          prefix <= metadata['update_index'] < recipe['update_budget'],
          'successor restart has no remaining owned updates')
    return _restore_checkpoint(checkpoint, model=model, optimizer=optimizer,
        recipe_sha256=restore_recipe, program_sha256=recipe['program_sha256'],
        admission_sha256=recipe['admission_sha256'], feature_roster_sha256=recipe['feature_roster_sha256'],
        expected_checkpoint_file_sha256=expected, case_order=recipe['case_order'],
        accumulation_cases=recipe['accumulation_cases'])


def _builtin_generation_required(successor):
    return successor is None


def _restore_fit_checkpoint(*, model, optimizer, recipe, recipe_sha256, output,
                            resume_checkpoint=None, resume_checkpoint_file_sha256=None,
                            successor=None):
    if successor is not None:
        return _restore_successor(successor, model=model, optimizer=optimizer,
            recipe=recipe, recipe_sha256=recipe_sha256, output=output,
            resume_checkpoint=resume_checkpoint,
            resume_checkpoint_file_sha256=resume_checkpoint_file_sha256)
    checkpoint = Path(resume_checkpoint).resolve(strict=True)
    _need(checkpoint.parent == output and output.is_dir(),
          'restart checkpoint must be inside this fit output')
    state = _restore_checkpoint(checkpoint, model=model,
        optimizer=optimizer, recipe_sha256=recipe_sha256,
        program_sha256=recipe['program_sha256'],
        admission_sha256=recipe['admission_sha256'],
        feature_roster_sha256=recipe['feature_roster_sha256'],
        expected_checkpoint_file_sha256=resume_checkpoint_file_sha256,
        case_order=recipe['case_order'],
        accumulation_cases=recipe['accumulation_cases'])
    _need(state['update_index'] < recipe['update_budget'],
          'restart has no remaining owned updates')
    return state


def _checkpoint_due(update_index, recipe, successor):
    origin = 0 if successor is None else successor['record']['parent_update_index']
    return ((update_index - origin) % recipe['checkpoint_every_updates'] == 0 or
            update_index == recipe['update_budget'])


def _deferred_assessment(output, *, successor, recipe_sha256, component_paths):
    record = {'schema': 'mfm-native-joint-separate-assessment-pending-v1',
        'fit_completed': True, 'assessment_completed': False,
        'builtin_generation_performed': False, 'qualification': 'none',
        'successor_contract_file_sha256': successor['file_sha256'],
        'parent_checkpoint_file_sha256': successor['record']['parent_checkpoint']['files']['checkpoint.json'],
        'recipe_file_sha256': recipe_sha256,
        'components': {Path(p).name: _file_sha256(p) for p in component_paths},
        'requirement': deepcopy(successor['record']['assessment'])}
    record['record_sha256'] = canonical_sha256(record)
    with (Path(output) / 'assessment-pending.json').open('xb') as stream:
        stream.write(canonical_json_bytes(record))
    return record


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
                           resume_checkpoint_file_sha256=None,
                           prepared_consumer_profile=None,
                           successor_contract_path=None,
                           successor_contract_file_sha256=None):
    """Fit the new joint stage from exact original FileRefs and frozen source.

    No candidate is qualified here. Paired construction records never grant
    training authority; every supplied program requires fresh physical corpus
    admission, exact author/source joins and a production frozen feature bank.
    An explicit processor metadata role preserves the original producer custody.
    """
    from transformers import AutoProcessor
    from cognitive_kernel.formation_v1_specialist import SpecialistConfig
    from cognitive_kernel.formation_joint_learning import FormationJointLearning
    from cognitive_kernel import formation_joint_learning as joint_source
    from scripts.mfm import export_joint_source_features as features
    from scripts.mfm import train_v1_formation_specialist as shared

    material_root = Path(material_root).resolve(strict=True)
    feature_root = Path(feature_root).resolve(strict=True)
    if prepared_consumer_profile is not None:
        for original in (Path(prepared_base_dir), Path(prepared_receipt_path)):
            _need(not any(p.is_symlink() or getattr(p, 'is_junction', lambda: False)()
                          for p in (original, *original.parents)),
                  'metadata consumer original path links are forbidden')
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
    broader_program=program.get('schema')==BROADER_PROGRAM_SCHEMA
    full_program=j6_program or broader_program
    _need(not broader_program or set(program)=={'schema','cases','private_original_modules',
      'source_migration','source_observation','connected_roster'},
      'broader nine-case program requires exact source observation/connected roster')
    _need(not j6_program or set(program)=={'schema','cases','private_original_modules','source_migration'},
      'original six-case program schema differs')
    _need(set(program) in ({'schema', 'cases'},
                           {'schema', 'cases', 'private_original_modules'},
                           {'schema', 'cases', 'private_original_modules',
                            'source_migration'},
                           {'schema','cases','private_original_modules','source_migration',
                            'source_observation','connected_roster'}) and
          program['schema'] in (PROGRAM_SCHEMA, J6_PROGRAM_SCHEMA,BROADER_PROGRAM_SCHEMA) and
          type(program['cases']) is list and program['cases'],
          'complete original training program differs')
    common_case_keys = {'case_id', 'profile', 'frozen', 'accepted', 'pair',
                        'max_target_tokens', 'variant_key'}
    _need(all(type(row) is dict and set(row) ==
              (common_case_keys | {'material15'} if full_program else common_case_keys) and
              type(row['case_id']) is str and
              require_identifier(row['case_id'], 'case_id') == row['case_id'] and
              type(row['variant_key']) is str and
              require_identifier(row['variant_key'], 'variant_key') ==
              row['variant_key'] and
              type(row['max_target_tokens']) is int and
              row['max_target_tokens'] > 0 for row in program['cases']),
          'original train case FileRef program differs')
    _need(all(row['profile'] in FULL_SOURCE_PROFILES
              for row in program['cases']) if broader_program else
          all(row['profile'] == 'j6-firstparty-successor'
              for row in program['cases']) if j6_program else
          all(row['profile'] in ('j4-p1', 'j5-p2-successor')
              for row in program['cases']),
          'supplied P2 is not independently admitted successor material')
    _need(not j6_program or (len(program['cases']) == 6 and
          len({row['case_id'] for row in program['cases']}) == 6),
          'complete six-cut J6 program required')
    uses_j5 = any(row['profile'] in ('j5-p2-successor', *FULL_SOURCE_PROFILES)
                  for row in program['cases'])
    _need(('private_original_modules' in program) == uses_j5,
          'J5 successor requires exact original module FILE roster')
    _need(('source_migration' in program) == full_program,
          'J6 requires one pinned six-cut source migration FILE')
    private = (_private_original_modules(program['private_original_modules'])
               if uses_j5 else None)
    j5_source = None if private is None else private['source']
    migration_raw = None
    if full_program:
        from scripts.mfm import formation_paired_teaching_v6 as j6_paired
        _, migration_raw = _material_bytes(material_root,
            program['source_migration'])
        _need(sha256(migration_raw).hexdigest() ==
              j6_paired.SOURCE_MIGRATION_FILE,
              'J6 actual six-cut source migration differs')
        connected=observation_raw=None
        if broader_program:
            from scripts.mfm import formation_broader_paired_teaching as broad
            _,connected_record=_material_json(material_root,program['connected_roster'])
            connected=broad.validate_connected_roster(material_root,connected_record,
              expected_roster_sha256=connected_record['record_sha256'],program_cases=program['cases'])
            _,observation_raw=_material_bytes(material_root,program['source_observation'])
            observation_record=broad.observation(observation_raw,sha256(observation_raw).hexdigest())
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
    _need((successor_contract_path is None) ==
          (successor_contract_file_sha256 is None),
          'successor contract path and external FILE pin are required together')
    _recipe(recipe, program_sha256=program_file_sha256,
            admission_sha256=admission_manifest_sha256,
            feature_roster_sha256=feature_roster_sha256,
            seed_manifest_sha256=seed_manifest_sha256,
            prepared_consumer_profile=prepared_consumer_profile,
            successor_contract_file_sha256=successor_contract_file_sha256)
    successor = (None if successor_contract_path is None else _successor_contract(
        successor_contract_path, successor_contract_file_sha256, recipe=recipe,
        output=output, prepared_consumer_profile=prepared_consumer_profile))
    if successor is not None and resume_checkpoint is None:
        _need(not output.exists() and resume_checkpoint_file_sha256 is None,
              'new successor output must be absent')
    _need(recipe['case_order'] == [row['case_id'] for row in program['cases']],
          'actual case order differs from immutable fit recipe')
    # Corpus authority precedes processor and any feature tensor/model load.
    # Constructed J6 records are fixed false and never grant this admission.
    if not full_program:
        admit_formation_corpus(admission_manifest_path,
            expected_sha256=admission_manifest_sha256, teacher_training=True,
            owner_authorization_ref=owner_authorization_ref)
    if broader_program:
        # Exact source/author/native conversion is a metadata boundary. False
        # lineage must refuse before tokenizer, tensor, prepared or seed load.
        preflight_broader_author_metadata(material_root=material_root,program=program,
          connected_record=connected_record,j5_source=j5_source,migration_raw=migration_raw,
          observation_raw=observation_raw,admission_manifest_path=admission_manifest_path,
          admission_manifest_sha256=admission_manifest_sha256,owner_authorization_ref=owner_authorization_ref)
    roster = features._read_record(feature_root / 'roster.json',
                                   features.SCHEMA, feature_roster_sha256)
    first, consumer_metadata = _verified_fit_prepared(None, receipt_path, prepared,
        prepared_receipt_file_sha256, prepared_receipt_sha256,
        parent_source_receipt_sha256, roster=roster, recipe=recipe,
        prepared_consumer_profile=prepared_consumer_profile)
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
    if successor is not None:
        _need(canonical_sha256(active_interfaces) ==
              successor['record']['assessment']['interface_pins_sha256'],
              'successor assessment native interface closure differs')
    feature_exporter_sha256 = _file_sha256(features.__file__)
    cases = []
    j6_materials = []
    for row in program['cases']:
        if full_program:
            is_broader=row['profile']=='broader-firstparty-successor'
            material_reader=read_broader_material15 if is_broader else read_j6_material15
            material15, source_descriptor = material_reader(material_root,
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
            if is_broader:
                _need(source_descriptor['observation_file_sha256']==sha256(observation_raw).hexdigest(),
                      'broader material observation differs from exact program source receipt')
                native_root=Path(source_descriptor['observation_file']).parent
                lowered=lower_broader_original_case(frozen=frozen,accepted=accepted,pair=pair,
                  source_descriptor=source_descriptor,observation_reader=lambda:observation_raw,
                  native_reader=lambda ref:(native_root/ref['path']).read_bytes(),
                  j5_source=j5_source,processor=processor,max_target_tokens=row['max_target_tokens'],
                  material15_file_sha256=row['material15'].get('sha256',row['material15'].get('file_sha256')))
            else:
                lowered = lower_j6_original_case(frozen=frozen,accepted=accepted,pair=pair,
                  source_descriptor=source_descriptor,migration_reader=lambda:migration_raw,
                  j5_source=j5_source,processor=processor,max_target_tokens=row['max_target_tokens'],
                  material15_file_sha256=row['material15'].get('sha256',row['material15'].get('file_sha256')))
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
    if broader_program:
        validate_broader_material_roster(material_root,connected_record,program['cases'],j6_materials,cases)
    admission, audit = preflight_admitted_cases(
        admission_manifest_path=admission_manifest_path,
        admission_manifest_sha256=admission_manifest_sha256,
        owner_authorization_ref=owner_authorization_ref, cases=cases)
    _verified_fit_prepared(first, receipt_path, prepared,
        prepared_receipt_file_sha256, prepared_receipt_sha256,
        parent_source_receipt_sha256, roster=roster, recipe=recipe,
        prepared_consumer_profile=prepared_consumer_profile,
        prior_metadata=consumer_metadata)
    if full_program:
        _, current_migration = _material_bytes(material_root,
            program['source_migration'])
        if broader_program:
            _bracket_broader_program(material_root,program,connected_record,observation_raw)
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
    if successor is not None or resume_checkpoint is not None:
        state = _restore_fit_checkpoint(model=model, optimizer=optimizer,
            recipe=recipe, recipe_sha256=recipe_file_sha256, output=output,
            resume_checkpoint=resume_checkpoint,
            resume_checkpoint_file_sha256=resume_checkpoint_file_sha256,
            successor=successor)
        start_update = state['update_index']
        consumed_cases = state['consumed_cases']
        history = deepcopy(state['history'])
        if successor is not None and resume_checkpoint is None:
            output.mkdir()
            semantics = {'schema': 'mfm-native-joint-successor-observation-semantics-v1',
                'successor_contract_file_sha256': successor_contract_file_sha256,
                'recipe_file_sha256': recipe_file_sha256,
                'parent_history_sha256': successor['record']['parent_history_sha256'],
                'loss_measurement_phase': 'pre-update',
                'meaning': 'Each case loss precedes its enclosing completed Adam update; history prefix is unchanged.',
                'separate_native_assessment_required': True}
            semantics['record_sha256'] = canonical_sha256(semantics)
            with (output / 'training-observation-semantics.json').open('xb') as stream:
                stream.write(canonical_json_bytes(semantics))
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
        if _checkpoint_due(update_index, recipe, successor):
            _save_checkpoint(output / f'checkpoint-update-{update_index:06d}',
                model=model, optimizer=optimizer, update_index=update_index,
                consumed_cases=consumed_cases, case_order=recipe['case_order'],
                recipe_sha256=recipe_file_sha256,
                program_sha256=program_file_sha256,
                admission_sha256=admission_manifest_sha256,
                feature_roster_sha256=feature_roster_sha256,
                history=history)
    if successor is not None:
        _bracket_successor(successor)
        _need(_file_sha256(recipe_path) == recipe_file_sha256,
              'successor child recipe changed during owned fit')
    _verified_fit_prepared(first, receipt_path, prepared,
        prepared_receipt_file_sha256, prepared_receipt_sha256,
        parent_source_receipt_sha256, roster=roster, recipe=recipe,
        prepared_consumer_profile=prepared_consumer_profile,
        prior_metadata=consumer_metadata)
    if full_program:
        _, current_migration = _material_bytes(material_root,
            program['source_migration'])
        if broader_program:
            _bracket_broader_program(material_root,program,connected_record,observation_raw)
        _need(current_migration == migration_raw,
              'J6 six-cut source migration changed during owned fit')
    _need(canonical_sha256(_json_file(program_path, program_file_sha256)) ==
          canonical_sha256(program) and
          _file_sha256(seed_weight_path) == seed_weight_sha256 and
          _file_sha256(seed_config_path) == seed_config_sha256 and
          _file_sha256(seed_manifest_path) == seed_manifest_sha256,
          'original program or zero-update seed changed during owned fit')
    for authored, original_case in zip(program['cases'], cases):
        reader = (_material_json if full_program else
                  lambda root, ref: (None, _record_ref(root, ref)))
        _need(all(canonical_sha256(reader(material_root, authored[key])[1]) ==
                  canonical_sha256(original_case[value])
                  for key, value in (('frozen', 'frozen'),
                                     ('accepted', 'accepted'), ('pair', 'pair'))),
              'physical original frozen/target/pair packet changed during fit')
        if full_program:
            material_reader=read_broader_material15 if authored['profile']=='broader-firstparty-successor' else read_j6_material15
            material15, descriptor = material_reader(material_root,
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
    if successor is not None:
        lineage = {'contract_file_sha256': successor_contract_file_sha256,
            'contract_record_sha256': successor['record']['record_sha256'],
            'parent_recipe_file_sha256': successor['record']['parent_recipe']['sha256'],
            'parent_trainer_file_sha256': successor['record']['parent_trainer_file_sha256'],
            'parent_checkpoint_files': deepcopy(successor['record']['parent_checkpoint']['files']),
            'parent_update_index': successor['record']['parent_update_index'],
            'additional_completed_updates': recipe['update_budget'] - successor['record']['parent_update_index'],
            'parent_history_sha256': successor['record']['parent_history_sha256'],
            'loss_measurement_phase': 'pre-update',
            'generation_policy': recipe['generation_policy'],
            'required_assessment': deepcopy(successor['record']['assessment'])}
        provenance['successor'] = lineage
        trained_source_pins['native_joint_successor'] = deepcopy(lineage)
    if consumer_metadata is not None:
        custody = _bind_processor_custody(consumer_metadata, processor)
        provenance['processor_custody'] = custody
        trained_source_pins['native_joint_processor_custody'] = custody
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
    if not _builtin_generation_required(successor):
        pending = _deferred_assessment(output, successor=successor,
            recipe_sha256=recipe_file_sha256, component_paths=component_paths)
        return {'component_paths': tuple(str(path) for path in component_paths),
                'history': history, 'admission_audit': audit,
                'corpus_id': admission.corpus_id, 'generated_consumer': None,
                'assessment_pending': pending, 'evidence_scope': 'unqualified_training'}
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
        caller=(first_case['frozen']['source']['caller'] if full_program else
                first_case['frozen']['caller']),
        caller_sha256=(first_case['frozen']['source']['caller']['record_sha256']
                       if full_program else
                       first_case['frozen']['caller']['record_sha256']),
        joint_alignment=(first_case['frozen']['source']['alignment'] if full_program
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
