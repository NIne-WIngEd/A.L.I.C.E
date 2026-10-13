"""Complete J6 source before a separately authored v1.6 original target.

This constructs material and geometry inputs. It never grants training rights,
Gemma capacity, independent gold, or a complete teacher label by default.
"""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import stat

from cognitive_kernel.canonical import (
    CognitiveKernelContractError, canonical_json_bytes, canonical_sha256,
    require_sha256)
from cognitive_kernel import formation_learning_v16 as learning
from . import formation_field_responsibility as policy_api
from . import formation_target_alignment as target_api

INPUT = 'mfm-paired-full-source-j6-input-v1'
TARGET = 'mfm-paired-full-source-j6-original-target-v1'
SAVED = 'mfm-paired-full-source-j6-saved-input-v1'
ANNOTATION_V1 = 'mfm-j6-authored-dimension-and-field-inventory-v1'
ANNOTATION_V2 = 'mfm-j6-authored-dimension-and-field-inventory-v2'
MAXIMUM = 52488  # proposed construction bound, not Gemma capacity
NATIVE = ('input_ids', 'attention_mask', 'mm_token_type_ids')
SOURCE_V6_FILE = '21ad69358dab645aa420cfa48aa07917393a100c3a5d856fe9b4006ac2f760c9'
SOURCE_J5_FILE = 'e1da3b95b96831066066f51dc429b118ba9735717230378c5f037dc7569612c7'
SOURCE_MIGRATION_FILE = '43a2e57e2f2b4f29dca0c4736378bcd65254f604c1629b3b5d494bd394d316f4'
ORIGINAL_OBSERVATION_V6_FILE = '19c63014fb55e0633985db3e555e21cb6c61a1c710a54bd2bbcc42382bedf8de'


def need(ok, message):
    if not ok:
        raise CognitiveKernelContractError(message)


def seal(row):
    return {**deepcopy(row), 'record_sha256': canonical_sha256(row)}


def original_json(raw):
    need(type(raw) is bytes, 'original reader must return bytes')

    def unique(pairs):
        row = {}
        for key, value in pairs:
            need(key not in row, 'duplicate original JSON key')
            row[key] = value
        return row

    try:
        return json.loads(raw.decode('utf-8', 'strict'), object_pairs_hook=unique,
                          parse_constant=lambda _: need(False, 'nonfinite JSON'))
    except (ValueError, UnicodeError) as exc:
        raise CognitiveKernelContractError('invalid original JSON') from exc


def pinned(raw, expected, schema):
    require_sha256(expected, 'external FILE pin')
    need(sha256(raw).hexdigest() == expected, 'original FILE differs')
    row = original_json(raw)
    need(type(row) is dict and row.get('schema') == schema and
         row.get('record_sha256') == canonical_sha256({k: v for k, v in row.items()
             if k != 'record_sha256'}), 'original record seal/schema differs')
    return row


def _source_modules(j5, v6):
    need(j5.__name__ == 'scripts.mfm.formation_joint_input_v5' and
         sha256(Path(j5.__file__).read_bytes()).hexdigest() == SOURCE_J5_FILE and
         sha256(Path(v6.__file__).read_bytes()).hexdigest() == SOURCE_V6_FILE,
         'actual J5/J6 source FILE differs')


def paired_code_binding():
    here = Path(__file__).resolve().parent
    return seal({'schema': 'mfm-paired-full-source-j6-code-binding-v1',
        'paired_source_file_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'target_geometry_file_sha256': sha256((here /
            'formation_joint_target_alignment_v6.py').read_bytes()).hexdigest(),
        'current_j6_source_file_sha256': SOURCE_V6_FILE,
        'original_j5_source_file_sha256': SOURCE_J5_FILE})


def _split(row):
    need(type(row) is dict and set(row) == {'case_id', 'split', 'lineage'} and
         type(row['case_id']) is str and bool(row['case_id']) and
         row['split'] in ('train', 'development', 'final') and
         type(row['lineage']) is dict, 'explicit split identity differs')


def input_core(j5, v6, source, source_file_sha256, migration):
    _source_modules(j5, v6)
    require_sha256(source_file_sha256, 'complete J6 source FILE')
    need(type(source) is dict and
         sha256(canonical_json_bytes(source) + b'\n').hexdigest() == source_file_sha256,
         'nested complete J6 source FILE differs')
    need(migration['current_j6_source_file_sha256'] == SOURCE_V6_FILE and
         migration['original_j6_source_file_sha256'] == ORIGINAL_OBSERVATION_V6_FILE and
         migration['original_j5_file_sha256'] == SOURCE_J5_FILE and
         migration['native_processor_reexecuted'] is False and
         migration['new_native_observation'] is False and
         migration['proposed_max_source_tokens'] == MAXIMUM,
         'source observation/migration ancestry differs')
    refs = [ref for ref in migration['migrated_source_files']
            if ref['path'] == source['cut_id'] + '.json']
    need(len(refs) == 1 and refs[0]['file_sha256'] == source_file_sha256,
         'source FILE absent from complete six-cut migration')
    v6.validate_frozen_source(j5, source, j5_file_sha256=SOURCE_J5_FILE,
                              proposed_max_source_tokens=MAXIMUM)
    alignment = source['alignment']
    need(type(source['proposed_max_source_tokens']) is int and
         source['proposed_max_source_tokens'] == MAXIMUM and
         type(alignment['source_tokens']) is int and
         0 < alignment['source_tokens'] <= MAXIMUM and
         source['native_batch'] == {key: [alignment[key]] for key in NATIVE} and
         source['construction_only'] is True and
         source['capacity_admitted'] is False and
         source['target_accessed'] is False and
         source['training_authorized'] is False and
         source['current_service_authenticated'] is False,
         'whole source/native/authority contract differs')
    return {'paired_code_binding': paired_code_binding(),
            'source': deepcopy(source), 'source_file_sha256': source_file_sha256,
            'source_migration_receipt_file_sha256': SOURCE_MIGRATION_FILE,
            'original_observation_j6_file_sha256': ORIGINAL_OBSERVATION_V6_FILE,
            'current_j6_source_file_sha256': SOURCE_V6_FILE,
            'original_j5_source_file_sha256': SOURCE_J5_FILE,
            'source_cache': deepcopy(alignment['cache']),
            'source_identity': deepcopy(alignment['source_identity']),
            'processor_binding': deepcopy(alignment['processor_binding']),
            'cut_id': source['cut_id'], 'cut_sha256': source['cut_sha256'],
            'world_sha256': source['world_sha256'],
            'native_batch': deepcopy(source['native_batch']),
            'source_tokens': alignment['source_tokens'],
            'proposed_max_source_tokens': MAXIMUM,
            'capacity_admitted': False, 'target_accessed': False,
            'training_authorized': False, 'current_service_authenticated': False}


def freeze_input(j5, v6, source_reader, migration_reader, *,
                 expected_source_file_sha256, split_row):
    """Open and validate the complete source before any target reader exists."""
    _split(split_row)
    migration = pinned(migration_reader(), SOURCE_MIGRATION_FILE,
                       'mfm-j6-six-source-code-binding-migration-v1')
    source_raw = source_reader()
    source = pinned(source_raw, expected_source_file_sha256, v6.FROZEN)
    need(source_raw == canonical_json_bytes(source) + b'\n',
         'noncanonical complete source FILE')
    return seal({'schema': INPUT, 'split_row': deepcopy(split_row),
                 **input_core(j5, v6, source, expected_source_file_sha256,
                              migration)})


def validate_input(j5, v6, row, *, expected_input_sha256,
                   expected_source_file_sha256, migration_reader):
    require_sha256(expected_input_sha256, 'frozen J6 paired input')
    need(type(row) is dict and row.get('schema') == INPUT and
         row.get('record_sha256') == expected_input_sha256 and
         canonical_sha256({k: v for k, v in row.items()
             if k != 'record_sha256'}) == expected_input_sha256,
         'J6 paired input seal differs')
    _split(row['split_row'])
    migration = pinned(migration_reader(), SOURCE_MIGRATION_FILE,
                       'mfm-j6-six-source-code-binding-migration-v1')
    rebuilt = seal({'schema': INPUT, 'split_row': row['split_row'],
        **input_core(j5, v6, row['source'], expected_source_file_sha256,
                     migration)})
    need(canonical_sha256(row) == canonical_sha256(rebuilt),
         'whole J6 paired source/native/cache/cut ancestry differs')
    return row


def _ordinary(path):
    path = Path(path).absolute()
    need('..' not in path.parts, 'saved input traversal refused')
    for index, part in enumerate((path, *path.parents)):
        info = part.lstat()
        need(not stat.S_ISLNK(info.st_mode) and
             not getattr(info, 'st_file_attributes', 0) & 0x400,
             'saved input alias/reparse refused')
        need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 if index == 0
             else stat.S_ISDIR(info.st_mode),
             'saved input path must be ordinary')
    return path


def publish_input(path, j5, v6, frozen, *, expected_source_file_sha256,
                  migration_reader):
    validate_input(j5, v6, frozen,
        expected_input_sha256=frozen['record_sha256'],
        expected_source_file_sha256=expected_source_file_sha256,
        migration_reader=migration_reader)
    raw = canonical_json_bytes(frozen) + b'\n'
    path = Path(path)
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    need(_ordinary(path).read_bytes() == raw, 'saved input physical readback differs')
    return seal({'schema': SAVED, 'input_file_sha256': sha256(raw).hexdigest(),
        'input_bytes': len(raw), 'input_record_sha256': frozen['record_sha256'],
        'source_file_sha256': expected_source_file_sha256,
        'source_migration_receipt_file_sha256': SOURCE_MIGRATION_FILE,
        'paired_code_binding': paired_code_binding(),
        'target_accessed': False, 'training_authorized': False})


def read_input(path, receipt, *, expected_receipt_sha256,
               expected_source_file_sha256, j5, v6, migration_reader):
    require_sha256(expected_receipt_sha256, 'saved input receipt')
    need(type(receipt) is dict and receipt.get('schema') == SAVED and
         receipt.get('record_sha256') == expected_receipt_sha256 and
         canonical_sha256({k: v for k, v in receipt.items()
             if k != 'record_sha256'}) == expected_receipt_sha256 and
         type(receipt.get('input_bytes')) is int and
         receipt['input_bytes'] > 0 and
         receipt['source_file_sha256'] == expected_source_file_sha256 and
         receipt['source_migration_receipt_file_sha256'] ==
             SOURCE_MIGRATION_FILE and
         canonical_sha256(receipt['paired_code_binding']) ==
             canonical_sha256(paired_code_binding()) and
         receipt['target_accessed'] is False and
         receipt['training_authorized'] is False,
         'saved input receipt/schema/source differs')
    path = _ordinary(path)
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) and
         len(raw) == receipt['input_bytes'] and
         sha256(raw).hexdigest() == receipt['input_file_sha256'],
         'saved input FILE changed or differs')
    row = original_json(raw)
    need(raw == canonical_json_bytes(row) + b'\n' and
         row.get('record_sha256') == receipt['input_record_sha256'],
         'saved input original bytes differ')
    return validate_input(j5, v6, row,
        expected_input_sha256=receipt['input_record_sha256'],
        expected_source_file_sha256=expected_source_file_sha256,
        migration_reader=migration_reader)


def _view_and_draft(row, view, draft, support, annotation,
                    view_file_sha256, draft_file_sha256,
                    support_file_sha256, annotation_file_sha256):
    source = row['source']
    need(view['schema'] == 'mfm-j6-current-cut-codex-authoring-view-v2' and
         view['cut_id'] == row['cut_id'] and
         view['cut_sha256'] == row['cut_sha256'] and
         view['world_sha256'] == row['world_sha256'] and
         view['source_freeze_file_sha256'] == row['source_file_sha256'] and
         view['source_freeze_record_sha256'] == source['record_sha256'] and
         view['source_alignment_record_sha256'] ==
             source['alignment']['record_sha256'] and
         view['context'] == source['caller']['context'] and
         view['opened_sources'] == source['caller']['opened_sources'] and
         view['candidate_inventory'] == source['meaning']['candidate_inventory'] and
         view['rendered_native_source_text'] == source['alignment']['rendered_text'] and
         view['source_tokens'] == row['source_tokens'] and
         view['proof_only_bodies_included'] is False and
         view['future_cut_bodies_included'] is False and
         view['target_included'] is False,
         'projected current-cut view/source differs')
    for key in ('episode_snapshot', 'task_snapshot'):
        proof = view[key]['historical_prefix_proof']
        need(set(proof) == {'schema', 'record_sha256', 'cut_sha256',
                'world_sha256', 'artifact_count', 'artifact_roster_sha256'} and
             proof['record_sha256'] ==
                source['meaning'][key]['historical_prefix_proof']['record_sha256'],
             'proof-only raw custody in author view')
    need(draft['case_id'] == row['split_row']['case_id'] and
         draft['split'] == row['split_row']['split'] and
         draft['context'] == view['context'] and
         draft['lineage']['current_cut_view_file_sha256'] == view_file_sha256 and
         draft['lineage']['source_freeze_file_sha256'] == row['source_file_sha256'] and
         draft['lineage']['cut_sha256'] == row['cut_sha256'] and
         draft['lineage']['world_sha256'] == row['world_sha256'] and
         support['current_cut_view_file_sha256'] == view_file_sha256 and
         support['source_freeze_file_sha256'] == row['source_file_sha256'] and
         annotation['view_file_sha256'] == view_file_sha256 and
         annotation['draft_file_sha256'] == draft_file_sha256 and
         support['supervised_eligible'] is False and
         annotation['supervised_eligible'] is False,
         'separate original author/support/annotation ancestry differs')
    need([item['ref_id'] for item in draft['sources']] ==
         [item['ref_id'] for item in view['opened_sources']] and
         all(a['content_b64'] == b['payload_base64'] for a, b in
             zip(draft['sources'], view['opened_sources'])),
         'authored original target source delivery differs')
    example = learning.learning_example_v16_from_record(draft)
    output = example.target.output_record()
    geometry = target_api.canonical_field_geometry(output,
        expected_target_sha256=canonical_sha256(output))
    policy = policy_api.normative_policy()
    authored = (annotation.get('authored_selection_fields', [])
                if annotation['schema'] == ANNOTATION_V2 else [])
    groups = (annotation['semantic_field_annotations'],
              annotation['unresolved_selection_fields'], authored,
              annotation['other_normative_fields'])
    pointers = [item['pointer'] for group in groups for item in group]
    need(annotation['normative_policy_sha256'] == policy['record_sha256'] and
         annotation['canonical_target_sha256'] == canonical_sha256(output) and
         annotation['total_target_nodes'] == len(geometry['nodes']) and
         len(pointers) == len(set(pointers)) == len(geometry['nodes']) and
         set(pointers) == {node['pointer'] for node in geometry['nodes']} and
         set(annotation['dimension_decisions']) ==
             set(learning.FULL_ROLE_DIMENSIONS) and
         all(annotation['dimension_decisions'][name]['state'] == state for
             name, state in draft['target']['adjudications'].items()),
         'authored target normative field/dimension inventory differs')
    return example, output, geometry


def convert_original_target(j5, v6, frozen, *, expected_input_sha256,
                            expected_source_file_sha256, migration_reader,
                            view_reader, expected_view_file_sha256,
                            draft_reader, expected_draft_file_sha256,
                            support_reader, expected_support_file_sha256,
                            annotation_reader, expected_annotation_file_sha256):
    """Open author material only after validating the complete frozen source."""
    validate_input(j5, v6, frozen, expected_input_sha256=expected_input_sha256,
        expected_source_file_sha256=expected_source_file_sha256,
        migration_reader=migration_reader)
    need(frozen['split_row']['split'] in ('train', 'development'),
         'final target remains unopened')
    view_raw = view_reader()
    view = pinned(view_raw, expected_view_file_sha256,
                  'mfm-j6-current-cut-codex-authoring-view-v2')
    draft_raw = draft_reader()
    require_sha256(expected_draft_file_sha256, 'original authored draft FILE')
    need(sha256(draft_raw).hexdigest() == expected_draft_file_sha256,
         'original authored draft FILE differs')
    draft = original_json(draft_raw)
    support_raw = support_reader()
    support = pinned(support_raw, expected_support_file_sha256,
                     'mfm-j6-firstparty-target-draft-support-v1')
    annotation_raw = annotation_reader()
    annotation_schema = original_json(annotation_raw).get('schema')
    need(annotation_schema in (ANNOTATION_V1, ANNOTATION_V2),
         'unsupported authored annotation version')
    annotation = pinned(annotation_raw, expected_annotation_file_sha256,
                        annotation_schema)
    example, output, _ = _view_and_draft(frozen, view, draft, support,
        annotation, expected_view_file_sha256, expected_draft_file_sha256,
        expected_support_file_sha256, expected_annotation_file_sha256)
    original = {'schema': learning.TARGET_SCHEMA_V16,
        'context': deepcopy(draft['context']),
        'source_ids': [item['ref_id'] for item in draft['sources']],
        'proposals': deepcopy(draft['target']['proposals']),
        'dispositions': deepcopy(draft['target']['dispositions']),
        'adjudications': deepcopy(draft['target']['adjudications'])}
    need(set(original) == {'schema', 'context', 'source_ids', 'proposals',
         'dispositions', 'adjudications'}, 'standalone original target shape')
    unknown = [name for name, state in example.adjudications if state == 'unknown']
    unresolved = deepcopy(annotation['unresolved_selection_fields'])
    copy_complete = annotation['whole_copy_alternatives_authored'] is True
    original_bytes = canonical_json_bytes(original) + b'\n'
    return seal({'schema': TARGET,
        'input_freeze_sha256': frozen['record_sha256'],
        'original_target_hex': original_bytes.hex(),
        'original_target_file_sha256': sha256(original_bytes).hexdigest(),
        'raw_author_draft_hex': draft_raw.hex(),
        'raw_author_draft_file_sha256': expected_draft_file_sha256,
        'raw_support_hex': support_raw.hex(),
        'support_file_sha256': expected_support_file_sha256,
        'raw_annotation_hex': annotation_raw.hex(),
        'annotation_file_sha256': expected_annotation_file_sha256,
        'projected_view_file_sha256': expected_view_file_sha256,
        'canonical_output_sha256': canonical_sha256(output),
        'unknown_dimensions': unknown,
        'unresolved_selection_fields': unresolved,
        'whole_copy_alternatives_authored': copy_complete,
        'source_cache': deepcopy(frozen['source_cache']),
        'state': 'incomplete' if unknown or unresolved or not copy_complete else 'complete',
        'author_origin': draft['status']['origin'],
        'author_id': draft['status']['author_id'],
        'author_producer_version': draft['status']['producer_version'],
        'author_model_reinvoked_for_projection': False,
        'independent_gold': False, 'authenticated_service': False,
        'training_authorized': False})


def convert_saved_target(path, receipt, *, expected_receipt_sha256,
                         expected_source_file_sha256, j5, v6,
                         migration_reader, view_reader,
                         expected_view_file_sha256, draft_reader,
                         expected_draft_file_sha256, support_reader,
                         expected_support_file_sha256, annotation_reader,
                         expected_annotation_file_sha256):
    """Physical source readback completes before opening author material."""
    frozen = read_input(path, receipt,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_source_file_sha256=expected_source_file_sha256,
        j5=j5, v6=v6, migration_reader=migration_reader)
    return convert_original_target(j5, v6, frozen,
        expected_input_sha256=frozen['record_sha256'],
        expected_source_file_sha256=expected_source_file_sha256,
        migration_reader=migration_reader,
        view_reader=view_reader,
        expected_view_file_sha256=expected_view_file_sha256,
        draft_reader=draft_reader,
        expected_draft_file_sha256=expected_draft_file_sha256,
        support_reader=support_reader,
        expected_support_file_sha256=expected_support_file_sha256,
        annotation_reader=annotation_reader,
        expected_annotation_file_sha256=expected_annotation_file_sha256)


def require_complete_labels(frozen, target):
    """No unknown-to-negative, ordinal guessing, or null copy labels."""
    need(type(target) is dict and target.get('schema') == TARGET and
         target.get('record_sha256') == canonical_sha256({k: v for k, v in
             target.items() if k != 'record_sha256'}) and
         target['input_freeze_sha256'] == frozen['record_sha256'] and
         target['source_cache'] == frozen['source_cache'] and
         target['state'] == 'complete' and
         not target['unknown_dimensions'] and
         not target['unresolved_selection_fields'] and
         target['whole_copy_alternatives_authored'] is True,
         'incomplete J6 author choices cannot produce labels')
    original = original_json(bytes.fromhex(target['original_target_hex']))
    need(set(original) == {'schema', 'context', 'source_ids', 'proposals',
         'dispositions', 'adjudications'} and
         sha256(bytes.fromhex(target['original_target_hex'])).hexdigest() ==
             target['original_target_file_sha256'],
         'standalone original target FILE differs')
    source = frozen['source']
    context, opened, _ = source['caller']['context'], source['caller']['opened_sources'], None
    need(original['context'] == context and
         original['source_ids'] == [item['ref_id'] for item in opened],
         'standalone original source/context differs')
    draft = original_json(bytes.fromhex(target['raw_author_draft_hex']))
    need(draft['target'] == {k: original[k] for k in
         ('schema', 'proposals', 'dispositions', 'adjudications')} and
         sha256(bytes.fromhex(target['raw_author_draft_hex'])).hexdigest() ==
             target['raw_author_draft_file_sha256'],
         'raw authored target differs')
    support_raw = bytes.fromhex(target['raw_support_hex'])
    annotation_raw = bytes.fromhex(target['raw_annotation_hex'])
    support = pinned(support_raw, target['support_file_sha256'],
                     'mfm-j6-firstparty-target-draft-support-v1')
    annotation_schema = original_json(annotation_raw).get('schema')
    need(annotation_schema in (ANNOTATION_V1, ANNOTATION_V2),
         'author annotation version differs')
    annotation = pinned(annotation_raw, target['annotation_file_sha256'],
                        annotation_schema)
    need(support['current_cut_view_file_sha256'] ==
             target['projected_view_file_sha256'] and
         support['source_freeze_file_sha256'] == frozen['source_file_sha256'] and
         annotation['view_file_sha256'] ==
             target['projected_view_file_sha256'] and
         annotation['draft_file_sha256'] ==
             target['raw_author_draft_file_sha256'] and
         annotation['canonical_target_sha256'] ==
             target['canonical_output_sha256'] and
         set(annotation['dimension_decisions']) ==
             set(learning.FULL_ROLE_DIMENSIONS) and
         all(annotation['dimension_decisions'][name]['state'] == state
             for name, state in original['adjudications'].items()) and
         annotation['unresolved_selection_fields'] ==
             target['unresolved_selection_fields'] and
         annotation['whole_copy_alternatives_authored'] is
             target['whole_copy_alternatives_authored'],
         'author annotation/support does not bind complete target choices')
    need(annotation_schema == ANNOTATION_V2 and
         type(annotation.get('authored_selection_fields')) is list and
         type(annotation.get('literal_copy_segments_declared')) is list,
         'author annotation lacks explicit completed choice version')
    example = learning.learning_example_v16_from_record(draft)
    output = learning.supervised_output_record_v16(example)
    need(canonical_sha256(output) == target['canonical_output_sha256'],
         'complete canonical output differs')
    geometry = target_api.canonical_field_geometry(output,
        expected_target_sha256=target['canonical_output_sha256'])
    policy = policy_api.normative_policy()
    candidates = source['meaning']['candidate_inventory']
    selected = annotation['authored_selection_fields']
    by_pointer = {item['pointer']: item for item in selected}
    need(len(by_pointer) == len(selected),
         'duplicate authored candidate ordinal')
    needed = set()
    for node in geometry['nodes']:
        rules = [rule for rule in policy['rules']
                 if policy_api._match(node['path'], rule['path'])]
        need(len(rules) == 1 and node['kind'] in rules[0]['kinds'],
             'author field policy differs')
        rule = rules[0]
        if (rule['responsibility'] == 'external_selection' and
                node['kind'] == 'string'):
            needed.add(node['pointer'])
            item = by_pointer.get(node['pointer'])
            eligible = [i for i, candidate in enumerate(candidates)
                if rule['candidate_path'] in candidate['eligible_paths'] and
                   candidate['ref_id'] == node['value']]
            need(type(item) is dict and set(item) ==
                {'pointer', 'selected_candidate_ordinal'} and
                type(item['selected_candidate_ordinal']) is int and
                item['selected_candidate_ordinal'] in eligible,
                'authored candidate ordinal missing or ineligible')
    need(set(by_pointer) == needed,
         'authored candidate ordinal inventory differs')
    need(annotation['whole_copy_alternatives_authored'] is True and
         all(type(item) is dict and
             type(item.get('source_occurrences')) is list and
             bool(item['source_occurrences']) for item in
             annotation['literal_copy_segments_declared']),
         'authored whole literal-copy alternative missing')
    return example, output
