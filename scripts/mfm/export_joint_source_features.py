"""Frozen complete text source states for the native joint formation decoder.

This consumes an already frozen, target-free J4/P1, J5/P2 or J6 source packet. It
does not acquire corpus material, admit rights, create targets, or fit a model.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from importlib.metadata import version
from types import SimpleNamespace
from uuid import uuid4

import torch
from safetensors.torch import load_file, save_file

from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256, require_identifier, require_sha256
from scripts.mfm import formation_joint_input_v4 as j4
from scripts.mfm import formation_source_alignment as source


SCHEMA = 'mfm-frozen-joint-source-feature-bank-v1'
CASE_SCHEMA = 'mfm-frozen-joint-source-feature-v1'
CASE_KEYS = frozenset({'case_id', 'profile', 'caller', 'meaning', 'source_kwargs',
                       'frozen_batch', 'frozen_alignment'})
TEXT_FIELDS = frozenset({'input_ids', 'attention_mask', 'token_type_ids',
                         'position_ids', 'mm_token_type_ids'})
J6_CASE_KEYS = frozenset({'case_id', 'profile', 'source_file', 'source_file_sha256',
                         'source_record_sha256', 'j5_file_sha256',
                         'proposed_max_source_tokens'})


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _digest(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _same(a, b):
    return canonical_sha256(a) == canonical_sha256(b)


def _sealed(core):
    return {**core, 'record_sha256': canonical_sha256(core)}


def _read_record(path, schema, expected):
    require_sha256(expected, 'expected record digest')
    raw = Path(path).read_bytes()
    def unique(pairs):
        row = {}
        for key, value in pairs:
            _need(key not in row, 'duplicate feature JSON key')
            row[key] = value
        return row
    row = json.loads(raw.decode('utf-8'), object_pairs_hook=unique)
    _need(type(row) is dict and row.get('schema') == schema and
          row.get('record_sha256') == expected and
          canonical_sha256({k: v for k, v in row.items() if k != 'record_sha256'}) == expected and
          canonical_json_bytes(row) == raw,
          'sealed feature record differs')
    return row


def _write_record(path, core):
    row = _sealed(core)
    with Path(path).open('xb') as stream:
        stream.write(canonical_json_bytes(row))
        stream.flush()
        os.fsync(stream.fileno())
    return row


def _source_api(profile, j5_source):
    if profile == 'j4-p1':
        _need(j5_source is None, 'J4 may not use private J5 source')
        return j4, j4.joint_source_batch_with_alignment_v4
    _need(profile == 'j5-p2' and j5_source is not None and
          j5_source.__name__ == 'scripts.mfm.formation_joint_input_v5' and
          callable(getattr(j5_source, 'joint_source_batch_with_alignment_v5', None)),
          'explicit original J5 source module required')
    return j5_source, j5_source.joint_source_batch_with_alignment_v5


def _batch_plain(batch):
    _need({'input_ids', 'attention_mask'} <= set(batch) <= TEXT_FIELDS,
          'unsupported or target-like source processor fields')
    values = {}
    for key in batch:
        tensor = batch[key]
        _need(type(tensor) is torch.Tensor and tensor.dtype == torch.int64 and
              tensor.ndim == 2 and tensor.shape[0] == 1,
              'complete source processor tensor differs')
        values[key] = tensor.tolist()
    _need(len(values['input_ids'][0]) > 0 and
          len(values['input_ids'][0]) == len(values['attention_mask'][0]) and
          all(type(v) is int and v >= 0 for v in values['input_ids'][0]) and
          all(type(v) is int and v == 1 for v in values['attention_mask'][0]),
          'complete text source IDs/mask differ')
    _need(all(len(row[0]) == len(values['input_ids'][0]) for row in values.values()),
          'native source field length differs')
    if 'mm_token_type_ids' in values:
        _need(all(type(value) is int and value == 0
                  for value in values['mm_token_type_ids'][0]),
              'text-only source modality field differs')
    return values


def _base_binding(base, weight_path, weight_sha, config_path, config_sha,
                  runtime, fixture_small_width):
    for value in (weight_sha, config_sha):
        require_sha256(value, 'frozen base FILE')
    _need(_digest(weight_path) == weight_sha and _digest(config_path) == config_sha,
          'actual frozen base FILE differs')
    _need(type(runtime) is dict and set(runtime) == {'torch_version', 'transformers_version'} and
          runtime['torch_version'] == torch.__version__, 'frozen runtime differs')
    if not fixture_small_width:
        _need(runtime['transformers_version'] == version('transformers'),
              'actual transformers runtime differs')
    _need(getattr(base.config, 'model_type', None) == 'gemma4_unified' and
          hasattr(base, 'model'), 'actual Gemma base interface required')
    width = getattr(getattr(base.config, 'text_config', None), 'hidden_size', None)
    _need(type(width) is int and width > 0 and (fixture_small_width or width == 3840),
          'actual source hidden width differs')
    if not fixture_small_width:
        _need(callable(getattr(base.config, 'to_json_string', None)) and
              _same(json.loads(Path(config_path).read_bytes().decode('utf-8')),
                    json.loads(base.config.to_json_string(use_diff=True))),
              'loaded base config differs from physical config')
    _need(not base.training and not base.model.training and
          all(not p.requires_grad and p.dtype == torch.bfloat16 for p in base.parameters()),
          'source base must be frozen BF16 and eval')
    devices = {p.device for p in base.parameters()}
    _need(len(devices) == 1, 'source base placement differs')
    return {'weight_file_sha256': weight_sha, 'config_file_sha256': config_sha,
            'hidden_width': width, 'runtime': runtime,
            'representation': 'base.model.last_hidden_state',
            'backbone_device': str(next(iter(devices))),
            'fixture_small_width': fixture_small_width}


def _preflight_case(case, processor, j5_source):
    if type(case) is dict and case.get('profile') == 'j6-source-only':
        return _preflight_j6(case, processor, j5_source)
    _need(type(case) is dict and set(case) == CASE_KEYS, 'target-free source case shape differs')
    name = case['case_id']
    _need(type(name) is str and require_identifier(name, 'case_id') == name,
          'noncanonical source case ID')
    module, build = _source_api(case['profile'], j5_source if case['profile'] == 'j5-p2' else None)
    kwargs = case['source_kwargs']
    _need(type(kwargs) is dict and set(kwargs) == {'caller_sha256', 'meaning_sha256',
          'render_policy', 'render_policy_sha256', 'processor_binding', 'source_identity',
          'source_identity_sha256', 'max_source_tokens'}, 'original source call differs')
    _need(_same(source.processor_binding(processor), kwargs['processor_binding']),
          'actual processor binding differs')
    _need(type(case['frozen_alignment']) is dict and
          case['frozen_alignment'].get('schema') == module.SCHEMA and
          type(case['frozen_batch']) is dict and
          {'input_ids', 'attention_mask'} <= set(case['frozen_batch']) <= TEXT_FIELDS,
          'original source profile differs')
    batch, actual = build(processor, case['caller'], case['meaning'], **kwargs)
    plain = _batch_plain(batch)
    _need(_digest(module.__file__) == actual['render_policy']['code_sha256'],
          'physical original source module differs from frozen policy')
    _need(_same(actual, case['frozen_alignment']) and
          _same(plain, case['frozen_batch']) and
          actual['record_sha256'] == canonical_sha256({k: v for k, v in actual.items()
                                                      if k != 'record_sha256'}) and
          _same(actual['cache'], case['frozen_alignment']['cache']) and
          _same({key: plain[key] for key in ('input_ids', 'attention_mask')},
                {'input_ids': [actual['input_ids']],
                 'attention_mask': [actual['attention_mask']]}),
          'complete independently frozen source differs')
    _need(actual['teacher_accessed'] is False and actual['native_locator_alignment'] is False,
          'unsupported source profile')
    return module, batch, actual, plain


def _preflight_j6(case, processor, j5_source):
    """Read the externally pinned complete observation; never retokenize it."""
    from scripts.mfm import formation_joint_input_v6 as j6

    _need(type(case) is dict and set(case) == J6_CASE_KEYS and
          case['profile'] == 'j6-source-only', 'target-free J6 source case differs')
    _need(type(case['case_id']) is str and
          require_identifier(case['case_id'], 'case_id') == case['case_id'],
          'noncanonical J6 case ID')
    _need(type(case['source_file']) is str and bool(case['source_file']),
          'physical J6 source path required')
    for key in ('source_file_sha256', 'source_record_sha256', 'j5_file_sha256'):
        require_sha256(case[key], key)
    raw = Path(case['source_file']).read_bytes()
    _need(sha256(raw).hexdigest() == case['source_file_sha256'],
          'physical J6 source FILE differs')
    def unique(pairs):
        row = {}
        for key, value in pairs:
            _need(key not in row, 'duplicate J6 source JSON key')
            row[key] = value
        return row
    frozen = json.loads(raw.decode('utf-8'), object_pairs_hook=unique)
    _need(type(frozen) is dict and frozen.get('record_sha256') ==
          case['source_record_sha256'], 'external J6 source record differs')
    j6.validate_frozen_source(j5_source, frozen,
        j5_file_sha256=case['j5_file_sha256'],
        proposed_max_source_tokens=case['proposed_max_source_tokens'])
    alignment = frozen['alignment']
    if processor is not None:
        _need(_same(source.processor_binding(processor), alignment['processor_binding']),
              'actual J6 processor binding differs')
    batch = {key: torch.tensor(frozen['native_batch'][key], dtype=torch.int64)
             for key in j6.NATIVE_FIELDS}
    return j6, batch, alignment, _batch_plain(batch)


def _j6_binding(case, alignment):
    return {key: case[key] for key in ('source_file_sha256', 'source_record_sha256',
            'j5_file_sha256', 'proposed_max_source_tokens')} | {
        'source_identity': alignment['source_identity'],
        'construction_only': True, 'capacity_admitted': False,
        'training_authorized': False, 'current_service_authenticated': False}


def _bracket_j6_files(cases):
    for case in cases:
        if case['profile'] == 'j6-source-only':
            _need(_digest(case['source_file']) == case['source_file_sha256'],
                  'physical J6 source changed during representation')


def _tensor_check(tensors, row, width):
    _need(set(tensors) == {'states', *row['frozen_batch']}, 'feature tensor roster differs')
    states, ids, mask = (tensors[k] for k in ('states', 'input_ids', 'attention_mask'))
    n = row['source_length']
    _need(states.dtype == torch.bfloat16 and tuple(states.shape) == (1, n, width) and
          ids.dtype == torch.int64 and mask.dtype == torch.int64 and
          tuple(ids.shape) == tuple(mask.shape) == (1, n) and
          bool(torch.isfinite(states).all()) and
          torch.equal(ids, torch.tensor(row['frozen_batch']['input_ids'])) and
          torch.equal(mask, torch.tensor(row['frozen_batch']['attention_mask'])),
          'complete feature tensor geometry or source values differ')
    _need(all(tensors[key].dtype == torch.int64 and
              tuple(tensors[key].shape) == (1, n) and
              torch.equal(tensors[key], torch.tensor(value, dtype=torch.int64))
              for key, value in row['frozen_batch'].items()),
          'complete native source fields differ')
    if 'mm_token_type_ids' in tensors:
        _need(bool(torch.all(tensors['mm_token_type_ids'] == 0)),
              'text-only source modality tensor differs')


def _export_bank(root, *, cases, processor, base, j5_source, base_weight_path,
                 base_weight_sha256, base_config_path, base_config_sha256,
                 runtime, fixture_small_width, prepared_binding, preflight=None):
    """Create one immutable complete target-free feature roster on the host.

    Fixture width is an explicit mechanics-only exception; production is 3840.
    Rights and material admission remain independent of this representation.
    """
    _need(type(fixture_small_width) is bool, 'fixture mode must be explicit')
    _need((fixture_small_width and prepared_binding is None) or
          (not fixture_small_width and type(prepared_binding) is dict and
           set(prepared_binding) == {'receipt_sha256', 'receipt_file_sha256',
                                     'parent_source_receipt_sha256',
                                     'source_weight_sha256', 'foundation_commit',
                                     'foundation_verifier_sha256',
                                     'foundation_inventory_sha256',
                                     'prepared_files', 'prepared_schema'}),
          'actual prepared clone binding required for production')
    _need(type(cases) is list and cases, 'complete source roster required')
    if not fixture_small_width:
        _need(all(row['profile'] != 'j5-p2' or
                  row['source_kwargs']['max_source_tokens'] == 32768
                  for row in cases),
              'retained private J5 complete source budget differs')
    binding = {**_base_binding(base, base_weight_path, base_weight_sha256,
                               base_config_path, base_config_sha256, runtime,
                               fixture_small_width),
               'prepared_base': prepared_binding}
    prepared = ([_preflight_case(row, processor, j5_source) for row in cases]
                if preflight is None else preflight)
    _need(type(prepared) is list and len(prepared) == len(cases),
          'complete source preflight differs')
    names = [row['case_id'] for row in cases]
    _need(len(set(names)) == len(names), 'duplicate source case')
    root = Path(root)
    _need(not root.exists(), 'immutable feature bank already exists')
    root.mkdir(parents=True, exist_ok=False)
    (root / 'cases').mkdir()
    receipts = []
    device = next(base.parameters()).device
    for case, (module, batch, alignment, plain) in zip(cases, prepared):
        with torch.inference_mode():
            states = base.model(**{k: v.to(device) for k, v in batch.items()},
                                use_cache=False, return_dict=True).last_hidden_state
        _need(type(states) is torch.Tensor, 'base representation missing')
        tensors = {'states': states.detach().cpu().contiguous(),
                   **{key: value.detach().cpu().contiguous()
                      for key, value in batch.items()}}
        directory = root / 'cases' / case['case_id']
        directory.mkdir()
        path = directory / 'features.safetensors'
        code = {'source_module_sha256': _digest(module.__file__),
                'exporter_sha256': _digest(__file__),
                'source_code_binding': alignment['code_binding'],
                'processor_binding': alignment['processor_binding']}
        core = {'schema': CASE_SCHEMA, 'case_id': case['case_id'], 'profile': case['profile'],
                'source_length': len(alignment['input_ids']),
                'alignment_sha256': alignment['record_sha256'], 'cache': alignment['cache'],
                'frozen_batch': plain, 'code': code, 'base': binding,
                'target_visible_to_backbone': False, 'whole_source': True}
        if case['profile'] == 'j6-source-only':
            core['j6_source'] = _j6_binding(case, alignment)
        _tensor_check(tensors, core, binding['hidden_width'])
        save_file(tensors, str(path))
        with path.open('r+b') as stream:
            os.fsync(stream.fileno())
        restored = load_file(str(path), device='cpu')
        _tensor_check(restored, core, binding['hidden_width'])
        _need(all(torch.equal(tensors[k], restored[k]) for k in tensors),
              'serialized source features differ on readback')
        receipt = _write_record(directory / 'receipt.json',
                                {**core, 'features_file_sha256': _digest(path)})
        receipts.append({'case_id': case['case_id'],
                         'receipt_sha256': receipt['record_sha256']})
    _bracket_j6_files(cases)
    roster = _write_record(root / 'roster.json',
                           {'schema': SCHEMA, 'base': binding, 'cases': receipts,
                            'target_visible_to_backbone': False,
                            'source_profiles': [row['profile'] for row in cases]})
    return roster


def export_bank(root, *, cases, processor, base, j5_source, base_weight_path,
                base_weight_sha256, base_config_path, base_config_sha256,
                runtime, fixture_small_width=False):
    """Untrained fixture mechanics only; production owns the prepared load."""
    _need(fixture_small_width is True,
          'caller-supplied base cannot attest production source weights')
    return _export_bank(root, cases=cases, processor=processor, base=base,
                        j5_source=j5_source, base_weight_path=base_weight_path,
                        base_weight_sha256=base_weight_sha256,
                        base_config_path=base_config_path,
                        base_config_sha256=base_config_sha256, runtime=runtime,
                        fixture_small_width=True, prepared_binding=None)


def _verified_prepared(prepared, receipt_path, root, expected_receipt_file_sha256,
                       expected_receipt_sha256, expected_parent_source_sha256):
    from scripts.mfm import train_v1_formation_specialist as shared

    _need(_digest(receipt_path) == expected_receipt_file_sha256,
          'prepared receipt physical FILE differs')
    args = SimpleNamespace(prepared_base_dir=root,
                           prepared_base_receipt=receipt_path)
    row = shared._prepared_base(args)
    valid = (row['schema'] == shared.CLONE_SCHEMA and row['role'] == 'mfm' and
             row['receipt_sha256'] == expected_receipt_sha256 and
             row['parent_source_receipt_sha256'] == expected_parent_source_sha256 and
             Path(row['snapshot_path']).resolve(strict=True) == root)
    if prepared is not None:
        valid = valid and _same(row, prepared)
    _need(valid,
          'prepared MFM clone or source ancestry differs')
    files = row['files']
    _need(type(files) is list and files and
          len({item['path'] for item in files}) == len(files),
          'complete prepared clone FILE roster differs')
    for item in files:
        path = root / item['path']
        _need(path.is_file() and path.stat().st_size == item['size'] and
              _digest(path) == item['sha256'],
              'prepared clone physical FILE roster differs')
    return row


def export_prepared_bank(root, *, prepared_base_dir, cases, j5_source,
                         prepared_receipt_path, expected_prepared_receipt_file_sha256,
                         expected_prepared_receipt_sha256,
                         expected_parent_source_receipt_sha256,
                         expected_cases_sha256,
                         device):
    """Host-side production entrypoint bound to one local prepared Gemma clone.

    Loading the processor and model from the same physically checked directory
    connects the supplied FILE pins to the actual representation call. This
    function performs no remote lookup or download.
    """
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    from scripts.mfm import train_v1_formation_specialist as shared

    prepared = Path(prepared_base_dir).resolve(strict=True)
    receipt_path = Path(prepared_receipt_path).resolve(strict=True)
    root = Path(root).resolve(strict=False)
    _need(prepared.is_dir() and type(device) is str and device,
          'local prepared base and explicit device required')
    _need(root.parent.is_dir() and not root.exists() and
          root != prepared and prepared not in root.parents and
          root != receipt_path and root not in receipt_path.parents,
          'new feature output must be outside prepared originals')
    for value in (expected_prepared_receipt_file_sha256,
                  expected_prepared_receipt_sha256,
                  expected_parent_source_receipt_sha256,
                  expected_cases_sha256):
        require_sha256(value, 'external prepared clone pin')
    _need(type(cases) is list and cases and
          canonical_sha256(cases) == expected_cases_sha256,
          'externally frozen complete source roster differs')
    _need(all(type(row) is dict and
              (row.get('profile') != 'j5-p2' or
               (type(row.get('source_kwargs')) is dict and
                type(row['source_kwargs'].get('max_source_tokens')) is int and
                row['source_kwargs']['max_source_tokens'] == 32768))
              for row in cases),
          'retained private J5 complete source budget differs')
    # Refuse changed physical J6 observations before any prepared weight loader.
    j6_preflight = {index: _preflight_j6(row, None, j5_source)
                    for index, row in enumerate(cases)
                    if row.get('profile') == 'j6-source-only'}
    first = _verified_prepared(None, receipt_path, prepared,
                               expected_prepared_receipt_file_sha256,
                               expected_prepared_receipt_sha256,
                               expected_parent_source_receipt_sha256)
    _need(first['schema'] == shared.CLONE_SCHEMA and
          first['parent_source_receipt_sha256'] == expected_parent_source_receipt_sha256,
          'prepared source recipe differs')
    files = {item['path']: item for item in first['files']}
    weight = prepared / 'model.safetensors'
    config = prepared / 'config.json'
    _need('model.safetensors' in files and 'config.json' in files and
          files['model.safetensors']['sha256'] == shared.SOURCE_WEIGHT_SHA256,
          'prepared clone source weight/config recipe differs')
    processor = AutoProcessor.from_pretrained(str(prepared), local_files_only=True,
                                              trust_remote_code=False)
    preflight = []
    for index, row in enumerate(cases):
        if index in j6_preflight:
            observed = j6_preflight[index]
            _need(_same(source.processor_binding(processor), observed[2]['processor_binding']),
                  'actual J6 processor binding differs')
            preflight.append(observed)
        else:
            preflight.append(_preflight_case(row, processor, j5_source))
    _bracket_j6_files(cases)
    names = [row['case_id'] for row in cases]
    _need(len(set(names)) == len(names) and
          canonical_sha256(cases) == expected_cases_sha256,
          'complete frozen source roster changed during preflight')
    _verified_prepared(first, receipt_path, prepared,
                       expected_prepared_receipt_file_sha256,
                       expected_prepared_receipt_sha256,
                       expected_parent_source_receipt_sha256)
    base = AutoModelForMultimodalLM.from_pretrained(
        str(prepared), trust_remote_code=False, local_files_only=True,
        use_safetensors=True, dtype=torch.bfloat16,
        device_map={'': device}, low_cpu_mem_usage=True)
    base.requires_grad_(False).eval()
    binding = {'receipt_sha256': first['receipt_sha256'],
               'receipt_file_sha256': expected_prepared_receipt_file_sha256,
               'parent_source_receipt_sha256': expected_parent_source_receipt_sha256,
               'source_weight_sha256': shared.SOURCE_WEIGHT_SHA256,
               'foundation_commit': shared.FOUNDATION_COMMIT,
               'foundation_verifier_sha256': shared.FOUNDATION_VERIFIER_SHA256,
               'foundation_inventory_sha256': shared.FOUNDATION_INVENTORY_SHA256,
               'prepared_files': first['files'], 'prepared_schema': first['schema']}
    staging = root.parent / ('.' + root.name + '.staging-' + uuid4().hex)
    roster = _export_bank(staging, cases=cases, processor=processor, base=base,
                          j5_source=j5_source, base_weight_path=weight,
                          base_weight_sha256=files['model.safetensors']['sha256'],
                          base_config_path=config,
                          base_config_sha256=files['config.json']['sha256'],
                          runtime={'torch_version': torch.__version__,
                                   'transformers_version': version('transformers')},
                          fixture_small_width=False, prepared_binding=binding,
                          preflight=preflight)
    _verified_prepared(first, receipt_path, prepared,
                       expected_prepared_receipt_file_sha256,
                       expected_prepared_receipt_sha256,
                       expected_parent_source_receipt_sha256)
    _need(not root.exists(), 'immutable feature output appeared during export')
    _bracket_j6_files(cases)
    staging.rename(root)
    return roster


def read_case(root, *, roster_sha256, case_id, expected_alignment_sha256,
              expected_cache, j5_source=None, allow_fixture=False,
              expected_j6_source=None):
    """Revalidate immutable source and complete representation at consumption."""
    require_identifier(case_id, 'case_id')
    require_sha256(expected_alignment_sha256, 'source alignment digest')
    root = Path(root)
    roster = _read_record(root / 'roster.json', SCHEMA, roster_sha256)
    _need(type(allow_fixture) is bool and
          (allow_fixture or roster['base']['fixture_small_width'] is False),
          'fixture representation cannot serve the actual joint fit')
    _need((roster['base']['fixture_small_width'] is True and
           roster['base']['prepared_base'] is None) or
          (roster['base']['fixture_small_width'] is False and
           type(roster['base']['prepared_base']) is dict),
          'feature source preparation binding differs')
    _need(set(root.iterdir()) == {root / 'roster.json', root / 'cases'} and
          {p.name for p in (root / 'cases').iterdir()} ==
          {row['case_id'] for row in roster['cases']}, 'complete feature roster differs')
    matches = [row for row in roster['cases'] if row['case_id'] == case_id]
    _need(len(matches) == 1, 'case absent from frozen feature roster')
    directory = root / 'cases' / case_id
    _need(set(directory.iterdir()) == {directory / 'receipt.json',
                                      directory / 'features.safetensors'},
          'feature case files differ')
    receipt = _read_record(directory / 'receipt.json', CASE_SCHEMA,
                           matches[0]['receipt_sha256'])
    if receipt['profile'] == 'j6-source-only':
        module, _, alignment, plain = _preflight_j6(expected_j6_source, None, j5_source)
        _need(expected_j6_source['case_id'] == case_id and
              _same(receipt['j6_source'], _j6_binding(expected_j6_source, alignment)) and
              alignment['record_sha256'] == expected_alignment_sha256 and
              _same(alignment['cache'], expected_cache) and
              _same(receipt['frozen_batch'], plain) and
              _same(receipt['code']['processor_binding'], alignment['processor_binding']),
              'complete physical J6 feature source binding differs')
        live_binding = alignment['code_binding']
    else:
        _need(expected_j6_source is None, 'J6 source cannot relabel an older feature profile')
        module, _ = _source_api(receipt['profile'], j5_source if receipt['profile'] == 'j5-p2' else None)
        live_binding = (module.joint_code_binding_v4() if receipt['profile'] == 'j4-p1'
                        else module.joint_code_binding_v5())
    _need(_digest(module.__file__) == receipt['code']['source_module_sha256'] and
          _digest(__file__) == receipt['code']['exporter_sha256'] and
          _same(receipt['code']['source_code_binding'], live_binding),
          'live original source interface differs')
    _need(receipt['case_id'] == case_id and _same(receipt['base'], roster['base']) and
          receipt['alignment_sha256'] == expected_alignment_sha256 and
          _same(receipt['cache'], expected_cache) and
          receipt['target_visible_to_backbone'] is False and
          receipt['whole_source'] is True and
          _digest(directory / 'features.safetensors') == receipt['features_file_sha256'],
          'independent source/feature binding differs')
    tensors = load_file(str(directory / 'features.safetensors'), device='cpu')
    _tensor_check(tensors, receipt, roster['base']['hidden_width'])
    if expected_j6_source is not None:
        _bracket_j6_files([expected_j6_source])
    return receipt, tensors
