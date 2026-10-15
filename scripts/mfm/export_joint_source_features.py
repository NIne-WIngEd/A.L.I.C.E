"""Frozen complete text source states for the native joint formation decoder.

This consumes an already frozen, target-free J4/P1, J5/P2 or J6 source packet. It
does not acquire corpus material, admit rights, create targets, or fit a model.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
import platform
from pathlib import Path
from importlib.metadata import version
from math import isfinite
from time import perf_counter,process_time
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
BROAD_CASE_KEYS = J6_CASE_KEYS | {'observation_file', 'observation_file_sha256',
                                'observation_record_sha256','whole_world_file',
                                'whole_world_file_sha256','whole_world_sha256'}
FULL_SOURCE_PROFILES = ('j6-source-only', 'broader-source-only')
CPU_FLASH_PROFILE = 'cpu-sdpa-flash-v1'
CPU_FLASH_OPERATOR = 'aten::_scaled_dot_product_flash_attention_for_cpu'
MATH_ATTENTION_OPERATOR = 'aten::_scaled_dot_product_attention_math'
CPU_POLICY_SCHEMA = 'mfm-source-cpu-sdpa-flash-execution-policy-v1'
CPU_FORWARD_SCHEMA = 'mfm-source-cpu-sdpa-flash-forward-v1'
CONFIG_IDENTITY_SCHEMA = 'mfm-loaded-source-config-identity-v1'
CONFIG_IDENTITY_CLASSES = {
    'config_class': 'transformers.models.gemma4_unified.configuration_gemma4_unified.Gemma4UnifiedConfig',
    'model_class': 'transformers.models.gemma4_unified.modeling_gemma4_unified.Gemma4UnifiedForConditionalGeneration',
    'backbone_class': 'transformers.models.gemma4_unified.modeling_gemma4_unified.Gemma4UnifiedModel',
    'text_model_class': 'transformers.models.gemma4_unified.modeling_gemma4_unified.Gemma4UnifiedTextModel'}
CONFIG_IDENTITY_SOURCES = {
    'configuration_utils': '02826ac9e987487b645135d8f1d747f60a7e567d1b8f72327de8514d02b6f15d',
    'modeling_utils': '38c2bd02ed7af229f54e2f02dae65663be7af29ed2c9498bd1c460cf60fbb62d',
    'gemma4_config': 'f484527a992041a950b86d433430739fdc0f6000c03af9d87ed8b122b4715119',
    'gemma4_model': 'ae181490a16d2f36cd8f74d4d268757a5388dafc421be0ccc5dbefeff3ba9e57'}


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _process_peak_rss_bytes():
    # Linux ru_maxrss is a process-lifetime high-water mark in KiB, including
    # model loading. It is not a reset per-forward allocation measurement.
    _need(platform.system()=='Linux','CPU flash resource measurement requires Linux')
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024


def _check_execution_profile(profile,device):
    if profile is None:return None
    _need(profile==CPU_FLASH_PROFILE and torch.device(device).type=='cpu' and
          torch.__version__.split('+')[0]=='2.7.1',
          'explicit CPU flash profile/device/PyTorch2.7.1 required')
    _process_peak_rss_bytes()
    return profile


def _cpu_flash_forward(base,batch,policy):
    """Observe the actual whole native text forward; forbid silent fallback."""
    from torch.nn.attention import sdpa_kernel,SDPBackend
    config=base.config.text_config
    _need(policy['profile']==CPU_FLASH_PROFILE and
          config._attn_implementation=='sdpa' and
          type(config.num_hidden_layers) is int and config.num_hidden_layers>0 and
          config.num_hidden_layers==policy['text_attention_layers'] and
          torch.get_num_threads()==policy['cpu_threads'] and
          set(batch)=={'input_ids','attention_mask','mm_token_type_ids'} and
          all(value.device.type=='cpu' for value in batch.values()),
          'CPU flash loaded SDPA/text layers/native inputs differ')
    start=perf_counter();cpu_start=process_time()
    with torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION), \
         torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        states=base.model(**batch,use_cache=False,return_dict=True).last_hidden_state
    elapsed=perf_counter()-start;cpu_elapsed=process_time()-cpu_start
    events={event.key:event for event in prof.key_averages()}
    flash=events.get(CPU_FLASH_OPERATOR);math=events.get(MATH_ATTENTION_OPERATOR)
    count=0 if flash is None else flash.count
    math_count=0 if math is None else math.count
    _need(count==policy['text_attention_layers'] and math_count==0,
          'CPU flash complete text-layer operators required; eager/math/partial execution refused')
    return states,{'cpu_flash_operator_count':count,'math_operator_count':math_count,
        'cpu_flash_operator_time_us':float(flash.cpu_time_total),
        'wall_seconds':elapsed,'cpu_seconds':cpu_elapsed}


def _make_execution_policy(base):
    config=base.config.text_config
    _need(config._attn_implementation=='sdpa' and type(config.num_hidden_layers) is int and
          config.num_hidden_layers>0,'CPU flash loaded text attention configuration differs')
    return _sealed({'schema':CPU_POLICY_SCHEMA,'profile':CPU_FLASH_PROFILE,'device':'cpu',
        'attention_implementation':'sdpa','backend':'FLASH_ATTENTION',
        'torch_version':str(torch.__version__),'transformers_version':version('transformers'),
        'python_version':platform.python_version(),'platform':platform.system(),
        'cpu_threads':torch.get_num_threads(),'text_attention_layers':config.num_hidden_layers,
        'peak_rss_measurement':'linux-getrusage-self-process-lifetime-high-water-bytes',
        'whole_source_single_forward':True})


def _validate_execution_records(roster,receipt):
    if 'execution_policy' not in roster and 'execution' not in receipt:return
    policy=roster.get('execution_policy');observed=receipt.get('execution')
    _need(type(policy) is dict and set(policy)=={'schema','profile','device',
          'attention_implementation','backend','torch_version','transformers_version',
          'python_version','platform','cpu_threads','text_attention_layers',
          'peak_rss_measurement','whole_source_single_forward','record_sha256'} and
          policy.get('record_sha256')==canonical_sha256({k:v for k,v in policy.items()
              if k!='record_sha256'}),'CPU flash sealed execution policy differs')
    _need(policy['schema']==CPU_POLICY_SCHEMA and policy['profile']==CPU_FLASH_PROFILE and
          policy['device']=='cpu' and policy['attention_implementation']=='sdpa' and
          policy['backend']=='FLASH_ATTENTION' and policy['platform']=='Linux' and
          type(policy['torch_version']) is str and policy['torch_version'].split('+')[0]=='2.7.1' and
          policy['torch_version']==roster['base']['runtime']['torch_version'] and
          policy['transformers_version']==roster['base']['runtime']['transformers_version'] and
          roster['base']['backbone_device']=='cpu' and
          type(policy['cpu_threads']) is int and policy['cpu_threads']>0 and
          type(policy['text_attention_layers']) is int and policy['text_attention_layers']>0 and
          type(policy['python_version']) is str and bool(policy['python_version']) and
          policy['peak_rss_measurement']=='linux-getrusage-self-process-lifetime-high-water-bytes' and
          policy['whole_source_single_forward'] is True,'CPU flash runtime/policy binding differs')
    _need(type(observed) is dict and set(observed)=={'schema','policy_sha256',
          'case_id','source_alignment_sha256','source_length',
          'cpu_flash_operator_count','math_operator_count','cpu_flash_operator_time_us',
          'wall_seconds','cpu_seconds','process_lifetime_peak_rss_before_bytes',
          'process_lifetime_peak_rss_after_bytes'},'CPU flash per-case execution evidence missing')
    _need(observed['schema']==CPU_FORWARD_SCHEMA and observed['policy_sha256']==policy['record_sha256'] and
          observed['case_id']==receipt['case_id'] and
          observed['source_alignment_sha256']==receipt['alignment_sha256'] and
          type(observed['source_length']) is int and observed['source_length']>0 and
          observed['source_length']==receipt['source_length'] and
          type(observed['cpu_flash_operator_count']) is int and
          observed['cpu_flash_operator_count']==policy['text_attention_layers'] and
          type(observed['math_operator_count']) is int and observed['math_operator_count']==0 and
          all(type(observed[key]) in (int,float) and isfinite(observed[key]) and observed[key]>=0
              for key in ('cpu_flash_operator_time_us','wall_seconds','cpu_seconds')) and
          observed['cpu_flash_operator_time_us']>0 and
          all(type(observed[key]) is int and observed[key]>0 for key in
              ('process_lifetime_peak_rss_before_bytes','process_lifetime_peak_rss_after_bytes')) and
          observed['process_lifetime_peak_rss_after_bytes']>=observed['process_lifetime_peak_rss_before_bytes'],
          'CPU flash per-case kernel/resource evidence differs')


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


def _loaded_config_identity(base, config_path, *, config_sha256, requested_attention=None):
    """Bind the actual typed model to a complete independently derived config."""
    from copy import deepcopy
    from inspect import getfile
    from transformers.configuration_utils import PreTrainedConfig
    from transformers.modeling_utils import _get_dtype
    from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedConfig
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedForConditionalGeneration, Gemma4UnifiedModel, Gemma4UnifiedTextModel)

    config_path = Path(config_path)
    require_sha256(config_sha256, 'physical source config FILE')
    raw = config_path.read_bytes()
    _need(sha256(raw).hexdigest() == config_sha256, 'physical source config FILE differs')
    physical = json.loads(raw.decode('utf-8'))
    _need(type(physical) is dict and physical.get('model_type') == 'gemma4_unified' and
          physical.get('architectures') == ['Gemma4UnifiedForConditionalGeneration'],
          'physical Gemma4 source architecture differs')
    config = getattr(base, 'config', None)
    _need(type(base) is Gemma4UnifiedForConditionalGeneration and
          type(config) is Gemma4UnifiedConfig and
          type(getattr(base, 'model', None)) is Gemma4UnifiedModel and
          type(getattr(base.model, 'language_model', None)) is Gemma4UnifiedTextModel and
          base.model.config is config and config.model_type == 'gemma4_unified',
          'actual Gemma4 model/config/backbone class differs')
    # Stock text model, layers, attention, MLP and rotary modules all retain
    # this same object. Root serialization cannot detect a detached live config.
    _need(all(module.config is config.text_config
              for module in base.model.language_model.modules() if hasattr(module, 'config')),
          'live Gemma4 text config alias differs')
    source_files = {
        'configuration_utils': _digest(getfile(PreTrainedConfig)),
        'modeling_utils': _digest(getfile(_get_dtype)),
        'gemma4_config': _digest(getfile(Gemma4UnifiedConfig)),
        'gemma4_model': _digest(getfile(Gemma4UnifiedForConditionalGeneration))}
    _need(source_files == CONFIG_IDENTITY_SOURCES and version('transformers') == '5.17.0',
          'pinned Gemma4 config normalization source differs')
    # Match the supplied producer path. Never borrow a locator or field from the
    # loaded object; these hashes describe this producer context, not every host.
    expected = Gemma4UnifiedConfig.from_dict(deepcopy(physical),
        name_or_path=str(config_path.parent))
    # The exact pinned explicit-dtype branch changes only this new config and
    # its subconfigs. No checkpoint paths/state dict or automatic dtype is used.
    expected, dtype = _get_dtype(torch.bfloat16, None, expected, None, None, True)
    _need(dtype is torch.bfloat16 and config.name_or_path == str(config_path.parent) and
          all(type(getattr(config, key, None)) is type(getattr(expected, key, None))
              for key in expected.sub_configs),
          'loaded Gemma4 config origin or typed subconfig differs')
    expected_full = json.loads(expected.to_json_string(use_diff=False))
    loaded_full = json.loads(config.to_json_string(use_diff=False))
    _need(_same(expected_full, loaded_full),
          'loaded full typed source config differs from independent expectation')
    observed_attention = {'root': config._attn_implementation,
                          'text': config.text_config._attn_implementation}
    _need(requested_attention in (None, 'sdpa') and
          (requested_attention is None or
           observed_attention == {'root': 'sdpa', 'text': 'sdpa'}),
          'loaded source config attention intent differs')
    classes = {'config_class': type(config), 'model_class': type(base),
               'backbone_class': type(base.model),
               'text_model_class': type(base.model.language_model)}
    return _sealed({'schema': CONFIG_IDENTITY_SCHEMA,
        'physical_config_sha256': config_sha256,
        'expected_full_config_sha256': canonical_sha256(expected_full),
        'loaded_full_config_sha256': canonical_sha256(loaded_full),
        **{key: value.__module__ + '.' + value.__qualname__ for key, value in classes.items()},
        'derivation': {'dtype': 'bfloat16', 'reference': 'fixed-Gemma4UnifiedConfig.from_dict',
            'dtype_resolution': 'transformers.modeling_utils._get_dtype-explicit-bfloat16',
            'path_context': 'supplied-producer-prepared-directory',
            'transformers_version': version('transformers'), 'source_files': source_files},
        'requested_attention': requested_attention, 'observed_attention': observed_attention})


def _validate_config_identity(roster, receipt):
    """Validate stored producer proof without consulting consumer runtime/path."""
    base = roster.get('base'); case_base = receipt.get('base')
    _need(type(base) is dict and type(case_base) is dict, 'config identity base metadata differs')
    policy = roster.get('execution_policy')
    required = policy is not None and base.get('fixture_small_width') is False
    if not required:
        _need('config_identity' not in base and 'config_identity' not in case_base,
              'config identity requires explicit production CPU policy')
        return
    record = base.get('config_identity')
    _need(type(record) is dict and set(record) == {'schema', 'physical_config_sha256',
          'expected_full_config_sha256', 'loaded_full_config_sha256', *CONFIG_IDENTITY_CLASSES,
          'derivation', 'requested_attention', 'observed_attention', 'record_sha256'} and
          record.get('record_sha256') == canonical_sha256({k:v for k,v in record.items()
              if k != 'record_sha256'}) and _same(record, case_base.get('config_identity')),
          'sealed producer config identity differs')
    for field in ('physical_config_sha256', 'expected_full_config_sha256', 'loaded_full_config_sha256'):
        require_sha256(record[field], 'producer config identity digest')
    _need(record['schema'] == CONFIG_IDENTITY_SCHEMA and
          record['physical_config_sha256'] == base.get('config_file_sha256') and
          record['expected_full_config_sha256'] == record['loaded_full_config_sha256'] and
          all(record[key] == value for key, value in CONFIG_IDENTITY_CLASSES.items()),
          'producer config identity raw/full/class binding differs')
    derivation = record['derivation']
    _need(type(derivation) is dict and set(derivation) == {'dtype', 'reference',
          'dtype_resolution', 'path_context', 'transformers_version', 'source_files'} and
          derivation['dtype'] == 'bfloat16' and
          derivation['reference'] == 'fixed-Gemma4UnifiedConfig.from_dict' and
          derivation['dtype_resolution'] == 'transformers.modeling_utils._get_dtype-explicit-bfloat16' and
          derivation['path_context'] == 'supplied-producer-prepared-directory' and
          derivation['transformers_version'] == '5.17.0' and
          derivation['transformers_version'] == base.get('runtime', {}).get('transformers_version') and
          _same(derivation['source_files'], CONFIG_IDENTITY_SOURCES) and
          type(policy) is dict and policy.get('attention_implementation') == 'sdpa' and
          record['requested_attention'] == 'sdpa' and
          record['observed_attention'] == {'root': 'sdpa', 'text': 'sdpa'},
          'producer config identity derivation/attention binding differs')


def _base_binding(base, weight_path, weight_sha, config_path, config_sha,
                  runtime, fixture_small_width, execution_policy=None):
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
    identity = None
    if not fixture_small_width:
        _need(execution_policy is None or (execution_policy.get('profile') == CPU_FLASH_PROFILE and
              execution_policy.get('attention_implementation') == 'sdpa'),
              'explicit loaded source config policy differs')
        identity = _loaded_config_identity(base, config_path, config_sha256=config_sha,
            requested_attention=None if execution_policy is None else 'sdpa')
    _need(not base.training and not base.model.training and
          all(not p.requires_grad and p.dtype == torch.bfloat16 for p in base.parameters()),
          'source base must be frozen BF16 and eval')
    devices = {p.device for p in base.parameters()}
    _need(len(devices) == 1, 'source base placement differs')
    binding = {'weight_file_sha256': weight_sha, 'config_file_sha256': config_sha,
            'hidden_width': width, 'runtime': runtime,
            'representation': 'base.model.last_hidden_state',
            'backbone_device': str(next(iter(devices))),
            'fixture_small_width': fixture_small_width}
    if execution_policy is not None and not fixture_small_width:
        binding['config_identity'] = identity
    return binding


def _preflight_case(case, processor, j5_source):
    if type(case) is dict and case.get('profile') == 'broader-source-only':
        return _preflight_broader(case, processor, j5_source)
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
    result = {key: case[key] for key in ('source_file_sha256', 'source_record_sha256',
            'j5_file_sha256', 'proposed_max_source_tokens')} | {
        'source_identity': alignment['source_identity'],
        'construction_only': True, 'capacity_admitted': False,
        'training_authorized': False, 'current_service_authenticated': False}
    if case['profile'] == 'broader-source-only':
        result.update({key:case[key] for key in ('observation_file_sha256',
          'observation_record_sha256','whole_world_file_sha256','whole_world_sha256')})
    return result


def _preflight_broader(case, processor, j5_source):
    """Consume the physical complete new observation, without a tokenizer."""
    from scripts.mfm import formation_joint_input_v6 as j6
    from scripts.mfm import formation_broader_paired_teaching as broad
    _need(type(case) is dict and set(case)==BROAD_CASE_KEYS and
          case['profile']=='broader-source-only','target-free broader source descriptor differs')
    require_identifier(case['case_id'],'broader case ID')
    observation_path=Path(case['observation_file']).resolve(strict=True)
    obs=broad.observation(observation_path.read_bytes(),case['observation_file_sha256'])
    _need(obs['record_sha256']==case['observation_record_sha256'] and
          case['j5_file_sha256']==broad.legacy.SOURCE_J5_FILE and
          case['proposed_max_source_tokens']==obs['proposed_max_source_tokens'],
          'broader physical observation descriptor differs')
    source_path=Path(case['source_file']).resolve(strict=True)
    raw=source_path.read_bytes();frozen=broad.pinned(raw,case['source_file_sha256'],j6.FROZEN)
    _need(raw==canonical_json_bytes(frozen)+b'\n' and
          frozen['record_sha256']==case['source_record_sha256'] and
          case['case_id']=='j6-reading-room-'+frozen['cut_id'],
          'broader physical complete source descriptor differs')
    root=observation_path.parent
    def native_reader(ref):
        name=ref['path'];path=(root/name).resolve(strict=True)
        _need(type(name) is str and '\\' not in name and
              not Path(name).is_absolute() and all(x not in ('','.','..') for x in name.split('/')) and
              path.is_relative_to(root),'broader native physical path differs')
        return path.read_bytes()
    broad.input_core(j5_source,j6,frozen,case['source_file_sha256'],obs,
                     case['observation_file_sha256'],native_reader,
                     Path(case['whole_world_file']).read_bytes(),case['whole_world_file_sha256'])
    _need(frozen['meaning']['episode_snapshot']['historical_prefix_proof']['world']['world_id']==
          json.loads(Path(case['whole_world_file']).read_bytes())['world_id'] and
          json.loads(Path(case['whole_world_file']).read_bytes())['world_sha256']==case['whole_world_sha256'],
          'broader complete whole-world descriptor differs')
    alignment=frozen['alignment']
    if processor is not None:
        _need(_same(source.processor_binding(processor),alignment['processor_binding']),
              'actual broader processor binding differs')
    batch={key:torch.tensor(frozen['native_batch'][key],dtype=torch.int64) for key in j6.NATIVE_FIELDS}
    return j6,batch,alignment,_batch_plain(batch)


def _preflight_full_source(case,processor,j5_source):
    return (_preflight_broader if case['profile']=='broader-source-only' else
            _preflight_j6)(case,processor,j5_source)


def _bracket_j6_files(cases):
    for case in cases:
        if case['profile'] in FULL_SOURCE_PROFILES:
            _need(_digest(case['source_file']) == case['source_file_sha256'],
                  'physical J6 source changed during representation')
            if case['profile']=='broader-source-only':
                from scripts.mfm import formation_broader_paired_teaching as broad
                path=Path(case['observation_file']);obs=broad.observation(path.read_bytes(),case['observation_file_sha256'])
                _need(_digest(case['whole_world_file'])==case['whole_world_file_sha256'],
                      'broader exact whole-world FILE changed during representation')
                entry=next(r for r in obs['source_files'] if 'j6-reading-room-'+r['cut_id']==case['case_id'])
                for key in ('default_native','offset_native'):
                    _need(_digest(path.parent/entry[key]['path'])==entry[key]['file_sha256'],
                          'broader complete native originals changed during representation')


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
                 runtime, fixture_small_width, prepared_binding, preflight=None,
                 execution_policy=None):
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
                               fixture_small_width, execution_policy=execution_policy),
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
        execution=None
        if execution_policy is None:
            with torch.inference_mode():
                states = base.model(**{k: v.to(device) for k, v in batch.items()},
                                    use_cache=False, return_dict=True).last_hidden_state
        else:
            before_rss=_process_peak_rss_bytes()
            states,observed=_cpu_flash_forward(base,{k:v.to(device) for k,v in batch.items()},execution_policy)
            execution={'schema':CPU_FORWARD_SCHEMA,'policy_sha256':execution_policy['record_sha256'],
                'case_id':case['case_id'],'source_alignment_sha256':alignment['record_sha256'],
                'source_length':len(alignment['input_ids']),
                **observed,'process_lifetime_peak_rss_before_bytes':before_rss,
                'process_lifetime_peak_rss_after_bytes':_process_peak_rss_bytes()}
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
        if case['profile'] in FULL_SOURCE_PROFILES:
            core['broader_source' if case['profile']=='broader-source-only' else
                 'j6_source'] = _j6_binding(case, alignment)
        if execution is not None:
            core['execution']=execution
            _validate_execution_records({'base':binding,'execution_policy':execution_policy},core)
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
    roster_core={'schema': SCHEMA, 'base': binding, 'cases': receipts,
                 'target_visible_to_backbone': False,
                 'source_profiles': [row['profile'] for row in cases]}
    if execution_policy is not None:roster_core['execution_policy']=execution_policy
    roster = _write_record(root / 'roster.json',roster_core)
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
                         device,execution_profile=None):
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
    _check_execution_profile(execution_profile,device)
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
    j6_preflight = {index: _preflight_full_source(row, None, j5_source)
                    for index, row in enumerate(cases)
                    if row.get('profile') in FULL_SOURCE_PROFILES}
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
    if execution_profile is not None:
        _need(all(set(observed[1])=={'input_ids','attention_mask','mm_token_type_ids'}
                  for observed in preflight),'CPU flash requires complete native3 text source fields')
    _need(len(set(names)) == len(names) and
          canonical_sha256(cases) == expected_cases_sha256,
          'complete frozen source roster changed during preflight')
    _verified_prepared(first, receipt_path, prepared,
                       expected_prepared_receipt_file_sha256,
                       expected_prepared_receipt_sha256,
                       expected_parent_source_receipt_sha256)
    attention_kwargs={} if execution_profile is None else {'attn_implementation':'sdpa'}
    base = AutoModelForMultimodalLM.from_pretrained(
        str(prepared), trust_remote_code=False, local_files_only=True,
        use_safetensors=True, dtype=torch.bfloat16,
        device_map={'': device}, low_cpu_mem_usage=True,**attention_kwargs)
    base.requires_grad_(False).eval()
    execution_policy=None if execution_profile is None else _make_execution_policy(base)
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
                          preflight=preflight,execution_policy=execution_policy)
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
    _validate_execution_records(roster,receipt)
    _validate_config_identity(roster,receipt)
    if receipt['profile'] in FULL_SOURCE_PROFILES:
        module, _, alignment, plain = _preflight_full_source(expected_j6_source, None, j5_source)
        _need(expected_j6_source['case_id'] == case_id and
              expected_j6_source['profile']==receipt['profile'] and
              _same(receipt['broader_source' if receipt['profile']=='broader-source-only' else
                            'j6_source'], _j6_binding(expected_j6_source, alignment)) and
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
