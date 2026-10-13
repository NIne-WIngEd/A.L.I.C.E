"""Build the actual S5 lexical -> native-reference -> complete joint seed.

This is a zero-joint-update component builder, never a fit or source encoder.
The production CLI accepts only the externally pinned full-size S5 recipe.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import subprocess
import sys
import tarfile
import tempfile
from time import perf_counter

import safetensors
from safetensors.torch import load_file, save_file
import torch

from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256
from cognitive_kernel.formation_v1_specialist import SpecialistConfig
from cognitive_kernel import formation_reference_specialist as reference
from cognitive_kernel import formation_joint_learning as joint


SCHEMA = 'mfm-native-joint-build-plan-v1'
S5_WEIGHT = '383bfc8497e616df1423ef996ebe1ecb649008ca98008ba7032c2c8b66f432df'
S5_WEIGHT_BYTES = 1070181192
S5_CONFIG = dict(base_hidden_size=3840, vocabulary_size=262144, width=768,
    layers=6, heads=12, max_target_tokens=8192, pad_token_id=0,
    start_token_id=2, end_token_id=1, logit_chunk_tokens=16)
S5_CURSOR = dict(epoch=20, next_case=0, step=45760)
S5_RECORD_FILES = {
    'component':'205ab7de1e4aadc9f5eacdcff976f2b8491c2eb2bb7a9335354169514ee566cf',
    'run':'21f441d48c54fc118cc1e9e7e332a726f15ea20d1212a6ea290958af3708c574',
    'checkpoint':'5371cfd7c473a07866e5291386c2777f03539e28eed3688bbe519cd3800232b2',
    'history':'20bd5d3d1367d6bbc2a7d772548ffa9283ffd88c10329f346d00b3407c10e077',
}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _pin(value):
    _need(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),
          'canonical SHA256 required')
    return value


def _seal(row):
    return {**row, 'record_sha256':canonical_sha256(row)}


def _record(row, schema=None):
    _need(type(row) is dict and row.get('record_sha256') == canonical_sha256(
        {k:v for k,v in row.items() if k!='record_sha256'}), 'original logical record seal differs')
    _pin(row['record_sha256'])
    if schema is not None:
        _need(row.get('schema')==schema, 'original record schema differs')
    return row


def read_json_bytes(raw):
    _need(type(raw) is bytes, 'JSON must be original bytes')
    def unique(pairs):
        row={}
        for key,value in pairs:
            _need(key not in row, 'duplicate original JSON key')
            row[key]=value
        return row
    def nonfinite(value):
        raise ValueError('nonfinite original JSON constant: '+value)
    def finite_float(value):
        number=float(value)
        _need(math.isfinite(number),'nonfinite original JSON number')
        return number
    return json.loads(raw.decode('utf-8','strict'),object_pairs_hook=unique,
                      parse_constant=nonfinite,parse_float=finite_float)


def _digest(path):
    value=sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(8*1024*1024):
            value.update(block)
    return value.hexdigest()


def _write_new(path, row):
    raw=canonical_json_bytes(row)
    with Path(path).open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())


def _save_new(path, state):
    path=Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.native-',delete=False) as stream:
        temporary=Path(stream.name)
    try:
        save_file({k:v.detach().cpu().contiguous() for k,v in state.items()},str(temporary))
        os.link(temporary,path)  # An existing destination refuses atomically.
    finally:
        temporary.unlink(missing_ok=True)


def _file(root, descriptor, *, logical=False):
    fields={'path','bytes','file_sha256'}|({'record_sha256'} if logical else set())
    _need(type(descriptor) is dict and set(descriptor)==fields, 'original FILE descriptor fields differ')
    relative=descriptor['path']
    _need(type(relative) is str and relative and '\\' not in relative, 'original relative path differs')
    parts=PurePosixPath(relative)
    _need(not parts.is_absolute() and not any(p in ('','..','.') for p in parts.parts) and
          parts.as_posix()==relative, 'original relative path is noncanonical')
    root=Path(root).resolve(strict=True)
    selected=root.joinpath(*parts.parts)
    _need(selected.resolve(strict=True).is_relative_to(root), 'original FILE escapes readonly parent')
    current=root
    for part in parts.parts:
        current=current/part
        _need(not current.is_symlink(), 'original FILE path contains a link')
    info=selected.stat()
    _need(stat.S_ISREG(info.st_mode) and type(descriptor['bytes']) is int and descriptor['bytes']>=0 and
          info.st_size==descriptor['bytes'] and _digest(selected)==_pin(descriptor['file_sha256']),
          'original FILE byte/hash contract differs')
    identity=dict(bytes=info.st_size,file_sha256=descriptor['file_sha256'],device=info.st_dev,
                  inode=info.st_ino,mtime_ns=info.st_mtime_ns,ctime_ns=info.st_ctime_ns)
    if logical:
        row=_record(read_json_bytes(selected.read_bytes()))
        _need(row['record_sha256']==_pin(descriptor['record_sha256']), 'externally expected original record differs')
        return selected,identity,row
    return selected,identity


def source_signature(expected):
    _need(type(expected) is dict and set(expected)=={'builder_file_sha256','interface_pins','reviewed_joint_file_sha256'},
          'builder source expectation fields differ')
    _need(_digest(__file__)==_pin(expected['builder_file_sha256']), 'current builder FILE differs')
    interfaces=joint._interface_pins()
    _need(type(expected['interface_pins']) is dict and interfaces==expected['interface_pins'],
          'current complete component interface closure differs')
    _need(interfaces.get('src/cognitive_kernel/formation_joint_learning.py')==
          _pin(expected['reviewed_joint_file_sha256']), 'actual joint producer differs from explicitly reviewed FILE')
    root=Path(__file__).resolve().parents[2]
    module_paths={};namespace_paths={}
    source_files={**interfaces,'scripts/mfm/build_native_joint_component.py':expected['builder_file_sha256']}
    for relative in source_files:
        parts=list(PurePosixPath(relative).parts)
        base=root/'src' if parts[0]=='src' else root
        if parts[0]=='src':parts=parts[1:]
        parts[-1]=PurePosixPath(parts[-1]).stem
        if parts[-1]=='__init__':parts=parts[:-1]
        name='.'.join(parts)
        _need(name not in module_paths,'ambiguous local import name in source closure')
        module_paths[name]=relative
        for count in range(1,len(parts)):
            namespace_paths.setdefault('.'.join(parts[:count]),set()).add(base.joinpath(*parts[:count]).resolve())
    package_roots={name.split('.')[0] for name in module_paths}
    origins={}
    for name,module in tuple(sys.modules.items()):
        if name.split('.')[0] not in package_roots:
            continue
        origin=getattr(module,'__file__',None)
        if origin is None:
            locations=getattr(getattr(module,'__spec__',None),'submodule_search_locations',None)
            _need(name not in module_paths and locations is not None and
                  bool(list(locations)) and all(Path(p).resolve() in namespace_paths.get(name,set()) for p in locations),
                  'local namespace package search path differs from source closure')
            origins[name]={'namespace_paths':sorted(Path(p).resolve().relative_to(root).as_posix() for p in locations)}
            continue
        path=Path(origin).resolve(strict=True)
        _need(path.is_relative_to(root), 'imported local module is outside the pinned source package')
        relative=path.relative_to(root).as_posix()
        pin=source_files.get(relative)
        spec_origin=getattr(getattr(module,'__spec__',None),'origin',None)
        _need(module_paths.get(name)==relative and pin is not None and _digest(path)==pin and
              spec_origin is not None and Path(spec_origin).resolve()==path,
              'actual imported local module name/FILE/spec origin differs')
        origins[name]=relative
    return dict(builder_file_sha256=expected['builder_file_sha256'],interface_pins=interfaces,
                reviewed_joint_file_sha256=expected['reviewed_joint_file_sha256'],
                imported_local_origins=origins)


def state_signatures(state):
    result={}
    for name,value in sorted(state.items()):
        _need(value.device.type=='cpu' and value.dtype==torch.float32 and torch.isfinite(value).all().item(),
              'state signature requires complete finite CPU FP32 tensors: '+name)
        contiguous=value.detach().contiguous()
        raw=memoryview(contiguous.view(torch.uint8).numpy()).cast('B')
        result[name]=dict(shape=list(value.shape),dtype='F32',bytes=len(raw),sha256=sha256(raw).hexdigest())
    return result


def _rng_pin():
    return sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()


def _cursor(value):
    _need(type(value) is dict and set(value)==set(S5_CURSOR) and
          all(type(v) is int and v>=0 for v in value.values()), 'actual original cursor integer types differ')
    return value


def _binding(value):
    _need(type(value) is dict and set(value)=={'run','seed','train_cases','batch_size','epochs'} and
          all(type(value[k]) is int and value[k]>0 for k in ('train_cases','batch_size','epochs')),
          'actual original checkpoint binding counter types differ')
    _pin(value['run']);_pin(value['seed'])
    return value


def _same(left,right):
    return canonical_json_bytes(left)==canonical_json_bytes(right)


def validate_plan(plan, expected, *, fixture_only=False):
    _record(plan,SCHEMA)
    _need(plan['record_sha256']==_pin(expected) and set(plan)=={'schema','purpose','component_id',
        'parent_kernel','parent','expected_parent','source','initialization','runtime','joint_updates',
        'training_authorized','full_MFM_qualified','record_sha256'}, 'externally pinned build plan differs')
    _need(type(plan['component_id']) is str and plan['component_id'] and
          type(plan['joint_updates']) is int and plan['joint_updates']==0 and
          plan['training_authorized'] is False and plan['full_MFM_qualified'] is False,
          'builder cannot train or grant rights/qualification')
    parent=plan['parent']; expected_parent=plan['expected_parent']
    _need(type(parent) is dict and set(parent)=={'weights','records'} and
          type(parent['records']) is dict and set(parent['records'])==set(S5_RECORD_FILES),
          'complete original parent record roster differs')
    _need(type(expected_parent) is dict and set(expected_parent)=={'config','config_canonical_sha256',
          'cursor','lexical_parameter_count','history_entry_count','history_session_count'},
          'original parent expectation fields differ')
    config=expected_parent['config']
    _need(type(config) is dict and set(config)==set(S5_CONFIG), 'complete embedded specialist config differs')
    cfg=SpecialistConfig(**config);cfg.validate()
    _need(canonical_sha256(cfg.record())==_pin(expected_parent['config_canonical_sha256']),
          'canonical embedded config pin differs')
    cursor=expected_parent['cursor']
    _need(type(cursor) is dict and set(cursor)==set(S5_CURSOR) and
          all(type(v) is int and v>=0 for v in cursor.values()), 'completed parent cursor types differ')
    for name in ('lexical_parameter_count','history_entry_count','history_session_count'):
        _need(type(expected_parent[name]) is int and expected_parent[name]>0, 'positive actual parent counts required')
    seeds=plan['initialization']
    _need(type(seeds) is dict and set(seeds)=={'reference_seed','copy_seed'} and
          all(type(v) is int and 0<=v<2**63 for v in seeds.values()) and seeds['reference_seed']!=seeds['copy_seed'],
          'explicit distinct fresh initialization seeds required')
    runtime=plan['runtime']
    _need(type(runtime) is dict and set(runtime)=={'device','threads','torch_version_family','safetensors_required'} and
          runtime['device']=='cpu' and type(runtime['threads']) is int and runtime['threads']==1 and
          type(runtime['torch_version_family']) is str and runtime['safetensors_required'] is True,
          'explicit CPU-only runtime contract required')
    kernel=plan['parent_kernel']
    _need(type(kernel) is dict and set(kernel)=={'ref','numeric_id','version','source_revision'} and
          type(kernel['numeric_id']) is int and type(kernel['version']) is int,
          'exact original parent kernel identity required')
    if fixture_only:
        _need(plan['purpose']=='tiny-mechanics-fixture','fixture purpose differs')
    else:
        _need(plan['purpose']=='full-size-S5-native-joint-build' and config==S5_CONFIG and cursor==S5_CURSOR and
              expected_parent['lexical_parameter_count']==267542272 and
              expected_parent['history_entry_count']==21 and expected_parent['history_session_count']==6 and
              kernel==dict(ref='mkrayanyan/rayan-mfm-full-fit-s5-20261011',numeric_id=138255915,version=1,
                  source_revision='268bf08760baf8a6032afca207c56e131ce20b81') and
              parent['weights']['file_sha256']==S5_WEIGHT and parent['weights']['bytes']==S5_WEIGHT_BYTES and
              all(parent['records'][k]['file_sha256']==pin for k,pin in S5_RECORD_FILES.items()) and
              runtime['torch_version_family']=='2.11',
              'production requires exact complete original S5 and reviewed native joint source')
    return cfg


def parent_snapshot(plan, root):
    weights,weight_identity=_file(root,plan['parent']['weights'])
    records={};signature={'weights':weight_identity}
    schemas={'component':'component','run':'run','checkpoint':'checkpoint','history':'history'}
    for name,description in plan['parent']['records'].items():
        _,identity,row=_file(root,description,logical=True)
        _record(row,'mfm-public-conditional-foundation-'+schemas[name]+'-v1')
        records[name]=row;signature[name]=identity
    component,run,checkpoint,history=(records[k] for k in ('component','run','checkpoint','history'))
    expected=plan['expected_parent'];pin=plan['parent']['weights']['file_sha256']
    for record in (component,run):
        actual_config=record.get('specialist_config')
        _need(type(actual_config) is dict and set(actual_config)==set(S5_CONFIG),'actual embedded config fields differ')
        SpecialistConfig(**actual_config).validate()
    _cursor(component.get('cursor'));_cursor(checkpoint.get('cursor'))
    _binding(checkpoint.get('binding'));_binding(history.get('binding'))
    _need(component['task']==run['task']=='conditional_meaning_to_text' and
          _same(component['specialist_config'],run['specialist_config']) and
          _same(run['specialist_config'],expected['config']) and
          canonical_sha256(component['specialist_config'])==expected['config_canonical_sha256'] and
          run['initialization']=='fresh-token-position-scale-0.02-v1' and
          component['qualified_for_product'] is False and run['qualified_for_product'] is False,
          'original task/config/initialization differs')
    _need(_same(component['cursor'],checkpoint['cursor']) and _same(checkpoint['cursor'],expected['cursor']) and
          component['specialist_sha256']==checkpoint['specialist_sha256']==pin and
          component['run_sha256']==run['record_sha256']==checkpoint['binding']['run'] and
          component['seed_sha256']==checkpoint['binding']['seed'] and
          component['checkpoint_sha256']==checkpoint['record_sha256'] and
          component['checkpoint_history_sha256']==history['record_sha256'] and
          _same(history['binding'],checkpoint['binding']) and _same(history['run'],run) and
          component['curriculum_sha256']==history['curriculum']['record_sha256'],
          'original completed checkpoint/config/history ancestry differs')
    _record(history['curriculum'],'mfm-public-conditional-foundation-curriculum-v1')
    _need(type(history['entries']) is list and len(history['entries'])==expected['history_entry_count'] and
          type(history['sessions']) is list and len(history['sessions'])==expected['history_session_count'] and
          type(history['session_index']) is int and history['session_index']==len(history['sessions'])-1,
          'whole original history counts differ')
    previous=None;lineage=[]
    for entry in history['entries']:
        _record(entry,'mfm-public-conditional-foundation-history-entry-v1')
        _record(entry['receipt'],'mfm-public-conditional-foundation-checkpoint-v1')
        _cursor(entry['receipt'].get('cursor'))
        _binding(entry['receipt'].get('binding'))
        _need(entry['previous_entry_sha256']==previous and _same(entry['receipt']['binding'],history['binding']) and
              type(entry['session_index']) is int and 0<=entry['session_index']<len(history['sessions']),
              'original history chain differs')
        previous=entry['record_sha256'];lineage.append(entry['receipt']['record_sha256'])
    for i,session in enumerate(history['sessions']):
        _record(session,'mfm-public-conditional-foundation-session-v1')
        _cursor(session.get('start_cursor'))
        _need(type(session['session_index']) is int and session['session_index']==i and
              session['run_sha256']==run['record_sha256'] and
              type(session['parent_entry_count']) is int and 0<=session['parent_entry_count']<len(history['entries']),
              'original session ancestry/counter types differ')
    _need(component['checkpoint_lineage']==lineage and _same(history['entries'][-1]['receipt'],checkpoint),
          'complete original component lineage differs')
    return weights,signature,records


def build_component(plan, *, expected_plan_sha256, parent_root, output_dir, fixture_only=False):
    output=Path(output_dir)
    _need(not output.resolve().is_relative_to(Path(parent_root).resolve()),'output cannot modify readonly parent')
    output.mkdir(mode=0o700,parents=False,exist_ok=False)
    stage='plan_preflight';started=perf_counter()
    cpu_rng=torch.get_rng_state().clone();threads=torch.get_num_threads();rng={}
    _write_new(output/'intent.json',_seal(dict(schema='mfm-native-joint-build-intent-v1',
        expected_plan_sha256=expected_plan_sha256,joint_updates=0,automatic_retry=False)))
    try:
        config=validate_plan(plan,expected_plan_sha256,fixture_only=fixture_only)
        stage='source_preflight';before_source=source_signature(plan['source'])
        stage='runtime_preflight'
        actual_family='.'.join(torch.__version__.split('.')[:2])
        _need(actual_family==plan['runtime']['torch_version_family'],'actual Torch family differs from explicit recipe')
        torch.set_num_threads(1)
        runtime=dict(python=platform.python_version(),python_executable=sys.executable,
            torch=torch.__version__,torch_file=str(Path(torch.__file__).resolve()),
            safetensors=safetensors.__version__,safetensors_file=str(Path(safetensors.__file__).resolve()),
            execution_device='cpu',threads=torch.get_num_threads(),optimizer_created=False,
            source_model_loaded=False,feature_payload_loaded=False,fixture_only=fixture_only)
        _write_new(output/'runtime.json',_seal(runtime))
        stage='parent_preflight';weights,before_parent,records=parent_snapshot(plan,parent_root)
        _write_new(output/'parent-before.json',_seal(dict(parent=before_parent,source=before_source,
            original_kernel=plan['parent_kernel'],original_config=records['component']['specialist_config'],
            completed_language_cursor=records['component']['cursor'])))
        stage='reference_initialize';rng['caller_before']=cpu_rng
        torch.random.default_generator.manual_seed(plan['initialization']['reference_seed'])
        model=reference.FormationReferenceSpecialist(config).cpu()
        rng['reference_after_initialize']=torch.get_rng_state().clone()
        inherited=set(model._inherited_keys)
        fresh_reference=state_signatures({k:v for k,v in model.state_dict().items() if k not in inherited})
        _need(sum(model.state_dict()[k].numel() for k in inherited)==plan['expected_parent']['lexical_parameter_count'],
              'actual full lexical parameter count differs')
        stage='reference_parent_load'
        lexical=state_signatures(load_file(str(weights),device='cpu'))
        model.load_parent_safetensors(weights,parent_sha256=plan['parent']['weights']['file_sha256'],
            config_sha256=plan['expected_parent']['config_canonical_sha256'])
        rng['reference_after_parent_load']=torch.get_rng_state().clone()
        reference_state=state_signatures(model.state_dict())
        _need({k:reference_state[k] for k in inherited}==lexical and
              {k:v for k,v in reference_state.items() if k not in inherited}==fresh_reference,
              'reference migration changed inherited lexical or fresh reference tensors')
        stage='reference_export';refdir=output/'reference';refdir.mkdir()
        _save_new(refdir/'specialist.safetensors',model.state_dict())
        _write_new(refdir/'config.json',config.record())
        parent_provenance=dict(kind='untrained_fixture' if fixture_only else 'unqualified_parent',
            task=reference.TASK,producer_sha256=_digest(reference.__file__),
            weight_file_sha256=_digest(refdir/'specialist.safetensors'),
            config_file_sha256=_digest(refdir/'config.json'),
            config_canonical_sha256=canonical_sha256(config.record()),lexical_parent_binding=deepcopy(model.parent_binding))
        _write_new(refdir/'provenance.json',parent_provenance)
        _write_new(refdir/'state-signatures.json',_seal(dict(tensors=reference_state,
            fresh_reference_tensors=fresh_reference,lexical_tensors=lexical)))
        del model
        stage='joint_initialize';torch.random.default_generator.manual_seed(plan['initialization']['copy_seed'])
        model=joint.FormationJointLearning(config).cpu()
        rng['copy_after_initialize']=torch.get_rng_state().clone()
        reference_keys=set(model._reference_parent_keys)
        fresh_copy=state_signatures({k:v for k,v in model.state_dict().items() if k not in reference_keys})
        stage='reference_to_copy_load'
        model.load_reference_parent_safetensors(refdir/'specialist.safetensors',
            parent_sha256=parent_provenance['weight_file_sha256'],parent_config_path=refdir/'config.json',
            parent_config_file_sha256=parent_provenance['config_file_sha256'],parent_task=reference.TASK,
            parent_provenance=parent_provenance,parent_provenance_sha256=canonical_sha256(parent_provenance))
        rng['copy_after_parent_load']=torch.get_rng_state().clone()
        complete=state_signatures(model.state_dict())
        _need({k:complete[k] for k in reference_keys}==reference_state and
              {k:v for k,v in complete.items() if k not in reference_keys}==fresh_copy,
              'copy migration changed inherited reference or fresh copy tensors')
        stage='joint_export';jointdir=output/'joint';jointdir.mkdir()
        provenance=dict(kind='zero-joint-update-seed',purpose=plan['purpose'],parent_kernel=plan['parent_kernel'],
            original_parent_files=plan['parent'],language_parent_cursor=records['component']['cursor'],
            language_parent_component_sha256=records['component']['record_sha256'],
            reference_parent=parent_provenance,copy_migration=deepcopy(model.parent_binding),
            initialization=plan['initialization'],runtime=runtime,joint_updates=0,
            new_reference_heads_trained=False,new_copy_heads_trained=False,training_authorized=False,
            full_MFM_qualified=False)
        manifest=model.export_component(jointdir/'specialist.safetensors',jointdir/'config.json',jointdir/'component.json',
            component_id=plan['component_id'],provenance=provenance,provenance_sha256=canonical_sha256(provenance),
            source_pins=before_source,source_pins_sha256=canonical_sha256(before_source),evidence_scope='untrained_mechanics')
        _write_new(jointdir/'state-signatures.json',_seal(dict(tensors=complete,fresh_copy_tensors=fresh_copy)))
        del model
        stage='joint_reload';torch.random.default_generator.manual_seed(plan['initialization']['copy_seed'])
        reloaded=joint.FormationJointLearning(config).cpu()
        rng['reload_after_initialize']=torch.get_rng_state().clone()
        reloaded.load_component(jointdir/'specialist.safetensors',jointdir/'config.json',jointdir/'component.json',
            weight_sha256=manifest['weight_file_sha256'],config_sha256=manifest['config_file_sha256'],
            manifest_sha256=_digest(jointdir/'component.json'),expected_provenance_sha256=canonical_sha256(provenance),
            expected_source_pins=before_source,expected_evidence_scope='untrained_mechanics')
        loaded=state_signatures(reloaded.state_dict())
        _need(loaded==complete,'whole exported component reload changed actual tensors')
        rng['reload_after_load']=torch.get_rng_state().clone();del reloaded
        stage='closing_immutability';_,after_parent,_=parent_snapshot(plan,parent_root)
        after_source=source_signature(plan['source'])
        _need(before_parent==after_parent and before_source==after_source,'parent/source changed during actual build')
        result=dict(schema='mfm-native-joint-component-build-result-v1',
            status='native-joint-seed-built-reloaded-unqualified',plan_sha256=expected_plan_sha256,
            parent_before=before_parent,parent_after=after_parent,source_before=before_source,source_after=after_source,
            inherited_lexical_tensor_count=len(lexical),inherited_reference_tensor_count=len(reference_state),
            reloaded_tensor_count=len(loaded),reference_fresh_heads_preserved=True,copy_fresh_heads_preserved=True,
            joint_manifest_file_sha256=_digest(jointdir/'component.json'),joint_weight_file_sha256=manifest['weight_file_sha256'],
            reference_weight_file_sha256=parent_provenance['weight_file_sha256'],runtime=runtime,
            initialization=plan['initialization'],original_language_cursor=records['component']['cursor'],
            joint_updates=0,optimizer_created=False,Gemma_or_features_loaded=False,training_authorized=False,
            full_MFM_qualified=False,seconds=perf_counter()-started)
    except Exception as exc:
        _write_new(output/'failure.json',_seal(dict(schema='mfm-native-joint-component-build-failure-v1',
            failure_stage=stage,error_type=type(exc).__name__,error=str(exc),
            plan_sha256=expected_plan_sha256,joint_updates=0,automatic_retry=False,seconds=perf_counter()-started)))
        raise
    finally:
        torch.set_rng_state(cpu_rng);torch.set_num_threads(threads)
    rng['caller_restored']=torch.get_rng_state().clone()
    _save_new(output/'rng-states.safetensors',rng)
    result['cpu_rng_restored']=torch.equal(cpu_rng,torch.get_rng_state())
    result['cpu_rng_sha256']=_rng_pin()
    result['rng_states_file_sha256']=_digest(output/'rng-states.safetensors')
    _write_new(output/'result.json',_seal(result))
    return result


MIGRATION_SCHEMA='mfm-native-joint-source-migration-plan-v1'
NATIVE_PARENT_FILES={
    'weights':'c3900ff29e162279de88c2456987e774fb8397530d994aaf7569e4c6a5f6a556',
    'manifest':'f5b4ee06c797693436434f0fdc93c9a0b3a8eabcd49af9eef3e42afc2fac9a9c',
    'config':'022f027b588563d839c7bbcf5526b78970e719e00281dee50b86770b5f414193',
    'result':'5afae35e60a3ee521b451d18f69e0a0fc15853724663e7878c8c2f31ac21f875',
}
NATIVE_SOURCE_FILES={
    'capsule':'a7cc9fc28624ba4f0fd7d6eab1bca27b45c0b7345866944bd43932ae78fa2323',
    'files':'bc995ef25a8a1118ce2e9ca403368edba11fdce7c344bb151d87df04736a255b',
    'build_plan':'f58ba135bef9ff919fa81597f74199700ff9d1f9961597aa7ab15365fca0dfca',
}

# Executed only in a separate isolated interpreter. All local imports originate
# from the extracted original package, never the currently loaded model modules.
_COLD_LOAD = r'''
import json,sys
from pathlib import Path
request=json.loads(Path(sys.argv[1]).read_bytes())
root=Path(request['source_root']).resolve()
sys.path[:0]=[str(root/'src'),str(root)]
from scripts.mfm import build_native_joint_component as b
import torch
torch.set_num_threads(1)
plan=request['plan'];parent=Path(request['parent_root'])
old_plan=b.read_json_bytes(Path(request['old_plan']).read_bytes())
before=b.source_signature(old_plan['source'])
paths={};identities={}
for key,desc in plan['parent'].items():
 paths[key],identities[key]=b._file(parent,desc)
manifest=b.read_json_bytes(paths['manifest'].read_bytes())
cfg=b.SpecialistConfig(**plan['expected_parent']['config']);cfg.validate()
model=b.joint.FormationJointLearning(cfg).cpu()
model.load_component(paths['weights'],paths['config'],paths['manifest'],weight_sha256=plan['parent']['weights']['file_sha256'],config_sha256=plan['parent']['config']['file_sha256'],manifest_sha256=plan['parent']['manifest']['file_sha256'],expected_provenance_sha256=plan['expected_parent']['provenance_sha256'],expected_source_pins=manifest['source_pins'],expected_evidence_scope='untrained_mechanics')
signatures=b.state_signatures(model.state_dict())
b._need(len(signatures)==plan['expected_parent']['tensor_count'] and sum(v.numel() for v in model.state_dict().values())==plan['expected_parent']['parameter_count'],'complete historical state count differs')
b._need(b.canonical_sha256(manifest['source_pins'])==plan['expected_parent']['source_pins_sha256'],'external historical source pin differs')
after=b.source_signature(old_plan['source'])
b._need(before==after and all(b._file(parent,desc)[1]==identities[key] for key,desc in plan['parent'].items()),'historical parent or source changed during cold load')
b._write_new(Path(request['report']),b._seal(dict(schema='mfm-native-joint-original-cold-load-v1',isolated_process=True,pid=__import__('os').getpid(),python_executable=sys.executable,torch=torch.__version__,interface_pins=b.joint._interface_pins(),source_before=before,source_after=after,parent_before=identities,parent_after=identities,tensors=signatures,tensor_count=len(signatures),joint_updates=0,optimizer_created=False,source_model_loaded=False)))
'''


def validate_migration_plan(plan,expected,*,fixture_only=False):
    _record(plan,MIGRATION_SCHEMA)
    _need(plan['record_sha256']==_pin(expected) and set(plan)=={'schema','purpose','component_id',
        'parent_kernel','parent','parent_source','expected_parent','source','runtime','joint_updates',
        'training_authorized','full_MFM_qualified','record_sha256'},'externally pinned migration plan differs')
    _need(type(plan['joint_updates']) is int and plan['joint_updates']==0 and
        plan['training_authorized'] is False and plan['full_MFM_qualified'] is False,
        'source migration cannot update weights or grant rights')
    _need(type(plan['parent']) is dict and set(plan['parent'])==set(NATIVE_PARENT_FILES) and
        type(plan['parent_source']) is dict and set(plan['parent_source'])==set(NATIVE_SOURCE_FILES),
        'complete original component and source package required')
    expected_parent=plan['expected_parent']
    _need(type(expected_parent) is dict and set(expected_parent)=={'config','cursor','tensor_count',
        'parameter_count','provenance_sha256','source_pins_sha256'},'complete migration expectations required')
    cfg=SpecialistConfig(**expected_parent['config']);cfg.validate()
    _need(_same(cfg.record(),expected_parent['config']),'exact typed parent config required')
    _cursor(expected_parent['cursor'])
    for name in ('tensor_count','parameter_count'):
        _need(type(expected_parent[name]) is int and expected_parent[name]>0,'positive complete state counts required')
    _pin(expected_parent['provenance_sha256']);_pin(expected_parent['source_pins_sha256'])
    runtime=plan['runtime']
    _need(type(runtime) is dict and set(runtime)=={'device','threads','torch_version_family','safetensors_required'} and runtime['device']=='cpu' and type(runtime['threads']) is int and runtime['threads']==1 and runtime['safetensors_required'] is True and runtime['torch_version_family']=='.'.join(torch.__version__.split('.')[:2]),'explicit actual CPU runtime required')
    if fixture_only:
        _need(plan['purpose']=='tiny-mechanics-fixture','fixture purpose differs')
    else:
        _need(plan['purpose']=='full-size-native-joint-source-migration' and _same(cfg.record(),S5_CONFIG) and
            _same(expected_parent['cursor'],S5_CURSOR) and expected_parent['tensor_count']==138 and expected_parent['parameter_count']==278395424 and
            _same(plan['parent_kernel'],dict(ref='mkrayanyan/rayan-mfm-native-joint-build-20261013',numeric_id=138478151,version=1,source_revision='273576d68d077f177593ac90004b9bcf4450df97f71874c3c155c1db9ee20966')) and
            all(plan['parent'][k]['file_sha256']==v for k,v in NATIVE_PARENT_FILES.items()) and
            all(plan['parent_source'][k]['file_sha256']==v for k,v in NATIVE_SOURCE_FILES.items()) and
            type(plan['parent']['weights']['bytes']) is int and plan['parent']['weights']['bytes']==1113595976 and runtime['torch_version_family']=='2.11',
            'production migration requires exact complete original138 state and package003')
    return cfg


def _migration_parent(plan,root):
    paths={};signature={}
    for key,desc in plan['parent'].items():paths[key],signature[key]=_file(root,desc)
    manifest=read_json_bytes(paths['manifest'].read_bytes());config=read_json_bytes(paths['config'].read_bytes())
    result=_record(read_json_bytes(paths['result'].read_bytes()),'mfm-native-joint-component-build-result-v1')
    expected=plan['expected_parent'];provenance=manifest['provenance']
    _need(_same(config,expected['config']) and manifest['config_file_sha256']==plan['parent']['config']['file_sha256'] and
        manifest['weight_file_sha256']==plan['parent']['weights']['file_sha256'] and manifest['provenance_sha256']==expected['provenance_sha256'] and
        canonical_sha256(provenance)==expected['provenance_sha256'] and canonical_sha256(manifest['source_pins'])==expected['source_pins_sha256'],
        'externally bound original component/config/provenance differs')
    _need(result['joint_manifest_file_sha256']==plan['parent']['manifest']['file_sha256'] and
        result['joint_weight_file_sha256']==plan['parent']['weights']['file_sha256'] and
        type(result['reloaded_tensor_count']) is int and result['reloaded_tensor_count']==expected['tensor_count'] and
        _same(_cursor(result['original_language_cursor']),expected['cursor']) and
        _same(_cursor(provenance['language_parent_cursor']),expected['cursor']) and
        _same(result['source_before'],result['source_after']) and _same(result['source_before'],manifest['source_pins']),
        'original complete build result/cursor/source join differs')
    for value in (result,provenance):
        _need(type(value.get('joint_updates')) is int and value['joint_updates']==0 and value.get('training_authorized') is False and value.get('full_MFM_qualified') is False,'original zero-update unqualified lineage required')
    _need(result['optimizer_created'] is False and result['Gemma_or_features_loaded'] is False and
        provenance['new_reference_heads_trained'] is False and provenance['new_copy_heads_trained'] is False,
        'original head/training status differs')
    return paths,signature,manifest,result


def _migration_source_package(plan,root):
    paths={};signature={}
    for key,desc in plan['parent_source'].items():paths[key],signature[key]=_file(root,desc)
    files=read_json_bytes(paths['files'].read_bytes());old_plan=_record(read_json_bytes(paths['build_plan'].read_bytes()),SCHEMA)
    _need(type(files) is dict and files,'whole physical original source roster required')
    for name,desc in files.items():
        relative=PurePosixPath(name)
        _need(type(name) is str and relative.as_posix()==name and not relative.is_absolute() and
            not any(p in ('..','.','') for p in relative.parts) and '\\' not in name and ':' not in name and
            relative.parts[0] in ('src','scripts') and type(desc) is dict and set(desc)=={'bytes','file_sha256'} and
            type(desc['bytes']) is int and desc['bytes']>=0,'ordinary original capsule member required')
        _pin(desc['file_sha256'])
    expected={**old_plan['source']['interface_pins'],'scripts/mfm/build_native_joint_component.py':old_plan['source']['builder_file_sha256']}
    _need(all(name in files and files[name]['file_sha256']==pin for name,pin in expected.items()),'original reviewed closure missing from capsule')
    return paths,signature,files,old_plan


def _check_extracted(root,files):
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    _need(actual==set(files),'complete extracted original source roster differs')
    for name,desc in files.items():_file(root,dict(path=name,**desc))


def _extract_original_source(capsule,files,output):
    output.mkdir(mode=0o700,exist_ok=False)
    with tarfile.open(capsule,'r:xz') as archive:
        members=archive.getmembers();names=[m.name for m in members]
        _need(len(names)==len(set(names)) and set(names)==set(files),'exact original capsule member roster differs')
        for member in members:
            expected=files[member.name]
            _need(member.isfile() and member.size==expected['bytes'],'ordinary original capsule body required')
            source=archive.extractfile(member);_need(source is not None,'original capsule body missing')
            with source:body=source.read()
            _need(sha256(body).hexdigest()==expected['file_sha256'],'original capsule body hash differs')
            target=output.joinpath(*PurePosixPath(member.name).parts);target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            with target.open('xb') as stream:stream.write(body)
    _check_extracted(output,files)


def migrate_component(plan,*,expected_plan_sha256,parent_root,source_package_root,output_dir,fixture_only=False):
    """Explicit whole-state lineage transition; both strict component gates remain."""
    output=Path(output_dir)
    _need(not any(output.resolve().is_relative_to(Path(root).resolve()) for root in (parent_root,source_package_root)),
        'migration output cannot modify readonly parent/source package')
    output.mkdir(mode=0o700,parents=False,exist_ok=False)
    stage='migration_plan';started=perf_counter();cpu_rng=torch.get_rng_state().clone();threads=torch.get_num_threads()
    _write_new(output/'intent.json',_seal(dict(schema='mfm-native-joint-source-migration-intent-v1',expected_plan_sha256=expected_plan_sha256,joint_updates=0,automatic_retry=False)))
    try:
        cfg=validate_migration_plan(plan,expected_plan_sha256,fixture_only=fixture_only)
        torch.set_num_threads(1);stage='current_source_preflight';before_source=source_signature(plan['source'])
        stage='complete_parent_preflight';paths,before_parent,parent_manifest,parent_result=_migration_parent(plan,parent_root)
        stage='original_source_package';old_paths,before_package,files,old_plan=_migration_source_package(plan,source_package_root)
        _need(_same(old_plan['source']['interface_pins'],parent_manifest['interface_pins']),'original package and parent interface closure differ')
        original_root=output/'original-source';_extract_original_source(old_paths['capsule'],files,original_root)
        stage='isolated_original_cold_load';request=output/'cold-load-request.json';report=output/'old-cold-load.json'
        _write_new(request,dict(source_root=str(original_root.resolve()),parent_root=str(Path(parent_root).resolve()),old_plan=str(old_paths['build_plan'].resolve()),plan=plan,report=str(report.resolve())))
        env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('KMP_DUPLICATE_LIB_OK',None);env['PYTHONNOUSERSITE']='1'
        with (output/'old-cold-load.log').open('xb') as log:
            executed=subprocess.run([sys.executable,'-I','-B','-c',_COLD_LOAD,str(request.resolve())],cwd=original_root,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
        _write_new(output/'cold-load-execution.json',_seal(dict(returncode=executed.returncode,bootstrap_sha256=sha256(_COLD_LOAD.encode()).hexdigest(),isolated_process=True,custom_timeout=False,automatic_retry=False)))
        _need(executed.returncode==0,'original strict cold load refused; original log retained')
        cold=_record(read_json_bytes(report.read_bytes()),'mfm-native-joint-original-cold-load-v1')
        _need(cold['isolated_process'] is True and cold['pid']!=os.getpid() and _same(cold['interface_pins'],parent_manifest['interface_pins']) and
            cold['parent_before']==before_parent and cold['parent_after']==before_parent and cold['source_before']==cold['source_after'],
            'isolated original load identity/bracket differs')
        _check_extracted(original_root,files)
        stage='complete_current_state_transfer';state=load_file(str(paths['weights']),device='cpu');signatures=state_signatures(state)
        _need(signatures==cold['tensors'],'original loaded tensor bytes differ from independent cold load')
        model=joint.FormationJointLearning(cfg).cpu();current=model.state_dict()
        _need(set(state)==set(current) and all(state[k].shape==current[k].shape and current[k].dtype==torch.float32 for k in state),'complete current tensor key/shape/dtype differs')
        _need(sum(v.numel() for v in state.values())==plan['expected_parent']['parameter_count'],'complete current parameter count differs')
        model.load_state_dict(state,strict=True)
        _need(state_signatures(model.state_dict())==signatures,'strict whole-state migration changed tensors')
        del state,current
        stage='current_component_export';jointdir=output/'joint';jointdir.mkdir()
        provenance=dict(kind='zero-update-whole-joint-source-lineage-migration',parent_kernel=plan['parent_kernel'],
            parent_files=plan['parent'],parent_source_package=plan['parent_source'],original_provenance=deepcopy(parent_manifest['provenance']),
            original_component_id=parent_manifest['component_id'],original_evidence_scope=parent_manifest['evidence_scope'],
            language_parent_cursor=deepcopy(plan['expected_parent']['cursor']),original_source_pins=deepcopy(parent_manifest['source_pins']),
            original_interface_pins=deepcopy(parent_manifest['interface_pins']),cold_load_file_sha256=_digest(report),
            complete_parent_state_preserved=True,joint_updates=0,new_reference_heads_trained=False,new_copy_heads_trained=False,
            optimizer_created=False,training_authorized=False,full_MFM_qualified=False)
        manifest=model.export_component(jointdir/'specialist.safetensors',jointdir/'config.json',jointdir/'component.json',component_id=plan['component_id'],
            provenance=provenance,provenance_sha256=canonical_sha256(provenance),source_pins=before_source,source_pins_sha256=canonical_sha256(before_source),evidence_scope=parent_manifest['evidence_scope'])
        del model
        stage='current_strict_reload';reloaded=joint.FormationJointLearning(cfg).cpu()
        reloaded.load_component(jointdir/'specialist.safetensors',jointdir/'config.json',jointdir/'component.json',weight_sha256=manifest['weight_file_sha256'],config_sha256=manifest['config_file_sha256'],manifest_sha256=_digest(jointdir/'component.json'),expected_provenance_sha256=canonical_sha256(provenance),expected_source_pins=before_source,expected_evidence_scope=parent_manifest['evidence_scope'])
        loaded=state_signatures(reloaded.state_dict());_need(loaded==signatures,'current strict reload changed complete original tensors');del reloaded
        _write_new(jointdir/'state-signatures.json',_seal(dict(tensors=loaded)))
        stage='closing_immutability';_,after_parent,_,_=_migration_parent(plan,parent_root);_,after_package,_,_=_migration_source_package(plan,source_package_root)
        after_source=source_signature(plan['source']);_check_extracted(original_root,files)
        _need(before_parent==after_parent and before_package==after_package and before_source==after_source,'parent or old/current source changed during migration')
        result=dict(schema='mfm-native-joint-source-migration-result-v1',status='complete-joint-source-lineage-migrated-reloaded-unqualified',
            plan_sha256=expected_plan_sha256,parent_before=before_parent,parent_after=after_parent,source_before=before_source,source_after=after_source,
            parent_source_before=before_package,parent_source_after=after_package,old_cold_load=cold,complete_parent_state_preserved=True,
            reloaded_tensor_count=len(loaded),joint_weight_file_sha256=manifest['weight_file_sha256'],joint_manifest_file_sha256=_digest(jointdir/'component.json'),
            original_language_cursor=deepcopy(plan['expected_parent']['cursor']),joint_updates=0,optimizer_created=False,Gemma_or_features_loaded=False,
            training_authorized=False,full_MFM_qualified=False,fixture_only=fixture_only,seconds=perf_counter()-started)
    except Exception as exc:
        _write_new(output/'failure.json',_seal(dict(schema='mfm-native-joint-source-migration-failure-v1',failure_stage=stage,error_type=type(exc).__name__,error=str(exc),plan_sha256=expected_plan_sha256,joint_updates=0,automatic_retry=False)))
        raise
    finally:
        torch.set_rng_state(cpu_rng);torch.set_num_threads(threads)
    _save_new(output/'rng-states.safetensors',{'caller_before':cpu_rng,'caller_restored':torch.get_rng_state().clone()})
    result['cpu_rng_restored']=torch.equal(cpu_rng,torch.get_rng_state());_write_new(output/'result.json',_seal(result));return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--plan-file-sha256',required=True)
    parser.add_argument('--parent-root',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--source-package-root',type=Path)
    args=parser.parse_args(argv)
    _need(_digest(args.plan)==_pin(args.plan_file_sha256),'actual externally pinned plan FILE differs')
    plan=read_json_bytes(args.plan.read_bytes())
    if plan.get('schema')==MIGRATION_SCHEMA:
        _need(args.source_package_root is not None,'explicit original source package required')
        result=migrate_component(plan,expected_plan_sha256=plan['record_sha256'],parent_root=args.parent_root,
            source_package_root=args.source_package_root,output_dir=args.output_dir)
        print(json.dumps(dict(status=result['status'],joint_weight_file_sha256=result['joint_weight_file_sha256'],
            reloaded_tensor_count=result['reloaded_tensor_count'],joint_updates=0)));return
    _need(args.source_package_root is None,'original source package only belongs to explicit migration mode')
    result=build_component(plan,expected_plan_sha256=plan['record_sha256'],parent_root=args.parent_root,output_dir=args.output_dir)
    print(json.dumps(dict(status=result['status'],joint_weight_file_sha256=result['joint_weight_file_sha256'],
        inherited_lexical_tensor_count=result['inherited_lexical_tensor_count'],
        inherited_reference_tensor_count=result['inherited_reference_tensor_count'],
        reloaded_tensor_count=result['reloaded_tensor_count'],joint_updates=0)))


if __name__=='__main__':
    main()
