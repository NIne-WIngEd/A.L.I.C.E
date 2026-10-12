"""Full-roster fresh public conditional MFM foundation construction.

This is an unqualified public sequence-learning artifact, not formation or
owner-independent qualification. No foundation model weights are loaded.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
from time import perf_counter
from uuid import uuid4

from cognitive_kernel.canonical import canonical_json_bytes, canonical_sha256, require_sha256
from cognitive_kernel.formation_conditional_task import bind_conditional_task
from cognitive_kernel.formation_decision_specialist_v16 import FormationDecisionSpecialist, INITIALIZATION
from cognitive_kernel.formation_v1_specialist import SpecialistConfig
from . import public_conditional_training as reader
from . import train_v1_formation_specialist as shared
from . import public_checkpoint_history as checkpoint_history
from .train_v16_cached_specialist import _cpu_tree, compare_tree

TASK = 'conditional_meaning_to_text'
RUN_SCHEMA = 'mfm-public-conditional-foundation-run-v1'
CURRICULUM_SCHEMA = 'mfm-public-conditional-foundation-curriculum-v1'
SEED_SCHEMA = 'mfm-public-conditional-foundation-seed-v1'
CHECKPOINT_SCHEMA = 'mfm-public-conditional-foundation-checkpoint-v1'
COMPONENT_SCHEMA = 'mfm-public-conditional-foundation-component-v1'
OBJECTIVE = 'sequence-mean-including-eos/reference-mean/semantic-group-mean-v1'
SAMPLER = 'python-local-random-seed-plus-epoch/encoded-train-order-v1'


def require_full_material(prepared):
    counts = (len(prepared.pairs), len(prepared.train), len(prepared.development), len(prepared.sources),
              len({p['semantic_input_sha256'] for p in prepared.train}),
              len({p['semantic_input_sha256'] for p in prepared.development}))
    if counts != (40868, 36597, 4271, 4862, 4356, 506):
        raise ValueError('complete public production roster differs')
    if prepared.census['base_hidden_size'] != 3840 or prepared.census['foundation_vocabulary'] != 262144:
        raise ValueError('complete public production geometry differs')
    if max(len(p['source_ids']) for p in prepared.pairs) != 71 or max(len(p['labels']) for p in prepared.pairs) != 84:
        raise ValueError('complete public observed sequence maxima differ')
    reader.case_weights(prepared.pairs, split='train')


def optimizer_step(model, optimizer, batch, *, prepared):
    import torch
    if getattr(model, 'task', None) != TASK or not hasattr(model, 'specialist'):
        raise ValueError('explicit public conditional task binding is required')
    batch = tuple(batch)
    weights = reader.case_weights(prepared.pairs, split='train')
    coefficients = reader.batch_coefficients(batch, weights, total_train_cases=len(prepared.train))
    if any(prepared.pair_by_case.get(p['case_id']) != p for p in batch):
        raise ValueError('public training pair membership differs')
    specialist = model.specialist
    device = next(specialist.parameters()).device
    if any(p.dtype != torch.float32 for p in specialist.parameters()):
        raise ValueError('public model arithmetic must be FP32')
    specialist.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for pair in batch:
        _, loss = model.forward(**reader.payload(prepared, pair, device=device))
        if not torch.isfinite(loss):
            raise ValueError('public sequence loss is nonfinite')
        (loss * coefficients[pair['case_id']]).backward()
        losses.append(float(loss.detach().cpu()))
    norm = torch.nn.utils.clip_grad_norm_(specialist.parameters(), 1., error_if_nonfinite=True)
    if not torch.isfinite(norm) or norm.item() <= 0:
        raise ValueError('public gradient must be finite and nonzero')
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if any(not torch.isfinite(p).all() for p in specialist.parameters()):
        raise ValueError('public AdamW produced nonfinite parameters')
    return {'objective_loss': math.fsum(coefficients[p['case_id']] * loss for p, loss in zip(batch, losses)),
            'sequence_mean_loss': math.fsum(losses) / len(losses), 'gradient_norm': norm.item(),
            'case_ids': [p['case_id'] for p in batch], 'label_tokens': sum(len(p['labels']) for p in batch)}


def _cursor(cursor, *, train_cases, batch_size, epochs):
    if set(cursor) != {'epoch', 'next_case', 'step'} or any(type(v) is not int or v < 0 for v in cursor.values()):
        raise ValueError('public checkpoint cursor fields differ')
    epoch, offset, step = (cursor[k] for k in ('epoch', 'next_case', 'step'))
    steps = math.ceil(train_cases / batch_size)
    if epoch > epochs or offset >= train_cases or offset % batch_size or (epoch == epochs and offset) or step != epoch * steps + offset // batch_size:
        raise ValueError('public checkpoint cursor is not a complete optimizer boundary')


def next_batch(prepared, *, seed, epochs, batch_size, cursor):
    if type(batch_size) is not int or batch_size < 1 or type(epochs) is not int or epochs < 1:
        raise ValueError('public complete epoch/batch allocation is invalid')
    _cursor(cursor, train_cases=len(prepared.train), batch_size=batch_size, epochs=epochs)
    if cursor['epoch'] == epochs:
        return (), dict(cursor)
    order = reader.epoch_order(prepared, seed=seed, epoch=cursor['epoch'])
    end = min(cursor['next_case'] + batch_size, len(order))
    batch = order[cursor['next_case']:end]
    following = {'epoch': cursor['epoch'] + int(end == len(order)),
                 'next_case': 0 if end == len(order) else end, 'step': cursor['step'] + 1}
    return batch, following


def _safe_root(root):
    root = Path(root)
    if any(p.is_symlink() or getattr(p, 'is_junction', lambda: False)() for p in (root, *root.parents)):
        raise ValueError('public run path links are forbidden')
    return root.resolve(strict=True)


def _safe_child(root, name):
    path = _safe_root(root) / name
    if path.is_symlink() or (path.exists() and getattr(path, 'is_junction', lambda: False)()):
        raise ValueError('public artifact links are forbidden')
    return path


def save_checkpoint(output_dir, specialist, optimizer, *, binding, cursor):
    """Public container using v1.6 parameter/Adam/Torch+CUDA RNG mechanics.

    Its schema and seed meaning are public-specific. Unlike the old finally
    cleanup, unfinished scratch remains as abrupt-exit evidence in this run.
    """
    import torch
    from safetensors.torch import save_file
    _cursor(cursor, train_cases=binding['train_cases'], batch_size=binding['batch_size'], epochs=binding['epochs'])
    _adam_cursor(optimizer.state_dict(), cursor)
    root = _safe_root(output_dir)
    final = _safe_child(root, f"checkpoint-{cursor['step']:08d}")
    if final.exists():
        raise ValueError('completed public checkpoint cannot be overwritten')
    staging = root / f'.checkpoint-{uuid4().hex}'
    staging.mkdir()
    save_file({k: v.detach().cpu().contiguous() for k, v in specialist.state_dict().items()}, str(staging / 'specialist.safetensors'))
    torch.save({'optimizer': optimizer.state_dict(), 'torch_rng_state': torch.random.get_rng_state(),
                'cuda_rng_states': torch.cuda.get_rng_state_all(), 'binding': binding, 'cursor': cursor}, staging / 'optimizer.pt')
    shared._write_new(staging / 'checkpoint.json', {'schema': CHECKPOINT_SCHEMA, 'binding': binding, 'cursor': cursor,
                      'specialist_sha256': shared._digest(staging / 'specialist.safetensors'),
                      'optimizer_sha256': shared._digest(staging / 'optimizer.pt')})
    staging.rename(final)
    return final


def verify_checkpoint(path, *, binding):
    path = _safe_root(path)
    if set(p.name for p in path.iterdir()) != {'specialist.safetensors', 'optimizer.pt', 'checkpoint.json'}:
        raise ValueError('public checkpoint files differ or are partial')
    for name in ('specialist.safetensors', 'optimizer.pt', 'checkpoint.json'):
        _safe_child(path, name)
    receipt = shared._read_sealed(path / 'checkpoint.json', CHECKPOINT_SCHEMA)
    if receipt['binding'] != binding or any(receipt[field] != shared._digest(path / name) for field, name in
            [('specialist_sha256', 'specialist.safetensors'), ('optimizer_sha256', 'optimizer.pt')]):
        raise ValueError('public checkpoint immutable binding or tensor digest differs')
    _cursor(receipt['cursor'], train_cases=binding['train_cases'], batch_size=binding['batch_size'], epochs=binding['epochs'])
    if path.name != f"checkpoint-{receipt['cursor']['step']:08d}":
        raise ValueError('public checkpoint name/cursor differs')
    return receipt


def resume_checkpoint(checkpoint_dir, output_dir, specialist, optimizer, *, binding, history=None):
    import torch
    from safetensors.torch import load_file
    root = _safe_root(output_dir)
    path = _safe_root(checkpoint_dir)
    if history is not None:
        receipt = history.authorize_latest(path, binding)
    elif path.parent != root or re.fullmatch(r'checkpoint-[0-9]{8}', path.name) is None:
        raise ValueError('public checkpoint is foreign or partial')
    else:
        complete = []
        for p in root.iterdir():
            if p.name.startswith('checkpoint-'):
                if re.fullmatch(r'checkpoint-[0-9]{8}', p.name) is None:
                    raise ValueError('unexpected public checkpoint path')
                complete.append((verify_checkpoint(p, binding=binding)['cursor']['step'], p))
        if not complete or max(complete)[1] != path:
            raise ValueError('resume requires latest complete public checkpoint')
        receipt = verify_checkpoint(path, binding=binding)
    state = torch.load(path / 'optimizer.pt', map_location='cpu', weights_only=True)
    if state['binding'] != binding or state['cursor'] != receipt['cursor']:
        raise ValueError('public optimizer container cursor or binding differs')
    _adam_cursor(state['optimizer'], receipt['cursor'])
    current_groups = optimizer.state_dict()['param_groups']
    saved_groups = state['optimizer']['param_groups']
    if len(current_groups) != len(saved_groups) or any(
            saved['params'] != current['params'] or
            {k: v for k, v in saved.items() if k != 'params'} !=
            {k: v for k, v in current.items() if k != 'params'}
            for saved, current in zip(saved_groups, current_groups)):
        raise ValueError('public optimizer exact recipe differs from bound current AdamW')
    specialist.load_state_dict(load_file(str(path / 'specialist.safetensors')), strict=True)
    optimizer.load_state_dict(state['optimizer'])
    torch.random.set_rng_state(state['torch_rng_state'])
    torch.cuda.set_rng_state_all(state['cuda_rng_states'])
    return dict(receipt['cursor'])


def _adam_cursor(state, cursor):
    """The serialized Adam counters must describe the actual completed updates."""
    import torch
    expected = cursor['step']
    parameters = {p for group in state['param_groups'] for p in group['params']}
    if (expected == 0 and state['state']) or (expected > 0 and set(state['state']) != parameters):
        raise ValueError('public Adam state and completed cursor differ')
    for row in state['state'].values():
        step = row.get('step')
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or not torch.isfinite(step) or step.item() != expected:
            raise ValueError('public Adam update counters and cursor differ')


METADATA_NAMES = ('config.json', 'processor_config.json', 'tokenizer.json', 'tokenizer_config.json',
                  'generation_config.json', 'README.md', '.gitattributes')


def verify_tokenizer_metadata(directory, receipt_path, expected_clone_sha256, census):
    """Authenticate original nonweight assets without opening any base weight."""
    root = _safe_root(directory)
    receipt_path = Path(receipt_path)
    if receipt_path.is_symlink():
        raise ValueError('clone receipt link is forbidden')
    receipt = reader.features._json(receipt_path.read_bytes())
    require_sha256(expected_clone_sha256, 'expected clone receipt')
    if receipt.get('schema') != 'alice-gemma4-v1-clone-v1' or receipt.get('role') != 'mfm' or \
            receipt.get('receipt_sha256') != expected_clone_sha256 or \
            canonical_sha256({k: v for k, v in receipt.items() if k != 'receipt_sha256'}) != expected_clone_sha256 or \
            census['prepared_clone_receipt_sha256'] != expected_clone_sha256:
        raise ValueError('public tokenizer original clone custody differs')
    files = {row['path']: row for row in receipt['files']}
    if len(files) != len(receipt['files']) or files['model.safetensors']['sha256'] != census['prepared_base_weight_sha256']:
        raise ValueError('public tokenizer original weight ancestry differs')
    # Processor auto-discovery must not consume arbitrary added code/assets.
    if set(p.name for p in root.iterdir()) != set(METADATA_NAMES):
        raise ValueError('public metadata-only snapshot asset roster differs')
    assets = []
    for name in METADATA_NAMES:
        path = _safe_child(root, name)
        row = files[name]
        if not path.is_file() or path.stat().st_size != row['size'] or shared._digest(path) != row['sha256']:
            raise ValueError('public tokenizer metadata asset custody differs')
        assets.append(row)
    return {'clone_receipt_sha256': expected_clone_sha256, 'assets': assets,
            'asset_roster_sha256': canonical_sha256(assets), 'base_weights_opened': False,
            'meaning': 'original pinned metadata subset; complete clone weight verification not performed here'}


def _memory(device):
    import torch
    result = {'host_peak_rss_bytes': None}
    if sys.platform != 'win32':
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['host_peak_rss_bytes'] = int(peak if sys.platform == 'darwin' else peak * 1024)
    else:
        import importlib.util
        if importlib.util.find_spec('psutil'):
            import psutil
            info = psutil.Process().memory_info()
            result['host_peak_rss_bytes'] = getattr(info, 'peak_wset', None)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                      cuda_total_bytes=torch.cuda.get_device_properties(device).total_memory)
    return result


def measure_construction(view, optimizer, prepared, *, output_dir, binding, seed):
    """Real construction controls, restored before any prescribed fit updates.

    Lower-level fixtures can exercise mechanics, never establish full resource
    fit. Production calls only after require_full_material and exact config.
    """
    import torch
    root = _safe_root(output_dir) / 'construction-controls'
    root.mkdir()
    initial = {'epoch': 0, 'next_case': 0, 'step': 0}
    seed_checkpoint = save_checkpoint(root, view.specialist, optimizer, binding=binding, cursor=initial)
    device = next(view.parameters()).device
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    source = max(prepared.train, key=lambda p: len(p['source_ids']))
    target = max(prepared.train, key=lambda p: len(p['labels']))
    joint = max(prepared.train, key=lambda p: len(p['source_ids']) + len(p['labels']))
    measurements = []
    for role, pair in [('longest-source', source), ('longest-target', target), ('longest-joint', joint)]:
        peaks = {}
        hook = optimizer.register_step_pre_hook(lambda *unused: peaks.update(backward=_memory(device)))
        started = perf_counter()
        try:
            metrics = optimizer_step(view, optimizer, [pair], prepared=prepared)
        finally:
            hook.remove()
        seconds = perf_counter() - started
        peaks['adam'] = _memory(device)
        measurements.append({'role': role, 'case_id': pair['case_id'], 'source_tokens': len(pair['source_ids']),
                             'label_tokens': len(pair['labels']), 'seconds': seconds,
                             'label_tokens_per_second': len(pair['labels']) / seconds, 'peaks': peaks,
                             'objective_loss': metrics['objective_loss']})
    # Restore the persisted seed/empty optimizer before replaying actual sampler
    # boundaries. Control checkpoint directories never share fitted lineage.
    resume_checkpoint(seed_checkpoint, root, view.specialist, optimizer, binding=binding)
    batch, cursor = next_batch(prepared, seed=seed, epochs=binding['epochs'], batch_size=binding['batch_size'], cursor=initial)
    batch_started = perf_counter()
    batch_metrics = optimizer_step(view, optimizer, batch, prepared=prepared)
    batch_seconds = perf_counter() - batch_started
    started = perf_counter()
    checkpoint = save_checkpoint(root, view.specialist, optimizer, binding=binding, cursor=cursor)
    io_seconds = perf_counter() - started
    batch2, following = next_batch(prepared, seed=seed, epochs=binding['epochs'], batch_size=binding['batch_size'], cursor=cursor)
    if not batch2 or [p['case_id'] for p in batch2] == [p['case_id'] for p in batch]:
        raise ValueError('construction replay requires a different next weighted batch')
    replay_started = perf_counter()
    expected = optimizer_step(view, optimizer, batch2, prepared=prepared)
    next_batch_seconds = perf_counter() - replay_started
    expected_model = _cpu_tree(view.state_dict())
    expected_adam = _cpu_tree(optimizer.state_dict())
    expected_rng = torch.random.get_rng_state().clone()
    expected_cuda = _cpu_tree(torch.cuda.get_rng_state_all())
    restored = resume_checkpoint(checkpoint, root, view.specialist, optimizer, binding=binding)
    replay, replay_cursor = next_batch(prepared, seed=seed, epochs=binding['epochs'], batch_size=binding['batch_size'], cursor=restored)
    actual = optimizer_step(view, optimizer, replay, prepared=prepared)
    if actual != expected or replay_cursor != following or replay != batch2:
        raise ValueError('actual construction continuation loss/order differs')
    difference = max(compare_tree(view.state_dict(), expected_model, atol=0, rtol=0),
                     compare_tree(optimizer.state_dict(), expected_adam, atol=0, rtol=0))
    compare_tree(torch.random.get_rng_state(), expected_rng, atol=0, rtol=0)
    compare_tree(_cpu_tree(torch.cuda.get_rng_state_all()), expected_cuda, atol=0, rtol=0)
    replay_peaks = _memory(device)
    del expected_model, expected_adam
    # Latest-checkpoint rule applies to normal continuation; this intentional
    # construction rollback uses authenticated step-zero container directly.
    _restore_control_seed(seed_checkpoint, view.specialist, optimizer, binding)
    return {'schema': 'mfm-public-conditional-foundation-construction-v1', 'measurements': measurements,
            'checkpoint_seconds': io_seconds, 'checkpoint_bytes': sum(p.stat().st_size for p in checkpoint.iterdir()),
            'allocation_batch': {'case_ids': batch_metrics['case_ids'], 'seconds': batch_seconds,
                                 'label_tokens': batch_metrics['label_tokens'],
                                 'label_tokens_per_second': batch_metrics['label_tokens'] / batch_seconds},
            'different_next_batch_seconds': next_batch_seconds, 'replay_peaks': replay_peaks,
            'replay_next_case_ids': [p['case_id'] for p in replay], 'replay_max_difference': difference,
            'control_updates_restored': True, 'control_checkpoint_sha256': verify_checkpoint(checkpoint, binding=binding)['record_sha256'],
            'full_material': len(prepared.pairs) == 40868, 'parameter_count': sum(p.numel() for p in view.parameters()),
            'resource_fit_claim': 'observed construction only; allocation/learning budget selection remains external'}


def _restore_control_seed(path, specialist, optimizer, binding):
    import torch
    from safetensors.torch import load_file
    verify_checkpoint(path, binding=binding)
    state = torch.load(path / 'optimizer.pt', map_location='cpu', weights_only=True)
    if state['binding'] != binding or state['cursor'] != {'epoch': 0, 'next_case': 0, 'step': 0} or state['optimizer']['state']:
        raise ValueError('construction initial seed/empty Adam container differs')
    specialist.load_state_dict(load_file(str(path / 'specialist.safetensors')), strict=True)
    optimizer.load_state_dict(state['optimizer'])
    torch.random.set_rng_state(state['torch_rng_state'])
    torch.cuda.set_rng_state_all(state['cuda_rng_states'])


def _generation(view, prepared, tokenizer, *, cap):
    """FP32 free generation and changed-source control, separate from likelihood."""
    import torch
    view.eval()
    device = next(view.parameters()).device
    selected = [prepared.development[0]]
    changed = next(p for p in prepared.development if p['source_ids'] != selected[0]['source_ids'])
    selected.append(changed)
    results = []
    with torch.inference_mode():
        for pair in selected:
            data = reader.payload(prepared, pair, device=device)
            _, likelihood = view.forward(**data)
            ids, ended, current = [], False, [2]
            for _ in range(cap):
                logits = view.next_token_logits(base_states=data['base_states'], source_mask=data['source_mask'],
                                                input_ids=torch.tensor([current], device=device))
                token = int(logits[0].argmax().item())
                if token == 1:
                    ended = True
                    break
                ids.append(token)
                current.append(token)
            results.append({'case_id': pair['case_id'], 'free_ids': ids, 'eos': ended,
                            'free_text': tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                            'teacher_sequence_mean_loss': float(likelihood.cpu())})
    return {'cases': results, 'source_change_changed_free_ids': results[0]['free_ids'] != results[1]['free_ids'],
            'decode_cap': cap, 'selection_used': False, 'qualification': 'unqualified'}


def _seed_control(root, model, run_digest, config):
    from safetensors.torch import save_file
    path = _safe_child(root, 'seed-control.safetensors')
    if path.exists():
        raise ValueError('public seeded control cannot be overwritten')
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(path))
    return shared._write_new(root / 'seed-control.json', {'schema': SEED_SCHEMA,
        'run_sha256': run_digest, 'specialist_config': config.record(), 'initialization': INITIALIZATION,
        'specialist_sha256': shared._digest(path), 'optimizer_steps': 0,
        'meaning': 'same-seed untrained fresh public conditional foundation control'})


def _verify_seed(root, run_digest, config):
    receipt = shared._read_sealed(_safe_child(root, 'seed-control.json'), SEED_SCHEMA)
    if receipt['run_sha256'] != run_digest or receipt['specialist_config'] != config.record() or \
            receipt['initialization'] != INITIALIZATION or receipt['optimizer_steps'] != 0 or \
            receipt['specialist_sha256'] != shared._digest(_safe_child(root, 'seed-control.safetensors')):
        raise ValueError('public run seed control binding differs')
    return receipt


def _validate_run_paths(root):
    allowed = {'worker.lock', 'run.json', 'curriculum.json', 'seed-control.json', 'seed-control.safetensors',
               'seed-generation.json', 'construction.json', 'construction-controls', 'component', 'session.json'}
    for path in root.iterdir():
        _safe_child(root, path.name)
        if path.name not in allowed and re.fullmatch(r'(checkpoint-[0-9]{8}|history-[0-9]{8}\.json|\.checkpoint-[A-Za-z0-9]+|\.record-[a-f0-9]+|execution-[a-f0-9]+\.json|\.component-[a-f0-9]+)', path.name) is None:
            raise ValueError('unexpected path in owned public run')
        if path.is_dir():
            for item in path.rglob('*'):
                if item.is_symlink() or getattr(item, 'is_junction', lambda: False)():
                    raise ValueError('linked scratch/artifact path in owned public run')


def _runtime(torch, transformers):
    return {'python': platform.python_version(), 'implementation': platform.python_implementation(),
            'torch': torch.__version__, 'transformers': transformers.__version__, 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'cuda_device_count': torch.cuda.device_count(),
            'threads': torch.get_num_threads(), 'model_precision': 'float32', 'source_precision': 'bfloat16',
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'shuffle_implementation_sha256': shared._digest(Path(random.__file__))}


def run(args):
    prepared = reader.load_public_training(args.census_dir, args.census_sha256, args.feature_bank, args.feature_bank_sha256)
    require_full_material(prepared)
    root = Path(args.output_dir)
    if args.resume_checkpoint:
        root = _safe_root(root)
    else:
        if root.exists() or root.is_symlink():
            raise ValueError('fresh public run requires a new owned output directory')
        _safe_root(root.parent)
        root.mkdir()
    with reader.features.worker_lock(root):
        return _run_locked(args, prepared)


def _run_locked(args, prepared):
    import torch
    import transformers
    from safetensors.torch import load_file, save_file
    started = perf_counter()
    for name in ('epochs', 'batch_size', 'save_every_steps', 'logit_chunk_tokens', 'generation_tokens'):
        if type(getattr(args, name)) is not int or getattr(args, name) < 1:
            raise ValueError('explicit positive public recipe allocation required')
    if args.batch_size >= len(prepared.train) or args.generation_tokens > 8192 or type(args.seed) is not int:
        raise ValueError('public allocation cannot omit different next-batch replay or exceed positions')
    for name in ('learning_rate', 'weight_decay', 'beta1', 'beta2', 'adam_eps'):
        if not math.isfinite(getattr(args, name)):
            raise ValueError('public AdamW fields must be finite')
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.adam_eps <= 0 or not 0 <= args.beta1 < 1 or not 0 <= args.beta2 < 1:
        raise ValueError('public AdamW recipe is invalid')
    if args.session_seconds is not None and (not math.isfinite(args.session_seconds) or args.session_seconds <= 0):
        raise ValueError('scheduling allowance must be finite and positive')
    if type(args.session_index) is not int or args.session_index < 0:
        raise ValueError('public execution session index must be nonnegative')
    if args.stop_step is not None and (type(args.stop_step) is not int or not 0 < args.stop_step <=
            args.epochs * math.ceil(len(prepared.train) / args.batch_size)):
        raise ValueError('public session stop must be an exact prescribed optimizer boundary')
    if args.resume_history and (args.resume_checkpoint or args.measure_only or not args.resume_history_sha256):
        raise ValueError('authenticated public history needs its pin and cannot import measurement or local resume')
    if not args.resume_history and (args.resume_history_sha256 or args.history_source or args.session_index != 0):
        raise ValueError('public source roots and continuation session require an authenticated history')
    if not args.provider.strip() or not args.allocation_ref.strip():
        raise ValueError('explicit provider and allocation reference required')
    device = torch.device(args.device)
    if args.device != 'cpu' and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', args.device) is None:
        raise ValueError('public device needs cpu or explicit CUDA ordinal')
    if device.type == 'cuda' and (not torch.cuda.is_available() or device.index >= torch.cuda.device_count()):
        raise ValueError('requested public CUDA device is unavailable')
    if device.type == 'cuda' and os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in {':4096:8', ':16:8'}:
        raise ValueError('exact CUDA replay requires an explicitly configured CUBLAS_WORKSPACE_CONFIG runtime')
    metadata = verify_tokenizer_metadata(args.prepared_base_dir, args.prepared_base_receipt, args.prepared_clone_sha256, prepared.census)
    processor = transformers.AutoProcessor.from_pretrained(str(args.prepared_base_dir), local_files_only=True, trust_remote_code=False)
    tokenizer = processor.tokenizer
    config_json = reader.features._json((args.prepared_base_dir / 'config.json').read_bytes())['text_config']
    pad = config_json['pad_token_id']
    if config_json['vocab_size'] != 262144 or config_json['hidden_size'] != 3840 or \
            len(tokenizer) != 262144 or tokenizer.bos_token_id != 2 or tokenizer.eos_token_id != 1 or tokenizer.pad_token_id != pad:
        raise ValueError('bound actual tokenizer vocabulary/BOS/EOS/pad differs')
    for pair in prepared.pairs:
        body = pair['labels'][:-1]
        text = tokenizer.decode(body, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if tokenizer.encode(text, add_special_tokens=False) != body:
            raise ValueError('bound encoded public target tokenizer roundtrip differs')
    config = SpecialistConfig(3840, 262144, 768, 6, 12, 8192, pad, 2, 1, args.logit_chunk_tokens)
    if require_sha256(args.config_sha256, 'config sha256') != canonical_sha256(config.record()):
        raise ValueError('explicit full producer config pin differs')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    runtime = _runtime(torch, transformers)
    runtime['arithmetic_device'] = {'type': device.type,
        'cuda_capability': list(torch.cuda.get_device_capability(device)) if device.type == 'cuda' else None,
        'cuda_model': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'cpu_machine': platform.machine() if device.type == 'cpu' else None}
    weights = reader.case_weights(prepared.pairs, split='train')
    curriculum = shared._seal({'schema': CURRICULUM_SCHEMA, 'task': TASK, 'objective': OBJECTIVE,
        'sampling': SAMPLER, 'weights': weights, 'weights_sha256': canonical_sha256(weights),
        'train_base_tuple_sha256': canonical_sha256([p['case_id'] for p in prepared.train]),
        'epochs': args.epochs, 'batch_size': args.batch_size, 'seed': args.seed,
        'train_cases': len(prepared.train), 'development_cases': len(prepared.development),
        'complete_train_label_tokens': sum(len(p['labels']) for p in prepared.train),
        'complete_development_label_tokens': sum(len(p['labels']) for p in prepared.development),
        'development_gradient_or_selection_effect': False, 'importance_coefficient': 'N/B * 1/(train-MRs * references-in-MR)',
        'epoch_order_sha256': [canonical_sha256([p['case_id'] for p in reader.epoch_order(prepared, seed=args.seed, epoch=e)]) for e in range(args.epochs)]})
    code_paths = [Path(__file__), Path(reader.__file__), Path(reader.features.__file__),
                  Path(sys.modules[FormationDecisionSpecialist.__module__].__file__),
                  Path(sys.modules[FormationDecisionSpecialist.__bases__[0].__module__].__file__),
                  Path(sys.modules[bind_conditional_task.__module__].__file__), Path(shared.__file__),
                  Path(reader.census_code.__file__), Path(sys.modules[_cpu_tree.__module__].__file__),
                  Path(checkpoint_history.__file__)]
    manifest = shared._seal({'schema': RUN_SCHEMA, 'task': TASK, 'objective': OBJECTIVE,
        'purpose': 'construction-measurement-only' if args.measure_only else 'full-roster-public-fit',
        'census_sha256': args.census_sha256, 'feature_bank_sha256': args.feature_bank_sha256,
        'encoded_pairs_sha256': prepared.census['encoded_pairs_sha256'], 'source_inputs_sha256': prepared.census['source_inputs_sha256'],
        'export_sha256': prepared.export['record_sha256'], 'metadata': metadata, 'specialist_config': config.record(),
        'initialization': INITIALIZATION, 'runtime': runtime, 'seed': args.seed, 'curriculum_sha256': curriculum['record_sha256'],
        'optimizer': {'name': 'AdamW', 'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay,
                      'betas': [args.beta1, args.beta2], 'eps': args.adam_eps, 'amsgrad': False, 'foreach': False, 'fused': False,
                      'gradient_clip_norm': 1.0}, 'save_every_steps': args.save_every_steps,
        'generation_tokens': args.generation_tokens, 'code': {p.name: shared._digest(p) for p in code_paths},
        'private_evidence_opened': False, 'final_payloads_opened': False, 'source_ancestry_frozen': True,
        'base_weights_loaded': False, 'qualified_for_product': False})
    root = Path(args.output_dir)
    accepted_history = None
    if args.resume_checkpoint:
        root = _safe_root(root)
        _validate_run_paths(root)
        if (root / 'session.json').exists():
            raise ValueError('indexed public fit requires authenticated history and a new session output root')
        if (root / 'component').exists() or args.measure_only:
            raise ValueError('completed or measurement-only run cannot resume fitting')
        if shared._read_sealed(root / 'run.json', RUN_SCHEMA) != manifest or shared._read_sealed(root / 'curriculum.json', CURRICULUM_SCHEMA) != curriculum:
            raise ValueError('public run/curriculum immutable recipe differs on resume')
        seed_receipt = _verify_seed(root, manifest['record_sha256'], config)
    else:
        if set(p.name for p in root.iterdir()) != {'worker.lock'}:
            raise ValueError('fresh locked public run contains unexpected files')
        shared._write_new(root / 'run.json', {k: v for k, v in manifest.items() if k != 'record_sha256'})
        shared._write_new(root / 'curriculum.json', {k: v for k, v in curriculum.items() if k != 'record_sha256'})
        if args.resume_history:
            sources = {}
            for value in args.history_source:
                index, separator, source = value.partition('=')
                if not separator or re.fullmatch(r'0|[1-9][0-9]*', index) is None or int(index) in sources or not source:
                    raise ValueError('public history source must be a unique SESSION_INDEX=READONLY_ROOT')
                sources[int(index)] = Path(source)
            prior = shared._read_sealed(_safe_root(args.resume_history), checkpoint_history.SCHEMA)
            provisional = {'run': manifest['record_sha256'], 'seed': prior['binding']['seed'], 'epochs': args.epochs,
                'batch_size': args.batch_size, 'train_cases': len(prepared.train)}
            accepted_history = checkpoint_history.load(args.resume_history, args.resume_history_sha256,
                sources=sources, manifest=manifest, curriculum=curriculum, binding=provisional)
            seed_receipt = accepted_history.record['origin']['seed']
            if args.session_index != accepted_history.record['session_index'] + 1:
                raise ValueError('public continuation session index differs from accepted predecessor')
    torch.manual_seed(args.seed)
    initialization_started = perf_counter()
    model = FormationDecisionSpecialist(config).to(device=device, dtype=torch.float32)
    initialization_seconds = perf_counter() - initialization_started
    initialization_memory = _memory(device)
    view = bind_conditional_task(model, task=TASK)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2), eps=args.adam_eps, foreach=False, fused=False)
    if not args.resume_checkpoint and not accepted_history:
        seed_receipt = _seed_control(root, model, manifest['record_sha256'], config)
    binding = {'run': manifest['record_sha256'], 'seed': seed_receipt['record_sha256'], 'epochs': args.epochs,
               'batch_size': args.batch_size, 'train_cases': len(prepared.train)}
    cursor = {'epoch': 0, 'next_case': 0, 'step': 0}
    if args.resume_checkpoint:
        construction = shared._read_sealed(root / 'construction.json', 'mfm-public-conditional-foundation-construction-v1')
        if construction['run_sha256'] != binding['run'] or not construction['control_updates_restored']:
            raise ValueError('public construction binding differs')
        cursor = resume_checkpoint(args.resume_checkpoint, root, model, optimizer, binding=binding)
    elif accepted_history:
        construction = accepted_history.record['origin']['construction']
        cursor = resume_checkpoint(accepted_history.latest_checkpoint, root, model, optimizer,
            binding=binding, history=accepted_history)
    else:
        seed_generation = _generation(view, prepared, tokenizer, cap=args.generation_tokens)
        shared._write_new(root / 'seed-generation.json', {'schema': 'mfm-public-conditional-foundation-generation-v1',
            'run_sha256': binding['run'], 'seed_sha256': binding['seed'], 'generation': seed_generation})
        construction = measure_construction(view, optimizer, prepared, output_dir=root, binding=binding, seed=args.seed)
        construction.update(model_initialization_seconds=initialization_seconds, model_initialization_memory=initialization_memory)
        construction = shared._write_new(root / 'construction.json', dict(construction, run_sha256=binding['run']))
    execution = {'schema': 'mfm-public-conditional-foundation-execution-v1', 'run_sha256': binding['run'],
        'provider': args.provider, 'allocation_ref': args.allocation_ref, 'device': args.device,
        'session_seconds': args.session_seconds, 'hardware': _memory(device), 'start_cursor': dict(cursor),
        'measure_only': args.measure_only, 'construction_sha256': construction['record_sha256']}
    execution.update(session_index=args.session_index, stop_step=args.stop_step,
        parent_history_sha256=accepted_history.record['record_sha256'] if accepted_history else None)
    if args.measure_only:
        execution.update(end_cursor=cursor, seconds=perf_counter() - started, status='construction-measured-no-fit')
        return shared._write_new(root / f'execution-{uuid4().hex}.json', execution)
    history = None if args.resume_checkpoint else checkpoint_history.start(root, manifest=manifest,
        curriculum=curriculum, binding=binding, session_index=args.session_index,
        allocation_ref=args.allocation_ref, parent=accepted_history)
    cursor, latest = fit_epochs(view, optimizer, prepared, output_dir=root, binding=binding,
        seed=args.seed, cursor=cursor, save_every_steps=args.save_every_steps,
        started=started, session_seconds=args.session_seconds, purpose=manifest['purpose'],
        stop_step=args.stop_step, history=history)
    if cursor['epoch'] == args.epochs:
        model.load_state_dict(load_file(str(latest / 'specialist.safetensors')), strict=True)
        learned_generation = _generation(view, prepared, tokenizer, cap=args.generation_tokens)
        component = publish_component(view, prepared, output_dir=root, binding=binding,
            curriculum=curriculum, cursor=cursor, checkpoint=latest, generation=learned_generation, history=history)
        execution['component_sha256'] = component['record_sha256']
    execution.update(end_cursor=cursor, seconds=perf_counter() - started,
        checkpoint_history_sha256=history.record['record_sha256'] if history else None,
        checkpoint_history_path=history.latest_path.name if history else None,
        checkpoint_sha256=verify_checkpoint(latest, binding=binding)['record_sha256'],
        status='full-roster-fit-complete-unqualified' if cursor['epoch'] == args.epochs else 'scheduled-complete-boundary-stop')
    return shared._write_new(root / f'execution-{uuid4().hex}.json', execution)


def publish_component(view, prepared, *, output_dir, binding, curriculum, cursor, checkpoint, generation, history=None):
    """Actual final component publication used by full production orchestration."""
    from safetensors.torch import save_file
    root = _safe_root(output_dir)
    model = view.specialist
    final = _safe_child(root, 'component')
    if final.exists():
        raise ValueError('completed public component cannot be overwritten')
    receipt = verify_checkpoint(checkpoint, binding=binding)
    if history is not None:
        history.authorize_latest(checkpoint, binding)
        if history.root != root or history.record['curriculum'] != curriculum:
            raise ValueError('public component requires its exact global fitting history')
    if _safe_root(checkpoint).parent != root or receipt['cursor'] != cursor or cursor['epoch'] != binding['epochs']:
        raise ValueError('public component requires its run-root final completed checkpoint')
    training = evaluate_objective(view, prepared, split='train', checkpoint=checkpoint, binding=binding)
    development = evaluate_objective(view, prepared, split='development', checkpoint=checkpoint, binding=binding)
    staging = root / f'.component-{uuid4().hex}'
    staging.mkdir()
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(staging / 'specialist.safetensors'))
    component = shared._write_new(staging / 'component.json', {'schema': COMPONENT_SCHEMA, 'task': TASK,
        'objective': OBJECTIVE, 'run_sha256': binding['run'], 'curriculum_sha256': curriculum['record_sha256'],
        'seed_sha256': binding['seed'], 'checkpoint_sha256': receipt['record_sha256'],
        'checkpoint_lineage': [e['receipt']['record_sha256'] for e in history.record['entries']] if history else
            [verify_checkpoint(p, binding=binding)['record_sha256'] for p in sorted(root.glob('checkpoint-*'))],
        'checkpoint_history_sha256': history.record['record_sha256'] if history else None,
        'specialist_sha256': shared._digest(staging / 'specialist.safetensors'), 'specialist_config': model.config.record(),
        'cursor': cursor, 'generation': generation, 'training_teacher_objective': training['value'],
        'training_objective_report': training, 'development_objective_report': development,
        'development_teacher_objective': development['value'],
        'development_selection_effect': False, 'qualified_for_product': False,
        'transfer': 'direct learned tensor parent; new full formation objective requires new task contract and optimizer'})
    staging.rename(final)
    return component


def evaluate_objective(view, prepared, *, split, checkpoint, binding):
    """Complete fixed-checkpoint J, never a changing-model minibatch estimate.

    Full-objective coefficients are recomputed from the entire declared roster;
    no N/B multiplier or token pooling belongs in this reporting reduction.
    """
    import torch
    from safetensors.torch import load_file
    if getattr(view, 'task', None) != TASK or not hasattr(view, 'specialist'):
        raise ValueError('exact objective reporting requires explicit public task binding')
    weights = reader.case_weights(prepared.pairs, split=split)
    selected = tuple(p for p in prepared.pairs if p['split'] == split)
    if selected != getattr(prepared, split):
        raise ValueError('exact objective report requires complete declared split roster')
    receipt = verify_checkpoint(checkpoint, binding=binding)
    persisted_weights = load_file(str(Path(checkpoint) / 'specialist.safetensors'))
    compare_tree(view.state_dict(), persisted_weights, atol=0, rtol=0)
    del persisted_weights
    model = view.specialist
    device = next(model.parameters()).device
    modes = [(module, module.training) for module in model.modules()]
    terms = []
    try:
        model.eval()
        # Evaluation cannot perturb a later continuation's CPU/CUDA RNG even
        # if an actual future processor/model implementation consumes randomness.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))), torch.no_grad():
            for pair in selected:
                _, loss = view.forward(**reader.payload(prepared, pair, device=device))
                if not torch.isfinite(loss):
                    raise ValueError('fixed-checkpoint public objective is nonfinite')
                terms.append(weights[pair['case_id']] * float(loss.detach().cpu()))
    finally:
        for module, mode in modes:
            module.training = mode
    return {'schema': 'mfm-public-conditional-foundation-objective-report-v1',
        'split': split, 'objective': OBJECTIVE, 'value': math.fsum(terms),
        'cases': len(selected), 'semantic_groups': len({p['semantic_input_sha256'] for p in selected}),
        'label_tokens': sum(len(p['labels']) for p in selected), 'weights_sha256': canonical_sha256(weights),
        'case_order_sha256': canonical_sha256([p['case_id'] for p in selected]),
        'checkpoint_sha256': receipt['record_sha256'], 'specialist_sha256': receipt['specialist_sha256'],
        'cursor': dict(receipt['cursor']), 'no_gradient': True, 'selection_effect': False}


def fit_epochs(view, optimizer, prepared, *, output_dir, binding, seed, cursor,
               save_every_steps, started, session_seconds, purpose, stop_step=None, history=None):
    """Prescribed fitting lifecycle, shared by the production orchestration."""
    if purpose != 'full-roster-public-fit':
        raise ValueError('measurement-only construction cannot enter or resume fitting')
    root = _safe_root(output_dir)
    _cursor(cursor, train_cases=binding['train_cases'], batch_size=binding['batch_size'], epochs=binding['epochs'])
    total_steps = binding['epochs'] * math.ceil(binding['train_cases'] / binding['batch_size'])
    if stop_step is not None and (type(stop_step) is not int or not cursor['step'] < stop_step <= total_steps):
        raise ValueError('explicit session stop step must be a future prescribed optimizer boundary')
    if history is not None and (history.root != root or history.record['binding'] != binding):
        raise ValueError('fitting history requires this exact owned session and run')
    latest = None
    if cursor['step'] == 0:
        # Construction has restored the persisted fresh model, empty Adam and
        # RNG. Publish that actual boundary in the fitting run before any step
        # (or scheduled stop); nested control checkpoints cannot resume fitting.
        latest = root / 'checkpoint-00000000'
        if latest.exists():
            verify_checkpoint(latest, binding=binding)
        else:
            latest = save_checkpoint(root, view.specialist, optimizer, binding=binding, cursor=cursor)
        if history is not None:
            history.append(latest)
    while cursor['epoch'] < binding['epochs']:
        if session_seconds is not None and perf_counter() - started >= session_seconds:
            break
        if stop_step is not None and cursor['step'] == stop_step:
            break
        batch, following = next_batch(prepared, seed=seed, epochs=binding['epochs'],
            batch_size=binding['batch_size'], cursor=cursor)
        optimizer_step(view, optimizer, batch, prepared=prepared)
        cursor = following
        if cursor['step'] % save_every_steps == 0 or cursor['epoch'] == binding['epochs']:
            latest = save_checkpoint(root, view.specialist, optimizer, binding=binding, cursor=cursor)
            if history is not None:
                history.append(latest)
    path = root / f"checkpoint-{cursor['step']:08d}"
    if not path.exists():
        latest = save_checkpoint(root, view.specialist, optimizer, binding=binding, cursor=cursor)
        if history is not None:
            history.append(latest)
    else:
        latest = path
        verify_checkpoint(latest, binding=binding)
        if history is not None:
            history.append(latest)
    return cursor, latest


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('census-dir', 'feature-bank', 'output-dir', 'prepared-base-dir', 'prepared-base-receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('census-sha256', 'feature-bank-sha256', 'allocation-ref', 'provider', 'config-sha256', 'prepared-clone-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', required=True)
    for name in ('epochs', 'batch-size', 'save-every-steps', 'seed', 'logit-chunk-tokens', 'generation-tokens'):
        parser.add_argument('--' + name, type=int, required=True)
    for name in ('learning-rate', 'weight-decay', 'beta1', 'beta2', 'adam-eps'):
        parser.add_argument('--' + name, type=float, required=True)
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--resume-history', type=Path)
    parser.add_argument('--resume-history-sha256')
    parser.add_argument('--history-source', action='append', default=[])
    parser.add_argument('--session-index', type=int, default=0)
    parser.add_argument('--stop-step', type=int)
    parser.add_argument('--session-seconds', type=float)
    parser.add_argument('--measure-only', action='store_true')
    return parser.parse_args(argv)


def _observation_record(path, file_sha256, schema):
    """Check FILE bytes and the actual writers' UTF-8 compact logical seal."""
    path = _safe_root(path)
    if not path.is_file() or shared._digest(path) != require_sha256(file_sha256, 'observation FILE pin'):
        raise ValueError('observation FILE digest differs')
    row = reader.features._json(path.read_bytes())
    if row.get('schema') != schema or not _observation_sealed(row):
        raise ValueError('observation schema or writer logical seal differs')
    return row


def _observation_sealed(row):
    return type(row) is dict and row == shared._seal({k: v for k, v in row.items() if k != 'record_sha256'})


def _observation_partitions(groups, edges, version):
    adjacency = {key: set() for key in groups}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    remaining, rows = set(groups), []
    while remaining:
        stack, members = [min(remaining)], set()
        while stack:
            key = stack.pop()
            if key not in members:
                members.add(key)
                stack.extend(adjacency[key] - members)
        remaining -= members
        component = sha256('\n'.join(sorted(members)).encode('ascii')).hexdigest()
        digest = sha256((version + '\n' + component).encode('ascii')).hexdigest()
        partition = 'calibration' if int(digest, 16) % 5 == 0 else 'audit'
        rows.extend({'semantic_input_sha256': key, 'component_sha256': component, 'partition': partition}
                    for key in sorted(members))
    return sorted(rows, key=lambda row: row['semantic_input_sha256'])


def read_observation_plan(plan_dir, proposal_sha256, approval_sha256, exposure_path, exposure_sha256, prepared) -> dict:
    """Authenticate the complete frozen cohort, actual-source fanout and graph.

    Source MR fields describe the declared input. Human references stay as
    custody associations and never become expected answers or decoder inputs.
    """
    from .prepare_public_conditional_material import semantic_mr, SLOTS
    root = _safe_root(plan_dir)
    proposal = _observation_record(_safe_child(root, 'proposal.json'), proposal_sha256,
                                  'mfm-public-development-observation-input-proposal-v1')
    approval = _observation_record(_safe_child(root, 'root-prospective-rubric-v2-approval.json'), approval_sha256,
                                  'mfm-root-prospective-semantic-workflow-approval-v1')
    if approval.get('decision') != 'APPROVED_PROSPECTIVE_SOURCE_FIDELITY_OBSERVATION_WORKFLOW_ONLY' or \
            any(approval.get(k) is not False for k in ('semantic_evaluator_admitted', 'automatic_score_or_threshold_admitted', 'qualified_for_product')):
        raise ValueError('observation prospective approval scope differs')
    rubric = _observation_record(_safe_child(root, 'semantic-adjudication-rubric-proposal-v2.json'),
        approval['proposal']['file_sha256'], 'mfm-public-development-semantic-adjudication-proposal-v2')
    if rubric['record_sha256'] != approval['proposal']['record_sha256']:
        raise ValueError('observation approved rubric logical pin differs')
    cohort = _observation_record(_safe_child(root, 'cohort.json'), proposal['cohort']['sha256'],
                                'mfm-public-development-observation-cohort-v1')
    controls = _observation_record(_safe_child(root, 'single-attribute-controls.json'),
        proposal['single_attribute_controls']['sha256'], 'mfm-observed-development-single-attribute-controls-v1')
    for name, record, binding in [('proposal.json', proposal, rubric['input_bindings']['proposal.json']),
                                 ('cohort.json', cohort, proposal['cohort']),
                                 ('single-attribute-controls.json', controls, proposal['single_attribute_controls'])]:
        if record['record_sha256'] != binding['logical_sha256'] or (root/name).stat().st_size != binding['bytes'] or \
                rubric['input_bindings'][name] != binding:
            raise ValueError('observation frozen file/logical binding differs')
    evidence = proposal['input_evidence']
    if cohort['census_sha256'] != prepared.census['record_sha256'] or \
            evidence['census']['logical_sha256'] != prepared.census['record_sha256'] or \
            evidence['encoded_pairs']['sha256'] != prepared.census['encoded_pairs_sha256'] or \
            evidence['source_inputs']['sha256'] != prepared.census['source_inputs_sha256'] or \
            proposal['accepted_bank_external_pin'] != prepared.bank['record_sha256'] or \
            rubric['accepted_bank_external_pin'] != prepared.bank['record_sha256'] or \
            any(cohort.get(k) is not False for k in ('semantic_gold_created', 'targets_rewritten', 'development_gradient_effect')):
        raise ValueError('observation public material or teaching boundary differs')
    cases = [p['case_id'] for p in prepared.development]
    if [p['case_id'] for p in cohort['pairs']] != cases or len(set(cases)) != len(cases) or \
            canonical_sha256(cases) != cohort['case_order_sha256']:
        raise ValueError('observation complete pair order differs')
    group_cases, source_cases = {}, {}
    for frozen, pair in zip(cohort['pairs'], prepared.development):
        key = canonical_sha256({'task': reader.census_code.TASK, 'source_ids': pair['source_ids']})
        expected = {'encoded_pair_record_sha256': canonical_sha256(pair), 'source_ids_sha256': canonical_sha256(pair['source_ids']),
                    'labels_sha256': canonical_sha256(pair['labels']), 'label_tokens': len(pair['labels']), 'source_key': key,
                    'semantic_input_sha256': pair['semantic_input_sha256'], 'reference_count': pair['semantic_reference_count'],
                    'source_payload_sha256': pair['source_payload_sha256']}
        if any(frozen.get(k) != value for k, value in expected.items()) or prepared.pair_by_case.get(pair['case_id']) != pair:
            raise ValueError('observation encoded pair/source custody differs')
        group_cases.setdefault(pair['semantic_input_sha256'], []).append(pair['case_id'])
        source_cases.setdefault(key, []).append(pair['case_id'])
    source_order = cohort['generation_source_keys']
    source_map = {s['source_key']: s for s in prepared.sources if s['split'] == 'development'}
    if source_order != sorted(source_cases) or set(source_map) != set(source_cases) or \
            canonical_sha256(source_order) != cohort['generation_source_order_sha256']:
        raise ValueError('observation complete exact source order differs')
    for key, source in source_map.items():
        if source['case_ids'] != sorted(source_cases[key]) or prepared.source_by_key.get(key) != source:
            raise ValueError('observation exact source fanout differs')
    groups = {g['semantic_input_sha256']: g for g in cohort['groups']}
    if len(groups) != len(cohort['groups']) or list(groups) != sorted(group_cases) or \
            canonical_sha256(list(groups)) != cohort['semantic_group_order_sha256']:
        raise ValueError('observation semantic group order differs')
    source_groups = {key: [] for key in source_order}
    for key, group in groups.items():
        expected_keys = sorted({p['source_key'] for p in cohort['pairs'] if p['semantic_input_sha256'] == key})
        if group['case_ids'] != sorted(group_cases[key]) or group['reference_count'] != len(group_cases[key]) or \
                group['source_keys'] != expected_keys or [m['source_key'] for m in group['original_mrs']] != expected_keys:
            raise ValueError('observation complete group/source associations differ')
        for mr in group['original_mrs']:
            source = source_map[mr['source_key']]
            parsed_slots = semantic_mr(mr['source_text'])
            if dict(parsed_slots) != group['observed_slots'] or canonical_sha256(parsed_slots) != key or \
                    mr['source_input_record_sha256'] != canonical_sha256(source) or \
                    mr['source_payload_sha256s'] != source['source_payload_sha256s']:
                raise ValueError('observation original declared source fields differ')
            source_groups[mr['source_key']].append(key)
    counts = {'pairs': len(cases), 'semantic_groups': len(groups), 'distinct_processed_sources': len(source_order),
              'label_tokens': sum(len(p['labels']) for p in prepared.development)}
    if counts != cohort['counts'] or proposal['measurement_protocol']['ce']['cases'] != counts['pairs'] or \
            proposal['measurement_protocol']['ce']['semantic_groups'] != counts['semantic_groups'] or \
            proposal['measurement_protocol']['ce']['label_tokens'] != counts['label_tokens']:
        raise ValueError('observation complete cohort counts differ')
    decode = proposal['measurement_protocol']['decode']
    required_decode = {'bos_token_id': 2, 'eos_token_id': 1, 'generation_tokens': 128, 'method': 'greedy_argmax',
        'gradient': False, 'malformed_outputs_retained': True, 'repair_or_base_head_fallback': False,
        'rng_and_model_mode_restored': True, 'skip_special_tokens': False, 'temperature_sampling': False,
        'tokenizer_cleanup': False, 'source_order': source_order,
        'output_unit': 'one output per distinct frozen development source_key per checkpoint'}
    if decode != required_decode or controls['cohort_file_sha256'] != proposal['cohort']['sha256'] or \
            controls['cohort_logical_sha256'] != cohort['record_sha256']:
        raise ValueError('observation frozen decode or contrast binding differs')
    # Recompute all naturally existing one-attribute edges, including zero-slot gaps.
    edges = {}
    keys = list(groups)
    for i, left in enumerate(keys):
        a = groups[left]['observed_slots']
        for right in keys[i+1:]:
            b = groups[right]['observed_slots']
            changed = [slot for slot in a if a[slot] != b.get(slot)]
            if set(a) == set(b) and len(changed) == 1:
                edges[left, right] = changed[0]
    observed_edges = {}
    for edge in controls['controls']:
        left, right = edge['left_semantic_input_sha256'], edge['right_semantic_input_sha256']
        slot = edges.get((left, right))
        if slot is None or (left, right) in observed_edges or edge['changed_slot'] != slot:
            raise ValueError('observation contrast edge differs or is repeated')
        for side, key in [('left', left), ('right', right)]:
            if edge[side+'_source_keys'] != groups[key]['source_keys'] or \
                    edge[side+'_reference_case_ids'] != groups[key]['case_ids'] or \
                    edge[side+'_value'] != groups[key]['observed_slots'][slot]:
                raise ValueError('observation contrast endpoint fanout/value differs')
        if edge['unchanged_slots'] != {k: v for k, v in groups[left]['observed_slots'].items() if k != slot}:
            raise ValueError('observation contrast unchanged source fields differ')
        observed_edges[left, right] = slot
    unmatched = sorted(set(groups) - {key for edge in edges for key in edge})
    slot_counts = {slot: sum(s == slot for s in edges.values()) for slot in SLOTS}
    if observed_edges != edges or controls['unmatched_group_sha256s'] != unmatched or \
            controls['counts']['controls'] != len(edges) or \
            controls['counts']['groups_with_controls'] != len(groups)-len(unmatched) or \
            controls['counts']['groups_without_controls'] != len(unmatched) or \
            any(controls['counts']['by_changed_slot'].get(slot, 0) != count for slot, count in slot_counts.items()) or \
            approval['frozen_cohort'] != {'groups': len(groups), 'pairs_per_checkpoint': len(cases),
                                         'existing_contrast_edges': len(edges), 'rewritten_targets': 0}:
        raise ValueError('observation contrast coverage or approved cohort differs')
    route = rubric['calibration_route_proposed']
    if route['partition_version'] != 'mfm-e2e-semantic-rubric-partition-v1':
        raise ValueError('observation partition version differs')
    rows = _observation_partitions(groups, edges, route['partition_version'])
    assignments = {row['semantic_input_sha256']: row['partition'] for row in rows}
    if canonical_sha256(rows) != route['actual_partition_counts']['assignment_sha256']:
        raise ValueError('observation frozen source partition digest differs')
    for partition in ('calibration', 'audit'):
        actual = {'groups': sum(v == partition for v in assignments.values()),
                  'pairs_per_checkpoint': sum(len(group_cases[k]) for k,v in assignments.items() if v == partition),
                  'edges': sum(assignments[a] == partition and assignments[b] == partition for a,b in edges)}
        for field, value in actual.items():
            approved_field = partition + '_' + ('pairs' if field == 'pairs_per_checkpoint' else field)
            if route['actual_partition_counts']['counts'][partition][field] != value or \
                    approval['source_only_partition'][approved_field] != value:
                raise ValueError('observation frozen partition counts differ')
    if any(assignments[a] != assignments[b] for a,b in edges) or \
            approval['source_only_partition']['cross_partition_edges'] != 0 or route['actual_partition_counts']['cross_partition_edges'] != 0:
        raise ValueError('observation contrast graph crosses partition')
    exposure = _observation_record(exposure_path, exposure_sha256, 'mfm-existing-development-output-exposure-ledger-v1')
    if exposure['cohort_file_sha256'] != proposal['cohort']['sha256'] or \
            exposure['assignment_sha256'] != canonical_sha256(rows) or \
            exposure['cohort_and_partition_unchanged'] is not True or exposure['drop_or_rebalance_performed'] is not False or \
            exposure['fixed_audit_groups'] != sum(v == 'audit' for v in assignments.values()) or \
            exposure['fixed_audit_outputs_both_conditions'] != 2*exposure['fixed_audit_groups']:
        raise ValueError('observation exposure ledger binding differs')
    exposed_audit = set()
    for row in exposure['known_exposures']:
        key = row['semantic_input_sha256']
        if key not in groups or row['partition'] != assignments[key] or row['case_id'] not in group_cases[key]:
            raise ValueError('observation known exposure membership differs')
        if assignments[key] == 'audit':
            exposed_audit.add(key)
    if len(exposed_audit) != exposure['known_exposed_audit_groups_in_these_contexts'] or \
            (exposed_audit and exposure['all_audit_outputs_untouched_claim_valid_for_these_contexts'] is not False):
        raise ValueError('observation known audit exposure cannot be concealed')
    return {'source_order': source_order, 'sources': source_map, 'source_groups': source_groups, 'groups': groups,
            'assignments': assignments, 'partition_rows': rows, 'decode': decode, 'counts': counts,
            'controls': controls, 'proposal': proposal, 'exposure': exposure,
            'bindings': {'proposal_file_sha256': proposal_sha256, 'approval_file_sha256': approval_sha256,
                         'rubric_file_sha256': approval['proposal']['file_sha256'], 'exposure_file_sha256': exposure_sha256,
                         'cohort_file_sha256': proposal['cohort']['sha256'], 'assignment_sha256': canonical_sha256(rows)}}


def _load_observation_weights(view, path, expected_sha256):
    """Verify every actual tensor before strict loading; never deserialize Adam."""
    import torch
    from safetensors.torch import load_file
    path = _safe_root(path)
    if shared._digest(path) != require_sha256(expected_sha256, 'observation actual weight bytes'):
        raise ValueError('observation actual weight digest differs')
    state = load_file(str(path), device='cpu')
    expected = view.state_dict()
    if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in state):
        raise ValueError('observation actual state keys or shapes differ')
    if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('observation actual weights require finite FP32 dtype')
    view.specialist.load_state_dict(state, strict=True)
    del state


def observe_public_development(view, prepared, tokenizer, plan, *, output_dir, condition, weight_sha256) -> dict:
    """Save every raw source-only output and expose only calibration text.

    Incremental/reference mismatch stops, retains diagnostic evidence and never
    silently falls back. Check first32 prefixes per source and every actual
    prefix through EOS/cap for the longest frozen source in each condition.
    """
    import torch
    from cognitive_kernel.formation_incremental_decoder import CachedFormationDecoder
    if getattr(view, 'task', None) != TASK or not hasattr(view, 'specialist'):
        raise ValueError('completed observation requires explicit public task binding')
    if any(p.dtype != torch.float32 for p in view.parameters()):
        raise ValueError('completed observation requires FP32 parameters')
    if re.fullmatch(r'[a-z0-9-]{1,80}', condition or '') is None:
        raise ValueError('observation opaque condition identifier is invalid')
    require_sha256(weight_sha256, 'observation weight pin')
    cap = plan['decode']['generation_tokens']
    if cap != 128 or cap > view.config.max_target_tokens or view.config.start_token_id != 2 or view.config.end_token_id != 1:
        raise ValueError('observation complete BOS/EOS/cap differs')
    longest_source = max(plan['source_order'], key=lambda key: len(plan['sources'][key]['source_ids']))
    root = _safe_root(output_dir)
    raw_path, calibration_path = root/f'raw-{condition}.jsonl', root/f'calibration-{condition}.jsonl'
    if any(p.exists() or p.is_symlink() for p in (raw_path, calibration_path, root/f'observation-{condition}.json')):
        raise ValueError('observation completed or partial evidence cannot be overwritten')
    device = next(view.parameters()).device
    modes = [(module, module.training) for module in view.specialist.modules()]
    started, completed, associations, steps, ended_count, compared = perf_counter(), 0, 0, 0, 0, 0
    active, parity, longest_check = None, None, None
    try:
        view.eval()
        with raw_path.open('xb') as raw, calibration_path.open('xb') as calibration, \
                torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))), \
                torch.inference_mode(), torch.autocast(device.type, enabled=False):
            for key in plan['source_order']:
                reference_budget = cap if key == longest_source else min(32, cap)
                active, parity = key, {'compared_greedy_prefixes': 0, 'max_abs_difference': 0.,
                    'atol': 1e-4, 'rtol': 1e-4, 'argmaxes_equal': True,
                    'reference_budget': reference_budget, 'longest_source_full_actual_prefix_check': key == longest_source}
                source = plan['sources'][key]
                receipt, tensors = reader.features.read_source(prepared.feature_bank, source,
                    prepared.export['record_sha256'], prepared.census['base_hidden_size'])
                if receipt['record_sha256'] != prepared.source_receipt_by_key[key]:
                    raise ValueError('observation source receipt differs from pinned complete bank')
                states = tensors['states'].detach().to(device)
                mask = tensors['attention_mask'].detach().to(device)
                # Target tensors and decoder_inputs are intentionally not constructed.
                decoder = CachedFormationDecoder(view, base_states=states, source_mask=mask)
                prefix, ids, eos, case_steps = [2], [], False, 0
                case_started = perf_counter()
                for index in range(cap):
                    logits = decoder.step(prefix[-1])
                    if logits.dtype != torch.float32 or not torch.isfinite(logits).all():
                        raise ValueError('observation generated logits must be finite FP32')
                    token = int(logits[0].argmax().item())
                    if index < reference_budget:
                        reference = view.next_token_logits(base_states=states, source_mask=mask,
                            input_ids=torch.tensor([prefix], dtype=torch.long, device=device))
                        finite = bool(torch.isfinite(reference).all())
                        difference = float((reference-logits).abs().max().cpu()) if finite else None
                        equal = bool(reference.dtype == torch.float32 and finite and
                            torch.allclose(reference, logits, atol=1e-4, rtol=1e-4) and
                            int(reference[0].argmax().item()) == token)
                        parity.update(compared_greedy_prefixes=index+1,
                            max_abs_difference=max(parity['max_abs_difference'], difference) if finite else None,
                            argmaxes_equal=equal, reference_finite=finite)
                        if not equal:
                            parity.update(reference_argmax=int(reference[0].argmax().item()), cached_argmax=token,
                                          failed_prefix_ids=list(prefix))
                            raise ValueError('observation actual FP32 cache/reference parity differs')
                    case_steps += 1
                    if token == 1:
                        eos = True
                        break
                    ids.append(token)
                    prefix.append(token)
                raw_text, decode_error = None, None
                try:
                    raw_text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                except Exception as exc:
                    decode_error = type(exc).__name__ + ': ' + str(exc)
                row = {'schema': 'mfm-public-completed-source-observation-v1', 'condition': condition,
                    'observation_id': canonical_sha256({'source_key': key, 'condition': condition}),
                    'source_key': key, 'source_receipt_sha256': receipt['record_sha256'],
                    'semantic_input_sha256s': plan['source_groups'][key], 'case_ids': source['case_ids'],
                    'raw_token_ids': ids, 'raw_output_text': raw_text, 'tokenizer_decode_error': decode_error,
                    'eos_observed': eos, 'cap_exhausted': not eos, 'decoder_steps': case_steps,
                    'parity': parity, 'seconds': perf_counter()-case_started,
                    'semantic_verdict': 'not-adjudicated', 'malformed_or_special_tokens_repaired': False}
                raw.write(canonical_json_bytes(row)+b'\n')
                raw.flush()
                partitions = {plan['assignments'][g] for g in plan['source_groups'][key]}
                if len(partitions) != 1:
                    raise ValueError('observation exact source associates across partitions')
                if partitions == {'calibration'}:
                    exposed = {**row, 'declared_source_fields': [
                        {'semantic_input_sha256': g, 'observed_slots': plan['groups'][g]['observed_slots'],
                         'original_mrs': plan['groups'][g]['original_mrs']} for g in plan['source_groups'][key]]}
                    calibration.write(canonical_json_bytes(exposed)+b'\n')
                    calibration.flush()
                completed += 1
                if key == longest_source:
                    longest_check = {'source_key': key, 'source_tokens': len(source['source_ids']),
                        'actual_decoder_steps': case_steps, 'compared_greedy_prefixes': parity['compared_greedy_prefixes'],
                        'eos_observed': eos, 'cap_exhausted': not eos, 'argmaxes_equal': True}
                associations += len(source['case_ids'])
                steps += case_steps
                ended_count += int(eos)
                compared += parity['compared_greedy_prefixes']
                del decoder, tensors, states, mask, logits
    except BaseException as exc:
        shared._write_new(root/f'observation-failure-{condition}.json', {
            'schema': 'mfm-public-completed-observation-failure-v1', 'condition': condition,
            'completed_sources': completed, 'active_source_key': active, 'parity': parity,
            'raw_file': raw_path.name, 'raw_file_sha256': shared._digest(raw_path) if raw_path.exists() else None,
            'error': type(exc).__name__ + ': ' + str(exc), 'automatic_retry': False})
        raise
    finally:
        for module, mode in modes:
            module.training = mode
    seconds = perf_counter()-started
    return shared._write_new(root/f'observation-{condition}.json', {
        'schema': 'mfm-public-completed-observation-condition-v1', 'condition': condition,
        'weight_sha256': weight_sha256, 'source_order_sha256': canonical_sha256(plan['source_order']),
        'sources': completed, 'pair_associations': associations, 'decoder_steps': steps,
        'eos_outputs': ended_count, 'cap_exhausted_outputs': completed-ended_count,
        'compared_greedy_prefixes': compared,
        'cache_method': 'FP32-first32-per-source-and-full-longest-actual-prefix-reference-checked',
        'longest_source_reference_check': longest_check,
        'raw_file': raw_path.name, 'raw_file_sha256': shared._digest(raw_path),
        'calibration_file': calibration_path.name, 'calibration_file_sha256': shared._digest(calibration_path),
        'seconds': seconds, 'decoder_steps_per_second': steps/seconds, 'memory': _memory(device),
        'target_visible_to_generation': False, 'optimizer_created': False, 'optimizer_updates': 0,
        'qualified_for_product': False, 'semantic_score_created': False})


def _read_completed_observation(args, prepared, metadata, plan):
    """Authenticate complete pinned metadata and declared direct tensor paths.

    This checks the accepted index, not every historical binary. It deliberately
    does not call History.load, restore an optimizer or open intermediate weights.
    """
    root, seed_root = _safe_root(args.completed_run_dir), _safe_root(args.seed_dir)
    component_root = _safe_root(args.component_dir)
    if component_root != root/'component' or seed_root == root:
        raise ValueError('observation explicit component/seed source directories differ')
    component = _observation_record(component_root/'component.json', args.component_file_sha256, COMPONENT_SCHEMA)
    manifest = _observation_record(root/'run.json', args.run_file_sha256, RUN_SCHEMA)
    curriculum = _observation_record(root/'curriculum.json', args.curriculum_file_sha256, CURRICULUM_SCHEMA)
    seed = _observation_record(seed_root/'seed-control.json', args.seed_file_sha256, SEED_SCHEMA)
    config = SpecialistConfig(**manifest['specialist_config'])
    if canonical_sha256(config.record()) != require_sha256(args.config_sha256, 'completed observation config pin') or \
            component['specialist_config'] != config.record() or seed['specialist_config'] != config.record() or \
            manifest['task'] != TASK or component['task'] != TASK or curriculum['task'] != TASK or \
            manifest['objective'] != OBJECTIVE or component['objective'] != OBJECTIVE or curriculum['objective'] != OBJECTIVE or \
            manifest['purpose'] != 'full-roster-public-fit' or manifest['initialization'] != INITIALIZATION or \
            seed['initialization'] != INITIALIZATION or seed['optimizer_steps'] != 0 or \
            curriculum['sampling'] != SAMPLER or curriculum['development_gradient_or_selection_effect'] is not False or \
            component['development_selection_effect'] is not False or component['qualified_for_product'] is not False or \
            any(manifest[k] is not False for k in ('base_weights_loaded', 'private_evidence_opened', 'final_payloads_opened', 'qualified_for_product')):
        raise ValueError('observation completed config/task/recipe or seed-control meaning differs')
    recipe = plan['proposal']['fit_program']
    if curriculum['epochs'] != recipe['epochs'] or curriculum['batch_size'] != recipe['batch_size'] or \
            curriculum['train_cases'] != len(prepared.train) or curriculum['development_cases'] != len(prepared.development) or \
            curriculum['seed'] != manifest['seed'] or manifest['curriculum_sha256'] != curriculum['record_sha256'] or \
            component['curriculum_sha256'] != curriculum['record_sha256'] or \
            component['run_sha256'] != manifest['record_sha256'] or seed['run_sha256'] != manifest['record_sha256'] or \
            component['seed_sha256'] != seed['record_sha256'] or manifest['metadata'] != metadata or \
            manifest['code']['train_public_conditional_foundation.py'] != plan['proposal']['source_code']['trainer']['git_export_sha256']:
        raise ValueError('observation exact completed run/curriculum/seed/source binding differs')
    for field, value in [('census_sha256', prepared.census['record_sha256']),
                         ('feature_bank_sha256', prepared.bank['record_sha256']),
                         ('encoded_pairs_sha256', prepared.census['encoded_pairs_sha256']),
                         ('source_inputs_sha256', prepared.census['source_inputs_sha256']),
                         ('export_sha256', prepared.export['record_sha256'])]:
        if manifest[field] != value:
            raise ValueError('observation completed public processor binding differs')
    binding = {'run': manifest['record_sha256'], 'seed': seed['record_sha256'], 'epochs': curriculum['epochs'],
               'batch_size': curriculum['batch_size'], 'train_cases': curriculum['train_cases']}
    _cursor(component['cursor'], train_cases=binding['train_cases'], batch_size=binding['batch_size'], epochs=binding['epochs'])
    if component['cursor']['epoch'] != binding['epochs'] or component['cursor']['next_case'] != 0:
        raise ValueError('observation requires fixed final completed E20 checkpoint')
    step = component['cursor']['step']
    checkpoint = _observation_record(root/f'checkpoint-{step:08d}'/'checkpoint.json', args.checkpoint_file_sha256, CHECKPOINT_SCHEMA)
    history = _observation_record(root/f'history-{step:08d}.json', args.history_file_sha256, checkpoint_history.SCHEMA)
    if checkpoint['binding'] != binding or checkpoint['cursor'] != component['cursor'] or \
            checkpoint['record_sha256'] != component['checkpoint_sha256'] or \
            checkpoint['specialist_sha256'] != component['specialist_sha256'] or \
            history['record_sha256'] != component['checkpoint_history_sha256'] or history['binding'] != binding or \
            history['run'] != manifest or history['curriculum'] != curriculum or \
            history['origin']['seed'] != seed or history['purpose'] != 'full-roster-public-fit':
        raise ValueError('observation completed checkpoint/history/origin differs')
    previous, last_step, last_session = None, -1, 0
    for entry in history['entries']:
        receipt = entry['receipt']
        if not _observation_sealed(entry) or entry['schema'] != checkpoint_history.ENTRY_SCHEMA or \
                not _observation_sealed(receipt) or receipt['schema'] != CHECKPOINT_SCHEMA or \
                receipt['binding'] != binding or entry['previous_entry_sha256'] != previous or \
                type(entry['session_index']) is not int or not last_session <= entry['session_index'] <= history['session_index']:
            raise ValueError('observation complete metadata history chain differs')
        _cursor(receipt['cursor'], train_cases=binding['train_cases'], batch_size=binding['batch_size'], epochs=binding['epochs'])
        current = receipt['cursor']['step']
        if current <= last_step or (last_step == -1 and current != 0) or entry['path'] != f'checkpoint-{current:08d}':
            raise ValueError('observation metadata history order differs')
        previous, last_step, last_session = entry['record_sha256'], current, entry['session_index']
    if not history['entries'] or history['entries'][-1]['receipt'] != checkpoint or \
            history['entries'][-1]['session_index'] != history['session_index'] or \
            component['checkpoint_lineage'] != [entry['receipt']['record_sha256'] for entry in history['entries']] or \
            history['entries'][0]['receipt']['specialist_sha256'] != seed['specialist_sha256'] or \
            len(history['sessions']) != history['session_index']+1:
        raise ValueError('observation complete final/step-zero history membership differs')
    for index, session in enumerate(history['sessions']):
        prior = [entry for entry in history['entries'] if entry['session_index'] < index]
        expected_cursor = prior[-1]['receipt']['cursor'] if prior else {'epoch': 0, 'next_case': 0, 'step': 0}
        if not _observation_sealed(session) or session['schema'] != checkpoint_history.SESSION_SCHEMA or \
                session['session_index'] != index or session['run_sha256'] != binding['run'] or \
                not session['allocation_ref'].strip() or session['parent_entry_count'] != len(prior) or \
                session['start_cursor'] != expected_cursor or (index == 0 and session['parent_history_sha256'] is not None):
            raise ValueError('observation complete lineage session metadata differs')
    for declared, expected in [(component['specialist_sha256'], args.final_weight_sha256),
                               (seed['specialist_sha256'], args.seed_weight_sha256)]:
        if declared != require_sha256(expected, 'completed observation actual tensor pin'):
            raise ValueError('observation declared final/seed tensor pin differs')
    final_path, seed_path = component_root/'specialist.safetensors', seed_root/'seed-control.safetensors'
    return {'config': config, 'component': component, 'run': manifest, 'curriculum': curriculum,
            'history': history, 'checkpoint': checkpoint, 'seed': seed,
            'weight_paths': {'final': final_path, 'seed': seed_path}, 'binding': binding,
            'history_verification': 'accepted complete sealed metadata; no fresh full binary ancestry audit'}


def _verify_observation_weight_bytes(records, args):
    """Unconditional runtime preflight, separate from metadata-only review."""
    for role in ('final', 'seed'):
        if shared._digest(_safe_root(records['weight_paths'][role])) != getattr(args, role+'_weight_sha256'):
            raise ValueError('observation actual final/seed tensor bytes differ')


def run_completed_observation(args) -> dict:
    """Native readonly completed-component observation; no Adam or base head."""
    import gc
    import torch
    import transformers
    started = perf_counter()
    if not args.provider.strip() or not args.allocation_ref.strip() or \
            (args.device != 'cpu' and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', args.device) is None):
        raise ValueError('observation explicit device/provider/allocation required')
    device = torch.device(args.device)
    if device.type == 'cuda' and (not torch.cuda.is_available() or device.index >= torch.cuda.device_count()):
        raise ValueError('observation CUDA ordinal is unavailable')
    if device.type == 'cuda' and os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in {':4096:8', ':16:8'}:
        raise ValueError('observation exact CUDA arithmetic requires configured CUBLAS_WORKSPACE_CONFIG')
    # Provider readonly placement is explicit; the observation output stays new.
    for path in (args.completed_run_dir, args.seed_dir):
        checkpoint_history.require_readonly_root(path)
    prepared = reader.load_public_training(args.census_dir, args.census_sha256, args.feature_bank, args.feature_bank_sha256)
    require_full_material(prepared)
    plan = read_observation_plan(args.plan_dir, args.proposal_sha256, args.approval_sha256,
                                 args.exposure_path, args.exposure_sha256, prepared)
    metadata = verify_tokenizer_metadata(args.prepared_base_dir, args.prepared_base_receipt, args.prepared_clone_sha256, prepared.census)
    records = _read_completed_observation(args, prepared, metadata, plan)
    _verify_observation_weight_bytes(records, args)
    config = records['config']
    if (config.base_hidden_size, config.vocabulary_size, config.width, config.layers, config.heads,
            config.max_target_tokens, config.start_token_id, config.end_token_id) != (3840, 262144, 768, 6, 12, 8192, 2, 1):
        raise ValueError('observation exact admitted full producer geometry differs')
    processor = transformers.AutoProcessor.from_pretrained(str(args.prepared_base_dir), local_files_only=True, trust_remote_code=False)
    tokenizer = processor.tokenizer
    text_config = reader.features._json((args.prepared_base_dir/'config.json').read_bytes())['text_config']
    if len(tokenizer) != config.vocabulary_size or tokenizer.bos_token_id != 2 or tokenizer.eos_token_id != 1 or \
            tokenizer.pad_token_id != config.pad_token_id or text_config['pad_token_id'] != config.pad_token_id or \
            text_config['hidden_size'] != config.base_hidden_size or text_config['vocab_size'] != config.vocabulary_size:
        raise ValueError('observation bound actual tokenizer/config differs')
    root = Path(args.output_dir)
    if root.exists() or root.is_symlink():
        raise ValueError('observation requires a new owned output directory')
    _safe_root(root.parent)
    root.mkdir(mode=0o700)
    conditions = ['condition-'+uuid4().hex, 'condition-'+uuid4().hex]
    # Randomize the opaque mapping using OS entropy; preserve it separately.
    random.SystemRandom().shuffle(conditions)
    mapping = dict(zip(conditions, ('final', 'seed')))
    old_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    old_deterministic = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    reports = []
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        with reader.features.worker_lock(root), torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            shared._write_new(root/'condition-mapping.json', {'schema': 'mfm-public-completed-observation-opaque-mapping-v1',
                'conditions': {condition: {'role': role, 'weight_sha256': getattr(args, role+'_weight_sha256')}
                               for condition, role in mapping.items()}, 'successful_blinding_established': False})
            shared._write_new(root/'protocol.json', {'schema': 'mfm-public-completed-observation-protocol-v1',
                'bindings': plan['bindings'], 'decode': plan['decode'], 'counts': plan['counts'],
                'partitions': plan['partition_rows'], 'contrasts': plan['controls'], 'exposure': plan['exposure'],
                'component_sha256': records['component']['record_sha256'],
                'history_sha256': records['history']['record_sha256'], 'history_verification': records['history_verification'],
                'direct_final_seed_bytes_verified': True,
                'producer_sha256': shared._digest(Path(__file__)), 'runtime': _runtime(torch, transformers),
                'provider': args.provider, 'allocation_ref': args.allocation_ref, 'device': str(device),
                'target_visible_to_generation': False, 'optimizer_created': False, 'qualified_for_product': False})
            # Execute opaque conditions in opaque-name order, so ordering alone
            # cannot disclose the final/seed role. Only one model is resident.
            for condition in sorted(mapping):
                role = mapping[condition]
                load_started = perf_counter()
                model = FormationDecisionSpecialist(config)
                view = bind_conditional_task(model, task=TASK)
                _load_observation_weights(view, records['weight_paths'][role], getattr(args, role+'_weight_sha256'))
                model.to(device=device, dtype=torch.float32)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                load_seconds = perf_counter()-load_started
                load_memory = _memory(device)
                report = observe_public_development(view, prepared, tokenizer, plan, output_dir=root,
                    condition=condition, weight_sha256=getattr(args, role+'_weight_sha256'))
                reports.append({'condition': condition, 'record_sha256': report['record_sha256'],
                                'load_seconds': load_seconds, 'load_memory': load_memory, 'memory': _memory(device)})
                del view, model
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            # A whole-file custody seal accompanies the per-condition seals.
            raw_path = root/'all-raw-observations.jsonl'
            with raw_path.open('xb') as stream:
                for condition in sorted(mapping):
                    with (root/f'raw-{condition}.jsonl').open('rb') as source:
                        for block in iter(lambda: source.read(8*1024*1024), b''):
                            stream.write(block)
            return shared._write_new(root/'observation.json', {
                'schema': 'mfm-public-completed-observation-run-v1', 'status': 'complete-raw-development-observation-unqualified',
                'condition_reports': reports, 'counts_per_condition': plan['counts'],
                'raw_file': raw_path.name, 'raw_file_sha256': shared._digest(raw_path),
                'protocol_file_sha256': shared._digest(root/'protocol.json'),
                'condition_mapping_file_sha256': shared._digest(root/'condition-mapping.json'),
                'known_exposed_audit_groups': plan['exposure']['known_exposed_audit_groups_in_these_contexts'],
                'all_audit_outputs_untouched_claim': False, 'semantic_score_created': False,
                'optimizer_created': False, 'optimizer_updates': 0, 'qualified_for_product': False,
                'seconds': perf_counter()-started})
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old_tf32
        torch.use_deterministic_algorithms(old_deterministic[0], warn_only=old_deterministic[1])


def observation_arguments(argv=None):
    """A completed-component purpose with no fitting or optimizer arguments."""
    parser = argparse.ArgumentParser(description='Observe the completed public foundation and its persisted same-run seed.')
    for name in ('census-dir', 'feature-bank', 'output-dir', 'prepared-base-dir', 'prepared-base-receipt',
                 'component-dir', 'completed-run-dir', 'seed-dir', 'plan-dir', 'exposure-path'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('census-sha256', 'feature-bank-sha256', 'config-sha256', 'prepared-clone-sha256',
                 'component-file-sha256', 'run-file-sha256', 'curriculum-file-sha256', 'history-file-sha256',
                 'checkpoint-file-sha256', 'seed-file-sha256', 'final-weight-sha256', 'seed-weight-sha256',
                 'proposal-sha256', 'approval-sha256', 'exposure-sha256', 'provider', 'allocation-ref', 'device'):
        parser.add_argument('--' + name, required=True)
    return parser.parse_args(argv)


if __name__ == '__main__':
    if sys.argv[1:2] == ['observe-completed']:
        print(json.dumps(run_completed_observation(observation_arguments(sys.argv[2:])), sort_keys=True))
    else:
        print(json.dumps(run(arguments()), sort_keys=True))
