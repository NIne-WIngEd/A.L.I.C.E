"""Full-roster fresh public conditional MFM foundation construction.

This is an unqualified public sequence-learning artifact, not formation or
owner-independent qualification. No foundation model weights are loaded.
"""
from __future__ import annotations

import argparse
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

from cognitive_kernel.canonical import canonical_sha256, require_sha256
from cognitive_kernel.formation_conditional_task import bind_conditional_task
from cognitive_kernel.formation_decision_specialist_v16 import FormationDecisionSpecialist, INITIALIZATION
from cognitive_kernel.formation_v1_specialist import SpecialistConfig
from . import public_conditional_training as reader
from . import train_v1_formation_specialist as shared
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


def resume_checkpoint(checkpoint_dir, output_dir, specialist, optimizer, *, binding):
    import torch
    from safetensors.torch import load_file
    root = _safe_root(output_dir)
    path = _safe_root(checkpoint_dir)
    if path.parent != root or re.fullmatch(r'checkpoint-[0-9]{8}', path.name) is None:
        raise ValueError('public checkpoint is foreign or partial')
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
               'seed-generation.json', 'construction.json', 'construction-controls', 'component'}
    for path in root.iterdir():
        _safe_child(root, path.name)
        if path.name not in allowed and re.fullmatch(r'(checkpoint-[0-9]{8}|\.checkpoint-[A-Za-z0-9]+|\.record-[a-f0-9]+|execution-[a-f0-9]+\.json|\.component-[a-f0-9]+)', path.name) is None:
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
                  Path(reader.census_code.__file__), Path(sys.modules[_cpu_tree.__module__].__file__)]
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
    if args.resume_checkpoint:
        root = _safe_root(root)
        _validate_run_paths(root)
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
    torch.manual_seed(args.seed)
    initialization_started = perf_counter()
    model = FormationDecisionSpecialist(config).to(device=device, dtype=torch.float32)
    initialization_seconds = perf_counter() - initialization_started
    initialization_memory = _memory(device)
    view = bind_conditional_task(model, task=TASK)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2), eps=args.adam_eps, foreach=False, fused=False)
    if not args.resume_checkpoint:
        seed_receipt = _seed_control(root, model, manifest['record_sha256'], config)
    binding = {'run': manifest['record_sha256'], 'seed': seed_receipt['record_sha256'], 'epochs': args.epochs,
               'batch_size': args.batch_size, 'train_cases': len(prepared.train)}
    cursor = {'epoch': 0, 'next_case': 0, 'step': 0}
    if args.resume_checkpoint:
        construction = shared._read_sealed(root / 'construction.json', 'mfm-public-conditional-foundation-construction-v1')
        if construction['run_sha256'] != binding['run'] or not construction['control_updates_restored']:
            raise ValueError('public construction binding differs')
        cursor = resume_checkpoint(args.resume_checkpoint, root, model, optimizer, binding=binding)
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
    if args.measure_only:
        execution.update(end_cursor=cursor, seconds=perf_counter() - started, status='construction-measured-no-fit')
        return shared._write_new(root / f'execution-{uuid4().hex}.json', execution)
    latest = None
    while cursor['epoch'] < args.epochs:
        if args.session_seconds is not None and perf_counter() - started >= args.session_seconds:
            break
        batch, following = next_batch(prepared, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, cursor=cursor)
        optimizer_step(view, optimizer, batch, prepared=prepared)
        cursor = following
        if cursor['step'] % args.save_every_steps == 0 or cursor['epoch'] == args.epochs:
            latest = save_checkpoint(root, model, optimizer, binding=binding, cursor=cursor)
    path = root / f"checkpoint-{cursor['step']:08d}"
    if not path.exists():
        latest = save_checkpoint(root, model, optimizer, binding=binding, cursor=cursor)
    else:
        latest = path
        verify_checkpoint(latest, binding=binding)
    if cursor['epoch'] == args.epochs:
        model.load_state_dict(load_file(str(latest / 'specialist.safetensors')), strict=True)
        learned_generation = _generation(view, prepared, tokenizer, cap=args.generation_tokens)
        development_weights = reader.case_weights(prepared.pairs, split='development')
        model.eval()
        development_loss = 0.
        with torch.inference_mode():
            for pair in prepared.development:
                _, loss = view.forward(**reader.payload(prepared, pair, device=device))
                development_loss += development_weights[pair['case_id']] * float(loss.cpu())
        staging = root / f'.component-{uuid4().hex}'
        staging.mkdir()
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(staging / 'specialist.safetensors'))
        component = shared._write_new(staging / 'component.json', {'schema': COMPONENT_SCHEMA, 'task': TASK,
            'objective': OBJECTIVE, 'run_sha256': binding['run'], 'curriculum_sha256': curriculum['record_sha256'],
            'seed_sha256': binding['seed'], 'checkpoint_sha256': verify_checkpoint(latest, binding=binding)['record_sha256'],
            'checkpoint_lineage': [verify_checkpoint(p, binding=binding)['record_sha256'] for p in sorted(root.glob('checkpoint-*'))],
            'specialist_sha256': shared._digest(staging / 'specialist.safetensors'), 'specialist_config': config.record(),
            'cursor': cursor, 'generation': learned_generation, 'development_teacher_objective': development_loss,
            'development_selection_effect': False, 'qualified_for_product': False,
            'transfer': 'direct learned tensor parent; new full formation objective requires new task contract and optimizer'})
        staging.rename(root / 'component')
        execution['component_sha256'] = component['record_sha256']
    execution.update(end_cursor=cursor, seconds=perf_counter() - started,
        checkpoint_sha256=verify_checkpoint(latest, binding=binding)['record_sha256'],
        status='full-roster-fit-complete-unqualified' if cursor['epoch'] == args.epochs else 'scheduled-complete-boundary-stop')
    return shared._write_new(root / f'execution-{uuid4().hex}.json', execution)


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
    parser.add_argument('--session-seconds', type=float)
    parser.add_argument('--measure-only', action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    print(json.dumps(run(arguments()), sort_keys=True))
