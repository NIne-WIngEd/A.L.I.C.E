"""Full-geometry MFM teacher construction on the corrected source view.

Same frozen source and original specialist geometry. Versioned local-identity
decision objective and calibrated fresh embeddings; no imported language head,
grammar mask, target truncation, memory write or qualification claim.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
from time import perf_counter

from cognitive_kernel.canonical import CognitiveKernelContractError
from cognitive_kernel.formation_decision_specialist_v16 import FormationDecisionSpecialist, INITIALIZATION
from cognitive_kernel.formation_decision_v16 import OBJECTIVE
from . import formation_source_view as source
from . import train_v16_cached_specialist as cached

training, shared = source.training, source.shared
RUN_SCHEMA = 'mfm-v16-decision-specialist-run-v1'
ARTIFACT_SCHEMA = 'mfm-v16-decision-specialist-artifact-v1'


def code_binding():
    import cognitive_kernel.formation_decision_specialist_v16 as model
    return {'source_view_reader': shared._digest(Path(source.__file__)),
            'decision_model': shared._digest(Path(model.__file__)),
            'trainer': shared._digest(Path(__file__)),
            **source.preparation.code_binding()}


def optimizer_step(specialist, optimizer, examples, *, prepared):
    import torch
    if not examples or any(e.split != 'train' for e in examples):
        raise CognitiveKernelContractError('only admitted training cases enter gradients')
    specialist.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for example in examples:
        payload = source.payload(prepared, example, device=prepared.device)
        _, loss = specialist(**payload)
        if not torch.isfinite(loss):
            raise CognitiveKernelContractError('decision loss is nonfinite')
        (loss / len(examples)).backward()
        losses.append(float(loss.detach().cpu()))
    norm = torch.nn.utils.clip_grad_norm_(specialist.parameters(), 1.0, error_if_nonfinite=True)
    if not torch.isfinite(norm) or norm.item() <= 0:
        raise CognitiveKernelContractError('decision specialist has no finite nonzero gradient')
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if any(not torch.isfinite(p).all() for p in specialist.parameters()):
        raise CognitiveKernelContractError('decision step produced nonfinite weights')
    return {'mean_loss': sum(losses) / len(losses), 'gradient_norm': norm.item(), 'train_examples': len(examples)}


def development_loss(specialist, prepared):
    import torch
    if not prepared.development or any(e.split != 'development' for e in prepared.development):
        raise CognitiveKernelContractError('decision development split differs')
    specialist.eval()
    total = count = 0
    with torch.no_grad():
        for example in prepared.development:
            payload = source.payload(prepared, example, device=prepared.device)
            _, loss = specialist(**payload)
            if not torch.isfinite(loss):
                raise CognitiveKernelContractError('decision development loss is nonfinite')
            tokens = payload['labels'].numel()
            total += float(loss.cpu()) * tokens
            count += tokens
    return total / count


def stress_examples(prepared, accumulation):
    stress = list(dict.fromkeys((*prepared.view['stress_train_case_ids'], *(
        e.case_id for e in prepared.train if e.case_id in source.preparation.prior.PUBLIC_SEED_PRIOR_MEMORY))))
    if not set(stress).issubset({e.case_id for e in prepared.train}) or not len(stress) <= accumulation <= len(prepared.train):
        raise CognitiveKernelContractError('decision stress/augmented cases do not fit complete accumulation')
    return sorted(prepared.train, key=lambda e: stress.index(e.case_id) if e.case_id in stress else len(stress))[:accumulation]


def probe(specialist, optimizer, prepared, args, digest):
    import torch
    from safetensors.torch import save_file
    examples = stress_examples(prepared, args.gradient_accumulation)
    before = cached._cpu_tree(specialist.state_dict())
    steps = [optimizer_step(specialist, optimizer, examples, prepared=prepared) for _ in range(2)]
    training._checkpoint(args.output_dir, specialist=specialist, optimizer=optimizer,
        run_digest=digest, epoch=0, next_case=len(examples), step=2)
    expected = optimizer_step(specialist, optimizer, examples, prepared=prepared)
    expected_model, expected_optimizer = cached._cpu_tree(specialist.state_dict()), cached._cpu_tree(optimizer.state_dict())
    training._resume(args.output_dir / 'checkpoint-00000002', args.output_dir, specialist, optimizer, digest)
    observed = optimizer_step(specialist, optimizer, examples, prepared=prepared)
    if abs(observed['mean_loss'] - expected['mean_loss']) > 1e-6:
        raise CognitiveKernelContractError('decision resumed next loss differs')
    model_diff = cached.compare_tree(specialist.state_dict(), expected_model)
    moment_diff = cached.compare_tree(optimizer.state_dict(), expected_optimizer)
    if all(torch.equal(value.cpu(), before[name]) for name, value in specialist.state_dict().items()):
        raise CognitiveKernelContractError('decision probe changed no weights')
    path = args.output_dir / 'probe-specialist.safetensors'
    save_file({k: v.detach().cpu().contiguous() for k, v in specialist.state_dict().items()}, str(path))
    return {'first_and_later_steps': steps, 'resumed_next_step': observed,
            'model_max_abs_difference': model_diff, 'optimizer_max_abs_difference': moment_diff,
            'checkpoint_reload_executed': True, 'backward_and_restart_passed': True,
            'optimizer_steps': 3, 'stress_case_ids': [e.case_id for e in examples],
            'probe_weights_sha256': shared._digest(path)}


def verify_probe(args, prepared, config):
    if args.probe_only:
        if args.probe_dir or args.probe_sha256:
            raise CognitiveKernelContractError('probe cannot depend on a probe')
        return None
    if not args.probe_dir or not args.probe_sha256:
        raise CognitiveKernelContractError('teacher fit requires externally pinned completed actual stress/restart probe')
    receipt = shared._read_sealed(args.probe_dir / 'component.json', ARTIFACT_SCHEMA)
    run = shared._read_sealed(args.probe_dir / 'run.json', RUN_SCHEMA)
    if receipt['record_sha256'] != args.probe_sha256 or receipt.get('run_sha256') != run['record_sha256'] or receipt.get('source_view_sha256') != prepared.view['record_sha256'] or receipt.get('specialist_config') != config.record() or receipt.get('probe_only') is not True or receipt.get('qualified_for_product') is not False or run.get('code_binding') != code_binding() or run.get('seed') != args.seed or run.get('gradient_accumulation') != args.gradient_accumulation or run.get('learning_rate') != args.learning_rate or run.get('device') != str(prepared.device) or run.get('initialization') != INITIALIZATION or run.get('objective') != OBJECTIVE:
        raise CognitiveKernelContractError('decision fit probe lineage differs')
    result = receipt.get('result', {})
    expected_stress = [e.case_id for e in stress_examples(prepared, args.gradient_accumulation)]
    if result.get('backward_and_restart_passed') is not True or result.get('checkpoint_reload_executed') is not True or result.get('optimizer_steps') != 3 or result.get('stress_case_ids') != expected_stress or shared._digest(args.probe_dir / 'probe-specialist.safetensors') != result.get('probe_weights_sha256'):
        raise CognitiveKernelContractError('decision fit stress/restart proof differs')
    training._verify_seed_control(args.probe_dir, run['record_sha256'], config)
    return receipt


def run(args):
    import torch
    from safetensors.torch import save_file, load_file
    prepared = source.load(args)
    config = prepared.config
    stress_receipt = verify_probe(args, prepared, config)
    torch.manual_seed(args.seed)
    if prepared.device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(prepared.device)
    specialist = FormationDecisionSpecialist(config).to(prepared.device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(specialist.parameters(), lr=args.learning_rate)
    manifest = {'schema': RUN_SCHEMA, 'objective': OBJECTIVE, 'initialization': INITIALIZATION,
        'source_view_sha256': prepared.view['record_sha256'], 'code_binding': code_binding(),
        'corpus_sha256': args.input_sha256, 'owner_authorization_ref': args.owner_authorization_ref,
        'specialist_config': config.record(), 'specialist_dtype': 'float32', 'device': str(prepared.device),
        'torch_version': torch.__version__, 'seed': args.seed, 'learning_rate': args.learning_rate,
        'gradient_accumulation': args.gradient_accumulation, 'epochs': args.epochs,
        'stress_probe_sha256': stress_receipt['record_sha256'] if stress_receipt else None,
        'probe_only': args.probe_only, 'teacher_fit': True, 'full_fit': False,
        'independent_gold': False, 'final_payloads_opened': False, 'qualified_for_product': False}
    digest = shared._record_hash(manifest)
    if args.resume_checkpoint:
        if args.probe_only or not args.output_dir.is_dir() or (args.output_dir / 'component.json').exists() or shared._read_sealed(args.output_dir / 'run.json', RUN_SCHEMA)['record_sha256'] != digest:
            raise CognitiveKernelContractError('decision resume needs matching incomplete run')
        training._verify_seed_control(args.output_dir, digest, config)
        epoch_cursor, case_cursor, steps = training._resume(args.resume_checkpoint, args.output_dir, specialist, optimizer, digest)
    else:
        args.output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
        shared._write_new(args.output_dir / 'run.json', manifest)
        seed = training._seed_control(args.output_dir, specialist, digest, config)
        if stress_receipt:
            stress_run = shared._read_sealed(args.probe_dir / 'run.json', RUN_SCHEMA)
            stress_seed = training._verify_seed_control(args.probe_dir, stress_run['record_sha256'], config)
            if seed['specialist_sha256'] != stress_seed['specialist_sha256']:
                raise CognitiveKernelContractError('decision fit seed differs from stress probe')
        epoch_cursor = case_cursor = steps = 0
    started = perf_counter()
    if args.probe_only:
        result = probe(specialist, optimizer, prepared, args, digest)
    else:
        for epoch in range(epoch_cursor, args.epochs):
            order = list(prepared.train)
            random.Random(args.seed + epoch).shuffle(order)
            cursor = case_cursor if epoch == epoch_cursor else 0
            if not 0 <= cursor <= len(order) or (cursor != len(order) and cursor % args.gradient_accumulation):
                raise CognitiveKernelContractError('decision resume cursor is not an optimizer boundary')
            while cursor < len(order):
                step = optimizer_step(specialist, optimizer, order[cursor:cursor + args.gradient_accumulation], prepared=prepared)
                cursor += step['train_examples']
                steps += 1
                pause = args.session_seconds is not None and perf_counter() - started >= args.session_seconds
                if steps % args.save_every_steps == 0 or pause or cursor == len(order):
                    training._checkpoint(args.output_dir, specialist=specialist, optimizer=optimizer,
                        run_digest=digest, epoch=epoch, next_case=cursor, step=steps)
                print(json.dumps({'optimizer_step': steps, 'epoch': epoch, 'train_cases_in_epoch': cursor, **step}), flush=True)
                if pause and not (epoch == args.epochs - 1 and cursor == len(order)):
                    return {'state': 'CHECKPOINTED_INCOMPLETE', 'optimizer_steps': steps,
                        'resume_checkpoint': str(args.output_dir / f'checkpoint-{steps:08d}'), 'qualified_for_product': False}
        path = args.output_dir / 'formation-specialist.safetensors'
        save_file({k: v.detach().cpu().contiguous() for k, v in specialist.state_dict().items()}, str(path))
        trained_loss = development_loss(specialist, prepared)
        specialist.load_state_dict(load_file(str(args.output_dir / 'seed-control.safetensors')))
        seed_loss = development_loss(specialist, prepared)
        specialist.load_state_dict(load_file(str(path)))
        seed = training._verify_seed_control(args.output_dir, digest, config)
        if shared._digest(path) == seed['specialist_sha256']:
            raise CognitiveKernelContractError('decision fit changed no weights')
        result = {'optimizer_steps': steps, 'all_train_epochs_completed': True,
            'trained_weights_sha256': shared._digest(path), 'diagnostic_development_teacher_forced_loss': trained_loss,
            'seeded_development_teacher_forced_loss': seed_loss, 'development_has_no_gradients': True,
            'generated_ability_established': False, 'independent_qualification': False}
    if prepared.device.type == 'cuda':
        torch.cuda.synchronize(prepared.device)
        result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(prepared.device)
    receipt = shared._write_new(args.output_dir / 'component.json', {
        'schema': ARTIFACT_SCHEMA, 'run_sha256': digest, 'source_view_sha256': prepared.view['record_sha256'],
        'specialist_config': config.record(), 'probe_only': args.probe_only,
        'teacher_fit': True, 'full_fit': False, 'qualified_for_product': False, 'result': result})
    return {'state': 'PROBE_COMPLETED' if args.probe_only else 'TEACHER_FIT_COMPLETED_UNQUALIFIED',
            'component_sha256': receipt['record_sha256'], 'seconds': perf_counter() - started, **result,
            'qualified_for_product': False}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('teacher-training-manifest', 'preflight-receipt', 'feature-bank', 'component-dir', 'source-view', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('input-sha256', 'owner-authorization-ref', 'feature-bank-sha256', 'component-sha256', 'source-view-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--gradient-accumulation', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--save-every-steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=73129)
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--probe-dir', type=Path)
    parser.add_argument('--probe-sha256')
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--session-seconds', type=int)
    args = parser.parse_args()
    if any(v < 1 for v in (args.epochs, args.gradient_accumulation, args.save_every_steps)) or not args.learning_rate > 0 or (args.session_seconds is not None and args.session_seconds < 1):
        raise CognitiveKernelContractError('invalid complete decision training recipe')
    return args


def main():
    os.umask(0o077)
    os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1', 'WANDB_DISABLED': 'true'})
    print(json.dumps(run(arguments()), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
