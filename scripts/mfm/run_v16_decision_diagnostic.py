"""Target-free matched generated diagnostics; preserve every invalid output.

Actual FP32 incremental decoding verifies its first 32 greedy prefix logits and
argmaxes against full-prefix inference for every case/control before caching
the remainder. No token penalties, grammar constraints or content repair.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from cognitive_kernel.canonical import canonical_json_bytes, CognitiveKernelContractError, require_sha256
from cognitive_kernel.formation_decision_v16 import compile_decision
from cognitive_kernel.formation_semantics_v16 import validate_formation_grounding_v16
from . import train_v16_decision_specialist as trainer
from . import formation_source_view as source
from .run_v16_cached_diagnostic import _unique_json

SCHEMA = 'mfm-v16-decision-generated-diagnostic-v1'


def greedy_fp32(specialist, states, mask, maximum):
    import torch
    from cognitive_kernel.formation_incremental_decoder import CachedFormationDecoder
    if type(maximum) is not int or not 1 <= maximum <= specialist.config.max_target_tokens or any(
            p.dtype != torch.float32 for p in specialist.parameters()):
        raise CognitiveKernelContractError('decision diagnostic requires complete FP32 decoding budget')
    specialist.eval()
    device = next(specialist.parameters()).device
    states, mask = states.to(device), mask.to(device)
    prefix, tokens = [specialist.config.start_token_id], []
    compared, maximum_difference = 0, 0.0
    started = perf_counter()
    with torch.inference_mode(), torch.autocast(device.type, enabled=False):
        cache = CachedFormationDecoder(specialist, base_states=states, source_mask=mask)
        current = specialist.config.start_token_id
        for index in range(maximum):
            logits = cache.step(current)
            if not torch.isfinite(logits).all():
                raise CognitiveKernelContractError('decision generated logits are nonfinite')
            if index < 32:
                reference = specialist.next_token_logits(base_states=states, source_mask=mask,
                    input_ids=torch.tensor([prefix], device=device))
                difference = float((reference - logits).abs().max().cpu())
                if not torch.allclose(reference, logits, atol=1e-4, rtol=1e-4) or reference.argmax().item() != logits.argmax().item():
                    raise CognitiveKernelContractError('actual FP32 incremental greedy prefix differs from reference')
                maximum_difference = max(maximum_difference, difference)
                compared += 1
            current = int(logits[0].argmax())
            if current == specialist.config.end_token_id:
                return tokens, True, {'compared_greedy_prefixes': compared, 'max_abs_difference': maximum_difference,
                                     'atol': 1e-4, 'rtol': 1e-4, 'argmaxes_equal': True, 'seconds': perf_counter() - started}
            tokens.append(current)
            prefix.append(current)
    return tokens, False, {'compared_greedy_prefixes': compared, 'max_abs_difference': maximum_difference,
                         'atol': 1e-4, 'rtol': 1e-4, 'argmaxes_equal': True, 'seconds': perf_counter() - started}


def validate(raw, ended, example, weights_sha256, run_id):
    result = {'raw_output_text': raw, 'eos_observed': ended, 'validation_status': 'invalid',
              'validation_error': None, 'exact_teacher_decision_agreement': None, 'proposal_count': 0}
    if not ended:
        result['validation_error'] = 'no EOS inside declared complete diagnostic budget'
        return result
    try:
        decision = json.loads(raw, object_pairs_hook=_unique_json)
        bundle = compile_decision(example.context, decision, artifact_sha256=weights_sha256, inference_run_id=run_id)
        validate_formation_grounding_v16(example.context, bundle, example.opened_sources)
    except (ValueError, TypeError, KeyError, CognitiveKernelContractError) as exc:
        result['validation_error'] = f'{type(exc).__name__}: {exc}'
        return result
    result.update(validation_status='grounded_teacher_proposal_only',
        proposal_count=len(bundle.proposals), exact_teacher_decision_agreement=decision == source.preparation.decision_for_example(example))
    return result


def run(args):
    from safetensors.torch import load_file
    import torch
    import cognitive_kernel.formation_incremental_decoder as incremental
    prepared = source.load(args)
    component = source.shared._read_sealed(args.decision_component_dir / 'component.json', trainer.ARTIFACT_SCHEMA)
    manifest = source.shared._read_sealed(args.decision_component_dir / 'run.json', trainer.RUN_SCHEMA)
    if component['record_sha256'] != require_sha256(args.decision_component_sha256, 'decision_component_sha256') or component.get('run_sha256') != manifest['record_sha256'] or component.get('source_view_sha256') != prepared.view['record_sha256'] or component.get('probe_only') is not False or component.get('qualified_for_product') is not False or component.get('specialist_config') != prepared.config.record() or manifest.get('code_binding') != trainer.code_binding() or manifest.get('initialization') != trainer.INITIALIZATION or manifest.get('objective') != trainer.OBJECTIVE or manifest.get('specialist_dtype') != 'float32' or manifest.get('source_view_sha256') != prepared.view['record_sha256']:
        raise CognitiveKernelContractError('decision generated component lineage differs')
    result = component.get('result', {})
    trained = args.decision_component_dir / 'formation-specialist.safetensors'
    seed = trainer.training._verify_seed_control(args.decision_component_dir, manifest['record_sha256'], prepared.config)
    if result.get('all_train_epochs_completed') is not True or source.shared._digest(trained) != result.get('trained_weights_sha256') or seed['specialist_sha256'] == result.get('trained_weights_sha256') or not prepared.view['longest_target_tokens'] <= args.max_new_tokens <= prepared.config.max_target_tokens:
        raise CognitiveKernelContractError('decision generated fit/weights/budget differs')
    args.output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    specialist = trainer.FormationDecisionSpecialist(prepared.config).to(prepared.device, dtype=torch.float32)
    digests, summaries = {}, {}
    for control, path, digest in (('trained', trained, result['trained_weights_sha256']),
        ('seeded-untrained', args.decision_component_dir / 'seed-control.safetensors', seed['specialist_sha256'])):
        specialist.load_state_dict(load_file(str(path)))
        file = args.output_dir / (control + '.jsonl')
        summaries[control] = {'cases': 0, 'invalid_outputs': 0, 'grounded_outputs': 0,
                             'proposals_in_grounded_outputs': 0, 'exact_teacher_agreements': 0}
        with file.open('x', encoding='utf-8') as stream:
            for example in prepared.development:
                root, export = prepared.locators[example.case_id]
                _, tensors = source.features.read_case(root, example, export, width=prepared.config.base_hidden_size)
                tokens, ended, parity = greedy_fp32(specialist, tensors['states'], tensors['attention_mask'], args.max_new_tokens)
                run_id = manifest['record_sha256'] + '-' + control
                record = {'case_id': example.case_id, 'split': 'development', 'control': control,
                    'weights_sha256': digest, 'source_view_sha256': prepared.view['record_sha256'],
                    'context_sha256': example.context.content_digest(), 'model_input_sha256': source.features.case_key(example),
                    'inference_run_id': run_id, 'generated_token_ids': tokens, 'incremental_parity': parity,
                    **validate(prepared.tokenizer.decode(tokens, skip_special_tokens=False), ended, example, digest, run_id)}
                stream.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
                stream.flush()
                summaries[control]['cases'] += 1
                summaries[control]['invalid_outputs' if record['validation_status'] == 'invalid' else 'grounded_outputs'] += 1
                summaries[control]['proposals_in_grounded_outputs'] += record['proposal_count']
                summaries[control]['exact_teacher_agreements'] += record['exact_teacher_decision_agreement'] is True
                print(json.dumps({'control': control, 'case_id': example.case_id, 'validation_status': record['validation_status'],
                                  'eos_observed': ended, 'generated_tokens': len(tokens), 'seconds': parity['seconds']}), flush=True)
        digests[control] = source.shared._digest(file)
    return source.shared._write_new(args.output_dir / 'diagnostic.json', {
        'schema': SCHEMA, 'component_sha256': component['record_sha256'], 'run_sha256': manifest['record_sha256'],
        'source_view_sha256': prepared.view['record_sha256'], 'outputs_sha256': digests, 'summary': summaries,
        'runner_sha256': source.shared._digest(Path(__file__)), 'incremental_decoder_sha256': source.shared._digest(Path(incremental.__file__)),
        'decode': {'method': 'FP32_incremental_first32_actual_greedy_reference_checked', 'max_new_tokens': args.max_new_tokens,
                   'device': str(prepared.device), 'specialist_dtype': 'float32', 'torch_version': torch.__version__,
                   'target_visible': False, 'matched_between_controls': True},
        'agreement_metric': 'exact_teacher_decision_fields_not_semantic_entailment',
        'independent_gold': False, 'final_payloads_opened': False, 'memory_gate_executed': False,
        'qualified_for_product': False, 'downstream_usefulness_established': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('teacher-training-manifest', 'preflight-receipt', 'feature-bank', 'component-dir', 'source-view', 'decision-component-dir', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('input-sha256', 'owner-authorization-ref', 'feature-bank-sha256', 'component-sha256', 'source-view-sha256', 'decision-component-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    os.umask(0o077)
    os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1', 'WANDB_DISABLED': 'true'})
    result = run(parser.parse_args())
    print(json.dumps({'decision_diagnostic_sha256': result['record_sha256'], 'summary': result['summary'], 'qualified_for_product': False}), flush=True)


if __name__ == '__main__':
    main()
