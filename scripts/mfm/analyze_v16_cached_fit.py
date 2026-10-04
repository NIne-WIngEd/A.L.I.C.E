"""Read-only source/target/objective diagnosis of the completed teacher fit.

No optimization, output repair, private activation or FINAL access. Full real
source/target lengths are retained. Counterfactual sources diagnose dependence;
the embedding intervention is an unfitted causal probe, never a new component.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
from time import perf_counter

from cognitive_kernel.canonical import canonical_json_bytes, CognitiveKernelContractError
from . import run_v16_cached_diagnostic as diagnostic

SCHEMA = "mfm-v16-cached-fit-root-analysis-v1"


def measure(model, payload, groups):
    import torch
    from torch.nn import functional as F
    if len(groups) != payload['labels'].numel():
        raise CognitiveKernelContractError('analysis token groups differ')
    model.eval()
    totals = {}
    correct = copied = 0
    losses = []
    with torch.inference_mode(), torch.autocast(payload['base_states'].device.type, enabled=False):
        hidden = model._decode(base_states=payload['base_states'],
            source_mask=payload['source_mask'], input_ids=payload['input_ids'])
        for start in range(0, hidden.shape[1], model.config.logit_chunk_tokens):
            end = min(start + model.config.logit_chunk_tokens, hidden.shape[1])
            logits = F.linear(hidden[:, start:end], model.token_embedding.weight, model.output_bias)
            labels = payload['labels'][:, start:end]
            nll = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none')
            predicted = logits.argmax(-1)
            correct += int(predicted.eq(labels).sum())
            copied += int(predicted.eq(payload['input_ids'][:, start:end]).sum())
            losses.extend(nll.tolist())
            for offset, loss in enumerate(nll.tolist()):
                item = totals.setdefault(groups[start + offset], {'tokens': 0, 'loss_sum': 0., 'correct': 0})
                item['tokens'] += 1
                item['loss_sum'] += loss
                item['correct'] += int(predicted[0, offset] == labels[0, offset])
            if start == 0:
                first = logits[0, 0].log_softmax(-1)
                first_measure = {'target_probability': float(first[labels[0, 0]].exp()),
                    'previous_input_probability': float(first[payload['input_ids'][0, 0]].exp()),
                    'entropy': float(-(first.exp() * first).sum())}
            if end == hidden.shape[1]:
                eos_log = logits[0, -1].log_softmax(-1)
                eos_probability = float(eos_log[model.config.end_token_id].exp())
    count = len(losses)
    return {'tokens': count, 'mean_loss': sum(losses) / count,
        'next_token_accuracy': correct / count, 'previous_input_copy_rate': copied / count,
        'first_position': first_measure, 'teacher_prefix_eos_probability': eos_probability,
        'groups': {key: {'tokens': value['tokens'], 'mean_loss': value['loss_sum'] / value['tokens'],
                        'accuracy': value['correct'] / value['tokens']} for key, value in totals.items()}}


def target_groups(tokenizer, example, maximum):
    training = diagnostic.cached.training
    answer = canonical_json_bytes(training.supervised_output_record_v16(example)).decode('utf-8')
    encoded = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
    _, labels = training._target_ids(tokenizer, example, maximum)
    if list(encoded['input_ids']) != labels[:-1]:
        raise CognitiveKernelContractError('analysis tokenizer alignment differs')
    spans = []
    # JSON scalars include numeric offsets/confidence, booleans and null;
    # treating these as punctuation would hide grounding-critical errors.
    scalar = r'"(?:[^"\\]|\\.)*"|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null'
    for match in re.finditer(scalar, answer):
        rest = answer[match.end():].lstrip()
        is_key = match.group().startswith('"') and rest.startswith(':')
        spans.append((match.start(), match.end(), 'key' if is_key else 'value'))
    groups = []
    for left, right in encoded['offset_mapping']:
        overlap = [(max(0, min(right, end) - max(left, start)), kind) for start, end, kind in spans]
        size, kind = max(overlap, default=(0, 'structure'))
        groups.append(kind if size else 'structure')
    return answer, labels, [*groups, 'eos']


def run(args):
    started = perf_counter()
    p = diagnostic.prepare_diagnostic(args)
    import torch
    from safetensors.torch import load_file
    from cognitive_kernel.formation_v1_specialist import FormationSpecialist
    torch.manual_seed(p.run_record['seed'])
    model = FormationSpecialist(p.config).to(p.device, dtype=torch.float32)
    counter = Counter()
    group_counts = Counter()
    lengths = []
    target_signatures = set()
    stress_ids = p.preflight['stress_train_case_ids']
    selected_train = [e for e in p.train if e.case_id in stress_ids]
    selected_train.extend(e for e in p.train if e.case_id not in stress_ids)
    selected = [*selected_train[:max(3, len(stress_ids))], *p.development[:3]]
    for example in (*p.train, *p.development):
        answer, labels, groups = target_groups(p.tokenizer, example, p.config.max_target_tokens)
        counter.update(labels)
        group_counts.update(groups)
        lengths.append(len(labels))
        target_signatures.add(answer)
        if p.tokenizer.decode(labels[:-1], skip_special_tokens=False) != answer:
            raise CognitiveKernelContractError('teacher target does not round-trip exactly')
    payloads = [diagnostic.cached._payload(args.feature_bank, e, p.export['record_sha256'],
        p.tokenizer, p.config, p.device) for e in selected]
    rows = []
    for condition in ('seeded-untrained', 'trained', 'untrained-embedding-scale-0.02'):
        name = 'seeded-untrained' if condition.startswith('untrained-') else condition
        model.load_state_dict(load_file(str(p.controls[name][0])))
        if condition.startswith('untrained-'):
            # A single intervention on original seed bytes, not trained weights.
            with torch.no_grad():
                model.token_embedding.weight.mul_(0.02)
                model.position_embedding.weight.mul_(0.02)
        embedding_std = float(model.token_embedding.weight.std())
        for index, (example, payload) in enumerate(zip(selected, payloads)):
            _, labels, groups = target_groups(p.tokenizer, example, p.config.max_target_tokens)
            if payload['labels'].flatten().tolist() != labels:
                raise CognitiveKernelContractError('analysis target shift differs')
            interventions = ('matched', 'zero-source', 'swapped-source') if condition == 'trained' else ('matched',)
            for intervention in interventions:
                changed = dict(payload)
                if intervention == 'zero-source':
                    changed['base_states'] = torch.zeros_like(payload['base_states'])
                elif intervention == 'swapped-source':
                    other = payloads[(index + 1) % len(payloads)]
                    changed['base_states'], changed['source_mask'] = other['base_states'], other['source_mask']
                result = measure(model, changed, groups)
                rows.append({'case_ordinal': index, 'split': example.split,
                    'model_input_sha256': diagnostic.features.case_key(example),
                    'condition': condition, 'source_intervention': intervention,
                    'embedding_std': embedding_std, 'source_tokens': changed['base_states'].shape[1],
                    'source_rms': float(changed['base_states'].float().square().mean().sqrt()), **result})
                print(json.dumps({'analysis_condition': condition, 'source_intervention': intervention,
                    'case_ordinal': index, 'mean_loss': result['mean_loss'],
                    'accuracy': result['next_token_accuracy']}, sort_keys=True), flush=True)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise CognitiveKernelContractError('analysis acquired gradients')
    return diagnostic.cached.training.shared._write_new(args.output, {
        'schema': SCHEMA, 'component_sha256': p.component['record_sha256'],
        'training_run_sha256': p.run_record['record_sha256'],
        'feature_bank_sha256': p.bank['record_sha256'], 'corpus_sha256': args.input_sha256,
        'analysis_sha256': diagnostic.cached.training.shared._digest(Path(__file__)),
        'corpus_statistics': {'train_cases': len(p.train), 'development_cases': len(p.development),
            'total_target_tokens': sum(lengths), 'unique_target_sequences': len(target_signatures),
            'unique_target_tokens': len(counter), 'token_groups': dict(group_counts),
            'minimum_target_length': min(lengths), 'maximum_target_length': max(lengths),
            'target_shift_and_tokenizer_roundtrip_verified': True},
        'uniform_vocabulary_loss': math.log(p.config.vocabulary_size), 'measurements': rows,
        'no_optimizer_or_weight_export': True, 'seconds': perf_counter() - started,
        'independent_gold': False, 'final_payloads_opened': False, 'qualified_for_product': False,
        'scope': 'bounded source/target/objective causal diagnosis; unfitted initialization intervention; no formation usefulness claim'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('teacher-training-manifest', 'preflight-receipt', 'feature-bank', 'component-dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('input-sha256', 'owner-authorization-ref', 'feature-bank-sha256', 'component-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    args = parser.parse_args()
    os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1', 'WANDB_DISABLED': 'true'})
    receipt = run(args)
    print(json.dumps({'analysis_sha256': receipt['record_sha256'], 'seconds': receipt['seconds']}), flush=True)


if __name__ == '__main__':
    main()
