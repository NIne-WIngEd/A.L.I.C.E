"""Read-only diagnosis of language/prefix and source-identity transfer.

Re-admits all corrected cases. Measures positive train-token exposure and
source token availability, full teacher-prefix accuracy, and already retained
greedy divergence from the exact teacher. Novel IDs here are not tokenizer OOVs. Copy availability
is a token-membership proxy, not a source-span/semantic or pointer-model result.
No optimization, next model, grammar repair, activation or FINAL.
"""
from collections import Counter
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from cognitive_kernel.canonical import canonical_json_bytes, CognitiveKernelContractError, require_sha256
from . import formation_source_view as source
from . import train_v16_decision_specialist as trainer
from . import analyze_v16_cached_fit as parent_analysis
from . import run_v16_decision_diagnostic as generated

SCHEMA = 'mfm-v16-decision-fit-language-analysis-v1'


def scalar_spans(record):
    """Serialize recursively, retaining exact field provenance for all scalars."""
    pieces, spans, position = [], [], 0
    def add(text, kind=None):
        nonlocal position
        pieces.append(text)
        if kind is not None:
            spans.append((position, position + len(text), kind))
        position += len(text)
    def emit(value, path):
        if type(value) is dict:
            add('{')
            for index, key in enumerate(sorted(value)):
                if index: add(',')
                add(canonical_json_bytes(key).decode('utf-8'), 'key')
                add(':')
                emit(value[key], (*path, key))
            add('}')
        elif type(value) is list:
            add('[')
            for index, item in enumerate(value):
                if index: add(',')
                emit(item, (*path, index))
            add(']')
        else:
            fields = [part for part in path if isinstance(part, str)]
            field = fields[-1] if fields else ''
            kind = 'value_text' if field == 'value_text' else (
                'reference_identity' if type(value) is str and field.endswith(('_ref', '_refs')) else 'value')
            add(canonical_json_bytes(value).decode('utf-8'), kind)
    emit(record, ())
    answer = ''.join(pieces)
    if answer.encode('utf-8') != canonical_json_bytes(record):
        raise CognitiveKernelContractError('field-span serialization differs from actual canonical decision')
    return answer, spans


def target_groups(tokenizer, example, maximum):
    answer, spans = scalar_spans(source.preparation.decision_for_example(example))
    encoded = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
    inputs, labels = source.preparation.target_ids(tokenizer, example, maximum)
    if list(encoded['input_ids']) != labels[:-1] or inputs[1:] != labels[:-1]:
        raise CognitiveKernelContractError('decision field analysis token shift differs')
    groups = []
    for left, right in encoded['offset_mapping']:
        overlap = [(max(0, min(right, end) - max(left, start)), kind) for start, end, kind in spans]
        size, kind = max(overlap, default=(0, 'structure'))
        groups.append(kind if size else 'structure')
    return answer, labels, [*groups, 'eos']


def exposure_counts(labels, groups, train_counts, source_ids):
    if len(labels) != len(groups):
        raise CognitiveKernelContractError('lexical exposure groups differ')
    result = {}
    for token, group in zip(labels, groups):
        row = result.setdefault(group, {'tokens': 0, 'no_positive_train_corpus_occurrence': 0,
            'no_positive_train_corpus_occurrence_present_in_source': 0, 'source_token_present': 0,
            'fewer_than_10_positive_train_corpus_occurrences': 0})
        row['tokens'] += 1
        unseen = train_counts[token] == 0
        available = token in source_ids
        row['no_positive_train_corpus_occurrence'] += unseen
        row['no_positive_train_corpus_occurrence_present_in_source'] += unseen and available
        row['source_token_present'] += available
        row['fewer_than_10_positive_train_corpus_occurrences'] += train_counts[token] < 10
    return result


def merge_counts(total, observed):
    for group, counts in observed.items():
        target = total.setdefault(group, Counter())
        target.update(counts)


def measure_with_exposure(model, payload, groups, train_counts):
    """Stratify the same forward pass; frequency is one corpus pass, not epochs."""
    labels = payload['labels'].flatten().tolist()
    if len(labels) != len(groups):
        raise CognitiveKernelContractError('stratified language token groups differ')
    buckets = ['0' if train_counts[token] == 0 else
               '1-9' if train_counts[token] < 10 else '10+' for token in labels]
    joint = [f'{field}|{bucket}' for field, bucket in zip(groups, buckets)]
    result = parent_analysis.measure(model, payload, joint)
    strata, totals = {}, {}
    for key, row in result['groups'].items():
        field, bucket = key.split('|')
        strata.setdefault(field, {})[bucket] = row
        total = totals.setdefault(field, {'tokens': 0, 'loss_sum': 0., 'correct': 0.})
        total['tokens'] += row['tokens']
        total['loss_sum'] += row['tokens'] * row['mean_loss']
        total['correct'] += row['tokens'] * row['accuracy']
    result['groups'] = {field: {'tokens': row['tokens'],
        'mean_loss': row['loss_sum'] / row['tokens'],
        'accuracy': row['correct'] / row['tokens']} for field, row in totals.items()}
    result['groups_by_positive_train_corpus_frequency'] = strata
    return result


def run(args):
    started = perf_counter()
    prepared = source.load(args)  # Enforces privacy before corpus/view reads.
    component = source.shared._read_sealed(args.decision_component_dir / 'component.json', trainer.ARTIFACT_SCHEMA)
    manifest = source.shared._read_sealed(args.decision_component_dir / 'run.json', trainer.RUN_SCHEMA)
    if component['record_sha256'] != require_sha256(args.decision_component_sha256, 'decision_component_sha256') or component.get('run_sha256') != manifest['record_sha256'] or component.get('source_view_sha256') != prepared.view['record_sha256'] or component.get('specialist_config') != prepared.config.record() or component.get('probe_only') is not False or component.get('qualified_for_product') is not False or manifest.get('code_binding') != trainer.code_binding() or manifest.get('objective') != trainer.OBJECTIVE or manifest.get('initialization') != trainer.INITIALIZATION or manifest.get('specialist_dtype') != 'float32':
        raise CognitiveKernelContractError('language diagnosis component lineage differs')
    result = component.get('result', {})
    weights = args.decision_component_dir / 'formation-specialist.safetensors'
    if result.get('all_train_epochs_completed') is not True or source.shared._digest(weights) != result.get('trained_weights_sha256'):
        raise CognitiveKernelContractError('language diagnosis needs completed pinned fit')
    diagnostic = source.shared._read_sealed(args.diagnostic_dir / 'diagnostic.json', generated.SCHEMA)
    if diagnostic['record_sha256'] != require_sha256(args.diagnostic_sha256, 'diagnostic_sha256') or diagnostic.get('component_sha256') != component['record_sha256'] or diagnostic.get('run_sha256') != manifest['record_sha256'] or diagnostic.get('source_view_sha256') != prepared.view['record_sha256'] or diagnostic.get('runner_sha256') != source.shared._digest(Path(generated.__file__)):
        raise CognitiveKernelContractError('language diagnosis generated receipt differs')
    output_file = args.diagnostic_dir / 'trained.jsonl'
    if source.shared._digest(output_file) != diagnostic['outputs_sha256']['trained']:
        raise CognitiveKernelContractError('language diagnosis retained output bytes differ')
    retained = [json.loads(line, object_pairs_hook=generated._unique_json) for line in output_file.read_text(encoding='utf-8').splitlines()]
    if len(retained) != len(prepared.development):
        raise CognitiveKernelContractError('language diagnosis retained development roster differs')
    train_counts, development_counts, all_targets = Counter(), Counter(), {}
    for example in (*prepared.train, *prepared.development):
        answer, labels, groups = target_groups(prepared.tokenizer, example, prepared.config.max_target_tokens)
        if prepared.tokenizer.decode(labels[:-1], skip_special_tokens=False) != answer:
            raise CognitiveKernelContractError('language diagnosis full target roundtrip differs')
        all_targets[example.case_id] = (labels, groups)
        (train_counts if example.split == 'train' else development_counts).update(labels)
    exposures = {'train': {}, 'development': {}}
    for example in (*prepared.train, *prepared.development):
        root, export = prepared.locators[example.case_id]
        _, tensors = source.features.read_case(root, example, export, width=prepared.config.base_hidden_size)
        ids = set(tensors['input_ids'][tensors['attention_mask'].bool()].tolist())
        labels, groups = all_targets[example.case_id]
        merge_counts(exposures[example.split], exposure_counts(labels, groups, train_counts, ids))
    divergence = []
    for row, example in zip(retained, prepared.development):
        if row.get('case_id') != example.case_id or row.get('split') != 'development' or row.get('control') != 'trained' or row.get('model_input_sha256') != source.features.case_key(example) or row.get('weights_sha256') != result['trained_weights_sha256']:
            raise CognitiveKernelContractError('language diagnosis retained case binding differs')
        labels, groups = all_targets[example.case_id]
        tokens = row['generated_token_ids']
        first = next((i for i, (observed, expected) in enumerate(zip(tokens, labels)) if observed != expected), None)
        if first is None and len(tokens) != len(labels) - int(row['eos_observed']):
            first = min(len(tokens), len(labels))
        comparison = row['incremental_parity']
        divergence.append({'case_id': example.case_id, 'first_exact_teacher_divergence_token': first,
            'field_kind': groups[first] if first is not None and first < len(groups) else None,
            'first_divergence_target_positive_train_corpus_count': train_counts[labels[first]] if first is not None and first < len(labels) else None,
            'inside_reference_checked_prefix': first is not None and first < comparison['compared_greedy_prefixes'] and comparison['argmaxes_equal'] is True,
            'distinct_generated_tokens': len(set(tokens)), 'eos_observed': row['eos_observed'],
            'retained_validation_status': row['validation_status']})
    import torch
    from safetensors.torch import load_file
    torch.manual_seed(manifest['seed'])
    model = trainer.FormationDecisionSpecialist(prepared.config).to(prepared.device, dtype=torch.float32)
    model.load_state_dict(load_file(str(weights)))
    selected_ids = list(dict.fromkeys((*prepared.view['stress_train_case_ids'], 'v16-seed-06', 'v16-seed-11')))
    selected_train = [e for e in prepared.train if e.case_id in selected_ids]
    selected_train.extend(e for e in prepared.train if e.case_id not in selected_ids)
    selected = [*selected_train[:max(3, len(selected_ids))], *prepared.development[:3]]
    measurements = []
    for example in selected:
        payload = source.payload(prepared, example, device=prepared.device)
        _, groups = all_targets[example.case_id]
        for intervention in ('matched', 'all-BOS-teacher-input', 'zero-source'):
            changed = dict(payload)
            if intervention == 'all-BOS-teacher-input':
                changed['input_ids'] = torch.full_like(payload['input_ids'], prepared.config.start_token_id)
            elif intervention == 'zero-source':
                changed['base_states'] = torch.zeros_like(payload['base_states'])
            measurement = measure_with_exposure(model, changed, groups, train_counts)
            measurements.append({'case_id': example.case_id, 'split': example.split,
                'intervention': intervention, **measurement})
            print(json.dumps({'case_id': example.case_id, 'intervention': intervention,
                'mean_loss': measurement['mean_loss'], 'accuracy': measurement['next_token_accuracy']}), flush=True)
    if any(p.grad is not None for p in model.parameters()):
        raise CognitiveKernelContractError('read-only language diagnosis acquired gradients')
    return source.shared._write_new(args.output, {
        'schema': SCHEMA, 'source_view_sha256': prepared.view['record_sha256'],
        'component_sha256': component['record_sha256'], 'run_sha256': manifest['record_sha256'],
        'diagnostic_sha256': diagnostic['record_sha256'], 'analyzer_sha256': source.shared._digest(Path(__file__)),
        'parent_measure_implementation_sha256': source.shared._digest(Path(parent_analysis.__file__)),
        'actual_runtime': {'torch_version': torch.__version__, 'device': str(prepared.device), 'dtype': 'float32'},
        'completed_training_epochs': manifest['epochs'],
        'frequency_basis': 'positive target occurrences in one complete admitted training corpus pass; every case was presented once per completed epoch',
        'positive_train_corpus_token_types': len(train_counts), 'positive_development_corpus_token_types': len(development_counts),
        'development_token_types_without_positive_train_corpus_occurrence': len(set(development_counts) - set(train_counts)),
        'exposure_by_split_and_field': exposures, 'retained_greedy_exact_teacher_divergence': divergence,
        'full_teacher_prefix_measurements': measurements, 'no_optimization_or_weight_export': True,
        'scope': 'bounded language/identity transfer diagnosis; gold-prefix and token-membership proxies are not semantic grounding',
        'independent_gold': False, 'qualified_for_product': False, 'final_payloads_opened': False,
        'seconds': perf_counter() - started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('teacher-training-manifest', 'preflight-receipt', 'feature-bank', 'component-dir', 'source-view', 'decision-component-dir', 'diagnostic-dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('input-sha256', 'owner-authorization-ref', 'feature-bank-sha256', 'component-sha256', 'source-view-sha256', 'decision-component-sha256', 'diagnostic-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    os.umask(0o077)
    os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1', 'WANDB_DISABLED': 'true'})
    result = run(parser.parse_args())
    print(json.dumps({'language_analysis_sha256': result['record_sha256'], 'seconds': result['seconds'], 'qualified_for_product': False}), flush=True)


if __name__ == '__main__':
    main()
