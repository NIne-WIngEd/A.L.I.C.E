"""Prepare a separately bound source/decision view; never train or qualify it.

Re-admit the original corpus and completed parent bank. Exact unchanged input
states remain in that immutable bank. Only explicitly augmented source inputs
are forwarded through the frozen, verified Gemma role clone. A complete view
records every case, new target/length/stress and transitive code binding. It is
not a legacy feature bank or a claim that legacy preflight covers new inputs.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import os
import re
from pathlib import Path
from time import perf_counter

from cognitive_kernel.canonical import canonical_json_bytes, CognitiveKernelContractError
from cognitive_kernel.formation_decision_v16 import decision_record, OBJECTIVE
from . import run_v16_cached_diagnostic as diagnostic
from . import formation_feature_bank as features
from . import v16_prior_memory_inputs as prior

SCHEMA = 'mfm-v16-contract-successor-source-view-v1'
PATCH_SCHEMA = 'mfm-v16-contract-successor-feature-patch-v1'
training = diagnostic.cached.training
shared = training.shared


def reference_input_gaps(example):
    """Literal identity availability, not semantic correctness or an alias resolver."""
    textual = {ref.ref_id for ref in example.context.base.evidence if ref.modality in {'text', 'code', 'structured'}}
    visible = (canonical_json_bytes(example.context.record()) + b'\n' + b'\n'.join(
        raw for ref_id, raw in example.opened_sources if ref_id in textual)).decode('utf-8')
    local = {value for p in example.target.proposals for value in (p.base.proposal_id, p.base.value_ref)}
    existing = {ref for p in example.target.proposals for ref in (*p.base.target_refs, *p.base.contradicts)}
    existing.update(ref for d in example.target.base.dispositions for ref in d.target_refs)
    existing.update(p.base.uncertainty_ref for p in example.target.proposals if p.base.uncertainty_ref)
    local.update(prior.local_uncertainty_allocations(example))
    return sorted(ref for ref in existing - local if not re.search(
        r'(?<![\w-])' + re.escape(ref) + r'(?![\w-])', visible))


def decision_for_example(example):
    return decision_record(example, local_uncertainty_refs=prior.local_uncertainty_allocations(example))


def expected_augmentation_roster(examples):
    """Authoring fixtures are a superset, not the admitted corpus roster."""
    return {e.case_id for e in examples if e.case_id in prior.PUBLIC_SEED_PRIOR_MEMORY}


def verify_source_runtime(export, torch_version, transformers_version):
    if export.get('torch_version') != torch_version or export.get('transformers_version') != transformers_version:
        raise CognitiveKernelContractError('successor source runtime differs from parent export')
    return {'torch_version': torch_version, 'transformers_version': transformers_version,
            'source_device': 'cpu', 'source_dtype': 'bfloat16', 'base_frozen': True}


def target_ids(tokenizer, example, maximum):
    answer = canonical_json_bytes(decision_for_example(example)).decode('utf-8')
    body = tokenizer.encode(answer, add_special_tokens=False)
    if not body or len(body) + 1 > maximum:
        raise CognitiveKernelContractError('complete decision exceeds target budget; never truncate')
    if tokenizer.decode(body, skip_special_tokens=False) != answer:
        raise CognitiveKernelContractError('complete decision does not round-trip')
    return [tokenizer.bos_token_id, *body], [*body, tokenizer.eos_token_id]


def code_binding():
    import cognitive_kernel.formation_decision_v16 as decision
    return {name: shared._digest(Path(module.__file__)) for name, module in (
        ('preparation', __import__(__name__, fromlist=['run'])), ('prior_memory_inputs', prior),
        ('decision_interface', decision), ('parent_diagnostic', diagnostic),
        ('parent_cached_trainer', diagnostic.cached), ('parent_trainer', training),
        ('parent_features', features), ('shared', shared))}


def prepare(args):
    started = perf_counter()
    # This verifies privacy before any corpus read, every parent feature and
    # code/processor/corpus binding, plus the historical failed component.
    p = diagnostic.prepare_diagnostic(args)
    originals = (*p.train, *p.development)
    examples = tuple(prior.augment_public_seed_example(e) for e in originals)
    root = args.output_dir
    if root.exists() or root.is_symlink():
        raise CognitiveKernelContractError('successor needs a new owner-controlled output directory')
    root.mkdir(mode=0o700, parents=True)
    import torch
    import transformers
    runtime = verify_source_runtime(p.export, torch.__version__, transformers.__version__)
    from transformers import AutoProcessor, AutoModelForMultimodalLM
    processor = AutoProcessor.from_pretrained(str(args.feature_bank / 'processor'),
                                              local_files_only=True, trust_remote_code=False)
    rows, lengths, stress = [], [], {}
    for original, example in zip(originals, examples):
        encoded = training.source_batch_v16(processor, example.context, example.opened_sources,
            p.preflight['max_source_tokens'], case_id=example.case_id)
        inputs, labels = target_ids(processor.tokenizer, example, p.config.max_target_tokens)
        source_length = encoded['input_ids'].shape[1]
        pair_count = source_length * len(inputs) * p.config.heads * p.config.layers
        changed = features.case_key(example) != features.case_key(original)
        if not changed:
            _, tensors = features.read_case(args.feature_bank, original, p.export['record_sha256'],
                                           width=p.config.base_hidden_size)
            if not torch.equal(tensors['input_ids'], encoded['input_ids']) or not torch.equal(tensors['attention_mask'], encoded['attention_mask']):
                raise CognitiveKernelContractError('unchanged source encoding differs from parent bank')
        gaps = reference_input_gaps(example)
        rows.append({'case_id': example.case_id, 'split': example.split,
            'parent_model_input_sha256': features.case_key(original),
            'model_input_sha256': features.case_key(example), 'source_augmented': changed,
            'decision_target_sha256': sha256(canonical_json_bytes(decision_for_example(example))).hexdigest(),
            'source_tokens': source_length, 'target_tokens': len(labels),
            'existing_reference_input_gaps': gaps})
        lengths.append(len(labels))
        if example.split == 'train':
            for name, size in (('source', source_length), ('target', len(labels)), ('joint', pair_count)):
                if name not in stress or size > stress[name][0]:
                    stress[name] = (size, example.case_id)
    expected = expected_augmentation_roster(originals)
    actual = {row['case_id'] for row in rows if row['source_augmented']}
    if actual != expected:
        raise CognitiveKernelContractError('explicit prior-memory augmentation roster differs')
    prepared = shared._prepared_base(args)
    if prepared['receipt_sha256'] != p.export['prepared_base_receipt_sha256']:
        raise CognitiveKernelContractError('successor role clone differs from parent export')
    patch_record = shared._write_new(root / 'export.json', {
        'schema': PATCH_SCHEMA, 'parent_feature_bank_sha256': p.bank['record_sha256'],
        'parent_export_sha256': p.export['record_sha256'],
        'parent_processor_preflight_sha256': p.preflight['record_sha256'],
        'corpus_sha256': args.input_sha256, 'code_binding': code_binding(),
        'actual_runtime': runtime, 'prepared_base_receipt_sha256': prepared['receipt_sha256'],
        'prepared_base_weight_sha256': p.export['prepared_base_weight_sha256'],
        'representation': 'base.model.last_hidden_state', 'base_frozen': True,
        'language_head_used': False, 'target_visible_to_backbone': False,
        'qualified_for_product': False})
    # Independently verify the role clone before loading source weights.
    base = AutoModelForMultimodalLM.from_pretrained(prepared['snapshot_path'],
        trust_remote_code=False, local_files_only=True, use_safetensors=True,
        dtype=torch.bfloat16, device_map={'': 'cpu'}, low_cpu_mem_usage=True)
    base.requires_grad_(False).eval()
    if getattr(base.config, 'model_type', None) != 'gemma4_unified' or not hasattr(base, 'model') or any(
        q.device.type != 'cpu' or q.dtype != torch.bfloat16 or q.requires_grad for q in base.parameters()):
        raise CognitiveKernelContractError('successor source model geometry/placement/freeze differs')
    for example, row in zip(examples, rows):
        if row['source_augmented']:
            encoded = training.source_batch_v16(processor, example.context, example.opened_sources,
                p.preflight['max_source_tokens'], case_id=example.case_id)
            with torch.no_grad():
                states = base.model(**encoded, use_cache=False, return_dict=True).last_hidden_state
            case = features.write_case(root, example, encoded, states, patch_record['record_sha256'])
            row['patch_case_receipt_sha256'] = case['record_sha256']
            print(json.dumps({'event': 'augmented_source_exported', 'case_id': example.case_id,
                              'source_tokens': row['source_tokens']}), flush=True)
            del states, encoded
    return shared._write_new(root / 'view.json', {
        'schema': SCHEMA, 'objective': OBJECTIVE, 'corpus_sha256': args.input_sha256,
        'owner_authorization_ref': args.owner_authorization_ref,
        'parent_component_sha256': p.component['record_sha256'],
        'parent_feature_bank_sha256': p.bank['record_sha256'],
        'parent_export_sha256': p.export['record_sha256'],
        'parent_preflight_sha256': p.preflight['record_sha256'],
        'patch_export_sha256': patch_record['record_sha256'], 'code_binding': code_binding(),
        'actual_runtime': runtime, 'prepared_base_receipt_sha256': prepared['receipt_sha256'],
        'train_cases': len(p.train), 'development_cases': len(p.development),
        'unchanged_source_cases_reused': len(rows) - len(actual), 'augmented_source_cases': len(actual),
        'cases': rows, 'complete': True, 'max_target_tokens': p.config.max_target_tokens,
        'longest_source_tokens': max(r['source_tokens'] for r in rows),
        'longest_target_tokens': max(lengths),
        'stress_train_case_ids': list(dict.fromkeys(value[1] for value in stress.values())),
        'peak_train_attention_pairs': stress['joint'][0],
        'unresolved_reference_input_gap_cases': sum(bool(r['existing_reference_input_gaps']) for r in rows),
        'gap_check_scope': 'literal ID availability; aliases/meaning are not judged',
        'independent_gold': False, 'final_payloads_opened': False, 'full_fit': False,
        'qualified_for_product': False, 'memory_gate_executed': False,
        'seconds': perf_counter() - started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('teacher-training-manifest', 'preflight-receipt', 'feature-bank', 'component-dir',
                 'prepared-base-dir', 'prepared-base-receipt', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('input-sha256', 'owner-authorization-ref', 'feature-bank-sha256', 'component-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    args = parser.parse_args()
    os.environ.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1', 'WANDB_DISABLED': 'true'})
    result = prepare(args)
    print(json.dumps({'source_view_sha256': result['record_sha256'],
        'unresolved_reference_input_gap_cases': result['unresolved_reference_input_gap_cases'],
        'seconds': result['seconds'], 'qualified_for_product': False}), flush=True)


if __name__ == '__main__':
    main()
