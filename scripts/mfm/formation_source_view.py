"""Verify the corrected source/target view before any specialist gradient."""
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from cognitive_kernel.canonical import canonical_json_bytes, CognitiveKernelContractError, require_sha256
from . import prepare_v16_contract_successor as preparation

features, training, shared = preparation.features, preparation.training, preparation.shared


def verify_rows(view, parent, examples):
    bindings = {
        'objective': preparation.OBJECTIVE, 'complete': True,
        'corpus_sha256': parent.preflight['corpus_sha256'],
        'owner_authorization_ref': parent.preflight['owner_authorization_ref'],
        'parent_component_sha256': parent.component['record_sha256'],
        'parent_feature_bank_sha256': parent.bank['record_sha256'],
        'parent_export_sha256': parent.export['record_sha256'],
        'parent_preflight_sha256': parent.preflight['record_sha256'],
        'prepared_base_receipt_sha256': parent.export['prepared_base_receipt_sha256'],
        'code_binding': preparation.code_binding(),
        'train_cases': len(parent.train), 'development_cases': len(parent.development),
        'max_target_tokens': parent.config.max_target_tokens,
        'independent_gold': False, 'final_payloads_opened': False, 'full_fit': False,
        'qualified_for_product': False, 'memory_gate_executed': False,
    }
    if any(view.get(name) != value for name, value in bindings.items()):
        raise CognitiveKernelContractError('source view ancestry/objective/code differs')
    originals = (*parent.train, *parent.development)
    rows = view.get('cases')
    if type(rows) is not list or len(rows) != len(examples) or len({r.get('case_id') for r in rows}) != len(rows):
        raise CognitiveKernelContractError('source view case roster differs')
    changed_count = 0
    for row, original, example in zip(rows, originals, examples):
        changed = features.case_key(original) != features.case_key(example)
        expected = {
            'case_id': example.case_id, 'split': example.split, 'source_augmented': changed,
            'parent_model_input_sha256': features.case_key(original),
            'model_input_sha256': features.case_key(example),
            'decision_target_sha256': sha256(canonical_json_bytes(preparation.decision_for_example(example))).hexdigest(),
            'existing_reference_input_gaps': preparation.reference_input_gaps(example),
        }
        if any(row.get(name) != value for name, value in expected.items()):
            raise CognitiveKernelContractError('source view input/decision row differs')
        if expected['existing_reference_input_gaps']:
            raise CognitiveKernelContractError('source view has unresolved existing reference inputs; diagnose before training')
        changed_count += changed
    if view.get('augmented_source_cases') != changed_count or view.get('unchanged_source_cases_reused') != len(rows) - changed_count or view.get('unresolved_reference_input_gap_cases') != 0:
        raise CognitiveKernelContractError('source view completion counts differ')


def load(args):
    # Original admission enforces owner identity/network isolation before reads.
    parent = preparation.diagnostic.prepare_diagnostic(args)
    import torch, transformers
    from transformers import AutoProcessor
    root = args.source_view
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise CognitiveKernelContractError('source view custody is a link')
    view = shared._read_sealed(root / 'view.json', preparation.SCHEMA)
    if view['record_sha256'] != require_sha256(args.source_view_sha256, 'source_view_sha256'):
        raise CognitiveKernelContractError('source view external producer pin differs')
    examples = tuple(preparation.prior.augment_public_seed_example(e) for e in (*parent.train, *parent.development))
    verify_rows(view, parent, examples)
    runtime = preparation.verify_source_runtime(parent.export, torch.__version__, transformers.__version__)
    patch = shared._read_sealed(root / 'export.json', preparation.PATCH_SCHEMA)
    expected = {
        'parent_feature_bank_sha256': parent.bank['record_sha256'],
        'parent_export_sha256': parent.export['record_sha256'],
        'parent_processor_preflight_sha256': parent.preflight['record_sha256'],
        'corpus_sha256': args.input_sha256, 'code_binding': preparation.code_binding(),
        'actual_runtime': runtime, 'prepared_base_receipt_sha256': parent.export['prepared_base_receipt_sha256'],
        'prepared_base_weight_sha256': parent.export['prepared_base_weight_sha256'],
        'representation': 'base.model.last_hidden_state', 'base_frozen': True,
        'language_head_used': False, 'target_visible_to_backbone': False, 'qualified_for_product': False,
    }
    if view.get('actual_runtime') != runtime or view.get('patch_export_sha256') != patch['record_sha256'] or any(
            patch.get(k) != value for k, value in expected.items()):
        raise CognitiveKernelContractError('source view patch/runtime binding differs')
    processor = AutoProcessor.from_pretrained(str(args.feature_bank / 'processor'), local_files_only=True, trust_remote_code=False)
    locators, stress = {}, {}
    for row, example in zip(view['cases'], examples):
        bank = root if row['source_augmented'] else args.feature_bank
        export = patch['record_sha256'] if row['source_augmented'] else parent.export['record_sha256']
        receipt, tensors = features.read_case(bank, example, export, width=parent.config.base_hidden_size)
        if row['source_augmented'] and row.get('patch_case_receipt_sha256') != receipt['record_sha256']:
            raise CognitiveKernelContractError('source view patch case receipt differs')
        encoded = training.source_batch_v16(processor, example.context, example.opened_sources,
            parent.preflight['max_source_tokens'], case_id=example.case_id)
        inputs, labels = preparation.target_ids(parent.tokenizer, example, parent.config.max_target_tokens)
        if not torch.equal(encoded['input_ids'], tensors['input_ids']) or not torch.equal(encoded['attention_mask'], tensors['attention_mask']) or row['source_tokens'] != encoded['input_ids'].shape[1] or row['target_tokens'] != len(labels):
            raise CognitiveKernelContractError('source view tokenization/length differs')
        locators[example.case_id] = (bank, export)
        if example.split == 'train':
            for name, size in (('source', row['source_tokens']), ('target', len(labels)),
                               ('joint', row['source_tokens'] * len(inputs) * parent.config.heads * parent.config.layers)):
                if name not in stress or size > stress[name][0]:
                    stress[name] = (size, example.case_id)
    expected_patch_cases = {features.case_key(e) for e, r in zip(examples, view['cases']) if r['source_augmented']}
    if {p.name for p in (root / 'cases').iterdir()} != expected_patch_cases or view.get('stress_train_case_ids') != list(dict.fromkeys(v[1] for v in stress.values())) or view.get('peak_train_attention_pairs') != stress['joint'][0] or view.get('longest_target_tokens') != max(r['target_tokens'] for r in view['cases']) or view.get('longest_source_tokens') != max(r['source_tokens'] for r in view['cases']):
        raise CognitiveKernelContractError('source view patches/stress coverage differs')
    return SimpleNamespace(parent=parent, view=view, patch=patch, locators=locators,
        train=tuple(e for e in examples if e.split == 'train'),
        development=tuple(e for e in examples if e.split == 'development'),
        tokenizer=parent.tokenizer, config=parent.config, device=parent.device)


def payload(prepared, example, *, device):
    import torch
    root, digest = prepared.locators[example.case_id]
    _, tensors = features.read_case(root, example, digest, width=prepared.config.base_hidden_size)
    inputs, labels = preparation.target_ids(prepared.tokenizer, example, prepared.config.max_target_tokens)
    return {'base_states': tensors['states'].to(device), 'source_mask': tensors['attention_mask'].to(device),
            'input_ids': torch.tensor([inputs], device=device), 'labels': torch.tensor([labels], device=device)}
