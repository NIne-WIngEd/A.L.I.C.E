"""Validated public conditional foundation inputs and group-balanced objective.

No weights or original/private corpora are opened. The complete independently
pinned census and complete frozen source bank are required before any payload.
Sequence losses must be means over every label token, including EOS. Uniform
without-replacement epoch sampling uses N/B importance coefficients, preserving
the sequence-mean, reference-mean, semantic-group-mean objective in expectation.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from pathlib import Path
import random
from typing import TypeAlias

from cognitive_kernel.canonical import canonical_sha256, require_sha256
from . import export_public_conditional_features as features
from . import preflight_public_conditional_material as census_code
from . import train_v1_formation_specialist as shared

Pair: TypeAlias = dict
PAIR_FIELDS = {'case_id', 'split', 'source_ids', 'decoder_inputs', 'labels',
               'semantic_input_sha256', 'semantic_reference_count', 'source_payload_sha256'}


@dataclass(frozen=True)
class PublicConditionalTraining:
    census: dict
    sources: tuple[dict, ...]
    pairs: tuple[Pair, ...]
    train: tuple[Pair, ...]
    development: tuple[Pair, ...]
    feature_bank: Path
    export: dict
    bank: dict
    pair_by_case: dict[str, Pair] = field(init=False, repr=False)
    source_by_key: dict[str, dict] = field(init=False, repr=False)
    source_receipt_by_key: dict[str, str] = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, 'pair_by_case', {p['case_id']: p for p in self.pairs})
        object.__setattr__(self, 'source_by_key', {s['source_key']: s for s in self.sources})
        object.__setattr__(self, 'source_receipt_by_key',
                           {s['source_key']: s['source_receipt_sha256'] for s in self.bank['sources']})


def _groups(pairs):
    pairs = tuple(pairs)
    if not pairs:
        raise ValueError('empty public pair roster')
    cases, group_splits, raw_splits = set(), {}, {}
    counts = Counter()
    for pair in pairs:
        case = pair.get('case_id')
        if type(case) is not str or not case:
            raise ValueError('public case identity is missing')
        if case in cases:
            raise ValueError('duplicate public case identity')
        cases.add(case)
        split = pair.get('split')
        if split not in {'train', 'development'}:
            raise ValueError('public pair split cannot open FINAL')
        group = pair.get('semantic_input_sha256')
        raw = pair.get('source_payload_sha256')
        for value in (group, raw):
            if require_sha256(value, 'public pair ancestry') != value:
                raise ValueError('public pair ancestry is not canonical')
        if group_splits.setdefault(group, split) != split or raw_splits.setdefault(raw, split) != split:
            raise ValueError('semantic or raw input ancestry crosses public splits')
        if split != ('development' if int(group, 16) % 10 == 0 else 'train'):
            raise ValueError('public semantic ancestry split rule differs')
        counts[group] += 1
    for pair in pairs:
        n = pair.get('semantic_reference_count')
        if type(n) is not int or n != counts[pair['semantic_input_sha256']]:
            raise ValueError('complete public semantic reference count differs')
    return pairs, counts


def case_weights(pairs, *, split: str) -> dict[str, float]:
    """Compute weights from the complete roster, checking both split ancestries."""
    if split not in {'train', 'development'}:
        raise ValueError('public objective split differs')
    pairs, counts = _groups(pairs)
    selected = [p for p in pairs if p['split'] == split]
    groups = {p['semantic_input_sha256'] for p in selected}
    if not groups:
        raise ValueError('public objective split is empty')
    return {p['case_id']: 1 / (len(groups) * counts[p['semantic_input_sha256']]) for p in selected}


def batch_coefficients(batch, weights, *, total_train_cases: int) -> dict[str, float]:
    """Multiply full-objective weights by N/B for a uniform training batch."""
    batch = tuple(batch)
    if type(total_train_cases) is not int or total_train_cases < 1 or total_train_cases != len(weights):
        raise ValueError('complete training case count differs from weights')
    if not batch or len(batch) > total_train_cases:
        raise ValueError('public training batch is empty or exceeds complete roster')
    if any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in weights.values()) or \
            not math.isclose(math.fsum(weights.values()), 1., rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('complete public objective weights must have unit positive mass')
    result = {}
    for pair in batch:
        case = pair.get('case_id')
        if pair.get('split') != 'train' or case not in weights or case in result:
            raise ValueError('batch requires unique cases from complete training weights')
        result[case] = weights[case] * total_train_cases / len(batch)
    return result


def _sequence(pair, census):
    if type(pair) is not dict or set(pair) != PAIR_FIELDS:
        raise ValueError('public encoded pair fields differ')
    vocabulary = census['foundation_vocabulary']
    source, inputs, labels = (pair[k] for k in ('source_ids', 'decoder_inputs', 'labels'))
    for tokens in (source, inputs, labels):
        if type(tokens) is not list or not tokens or any(type(t) is not int or not 0 <= t < vocabulary for t in tokens):
            raise ValueError('public sequence token IDs exceed vocabulary or lack tokens')
    if len(source) > census['max_source_tokens'] or len(labels) > census['max_target_tokens'] or \
            len(inputs) != len(labels) or inputs[0] != 2 or labels[-1] != 1 or \
            inputs[1:] != labels[:-1] or len(labels) < 2 or any(t in (1, 2) for t in labels[:-1]):
        raise ValueError('public complete BOS/EOS shifted sequence differs; never truncate')


def _export_binding(census):
    # Reconstruct the producer's immutable binding rather than trusting flags
    # or a self-consistent nested census supplied by an arbitrary export.
    return shared._seal({
        'schema': features.EXPORT_SCHEMA, 'census_sha256': census['record_sha256'], 'census': census,
        'source_inputs_sha256': census['source_inputs_sha256'],
        'prepared_clone_receipt_sha256': census['prepared_clone_receipt_sha256'],
        'prepared_base_weight_sha256': census['prepared_base_weight_sha256'],
        'exporter_sha256': shared._digest(Path(features.__file__)),
        'torch_version': census['torch_version'], 'transformers_version': census['transformers_version'],
        'representation': 'base.model.last_hidden_state', 'backbone_dtype': 'bfloat16',
        'backbone_device': 'cpu', 'base_frozen': True, 'language_head_used': False,
        'target_visible_to_backbone': False, 'private_evidence_opened': False,
        'optimizer_created': False, 'authority_written': False, 'final_payloads_opened': False,
        'qualified_for_product': False})


def load_public_training(census_dir: Path, census_sha256: str, feature_bank: Path,
                         feature_bank_sha256: str) -> PublicConditionalTraining:
    census_dir, feature_bank = Path(census_dir), Path(feature_bank)
    census, sources = features.read_census(census_dir, census_sha256)
    features._positive(census.get('max_target_tokens'), 'complete target budget')
    require_sha256(census.get('encoded_pairs_sha256'), 'encoded pair digest')
    path = features._within(census_dir, 'encoded-pairs.jsonl')
    if shared._digest(path) != census['encoded_pairs_sha256']:
        raise ValueError('public encoded pair file digest differs')
    with path.open('rb') as stream:
        pairs = tuple(features._json(line) for line in stream)
    for pair in pairs:
        _sequence(pair, census)
    _groups(pairs)
    if census_code.token_statistics(pairs) != census.get('statistics'):
        raise ValueError('complete public pair statistics differ from census')
    expected_sources = census_code.source_inputs(pairs)
    if expected_sources != sources:
        raise ValueError('complete public pair/source roster or ancestry differs')
    export = features._read_sealed(features._within(feature_bank, 'export.json'), features.EXPORT_SCHEMA)
    if export != _export_binding(census):
        raise ValueError('public source-only export immutable binding differs')
    require_sha256(feature_bank_sha256, 'feature_bank_sha256')
    pinned_bank = features._read_sealed(features._within(feature_bank, 'bank.json'), features.BANK_SCHEMA)
    if pinned_bank['record_sha256'] != feature_bank_sha256:
        raise ValueError('external public bank digest differs')
    bank = features.verify_bank(feature_bank, census, sources, export)
    train = tuple(p for p in pairs if p['split'] == 'train')
    development = tuple(p for p in pairs if p['split'] == 'development')
    if not train or not development:
        raise ValueError('complete public census needs teaching and development splits')
    return PublicConditionalTraining(census, tuple(sources), pairs, train, development,
                                     feature_bank, export, bank)


def payload(prepared, pair, *, device) -> dict:
    """Recheck exact receipt/tensor custody at consumption, returning frozen states."""
    import torch
    if prepared.pair_by_case.get(pair.get('case_id')) != pair:
        raise ValueError('public pair membership differs from validated census')
    key = canonical_sha256({'task': census_code.TASK, 'source_ids': pair['source_ids']})
    source = prepared.source_by_key.get(key)
    if source is None or pair['case_id'] not in source['case_ids']:
        raise ValueError('public pair/source membership differs')
    receipt, tensors = features.read_source(prepared.feature_bank, source,
                                           prepared.export['record_sha256'], prepared.census['base_hidden_size'])
    if receipt['record_sha256'] != prepared.source_receipt_by_key[key]:
        raise ValueError('public source receipt differs from externally pinned bank')
    return {'base_states': tensors['states'].detach().to(device),
            'source_mask': tensors['attention_mask'].detach().to(device),
            'input_ids': torch.tensor([pair['decoder_inputs']], dtype=torch.long, device=device),
            'labels': torch.tensor([pair['labels']], dtype=torch.long, device=device)}


def epoch_order(prepared, *, seed: int, epoch: int) -> tuple[Pair, ...]:
    """Deterministic uniform complete training permutation; no prefix curriculum."""
    if type(seed) is not int or type(epoch) is not int or epoch < 0:
        raise ValueError('public epoch seed must be integer and epoch nonnegative')
    order = list(prepared.train)
    random.Random(canonical_sha256({'seed': seed, 'epoch': epoch,
                                   'sampler': 'public-uniform-complete-permutation-v1'})).shuffle(order)
    return tuple(order)
