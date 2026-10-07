"""Source-only sequence teaching and balanced full formation construction.

These pure helpers consume already admitted, source-validated examples. They
neither admit rights nor read teacher targets, a Gemma language head or FINAL.
The construction strata describe the recovered 432-case training snapshot;
they are not a future corpus or role-capability ceiling. A new corpus requires
an explicitly versioned construction, not silently relaxed roster checks.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import math
from numbers import Real
import re

from cognitive_kernel.canonical import (
    CognitiveKernelContractError, canonical_json_bytes, canonical_sha256,
)


SCHEMA = 'mfm-source-sequence-curriculum-v1'
OBJECTIVE = 'mfm-full-role-plus-source-sequence-29-strata-v1'
TEXTUAL_MODALITIES = frozenset({'text', 'code', 'structured'})
DIVERSE_CASE_IDS = tuple(f'v16-seed-{i:02}' for i in (*range(1, 13), 19, 20, 21)) + (
    'role-source-person-revocation',)
CRAFTS = ('atlas', 'ceramics', 'orchard', 'theatre', 'harbor', 'library', 'garden', 'radio')
SITES = ('east', 'west', 'north', 'south')
OPERATIONS = (
    'norm-agreed', 'norm-unsettled', 'episode-mission', 'plan-outcome',
    'procedure-revision', 'source-person', 'revocation-deletion',
    'record-correction', 'owner-deletion', 'sensitive-boundary',
    'assistant-self', 'hypothetical-code', 'conflicting-outside',
)
_LONG_CASE = re.compile(
    r'long-(?:' + '|'.join(map(re.escape, CRAFTS)) + r')-(?:' +
    '|'.join(map(re.escape, SITES)) + r')-(' +
    '|'.join(map(re.escape, OPERATIONS)) + r')\Z')
EXPECTED_TRAIN_CASE_IDS = frozenset((*DIVERSE_CASE_IDS, *(
    f'long-{craft}-{site}-{operation}'
    for craft in CRAFTS for site in SITES for operation in OPERATIONS)))


def lexical_source_record(example) -> dict:
    """Preserve every opened textual source in its original evidence order.

    No call to example.validate() is made: that method reads teacher targets.
    The caller must perform the existing full admission before invoking this
    helper. Local custody/order checks do not replace that authority boundary.
    Native modalities remain available to formation; this auxiliary task
    explicitly rejects them rather than decoding or silently dropping them.
    """
    if example.split not in {'train', 'development'}:
        raise CognitiveKernelContractError('FINAL cannot supply lexical learning targets')
    evidence = tuple(example.context.base.evidence)
    opened = tuple(example.opened_sources)
    expected_refs = tuple(ref.ref_id for ref in evidence)
    if (not evidence or len(expected_refs) != len(set(expected_refs)) or
            tuple(ref_id for ref_id, _ in opened) != expected_refs):
        raise CognitiveKernelContractError('lexical sources must cover exact evidence order')
    sources = []
    for ref, (ref_id, raw) in zip(evidence, opened):
        if ref.modality not in TEXTUAL_MODALITIES:
            raise CognitiveKernelContractError('lexical auxiliary requires textual source modality')
        if type(raw) is not bytes or sha256(raw).hexdigest() != ref.content_digest:
            raise CognitiveKernelContractError('lexical source bytes differ from admitted custody')
        try:
            text = raw.decode('utf-8', errors='strict')
        except UnicodeDecodeError as exc:
            raise CognitiveKernelContractError('lexical source is not exact UTF-8') from exc
        sources.append({'ref_id': ref_id, 'text': text})
    return {'sources': sources}


def lexical_target_ids(tokenizer, example, maximum: int) -> tuple[list[int], list[int]]:
    """Encode complete canonical source JSON, shift BOS and terminate once."""
    if type(maximum) is not int or maximum < 1:
        raise CognitiveKernelContractError('lexical target budget must be a positive integer')
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    if (type(bos) is not int or type(eos) is not int or bos < 0 or eos < 0 or bos == eos):
        raise CognitiveKernelContractError('lexical tokenizer needs distinct BOS/EOS identities')
    answer = canonical_json_bytes(lexical_source_record(example)).decode('utf-8')
    body = tokenizer.encode(answer, add_special_tokens=False)
    if (not isinstance(body, (list, tuple)) or not body or
            any(type(token) is not int or token < 0 for token in body)):
        raise CognitiveKernelContractError('lexical target tokenization is invalid')
    body = list(body)
    if eos in body:
        raise CognitiveKernelContractError('lexical source literal produces interior EOS')
    if len(body) + 1 > maximum:
        raise CognitiveKernelContractError('complete lexical target exceeds budget; never truncate')
    if tokenizer.decode(body, skip_special_tokens=False) != answer:
        raise CognitiveKernelContractError('complete lexical target does not round-trip')
    return [bos, *body], [*body, eos]


def construction_stratum(case_id: str) -> str:
    """Use the audited complete identifier grammar, including dashed suffixes."""
    if type(case_id) is not str:
        raise CognitiveKernelContractError('construction case identity is invalid')
    if case_id in DIVERSE_CASE_IDS:
        return 'diverse:' + case_id
    matched = _LONG_CASE.fullmatch(case_id)
    if matched is None:
        raise CognitiveKernelContractError('case is outside the declared recovered construction')
    return 'longitudinal:' + matched.group(1)


def balanced_case_weights(examples) -> dict:
    """Each of 29 strata has mass 1/29; hosts divide their stratum's mass."""
    rows = tuple(examples)
    ids = tuple(example.case_id for example in rows)
    if (len(ids) != len(set(ids)) or set(ids) != EXPECTED_TRAIN_CASE_IDS or
            any(example.split != 'train' for example in rows)):
        raise CognitiveKernelContractError('balanced construction requires complete unique train roster')
    strata = {case_id: construction_stratum(case_id) for case_id in sorted(ids)}
    counts = dict(sorted(Counter(strata.values()).items()))
    if len(counts) != 29 or sorted(counts.values()) != [1] * 16 + [32] * 13:
        raise CognitiveKernelContractError('balanced construction stratum counts differ')
    weights = {case_id: 1. / (29 * counts[stratum]) for case_id, stratum in strata.items()}
    return {'weights': weights, 'group_counts': counts, 'strata': strata,
            'roster_sha256': canonical_sha256({'objective': OBJECTIVE, 'cases': strata})}


def _loss_number(value) -> float:
    # A scalar tensor is inspected without detaching the returned objective.
    detached = value.detach() if hasattr(value, 'detach') else value
    if hasattr(detached, 'numel'):
        if detached.numel() != 1 or tuple(detached.shape) != ():
            raise CognitiveKernelContractError('balanced task loss must be scalar')
        number = float(detached)
    elif isinstance(detached, Real) and not isinstance(detached, bool):
        number = float(detached)
    else:
        raise CognitiveKernelContractError('balanced task loss must be numeric')
    if not math.isfinite(number) or number < 0:
        raise CognitiveKernelContractError('balanced task loss must be finite and nonnegative')
    return number


def balanced_objective(losses_by_case, weights_by_case):
    """Sum w_case * (complete formation CE + complete source sequence CE)/2.

    This computes the complete training objective, not a batch normalization or
    checkpoint selector. Scalar tensor inputs preserve their gradient graph.
    """
    if (set(losses_by_case) != EXPECTED_TRAIN_CASE_IDS or
            set(weights_by_case) != EXPECTED_TRAIN_CASE_IDS):
        raise CognitiveKernelContractError('balanced objective needs the complete train loss roster')
    counts = Counter(construction_stratum(case_id) for case_id in weights_by_case)
    terms = []
    for case_id in sorted(weights_by_case):
        weight = weights_by_case[case_id]
        if (not isinstance(weight, Real) or isinstance(weight, bool) or
                not math.isfinite(weight) or not math.isclose(
                    weight, 1. / (29 * counts[construction_stratum(case_id)]),
                    rel_tol=1e-12, abs_tol=0.)):
            raise CognitiveKernelContractError('balanced construction case weight differs')
        pair = losses_by_case[case_id]
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise CognitiveKernelContractError('balanced objective requires both complete task losses')
        formation, lexical = pair
        _loss_number(formation)
        _loss_number(lexical)
        terms.append(weight * (formation + lexical) / 2.)
    return sum(terms)


def curriculum_receipt(examples, tokenizer, maximum: int) -> dict:
    """Bind exact source text/token targets without including raw text.

    Token identities/frequencies and case/source hashes belong in the caller's
    existing protected custody. The caller binds tokenizer and admission pins.
    This is construction evidence, never source rights or independent gold.
    """
    rows = tuple(examples)
    balance = balanced_case_weights(rows)
    frequency = Counter()
    cases = []
    for example in sorted(rows, key=lambda row: row.case_id):
        raw = canonical_json_bytes(lexical_source_record(example))
        _, labels = lexical_target_ids(tokenizer, example, maximum)
        frequency.update(labels)
        cases.append({
            'case_id': example.case_id,
            'construction_stratum': balance['strata'][example.case_id],
            'source_text_sha256': sha256(raw).hexdigest(),
            'sources': [{'ref_id': ref_id, 'sha256': sha256(payload).hexdigest()}
                        for ref_id, payload in example.opened_sources],
            'lexical_target_token_ids_sha256': canonical_sha256(labels),
            'lexical_target_tokens': len(labels),
        })
    receipt = {
        'schema': SCHEMA, 'objective': OBJECTIVE,
        'train_cases': len(cases), 'construction_strata': len(balance['group_counts']),
        'roster_sha256': balance['roster_sha256'],
        'group_counts': balance['group_counts'], 'weights': balance['weights'],
        'cases': cases, 'lexical_target_tokens': sum(frequency.values()),
        'maximum_lexical_target_tokens': max(row['lexical_target_tokens'] for row in cases),
        'positive_token_frequency': {str(token): frequency[token] for token in sorted(frequency)},
        'frequency_basis': 'positive lexical labels in one complete train corpus pass, including EOS',
        'teacher_output_used': False, 'language_head_used': False,
        'independent_gold': False, 'qualified_for_product': False,
    }
    return {**receipt, 'record_sha256': canonical_sha256(receipt)}
