"""Target-bearing frozen diagnostics of N1 conditional channels into real N2/N3.

The caller verifies native case/source authority and maps partner facts to the
recipient frame's exact IDs before using this helper. Omit unknown facts. These
interventions are not authentic inferred outputs, training, or qualification.
"""
from collections.abc import Mapping
from dataclasses import fields, replace
import math

import torch

from src.alice_personality.identity.codec import fingerprint
from src.alice_personality.identity.contracts import IdentityError, IdentityRepresentation
from src.alice_personality.identity.model import IdentityModel


CONDITIONAL_FIELDS = ('concept_applicability', 'concept_uncertainty',
                      'concept_source_support', 'concept_source_disconfirmation')


def _require(condition, message):
    if not condition:
        raise IdentityError(message)


def _check_tensor_tree(value):
    if isinstance(value, torch.Tensor):
        _require(not value.requires_grad and value.grad_fn is None,
                 'cached diagnostic tensors must be detached')
        _require(not value.is_floating_point() or bool(torch.isfinite(value).all()),
                 'cached diagnostic tensors must be finite')
    elif isinstance(value, Mapping):
        for item in value.values():
            _check_tensor_tree(item)
    elif isinstance(value, IdentityRepresentation):
        for field in fields(value):
            _check_tensor_tree(getattr(value, field.name))


def _coordinates(representation, frame):
    _require(isinstance(representation, IdentityRepresentation), 'native cached N1 representation required')
    _require(len(frame.concepts.entry_ids) == len(frame.sources.entry_ids) == 1,
             'conditional diagnostic requires one native frame')
    concepts, sources = frame.concepts.entry_ids[0], frame.sources.entry_ids[0]
    _require(len(set(concepts)) == len(concepts) and len(set(sources)) == len(sources),
             'diagnostic coordinates must be unique')
    _require(tuple(representation.concept_mask.shape) == (1, len(concepts))
             and representation.concept_mask.dtype == torch.bool
             and tuple(representation.concept_source_mask.shape) == (1, len(concepts), len(sources))
             and representation.concept_source_mask.dtype == torch.bool,
             'cached authority mask geometry differs from frame')
    _check_tensor_tree(representation)
    for name in CONDITIONAL_FIELDS:
        mask = representation.concept_source_mask if 'source_' in name else representation.concept_mask
        value, logits = getattr(representation, name), getattr(representation, name + '_logits')
        _require(value.is_floating_point() and logits.is_floating_point()
                 and value.shape == logits.shape == mask.shape
                 and value.dtype == logits.dtype and value.device == logits.device == mask.device,
                 'cached conditional field geometry or dtype differs')
        _require(bool(((value >= 0) & (value <= 1)).all()), 'cached conditional probability is invalid')
        absent = 1 if name == 'concept_uncertainty' else 0
        _require(not bool(((value != absent) & ~mask).any()),
                 'cached conditional values cannot grant authority')
    return {key: i for i, key in enumerate(concepts)}, {key: i for i, key in enumerate(sources)}


def patch_conditionals(representation, frame, values_by_field):
    """Return (replacement, metadata), changing only explicitly supplied facts.

    ``values_by_field`` maps one of the four conditional field names to
    ``{(concept_id,): probability}`` or, for source relations,
    ``{(concept_id, source_id): probability}``. All IDs refer to this frame.
    Unknown observations must be omitted, not replaced with guessed targets.

    N2 consumes exact reviewed probabilities, including 0/1. For changed binary
    endpoints only, the paired audit logit uses logit(eps) or logit(1-eps), with
    eps from the probability dtype. This finite serialization convention is
    explicitly diagnostic, not an inferred confidence or calibrated logit.
    Same-value patches preserve the existing logit bit-for-bit for a true sham.
    """
    concepts, sources = _coordinates(representation, frame)
    _require(isinstance(values_by_field, Mapping) and set(values_by_field) <= set(CONDITIONAL_FIELDS),
             'only four reviewed conditional channels may be patched')
    updates, counts = {}, {}
    for name, facts in values_by_field.items():
        _require(isinstance(facts, Mapping), 'conditional facts require an exact pointer mapping')
        relation = 'source_' in name
        mask = representation.concept_source_mask if relation else representation.concept_mask
        original = getattr(representation, name)
        values, logits = original.clone(), getattr(representation, name + '_logits').clone()
        applied = changed = endpoints = finite_endpoint_logits = 0
        epsilon = torch.finfo(original.dtype).eps
        for pointers, target in facts.items():
            _require(type(pointers) is tuple and len(pointers) == (2 if relation else 1)
                     and all(type(p) is str for p in pointers), 'conditional fact pointer geometry differs')
            _require(pointers[0] in concepts and (not relation or pointers[1] in sources),
                     'conditional fact names an unknown recipient pointer')
            _require(type(target) in (int, float) and math.isfinite(target) and 0 <= target <= 1,
                     'available conditional targets must be finite probabilities')
            index = (0, concepts[pointers[0]], sources[pointers[1]]) if relation else (0, concepts[pointers[0]])
            _require(bool(mask[index]), 'conditional target cannot patch a forbidden authority coordinate')
            value = original.new_tensor(target)
            applied += 1
            endpoint = bool(value == 0 or value == 1)
            endpoints += endpoint
            if bool(value != original[index]):
                changed += 1
                values[index] = value
                audit_probability = value.clamp(epsilon, 1 - epsilon) if endpoint else value
                logits[index] = torch.logit(audit_probability)
                finite_endpoint_logits += endpoint
        if facts:
            updates[name], updates[name + '_logits'] = values, logits
        counts[name] = {'applied_facts': applied, 'changed_facts': changed,
                        'binary_endpoint_facts': endpoints, 'finite_endpoint_logits': finite_endpoint_logits,
                        'audit_logit_endpoint_epsilon': epsilon}
    patched = replace(representation, **updates)
    _coordinates(patched, frame)
    metadata = {'schema': 'alice-personality-conditional-channel-patch-v1',
                'fields': counts, 'target_bearing_diagnostic': True, 'authentic_inference': False,
                'independent_qualification': False, 'new_gradient_steps': 0,
                'unknown_facts_preserved': True,
                'endpoint_logit_convention': 'exact_probability_with_dtype_epsilon_finite_audit_logit',
                **{key: sum(row[key] for row in counts.values()) for key in
                   ('applied_facts', 'changed_facts', 'binary_endpoint_facts', 'finite_endpoint_logits')}}
    return patched, metadata


def _require_frozen(model):
    _require(isinstance(model, IdentityModel) and model.phase == 'inference'
             and not any(module.training for module in model.modules()),
             'conditional replay requires the native inference model')
    _require(all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters()),
             'conditional replay requires frozen parameters without gradients')


def replay_with_representation(model, frame, representation, query):
    """Replay real N2/N3 using cached N1 output; return an unqualified packet.

    The caller must label patched packets as target-bearing diagnostic results,
    never authentic model inferences. No N1 forward is recomputed. This temporary
    method substitution requires exclusive use of the model for the call.
    """
    _require_frozen(model)
    frame.validate(model.config)
    _coordinates(representation, frame)
    _require(isinstance(query, torch.Tensor) and query.shape == (1, model.config.learned_width)
             and query.dtype == representation.concepts.dtype and query.device == representation.concepts.device,
             'cached native query geometry differs')
    _check_tensor_tree(query)
    before = fingerprint(model.state_dict())
    cached = fingerprint((representation, query))
    authority = fingerprint({'source_records': frame.source_records, 'concept_records': frame.concept_records,
        'candidate_ids': frame.candidates.entry_ids, 'candidate_authority_mask': frame.candidate_authority_mask,
        'candidate_context_mask': frame.candidate_context_mask})
    had_override = 'forward' in model.n1.__dict__
    previous = model.n1.__dict__.get('forward')
    calls = 0

    def cached_forward(actual_frame):
        nonlocal calls
        _require(actual_frame is frame, 'cached N1 replay received a different frame')
        calls += 1
        return representation, query

    try:
        model.n1.forward = cached_forward
        with torch.no_grad():
            packet = model(frame)
        _require(calls == 1 and packet.representation is representation,
                 'native replay must consume exactly one cached N1 output')
        packet.validate()
        return packet
    finally:
        if had_override:
            model.n1.forward = previous
        else:
            del model.n1.forward
        _require_frozen(model)
        _require(fingerprint(model.state_dict()) == before, 'conditional replay changed model state')
        _require(fingerprint((representation, query)) == cached, 'conditional replay changed cached N1 state')
        _require(fingerprint({'source_records': frame.source_records, 'concept_records': frame.concept_records,
            'candidate_ids': frame.candidates.entry_ids, 'candidate_authority_mask': frame.candidate_authority_mask,
            'candidate_context_mask': frame.candidate_context_mask}) == authority,
            'conditional replay changed frame authority')
