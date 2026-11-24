"""Passive native-stage observations; temporary exact frozen readout reuse.

Tensor observations stay in the protected caller. Effects are descriptive;
neither a small norm nor attention output alone establishes a unique cause.
"""
from contextlib import contextmanager
from dataclasses import fields

import torch

from src.alice_personality.identity.codec import fingerprint
from src.alice_personality.identity.contracts import IdentityError


def require(condition, message):
    if not condition:
        raise IdentityError(message)


def frozen(model):
    require(model.phase == 'inference' and not any(m.training for m in model.modules()),
            'stage observation requires native inference mode')
    require(all(not p.requires_grad and p.grad is None for p in model.parameters()),
            'stage observation requires frozen model parameters')


def bank_version(bank):
    """Detect accidental tensor mutation within this exclusive immediate session."""
    return tuple((field.name, id(value), value._version, tuple(value.shape), value.dtype, value.device)
                 for field in fields(bank) if isinstance(value := getattr(bank, field.name), torch.Tensor))


def capture_stages(model, frame, call):
    """Call the actual native route with passive hooks; return result, stages, cache.

    Hooks return None and never change module arguments, outputs or execution
    mode. A cache is local to this model/frame/query, not a persistent feature
    store or evidence authority. Model and cached tensor bytes are checked.
    """
    frozen(model)
    state = fingerprint(model.state_dict())
    stages, handles, readouts, native = {}, [], [], {}
    candidate_mask = frame.candidates.entry_mask & frame.candidate_authority_mask & frame.candidate_context_mask

    def record(name, value, mask, centered=False):
        require(name not in stages, 'native stage unexpectedly repeated')
        require(value.is_floating_point() and bool(torch.isfinite(value).all()), 'nonfinite stage tensor')
        require(mask.dtype == torch.bool and value.shape[:mask.ndim] == mask.shape,
                'stage mask geometry differs')
        stages[name] = (value.detach().clone(), mask.detach().clone(), centered)

    def entered(_module, args):
        require(args[0] is frame and not native, 'native N2 frame differs or repeats')
        rep = args[1]
        native['rep'] = rep
        for name in ('concept_applicability', 'concept_uncertainty',
                     'concept_source_support', 'concept_source_disconfirmation'):
            mask = rep.concept_source_mask if 'source_' in name else rep.concept_mask
            record('n1.' + name, getattr(rep, name), mask)

    def readout(_module, args, result):
        bank, query = args
        require(all(isinstance(t, torch.Tensor) and not t.requires_grad for t in result),
                'readout cache requires detached native tensors')
        readouts.append((bank, bank_version(bank), fingerprint(query),
                         tuple(t.detach().clone() for t in result)))
        if len(readouts) == 1:
            require(bank is frame.candidates, 'native candidate readout must be first')
            record('candidate.entry', result[1], candidate_mask, True)

    def memory(_module, args, kwargs):
        if 'memory.views' in stages:
            return
        rep = native['rep']
        values, mask = args[1], ~kwargs['key_padding_mask']
        query_end = rep.query_mask.shape[1]
        source_end = query_end + rep.source_tokens.flatten(1, 2).shape[1]
        require(query_end < source_end <= values.shape[1], 'native memory segments differ')
        for name, start, end in (('query', 0, query_end), ('source_tokens', query_end, source_end),
                                  ('views', source_end, values.shape[1])):
            record('memory.' + name, values[:, start:end], mask[:, start:end])

    def hook(name, tuple_output=False, concept=False):
        def observe(_module, _args, result):
            record(name, result[0] if tuple_output else result,
                   native['rep'].concept_mask if concept else candidate_mask, not concept)
        return observe

    def conditional_input(name):
        def observe(_module, args):
            record(name, args[0], native['rep'].concept_mask)
        return observe

    def preference_output(_module, _args, result):
        record('scalar.preference', result[..., :1], candidate_mask, True)

    try:
        handles.append(model.n2.register_forward_pre_hook(entered))
        handles.append(model.n2.readout.register_forward_hook(readout))
        handles.append(model.n2.conditional_embedding.register_forward_pre_hook(conditional_input('conditional.signals')))
        handles.append(model.n2.conditional_relation_embedding.register_forward_pre_hook(conditional_input('conditional.messages')))
        handles.append(model.n2.conditional_embedding.register_forward_hook(hook('conditional.embedding', concept=True)))
        handles.append(model.n2.conditional_relation_embedding.register_forward_hook(hook('conditional.relation', concept=True)))
        for i, layer in enumerate(model.n2.layers):
            handles.append(layer.context.register_forward_pre_hook(memory, with_kwargs=True))
            for name in ('context', 'context_norm', 'alternatives', 'alternative_norm', 'feedforward', 'output_norm'):
                handles.append(getattr(layer, name).register_forward_hook(
                    hook(f'layer.{i}.{name}', name in ('context', 'alternatives'))))
        for name, layer in zip(('linear', 'activation', 'output'), model.n2.scalar):
            handles.append(layer.register_forward_hook(hook('scalar.' + name)))
        handles.append(model.n2.scalar[-1].register_forward_hook(preference_output))
        with torch.no_grad():
            result = call()
        require(len(readouts) == 1 + len(frame.labels), 'incomplete native readout capture')
        require(len(stages) == 16 + 6 * len(model.n2.layers), 'incomplete native stage capture')
        return result, stages, {'model': model, 'state': state, 'frame': frame, 'readouts': readouts,
                                'output_fingerprint': fingerprint([r[3] for r in readouts])}
    finally:
        for handle in reversed(handles):
            handle.remove()
        frozen(model)
        require(fingerprint(model.state_dict()) == state, 'stage observation changed model state')


@contextmanager
def replay_readout(model, cache):
    """Reuse only exact projections in an exclusive single-forward session.

    Frozen source files and frame bindings remain the owner's responsibility.
    Version counters detect accidental in-place mutation, not adversarial .data
    writes. No untrusted code or concurrent user may mutate this frame/session.
    """
    frozen(model)
    require(cache['model'] is model and fingerprint(model.state_dict()) == cache['state'],
            'readout cache model differs')
    expected = cache['output_fingerprint']
    require(fingerprint([r[3] for r in cache['readouts']]) == expected, 'readout cache changed')
    target = model.n2.readout
    had_override = 'forward' in target.__dict__
    old = target.__dict__.get('forward')
    calls = 0

    def cached(bank, query):
        nonlocal calls
        require(calls < len(cache['readouts']), 'extra native readout call')
        original_bank, versions, query_sha, output = cache['readouts'][calls]
        require(bank is original_bank and bank_version(bank) == versions, 'readout cache bank differs')
        require(fingerprint(query) == query_sha, 'readout cache query differs')
        calls += 1
        return output

    try:
        target.forward = cached
        yield
        require(calls == len(cache['readouts']), 'incomplete native readout replay')
    finally:
        if had_override:
            target.forward = old
        else:
            del target.forward
        frozen(model)
        require(fingerprint(model.state_dict()) == cache['state'], 'readout replay changed model state')
        require(fingerprint([r[3] for r in cache['readouts']]) == expected, 'readout replay changed cached output')


def compare_stages(before, after):
    """Masked finite effects, with candidate-centered effects where applicable."""
    require(set(before) == set(after), 'stage sets differ')
    result = {}
    for name, (left, mask, centered) in before.items():
        right, other_mask, other_centered = after[name]
        require(left.shape == right.shape and left.dtype == right.dtype, 'stage tensor geometry differs')
        require(torch.equal(mask, other_mask) and centered == other_centered, 'stage mask differs')
        allowed = mask.reshape(*mask.shape, *((1,) * (left.ndim - mask.ndim))).expand_as(left)
        a, b = left.double()[allowed], right.double()[allowed]
        require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), 'stage effects must be finite')
        delta = b - a
        rms = lambda x: float(x.square().mean().sqrt()) if x.numel() else 0.
        baseline_rms, delta_rms = rms(a), rms(delta)
        row = {'elements': int(a.numel()), 'changed_elements': int((delta != 0).sum()),
               'before_rms': baseline_rms, 'after_rms': rms(b), 'delta_rms': delta_rms,
               'max_absolute_delta': float(delta.abs().max()) if delta.numel() else 0.,
               'relative_delta_rms': delta_rms / baseline_rms if baseline_rms else None}
        if centered:
            require(mask.ndim == 2 and left.ndim == 3, 'candidate centering geometry differs')
            def center(t):
                value = t.double() * mask[..., None]
                return value - value.sum(1, keepdim=True) / mask.sum(1).clamp_min(1)[:, None, None]
            ca, cb = center(left)[allowed], center(right)[allowed]
            row.update(candidate_centered_before_rms=rms(ca), candidate_centered_after_rms=rms(cb),
                       candidate_centered_delta_rms=rms(cb-ca))
        result[name] = row
    return result
