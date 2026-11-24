"""Read-only N1/N2 interventions, explicitly separate from authentic responses.

These measurements establish path sensitivity, not correct conditional meaning,
independent behavioral qualification, or permission to change model weights.
"""
from dataclasses import replace

import torch

from src.alice_personality.identity.codec import fingerprint
from src.alice_personality.identity.contracts import IdentityError
from src.alice_personality.identity.model import masked_softmax


def measure_conditional_path(model, frame, *, concept_id, donor_frame=None):
    """Hold source/candidate authority and weights fixed while varying one path.

    Applicability clamps target the named concept only. Query replacement changes
    that channel only; host/relationship concepts and other context remain fixed.
    Synthetic latent mixtures are never serialized as native IDP or gold.
    """
    if model.phase != 'inference' or model.training:
        raise IdentityError('conditional measurements require the actual inference mode')
    frame.validate(model.config)
    if len(frame.concepts.entry_ids) != 1 or concept_id not in frame.concepts.entry_ids[0]:
        raise IdentityError('measurement requires one frame with the named concept')
    column = frame.concepts.entry_ids[0].index(concept_id)
    before = fingerprint(model.state_dict())
    authority = fingerprint({'source_records':frame.source_records,
        'concept_records':frame.concept_records, 'candidate_ids':frame.candidates.entry_ids,
        'candidate_authority_mask':frame.candidate_authority_mask,
        'candidate_context_mask':frame.candidate_context_mask})
    with torch.no_grad():
        representation, query = model.n1(frame)
        def read(rep, current_query):
            raw = model.n2(frame, rep, current_query)
            scalar, mask = raw['scalar'], raw['candidate_mask']
            if not bool(torch.isfinite(scalar).all()):
                raise IdentityError('conditional measurement produced a nonfinite native scalar')
            return {'co_valid_probabilities':(model.n3.scale('scalar',scalar[...,1]).sigmoid()*mask).tolist(),
                'contraindications':(model.n3.scale('scalar',scalar[...,4]).sigmoid()*mask).tolist(),
                'preferences':masked_softmax(model.n3.scale('preference',scalar[...,0]),mask).tolist(),
                'raw_n2_scalar':scalar.tolist(), 'authority_frame_sha256':authority}
        baseline = read(representation, query)
        modes = {}
        for name, value in (('applicability_zero',0.),('applicability_one',1.)):
            applicability = representation.concept_applicability.clone()
            applicability[:,column] = value
            modes[name] = read(replace(representation, concept_applicability=applicability),query)
        # Remove only N2's learned conditional message. Full N1 representations
        # and their source/view permissions remain as the native model produced.
        had_override = 'conditional_memory' in model.n2.__dict__
        previous = model.n2.__dict__.get('conditional_memory')
        try:
            model.n2.conditional_memory = lambda rep: (rep.concepts,rep.views)
            modes['conditional_messages_removed'] = read(representation,query)
        finally:
            if had_override:
                model.n2.conditional_memory = previous
            else:
                del model.n2.conditional_memory
        if donor_frame is not None:
            donor_frame.validate(model.config)
            donor, donor_query = model.n1(donor_frame)
            if donor_query.shape != query.shape:
                raise IdentityError('query donor must share native query geometry')
            modes['query_channel_swap'] = read(replace(representation,
                query_tokens=donor.query_tokens,query_mask=donor.query_mask),donor_query)
    if fingerprint(model.state_dict()) != before:
        raise IdentityError('conditional measurement changed the scientific model')
    return {'schema':'alice-personality-conditional-path-measurement-v1',
        'model_state_sha256':before,'candidate_ids':frame.candidates.entry_ids,
        'concept_id':concept_id,'operations':['baseline',*modes],
        'baseline':baseline,'modes':modes,'authentic_response':False,
        'independent_qualification':False,'additional_gradient_steps':0}
