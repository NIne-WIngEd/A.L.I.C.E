"""Deterministic named target geometry from declared frame authority only."""
from dataclasses import dataclass

import torch

from ..identity.contracts import CALIBRATION_FAMILIES, HEAD_FAMILIES, SOURCE_KINDS
from ..identity.model import IdentityRepresentationModel
from .contracts import TeachingError


@dataclass(frozen=True)
class TargetLayout:
    axes: tuple[tuple[str | None, ...], ...]
    allowed: torch.Tensor
    distribution: bool = False
    probability: bool = True


def target_layouts(frame, phase):
    """No forward, inferred target, feature scoring or learned authority."""
    if frame.query.states.shape[0] != 1:
        raise TeachingError('named-case materialization requires one complete producer document')
    source_masks = IdentityRepresentationModel.source_masks(frame)
    concept_masks, _ = IdentityRepresentationModel._concept_authority(frame, source_masks)
    source = torch.stack(tuple(source_masks.values())).any(0)
    concept = torch.stack(tuple(concept_masks.values())).any(0)
    identity = source_masks['identity'].any(-1)
    candidate = frame.candidates.entry_mask & frame.candidate_authority_mask & frame.candidate_context_mask
    ids = lambda bank: bank.entry_ids[0]
    s, k, c = ids(frame.sources), ids(frame.concepts), ids(frame.candidates)
    if phase == 'n1':
        alignment = torch.zeros((1, len(k), len(s)), dtype=torch.bool, device=source.device)
        for view in source_masks:
            alignment |= concept_masks[view][..., None] & source_masks[view][:, None]
        return {'concept_source_alignment': TargetLayout((k, s), alignment, True),
                'provenance': TargetLayout((s, SOURCE_KINDS), source[..., None].expand(-1, -1, len(SOURCE_KINDS)), True),
                'concept_applicability': TargetLayout((k,), concept),
                'concept_uncertainty': TargetLayout((k,), concept),
                'concept_source_support': TargetLayout((k, s), alignment),
                'concept_source_disconfirmation': TargetLayout((k, s), alignment)}
    if phase not in {'n2', 'calibration'}:
        raise TeachingError('target layouts require an implemented teaching phase')
    layouts = {'preferences': TargetLayout((c,), candidate, True)}
    scalar_names = {'co_valid_probabilities', 'evidence_sufficiency', 'uncertainty', 'contraindications',
                    'failure_tail_risk', 'owner_fidelity_uncertainty'}
    for name in scalar_names:
        if name != 'owner_fidelity_uncertainty' or phase == 'calibration':
            layouts[name] = TargetLayout((c,), candidate)
    for family in HEAD_FAMILIES:
        labels = ids(frame.labels[family])
        mask = candidate[..., None] & frame.labels[family].entry_mask[:, None]
        if family != 'voice' or phase == 'calibration':
            layouts['head:' + family] = TargetLayout((c, labels), mask, family == 'stance')
    voice = ids(frame.labels['voice'])
    voice_mask = candidate[..., None] & frame.labels['voice'].entry_mask[:, None]
    layouts['voice_control_confidence'] = TargetLayout((c, voice), voice_mask & identity[:, None, None])
    if phase == 'n2':
        layouts['voice_control_values'] = TargetLayout((c, voice), voice_mask, probability=False)
        for name in ('evidence_pointers', 'historical_evidence_pointers', 'alice_experience_pointers'):
            allowed = source.clone()
            if name != 'evidence_pointers':
                for col, record in enumerate(frame.source_records[0]):
                    allowed[0, col] &= record is not None and (record.historical_truth_allowed
                        if name == 'historical_evidence_pointers' else record.alice_lived_memory)
            layouts[name] = TargetLayout((c, s), candidate[..., None] & allowed[:, None], True)
        layouts['concept_pointers'] = TargetLayout((c, k), candidate[..., None] & concept[:, None], True)
    pairs = candidate[..., :, None] & candidate[..., None, :]
    pairs &= ~torch.eye(len(c), dtype=torch.bool, device=candidate.device)
    layouts['candidate_pair_probabilities'] = TargetLayout((c, c), pairs)
    values = ids(frame.labels['values'])
    value_mask = candidate[..., None] & frame.labels['values'].entry_mask[:, None]
    value_pairs = value_mask[..., :, None] & value_mask[..., None, :]
    value_pairs &= ~torch.eye(len(values), dtype=torch.bool, device=candidate.device)
    layouts['value_tradeoff_probabilities'] = TargetLayout((c, values, values), value_pairs)
    return {name: layout for name, layout in layouts.items() if phase != 'calibration' or name in CALIBRATION_FAMILIES}
