"""Actual masked N1, N2 and separate N3 objectives for reviewed facts only."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..identity.contracts import CALIBRATION_FAMILIES, HEAD_FAMILIES, SOURCE_KINDS
from .contracts import ReviewedTarget, TeachingBatch, TeachingError


def _clean(target: ReviewedTarget):
    values = torch.where(target.available, target.values, torch.zeros_like(target.values))
    certainty = torch.where(target.available, 1 - target.uncertainty, torch.zeros_like(target.uncertainty))
    return values.detach(), certainty.detach()


def _mean(loss, weights):
    return (loss * weights).sum() / weights.sum().clamp_min(torch.finfo(loss.dtype).tiny)


def _distribution(scores, target, allowed):
    target.validate(tuple(scores.shape), allowed)
    values, weights = _clean(target)
    active = target.available.any(-1)
    if bool((torch.abs(values.sum(-1)[active] - 1) > 1e-5).any()):
        raise TeachingError("reviewed distribution must sum to one on its explicitly known choices")
    logits = scores.float().masked_fill(~target.available, -torch.finfo(torch.float32).max)
    losses = -(values.float() * F.log_softmax(logits, dim=-1)).sum(-1)
    certainty = weights.sum(-1) / target.available.sum(-1).clamp_min(1)
    return _mean(losses, certainty), int(active.sum())


def _probability(values, target, allowed, *, logits=False):
    target.validate(tuple(values.shape), allowed)
    desired, certainty = _clean(target)
    if logits:
        loss = F.binary_cross_entropy_with_logits(values.float(), desired.float(), reduction="none")
    else:
        loss = F.binary_cross_entropy(values.float().clamp(0, 1), desired.float(), reduction="none")
    return _mean(loss, certainty), int(target.available.sum())


def _source_available(packet):
    count = packet.representation.source_states.shape[1]
    return torch.stack(tuple(mask[:, :count] for mask in packet.representation.view_masks.values())).any(0)


class IdentityObjectives(nn.Module):
    """Teaching-only provenance readout plus full native packet objectives.

    Runtime source authority remains deterministic in the identity core. This
    auxiliary readout is first-party, checkpointed, and trained only in N1.
    No target is derived automatically from the frame or compiler output.
    """
    def __init__(self, learned_width: int):
        super().__init__()
        self.provenance = nn.Linear(learned_width, len(SOURCE_KINDS))

    def forward(self, packet, batch: TeachingBatch, *, loss_weights=None):
        losses, coverage = {}, {}
        source_mask = _source_available(packet)
        candidate = packet.candidate_mask
        scalar_names = {"co_valid_probabilities", "evidence_sufficiency", "uncertainty", "contraindications",
                        "failure_tail_risk", "owner_fidelity_uncertainty"}
        if batch.phase == "n1":
            supported = {"concept_source_alignment", "provenance"}
        elif batch.phase == "calibration":
            supported = CALIBRATION_FAMILIES
        else:
            supported = {"preferences", *(scalar_names - {"owner_fidelity_uncertainty"}), "value_tradeoff_probabilities",
                         "candidate_pair_probabilities",
                         "evidence_pointers", "historical_evidence_pointers",
                         "alice_experience_pointers", "concept_pointers", "voice_control_values",
                         "voice_control_confidence", *("head:" + name for name in HEAD_FAMILIES if name != "voice")}
        if set(batch.targets) - supported:
            raise TeachingError("target field is not implemented in the selected teaching phase")
        if loss_weights and set(loss_weights) - (set(batch.targets) | {"contrast"}):
            raise TeachingError("recipe cannot silently weight absent reviewed targets")
        if loss_weights and "contrast" in loss_weights and not batch.contrasts:
            raise TeachingError("contrast recipe weight requires explicit reviewed contrast facts")
        for name, target in batch.targets.items():
            if name == "concept_source_alignment":
                representation = packet.representation
                result = _distribution(representation.concept_source_logits, target,
                                       representation.concept_source_mask)
            elif name == "provenance":
                scores = self.provenance(packet.representation.source_states)
                allowed = source_mask[..., None].expand_as(scores)
                result = _distribution(scores, target, allowed)
                # A reviewed provenance fact cannot contradict the immutable
                # source declaration; unlike an uncertain behavioral preference.
                for row, records in enumerate(batch.frame.source_records):
                    for col, record in enumerate(records):
                        if record is not None and bool(target.available[row, col].any()):
                            truth = SOURCE_KINDS.index(record.source_kind)
                            values, _ = _clean(target)
                            if float(values[row, col, truth]) != 1:
                                raise TeachingError("reviewed provenance contradicts source-kind authority")
            elif name == "preferences":
                result = _distribution(packet.preference_logits, target, candidate)
            elif name == "candidate_pair_probabilities":
                allowed = candidate[..., :, None] & candidate[..., None, :]
                allowed &= ~torch.eye(candidate.shape[-1], dtype=torch.bool, device=candidate.device)
                margins = packet.preference_logits[..., :, None] - packet.preference_logits[..., None, :]
                target.validate(tuple(margins.shape), allowed)
                both = target.available & target.available.transpose(-1, -2)
                pair_sum = target.values + target.values.transpose(-1, -2)
                if bool((torch.abs(pair_sum[both] - 1) > 1e-5).any()):
                    raise TeachingError("reviewed candidate-pair directions must be complementary when both are known")
                result = _probability(margins, target, allowed, logits=True)
            elif name == "value_tradeoff_probabilities":
                # Gold describes an explicit context-conditioned preference
                # probability for this ordered value pair, not a raw logit.
                # Unknown pair directions remain unknown rather than copied.
                head_mask = packet.heads["values"].mask
                allowed = head_mask[..., :, None] & head_mask[..., None, :]
                diagonal = torch.eye(head_mask.shape[-1], dtype=torch.bool, device=head_mask.device)
                allowed &= ~diagonal
                target.validate(tuple(packet.value_tradeoff_margins.shape), allowed)
                both = target.available & target.available.transpose(-1, -2)
                pair_sum = target.values + target.values.transpose(-1, -2)
                if bool((torch.abs(pair_sum[both] - 1) > 1e-5).any()):
                    raise TeachingError("reviewed value-pair directions must be complementary when both are known")
                result = _probability(packet.value_tradeoff_margins, target, allowed, logits=True)
            elif name.startswith("head:"):
                family = name.split(":", 1)[1]
                head = packet.heads[family]
                result = _distribution(head.scores, target, head.mask) if family == "stance" else \
                    _probability(head.scores, target, head.mask, logits=True)
            elif name in scalar_names:
                value = getattr(packet, name)
                result = _probability(value, target, candidate)
                missing = target.available & ~packet.identity_grounding_available[:, None]
                required = 0 if name == "evidence_sufficiency" else 1
                if name in {"evidence_sufficiency", "uncertainty", "failure_tail_risk", "owner_fidelity_uncertainty"} \
                        and bool((target.values[missing] != required).any()):
                    raise TeachingError("reviewed targets cannot overrule deterministic absent identity grounding")
            elif name.endswith("pointers"):
                value = getattr(packet, name)
                if name == "concept_pointers":
                    allowed = candidate[..., None] & packet.representation.concept_mask[:, None]
                else:
                    available = source_mask.clone()
                    if name != "evidence_pointers":
                        for row, records in enumerate(batch.frame.source_records):
                            for col, record in enumerate(records):
                                available[row, col] &= record is not None and (
                                    record.historical_truth_allowed if name == "historical_evidence_pointers"
                                    else record.alice_lived_memory)
                    allowed = candidate[..., None] & available[:, None]
                result = _distribution(value.clamp_min(torch.finfo(value.dtype).tiny).log(), target, allowed)
            elif name == "voice_control_confidence":
                result = _probability(packet.voice_control_confidence, target,
                                      packet.heads["voice"].mask & packet.identity_grounding_available[:, None, None])
            elif name == "voice_control_values":
                value = packet.voice_control_values
                target.validate(tuple(value.shape), packet.heads["voice"].mask, probability=False)
                low, span = torch.zeros_like(value), torch.ones_like(value)
                for row, specs in enumerate(batch.frame.label_specs["voice"]):
                    for col, spec in enumerate(specs):
                        if spec is not None:
                            low[row, :, col], span[row, :, col] = spec.minimum, spec.maximum - spec.minimum
                desired, certainty = _clean(target)
                if bool((((target.values < low) | (target.values > low + span)) & target.available).any()):
                    raise TeachingError("reviewed voice targets exceed their explicit portable-unit ranges")
                loss = F.smooth_l1_loss((value - low) / span, (desired - low) / span, reduction="none")
                result = _mean(loss, certainty), int(target.available.sum())
            else:
                raise TeachingError("unsupported reviewed target")
            losses[name], coverage[name] = result
        if not any(bool(((1 - target.uncertainty)[target.available] > 0).any()) for target in batch.targets.values()):
            raise TeachingError("teaching needs at least one available reviewed fact with nonzero certainty")
        total = sum(value * (loss_weights or {}).get(name, 1.0) for name, value in losses.items())
        return total, losses, coverage

    @staticmethod
    def contrast_loss(first, second, constraint, batch):
        """Only an externally reviewed relevance label supplies this objective."""
        if constraint.relation not in {"relevant", "irrelevant"} or first.candidate_ids != second.candidate_ids:
            raise TeachingError("contrast needs explicit relevance and matching runtime candidate pointers")
        if constraint.field.startswith("head:"):
            family = constraint.field.split(":", 1)[1]
            if family not in first.heads or first.heads[family].specs != second.heads[family].specs:
                raise TeachingError("contrast semantic head schema differs")
            left, right = first.heads[family].activations, second.heads[family].activations
            allowed = first.heads[family].mask & second.heads[family].mask
        elif constraint.field in {"preferences", "co_valid_probabilities", "evidence_sufficiency", "uncertainty",
                                  "contraindications", "failure_tail_risk", "owner_fidelity_uncertainty"}:
            left, right = getattr(first, constraint.field), getattr(second, constraint.field)
            allowed = first.candidate_mask & second.candidate_mask
        else:
            raise TeachingError("contrast field needs an explicit implemented behavioral metric")
        mask = constraint.position_mask
        shape = tuple(left.shape)
        count = left.shape[0]
        if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != shape
                or mask.device != left.device or bool((mask & ~allowed).any())
                or not isinstance(constraint.available, torch.Tensor) or constraint.available.dtype != torch.bool
                or tuple(constraint.available.shape) != (count,) or constraint.available.device != left.device):
            raise TeachingError("contrast availability must address actual common output positions")
        if bool((constraint.available & ~mask.flatten(1).any(-1)).any()):
            raise TeachingError("available contrast needs explicitly selected output positions")
        target = ReviewedTarget(constraint.distance_bound, constraint.available, constraint.uncertainty)
        target.validate((count,), torch.ones_like(constraint.available))
        distance = ((left - right).abs() * mask).flatten(1).sum(-1) / mask.flatten(1).sum(-1).clamp_min(1)
        desired, certainty = _clean(target)
        loss = torch.relu(desired - distance) if constraint.relation == "relevant" else torch.relu(distance - desired)
        return _mean(loss, certainty)
