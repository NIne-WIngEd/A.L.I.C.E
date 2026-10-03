"""First-party N1/N2/N3 mechanics; random initialization is not personality.

Learned operating dimensions are checkpoint choices, not semantic ceilings.
Provider tensors are detached at admission. Authority masks are deterministic
and cannot be modified by attention, graph scores or calibration.
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from .contracts import (CONCEPT_VIEWS, EDGE_ROLES, HEAD_FAMILIES, SOURCE_KINDS, CalibrationBatch,
                        FrozenTokenBank, HeadOutput, IdentityDecisionPacket,
                        IdentityError, IdentityFrame, IdentityModelConfig,
                        IdentityRepresentation)


def masked_softmax(scores: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    """Unavailable/all-missing choices get exactly zero, never fake mass."""
    masked = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)
    weights = torch.softmax(masked, dim=dim) * mask.to(scores.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(torch.finfo(scores.dtype).tiny)


class _LayerTokenReadout(nn.Module):
    def __init__(self, config: IdentityModelConfig):
        super().__init__()
        width = config.learned_width
        self.projection = nn.Linear(config.provider_width, width)
        self.depth_projection = nn.Linear(2, width, bias=False)
        self.condition = nn.Linear(width, width, bias=False)
        self.layer_score = nn.Linear(width, 1, bias=False)
        self.token_score = nn.Linear(width, 1, bias=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, bank: FrozenTokenBank, query: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Detach even if an accidental producer leaves requires_grad enabled.
        value = bank.states.detach().to(dtype=self.projection.weight.dtype)
        projected = self.projection(value)
        layers = value.shape[2]
        depth = torch.linspace(0, 1, layers, device=value.device, dtype=projected.dtype)
        coordinates = torch.stack((depth, torch.cos(depth * torch.pi)), -1)
        position = self.depth_projection(coordinates)[None, None, :, None]
        conditioned = self.condition(query)[:, None, None, None]
        scores = self.layer_score(torch.tanh(projected + position + conditioned)).squeeze(-1)
        mask = bank.token_mask[:, :, None].expand_as(scores)
        layer_weights = masked_softmax(scores, mask, dim=2)
        tokens = self.norm((projected * layer_weights[..., None]).sum(2))
        tokens = tokens * bank.token_mask[..., None]
        token_scores = self.token_score(torch.tanh(tokens + conditioned.squeeze(2))).squeeze(-1)
        token_weights = masked_softmax(token_scores, bank.token_mask)
        entries = (tokens * token_weights[..., None]).sum(2) * bank.entry_mask[..., None]
        return tokens, entries, layer_weights


class _DirectedGraphLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(width * 3, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.score = nn.Linear(width * 3, 1)
        self.norm = nn.LayerNorm(width)

    def forward(self, sources: Tensor, concepts: Tensor, relations: Tensor,
                frame: IdentityFrame, edge_allowed: Tensor, concept_allowed: Tensor) -> Tensor:
        if frame.graph.senders.shape[1] == 0:
            return concepts * concept_allowed[..., None]
        batch, count, width = concepts.shape
        nodes = torch.cat((sources, concepts), 1)
        sender_indices = frame.graph.senders.clamp(0, nodes.shape[1] - 1)
        target_indices = (frame.graph.receivers - sources.shape[1]).clamp(0, count - 1)
        relation_indices = frame.graph.relation_indices.clamp(0, relations.shape[1] - 1)
        gather = lambda bank, indices: bank.gather(1, indices[..., None].expand(-1, -1, width))
        endpoints = torch.cat((gather(nodes, sender_indices), gather(concepts, target_indices),
                               gather(relations, relation_indices)), -1)
        scores = self.score(endpoints).squeeze(-1)
        scores = scores.masked_fill(~edge_allowed, -torch.finfo(scores.dtype).max)
        maxima = torch.full((batch, count), -torch.finfo(scores.dtype).max,
                            device=scores.device, dtype=scores.dtype)
        maxima.scatter_reduce_(1, target_indices, scores, reduce="amax", include_self=True)
        exponent = torch.exp(scores - maxima.gather(1, target_indices)) * edge_allowed
        totals = torch.zeros_like(maxima).scatter_add_(1, target_indices, exponent)
        weights = exponent / totals.gather(1, target_indices).clamp_min(torch.finfo(scores.dtype).tiny)
        updates = torch.zeros_like(concepts).scatter_add_(
            1, target_indices[..., None].expand(-1, -1, width),
            self.message(endpoints) * weights[..., None])
        return self.norm(concepts + updates) * concept_allowed[..., None]


class IdentityRepresentationModel(nn.Module):
    """N1: governed concepts, full token readouts and distinct evidence views."""
    def __init__(self, config: IdentityModelConfig):
        super().__init__()
        self.config = config
        self.readout = _LayerTokenReadout(config)
        self.query_seed = nn.Parameter(torch.randn(config.learned_width) * .02)
        self.query_attention = nn.MultiheadAttention(config.learned_width, config.attention_heads, batch_first=True)
        self.source_kind = nn.Embedding(len(SOURCE_KINDS), config.learned_width)
        self.concept_view = nn.Embedding(len(CONCEPT_VIEWS), config.learned_width)
        self.residual_kind = nn.Embedding(2, config.learned_width)
        self.concept_evidence_query = nn.Linear(config.learned_width, config.learned_width, bias=False)
        self.source_evidence_key = nn.Linear(config.learned_width, config.learned_width, bias=False)
        self.graph = nn.ModuleList(_DirectedGraphLayer(config.learned_width) for _ in range(config.graph_layers))

    @staticmethod
    def source_masks(frame: IdentityFrame) -> dict[str, Tensor]:
        masks = {view: torch.zeros_like(frame.sources.entry_mask) for view in CONCEPT_VIEWS}
        for row, records in enumerate(frame.source_records):
            for col, source in enumerate(records):
                if source is None:
                    continue
                masks["identity"][row, col] = source.identity_core_allowed
                if source.context_allowed:
                    view = {"HOST": "host", "RELATIONSHIP": "relationship", "AEXP": "self", "ASELF": "self"}.get(source.source_kind, "context")
                    masks[view][row, col] = True
        return masks

    @staticmethod
    def _concept_authority(frame: IdentityFrame, source_masks: Mapping[str, Tensor]) -> tuple[dict, dict]:
        s, k = frame.sources.states.shape[1], frame.concepts.states.shape[1]
        concepts = {view: torch.zeros_like(frame.concepts.entry_mask) for view in CONCEPT_VIEWS}
        intended = {view: torch.zeros_like(frame.concepts.entry_mask) for view in CONCEPT_VIEWS}
        roles = {role: torch.zeros_like(frame.graph.edge_mask) for role in EDGE_ROLES}
        for row, records in enumerate(frame.concept_records):
            for col, record in enumerate(records):
                if record is not None:
                    intended[record.view][row, col] = True
        for row, edge_roles in enumerate(frame.graph.roles):
            for col, role in enumerate(edge_roles):
                if role is not None:
                    roles[role][row, col] = True
        sender = frame.graph.senders.clamp(0, s + k - 1)
        target = (frame.graph.receivers - s).clamp(0, k - 1)
        view_edges = {}
        for view in CONCEPT_VIEWS:
            role_mask = (roles["identity_support"] | roles["inference_support"]
                         if view == "identity" else roles["context_only"])
            eligible = frame.graph.edge_mask & role_mask & intended[view].gather(1, target)
            # Authorization follows complete source lineage, independent of
            # learned graph depth or score. Unsupported cycles gain nothing.
            for _ in range(k):
                admitted = torch.cat((source_masks[view], concepts[view]), 1).gather(1, sender)
                incoming = torch.zeros_like(concepts[view], dtype=torch.long).scatter_add_(
                    1, target, (eligible & admitted).long()).bool()
                expanded = concepts[view] | incoming
                if torch.equal(expanded, concepts[view]):
                    break
                concepts[view] = expanded
            admitted = torch.cat((source_masks[view], concepts[view]), 1).gather(1, sender)
            view_edges[view] = eligible & admitted
        return concepts, view_edges

    def forward(self, frame: IdentityFrame) -> tuple[IdentityRepresentation, Tensor]:
        frame.validate(self.config)
        batch = frame.query.states.shape[0]
        seed = self.query_seed[None].expand(batch, -1)
        query_tokens, _, _ = self.readout(frame.query, seed)
        query_tokens = query_tokens.flatten(1, 2)
        query_mask = frame.query.token_mask.flatten(1, 2)
        query, _ = self.query_attention(seed[:, None], query_tokens, query_tokens,
                                        key_padding_mask=~query_mask, need_weights=False)
        query = query[:, 0]
        source_tokens, sources, layer_weights = self.readout(frame.sources, query)
        _, concepts, _ = self.readout(frame.concepts, query)
        _, relations, _ = self.readout(frame.relations, query)
        kinds = torch.zeros_like(frame.sources.entry_mask, dtype=torch.long)
        views = torch.zeros_like(frame.concepts.entry_mask, dtype=torch.long)
        residual = torch.zeros_like(views)
        for row, records in enumerate(frame.source_records):
            for col, record in enumerate(records):
                if record is not None:
                    kinds[row, col] = SOURCE_KINDS.index(record.source_kind)
        for row, records in enumerate(frame.concept_records):
            for col, record in enumerate(records):
                if record is not None:
                    views[row, col] = CONCEPT_VIEWS.index(record.view)
                    residual[row, col] = int(record.residual)
        sources = sources + self.source_kind(kinds)
        concepts = concepts + self.concept_view(views) + self.residual_kind(residual)
        source_masks = self.source_masks(frame)
        concept_masks, graph_edges = self._concept_authority(frame, source_masks)
        stream_states, stream_masks = {}, {}
        combined_concepts = torch.zeros_like(concepts)
        combined_concept_mask = torch.zeros_like(frame.concepts.entry_mask)
        for view in CONCEPT_VIEWS:
            source_view = sources * source_masks[view][..., None]
            concept_view = concepts * concept_masks[view][..., None]
            for layer in self.graph:
                concept_view = layer(source_view, concept_view, relations, frame,
                                     graph_edges[view], concept_masks[view])
            stream_states[view] = torch.cat((source_view, concept_view), 1)
            stream_masks[view] = torch.cat((source_masks[view], concept_masks[view]), 1)
            combined_concepts = combined_concepts + concept_view
            combined_concept_mask |= concept_masks[view]
        source_available = torch.stack(tuple(source_masks.values())).any(0)
        concept_source_mask = torch.zeros((batch, frame.concepts.states.shape[1], frame.sources.states.shape[1]),
                                          device=sources.device, dtype=torch.bool)
        for view in CONCEPT_VIEWS:
            concept_source_mask |= concept_masks[view][..., None] & source_masks[view][:, None]
        concept_source_logits = torch.einsum("bkh,bsh->bks", self.concept_evidence_query(combined_concepts),
                                             self.source_evidence_key(sources))
        concept_source_probabilities = masked_softmax(concept_source_logits, concept_source_mask)
        source_tokens = source_tokens * source_available[..., None, None]
        sources = sources * source_available[..., None]
        representation = IdentityRepresentation(
            source_tokens, sources, combined_concepts, combined_concept_mask,
            stream_states, stream_masks, query_tokens, query_mask,
            source_masks["identity"].any(-1), layer_weights,
            concept_source_logits * concept_source_mask, concept_source_probabilities, concept_source_mask)
        return representation, query


class _CandidateReadoutLayer(nn.Module):
    def __init__(self, config: IdentityModelConfig):
        super().__init__()
        self.config = config
        width = config.learned_width
        self.context = nn.MultiheadAttention(width, config.attention_heads, batch_first=True)
        self.alternatives = nn.MultiheadAttention(width, config.attention_heads, batch_first=True)
        self.context_norm = nn.LayerNorm(width)
        self.alternative_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))
        self.output_norm = nn.LayerNorm(width)

    def forward(self, candidates: Tensor, mask: Tensor, memory: Tensor, memory_mask: Tensor) -> Tensor:
        attended, _ = self.context(candidates, memory, memory, key_padding_mask=~memory_mask, need_weights=False)
        states = self.context_norm(candidates + attended) * mask[..., None]
        safe_mask = mask.clone()
        safe_mask[~safe_mask.any(-1), 0] = True
        alternatives, _ = self.alternatives(states, states, states, key_padding_mask=~safe_mask, need_weights=False)
        states = self.alternative_norm(states + alternatives) * mask[..., None]
        return self.output_norm(states + self.feedforward(states)) * mask[..., None]


class _SemanticHead(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.candidate = nn.Linear(width, width)
        self.label = nn.Linear(width, width)
        self.score = nn.Linear(width, 1, bias=False)

    def forward(self, candidates: Tensor, labels: Tensor) -> Tensor:
        return self.score(torch.tanh(self.candidate(candidates)[:, :, None]
                                     + self.label(labels)[:, None])).squeeze(-1)


class IdentityJudgmentModel(nn.Module):
    """N2: alternatives and complete semantic heads, not candidate-ID classes."""
    def __init__(self, config: IdentityModelConfig):
        super().__init__()
        self.config = config
        self.readout = _LayerTokenReadout(config)
        self.layers = nn.ModuleList(_CandidateReadoutLayer(config) for _ in range(config.readout_layers))
        self.heads = nn.ModuleDict({"family_" + family: _SemanticHead(config.learned_width) for family in HEAD_FAMILIES})
        self.scalar = nn.Sequential(nn.Linear(config.learned_width, config.learned_width), nn.GELU(),
                                    nn.Linear(config.learned_width, 7))
        self.evidence_query = nn.Linear(config.learned_width, config.learned_width, bias=False)
        self.evidence_key = nn.Linear(config.learned_width, config.learned_width, bias=False)
        self.concept_query = nn.Linear(config.learned_width, config.learned_width, bias=False)
        self.concept_key = nn.Linear(config.learned_width, config.learned_width, bias=False)
        self.voice_confidence = _SemanticHead(config.learned_width)

    def forward(self, frame: IdentityFrame, n1: IdentityRepresentation, query: Tensor) -> dict:
        frame.validate(self.config)
        _, candidates, _ = self.readout(frame.candidates, query)
        mask = frame.candidates.entry_mask & frame.candidate_authority_mask & frame.candidate_context_mask
        tokens = n1.source_tokens.flatten(1, 2)
        source_available = torch.stack(tuple(IdentityRepresentationModel.source_masks(frame).values())).any(0)
        token_mask = (frame.sources.token_mask & source_available[..., None]).flatten(1, 2)
        memory = torch.cat((n1.query_tokens, tokens, *n1.views.values()), 1)
        memory_mask = torch.cat((n1.query_mask, token_mask, *n1.view_masks.values()), 1)
        for layer in self.layers:
            candidates = layer(candidates, mask, memory, memory_mask)
        heads = {}
        label_states = {}
        for family, bank in frame.labels.items():
            _, labels, _ = self.readout(bank, query)
            label_states[family] = labels
            heads[family] = self.heads["family_" + family](candidates, labels)
        evidence_logits = torch.einsum("bch,bsh->bcs", self.evidence_query(candidates), self.evidence_key(n1.source_states))
        concept_logits = torch.einsum("bch,bkh->bck", self.concept_query(candidates), self.concept_key(n1.concepts))
        return {"candidates": candidates, "candidate_mask": mask, "heads": heads,
                "scalar": self.scalar(candidates), "evidence_logits": evidence_logits,
                "concept_logits": concept_logits, "source_mask": source_available,
                "voice_confidence": self.voice_confidence(candidates, label_states["voice"])}


class IdentityCalibration(nn.Module):
    """N3 state lives separately; no input/provenance/graph rewriting API."""
    def __init__(self):
        super().__init__()
        self.log_temperatures = nn.ParameterDict({"temperature_" + name: nn.Parameter(torch.zeros(()))
                                                 for name in (*HEAD_FAMILIES, "preference", "scalar", "voice_confidence")})
        self.risk_adjustment = nn.Linear(3, 3)
        nn.init.zeros_(self.risk_adjustment.weight)
        nn.init.zeros_(self.risk_adjustment.bias)

    def scale(self, name: str, values: Tensor) -> Tensor:
        return values * self.log_temperatures["temperature_" + name].clamp(-7, 7).neg().exp()

    def risks(self, scalar: Tensor, margins: Tensor) -> Tensor:
        signals = torch.stack((margins.abs(), scalar[..., 3], scalar[..., 2]), -1)
        return self.scale("scalar", scalar[..., (3, 5, 6)]) + self.risk_adjustment(signals)


class IdentityModel(nn.Module):
    """Implemented trainable EIPM core; default phase is frozen inference.

    No acceptance flag or checkpoint API can certify learned identity. External
    source compilation, objectives and qualification must be implemented and
    evidenced separately. All exports remain UNTRAINED/UNQUALIFIED.
    """
    def __init__(self, config: IdentityModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.n1 = IdentityRepresentationModel(config)
        self.n2 = IdentityJudgmentModel(config)
        self.n3 = IdentityCalibration()
        self.phase = "inference"
        self._calibration_batch = None
        self.set_phase("inference")

    def set_phase(self, phase: str, *, adapt_n1: bool = False,
                  calibration_batch: CalibrationBatch | None = None) -> None:
        if phase not in {"inference", "n1", "n2", "calibration"} or type(adapt_n1) is not bool:
            raise IdentityError("phase must be inference, n1, n2 or calibration with a boolean adaptation flag")
        if adapt_n1 and phase != "n2":
            raise IdentityError("N1 continuation is an explicit N2-phase option")
        if phase == "calibration":
            if not isinstance(calibration_batch, CalibrationBatch):
                raise IdentityError("calibration phase requires explicitly supplied calibration data")
            calibration_batch.validate(self.config)
        elif calibration_batch is not None:
            raise IdentityError("calibration labels cannot enter N1/N2/inference phases")
        self._calibration_batch = calibration_batch
        self.phase = phase
        self.zero_grad(set_to_none=True)
        self.requires_grad_(False)
        self.n1.requires_grad_(phase == "n1" or (phase == "n2" and adapt_n1))
        self.n2.requires_grad_(phase == "n2")
        self.n3.requires_grad_(phase == "calibration")
        super().train(phase != "inference")
        self.n1.train(phase == "n1" or (phase == "n2" and adapt_n1))
        self.n2.train(phase == "n2")
        self.n3.train(phase == "calibration")

    def train(self, mode: bool = True) -> "IdentityModel":
        if type(mode) is not bool:
            raise IdentityError("training mode must be an actual boolean")
        if mode and self.phase == "inference":
            raise IdentityError("select an explicit learned or calibration phase before training mode")
        if not mode:
            super().train(False)
        else:
            self.n1.train(any(parameter.requires_grad for parameter in self.n1.parameters()))
            self.n2.train(any(parameter.requires_grad for parameter in self.n2.parameters()))
            self.n3.train(self.phase == "calibration")
            self.training = True
        return self

    def forward(self, frame: IdentityFrame) -> IdentityDecisionPacket:
        if self.phase == "calibration" and frame is not self._calibration_batch.frame:
            raise IdentityError("N3 forward must use its admitted calibration frame")
        frame.validate(self.config)
        if frame.query.states.device != next(self.parameters()).device:
            raise IdentityError("frame provider tensors and learned core must share a device")
        representation, query = self.n1(frame)
        raw = self.n2(frame, representation, query)
        mask = raw["candidate_mask"]
        scalar = raw["scalar"]
        preference_logits = self.n3.scale("preference", scalar[..., 0])
        preferences = masked_softmax(preference_logits, mask)
        # Candidate margins do not imply epistemic certainty, particularly for
        # a singleton. Uncertainty and evidence sufficiency remain separate.
        count = mask.shape[1]
        rivals = mask[:, None].expand(-1, count, -1) & ~torch.eye(count, device=mask.device, dtype=torch.bool)[None]
        rival_scores = preference_logits[:, None].expand(-1, count, -1).masked_fill(~rivals, -torch.finfo(scalar.dtype).max)
        margins = torch.where(rivals.any(-1), preference_logits - rival_scores.max(-1).values,
                              torch.zeros_like(preference_logits))
        risks = torch.sigmoid(self.n3.risks(scalar, margins))
        heads = {}
        for family, scores in raw["heads"].items():
            scores = self.n3.scale(family, scores)
            head_mask = mask[..., None] & frame.labels[family].entry_mask[:, None]
            activations = masked_softmax(scores, head_mask) if family == "stance" else torch.sigmoid(scores) * head_mask
            heads[family] = HeadOutput(scores * head_mask, activations, head_mask, frame.label_specs[family])
        source_mask = raw["source_mask"][:, None] & mask[..., None]
        evidence = masked_softmax(raw["evidence_logits"], source_mask)
        historical = torch.zeros_like(raw["source_mask"])
        experience = torch.zeros_like(historical)
        for row, records in enumerate(frame.source_records):
            for col, record in enumerate(records):
                if record is not None:
                    historical[row, col] = record.historical_truth_allowed
                    experience[row, col] = record.alice_lived_memory
        historical_pointers = masked_softmax(raw["evidence_logits"], source_mask & historical[:, None])
        experience_pointers = masked_softmax(raw["evidence_logits"], source_mask & experience[:, None])
        concept_pointers = masked_softmax(raw["concept_logits"], representation.concept_mask[:, None] & mask[..., None])
        grounding = representation.identity_available
        grounded_candidates = mask & grounding[:, None]
        sufficiency = torch.sigmoid(self.n3.scale("scalar", scalar[..., 2])) * grounded_candidates
        uncertainty = torch.where(grounded_candidates, risks[..., 0], torch.ones_like(sufficiency))
        voice_values = torch.zeros_like(heads["voice"].activations)
        for row, specs in enumerate(frame.label_specs["voice"]):
            for col, spec in enumerate(specs):
                if spec is not None:
                    voice_values[row, :, col] = (spec.minimum + heads["voice"].activations[row, :, col]
                                                 * (spec.maximum - spec.minimum))
        voice_values *= heads["voice"].mask
        voice_confidence = torch.sigmoid(self.n3.scale("voice_confidence", raw["voice_confidence"])) \
            * heads["voice"].mask * grounded_candidates[..., None]
        value_scores = heads["values"].scores
        pair_mask = heads["values"].mask[..., :, None] & heads["values"].mask[..., None, :]
        value_tradeoffs = (value_scores[..., :, None] - value_scores[..., None, :]) * pair_mask
        packet = IdentityDecisionPacket(
            candidate_ids=frame.candidates.entry_ids, source_records=frame.source_records,
            concept_ids=frame.concepts.entry_ids, concept_records=frame.concept_records, candidate_mask=mask,
            preference_logits=preference_logits * mask, preferences=preferences,
            preference_margins=margins * mask,
            co_valid_probabilities=torch.sigmoid(self.n3.scale("scalar", scalar[..., 1])) * mask,
            heads=heads, value_tradeoff_margins=value_tradeoffs,
            voice_control_values=voice_values, voice_control_confidence=voice_confidence,
            evidence_pointers=evidence, historical_evidence_pointers=historical_pointers,
            alice_experience_pointers=experience_pointers, concept_pointers=concept_pointers,
            evidence_sufficiency=sufficiency, uncertainty=uncertainty,
            contraindications=torch.sigmoid(self.n3.scale("scalar", scalar[..., 4])) * mask,
            failure_tail_risk=torch.where(grounded_candidates, risks[..., 1], torch.ones_like(sufficiency)),
            owner_fidelity_uncertainty=torch.where(grounded_candidates, risks[..., 2], torch.ones_like(sufficiency)),
            identity_grounding_available=grounding, latent_candidates=raw["candidates"],
            representation=representation)
        packet.validate()
        return packet

    def calibration_loss(self, batch: CalibrationBatch) -> Tensor:
        """Actual calibration objective, callable only for admitted N3 data.

        This does not run an optimizer or certify source/owner qualification.
        The caller owns independent target receipts and calibration holdouts.
        """
        if self.phase != "calibration" or batch is not self._calibration_batch:
            raise IdentityError("calibration loss requires its explicit admitted calibration phase")
        batch.validate(self.config)
        packet = self(batch.frame)
        total = torch.zeros((), device=packet.uncertainty.device)
        for name, targets in batch.targets.items():
            family_mask = batch.target_mask if batch.target_masks is None else batch.target_masks[name]
            if not bool(family_mask.any()):
                continue
            clean = torch.where(family_mask, targets.detach(), torch.zeros_like(targets))
            if name in {"preferences", "head:stance"}:
                scores = packet.preference_logits if name == "preferences" else packet.heads["stance"].scores
                logits = scores.float().masked_fill(~family_mask, -torch.finfo(torch.float32).max)
                loss = -(clean.float() * nn.functional.log_softmax(logits, dim=-1)).sum(-1)
                total = total + loss[family_mask.any(-1)].mean()
            elif name.startswith("head:"):
                values = packet.heads[name.split(":", 1)[1]].scores[family_mask]
                total = total + nn.functional.binary_cross_entropy_with_logits(values, clean.to(values.dtype)[family_mask])
            elif name == "candidate_pair_probabilities":
                margins = packet.preference_logits[..., :, None] - packet.preference_logits[..., None, :]
                total = total + nn.functional.binary_cross_entropy_with_logits(margins[family_mask], clean[family_mask])
            elif name == "value_tradeoff_probabilities":
                total = total + nn.functional.binary_cross_entropy_with_logits(packet.value_tradeoff_margins[family_mask], clean[family_mask])
            else:
                values = getattr(packet, name)[family_mask]
                total = total + nn.functional.binary_cross_entropy(values, clean.to(values.dtype)[family_mask])
        return total
