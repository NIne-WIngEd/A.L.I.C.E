from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


QSRE_FALLBACK = 0
QSRE_RELATIONAL = 1
QSRE_DEFER = 2


@dataclass(frozen=True)
class QSRETensorInputs:
    """Shape contract for QSRE mechanics.

    Axes B/L/Tq/F/Tf/E and feature widths are intentionally dynamic.
    This class validates interfaces only; it has no learned parameters.
    """

    query_hidden_states: Tensor
    query_token_mask: Tensor
    field_token_states: Tensor
    field_token_mask: Tensor
    field_base_state: Tensor
    field_struct_state: Tensor
    field_metadata: Tensor
    field_valid_mask: Tensor
    edge_index: Tensor
    edge_relation_state: Tensor
    edge_metadata: Tensor
    edge_valid_mask: Tensor

    def validate(self) -> dict[str, int]:
        if self.query_hidden_states.ndim != 4:
            raise ValueError("query_hidden_states must be [B,L,Tq,Ds]")
        batch, layers, query_tokens, _semantic = self.query_hidden_states.shape
        if self.query_token_mask.shape != (batch, query_tokens):
            raise ValueError("query_token_mask shape drift")

        if self.field_token_states.ndim != 4:
            raise ValueError("field_token_states must be [B,F,Tf,Ds]")
        fb, fields, field_tokens, field_semantic = self.field_token_states.shape
        if fb != batch:
            raise ValueError("field batch drift")
        if field_semantic != self.query_hidden_states.size(-1):
            raise ValueError("query/field semantic width drift")
        if self.field_token_mask.shape != (batch, fields, field_tokens):
            raise ValueError("field_token_mask shape drift")

        for name, tensor in (
            ("field_base_state", self.field_base_state),
            ("field_struct_state", self.field_struct_state),
            ("field_metadata", self.field_metadata),
        ):
            if tensor.ndim != 3 or tensor.shape[:2] != (batch, fields):
                raise ValueError(f"{name} must be [B,F,D]")

        if self.field_valid_mask.shape != (batch, fields):
            raise ValueError("field_valid_mask shape drift")

        if self.edge_index.ndim != 3 or self.edge_index.size(-1) != 2:
            raise ValueError("edge_index must be [B,E,2]")
        eb, edges, _ = self.edge_index.shape
        if eb != batch:
            raise ValueError("edge batch drift")

        for name, tensor in (
            ("edge_relation_state", self.edge_relation_state),
            ("edge_metadata", self.edge_metadata),
        ):
            if tensor.ndim != 3 or tensor.shape[:2] != (batch, edges):
                raise ValueError(f"{name} must be [B,E,D]")

        if self.edge_valid_mask.shape != (batch, edges):
            raise ValueError("edge_valid_mask shape drift")

        if self.edge_index.dtype not in (torch.int32, torch.int64):
            raise ValueError("edge_index must use integer dtype")
        if self.query_token_mask.dtype != torch.bool:
            raise ValueError("query_token_mask must be bool")
        if self.field_token_mask.dtype != torch.bool:
            raise ValueError("field_token_mask must be bool")
        if self.field_valid_mask.dtype != torch.bool:
            raise ValueError("field_valid_mask must be bool")
        if self.edge_valid_mask.dtype != torch.bool:
            raise ValueError("edge_valid_mask must be bool")

        if bool(self.edge_valid_mask.any()):
            valid_edges = self.edge_index[self.edge_valid_mask]
            if int(valid_edges.min()) < 0 or int(valid_edges.max()) >= fields:
                raise ValueError("valid edge endpoint outside field range")

        return {
            "batch": batch,
            "layers": layers,
            "query_tokens": query_tokens,
            "fields": fields,
            "field_tokens": field_tokens,
            "edges": edges,
        }


@dataclass(frozen=True)
class QSREQueryOperatorState:
    continuous: Tensor
    relation_anchor: Tensor
    role_anchor: Tensor
    temporal_status: Tensor
    applicability: Tensor
    uncertainty: Tensor
    layer_weight: Tensor

    def validate(self, *, batch: int, layers: int) -> dict[str, int]:
        for name, tensor in (
            ("continuous", self.continuous),
            ("relation_anchor", self.relation_anchor),
            ("role_anchor", self.role_anchor),
            ("temporal_status", self.temporal_status),
            ("uncertainty", self.uncertainty),
        ):
            if tensor.ndim != 2 or tensor.size(0) != batch:
                raise ValueError(f"{name} must be [B,D]")

        if self.applicability.shape != (batch,):
            raise ValueError("applicability must be [B]")
        if self.layer_weight.shape != (batch, layers):
            raise ValueError("layer_weight must be [B,L]")

        return {
            "relation_anchors": self.relation_anchor.size(1),
            "role_anchors": self.role_anchor.size(1),
            "operator_width": self.continuous.size(1),
        }


def masked_sparsemax(logits: Tensor, mask: Tensor, *, dim: int = -1) -> Tensor:
    """Reference exact-zero sparse projection with adaptive support size.

    This is a mechanics implementation, not the final optimized training kernel.
    Rows with no valid entries fail closed to all-zero support.
    """
    if logits.shape != mask.shape:
        raise ValueError("logits/mask shape mismatch")
    if mask.dtype != torch.bool:
        raise ValueError("mask must be bool")

    dim = dim if dim >= 0 else logits.ndim + dim
    if not (0 <= dim < logits.ndim):
        raise ValueError("invalid sparsemax dimension")

    moved_logits = logits.movedim(dim, -1)
    moved_mask = mask.movedim(dim, -1)
    width = moved_logits.size(-1)

    flat_logits = moved_logits.reshape(-1, width)
    flat_mask = moved_mask.reshape(-1, width)
    flat_out = torch.zeros_like(flat_logits)

    for row in range(flat_logits.size(0)):
        valid = flat_mask[row]
        if not bool(valid.any()):
            continue
        values = flat_logits[row][valid]
        sorted_values, _ = torch.sort(values, descending=True)
        cumsum = torch.cumsum(sorted_values, dim=0)
        k = torch.arange(
            1,
            sorted_values.numel() + 1,
            device=sorted_values.device,
            dtype=sorted_values.dtype,
        )
        support = 1 + k * sorted_values > cumsum
        k_star = int(support.sum().item())
        if k_star <= 0:
            raise RuntimeError("sparsemax support search failed")
        tau = (cumsum[k_star - 1] - 1) / k_star
        projected = torch.clamp(values - tau, min=0)
        total = projected.sum()
        if not bool(total > 0):
            raise RuntimeError("sparsemax produced zero valid mass")
        projected = projected / total
        flat_out[row][valid] = projected

    out = flat_out.reshape(moved_logits.shape)
    return out.movedim(-1, dim)


def role_incidence(
    edge_index: Tensor,
    edge_valid_mask: Tensor,
    *,
    num_fields: int,
) -> tuple[Tensor, Tensor]:
    """Return explicit SOURCE and TARGET incidence [B,F,E]."""
    if edge_index.ndim != 3 or edge_index.size(-1) != 2:
        raise ValueError("edge_index must be [B,E,2]")
    if edge_valid_mask.shape != edge_index.shape[:2]:
        raise ValueError("edge_valid_mask shape drift")
    if edge_valid_mask.dtype != torch.bool:
        raise ValueError("edge_valid_mask must be bool")
    if num_fields < 0:
        raise ValueError("num_fields must be non-negative")

    batch, edges, _ = edge_index.shape
    if edges == 0:
        empty = torch.zeros(
            batch,
            num_fields,
            0,
            dtype=torch.bool,
            device=edge_index.device,
        )
        return empty, empty.clone()

    if bool(edge_valid_mask.any()):
        valid_edges = edge_index[edge_valid_mask]
        if int(valid_edges.min()) < 0 or int(valid_edges.max()) >= num_fields:
            raise ValueError("valid edge endpoint outside field range")

    source_idx = edge_index[..., 0].clamp(min=0, max=max(num_fields - 1, 0))
    target_idx = edge_index[..., 1].clamp(min=0, max=max(num_fields - 1, 0))

    source = F.one_hot(source_idx.long(), num_classes=num_fields).permute(0, 2, 1).bool()
    target = F.one_hot(target_idx.long(), num_classes=num_fields).permute(0, 2, 1).bool()
    valid = edge_valid_mask[:, None, :]
    return source & valid, target & valid


def support_local_field_mask(
    field_valid_mask: Tensor,
    field_support_weight: Tensor,
    edge_index: Tensor,
    edge_support_weight: Tensor,
    edge_valid_mask: Tensor,
) -> Tensor:
    """Fields are active if directly supported or incident to an active edge."""
    if field_valid_mask.shape != field_support_weight.shape:
        raise ValueError("field support shape drift")
    if edge_support_weight.shape != edge_valid_mask.shape:
        raise ValueError("edge support shape drift")
    if edge_index.shape[:2] != edge_support_weight.shape:
        raise ValueError("edge index/support shape drift")
    if field_valid_mask.dtype != torch.bool or edge_valid_mask.dtype != torch.bool:
        raise ValueError("valid masks must be bool")

    num_fields = field_valid_mask.size(1)
    source, target = role_incidence(
        edge_index,
        edge_valid_mask,
        num_fields=num_fields,
    )
    active_edge = (edge_support_weight > 0) & edge_valid_mask
    incident = ((source | target) & active_edge[:, None, :]).any(dim=-1)
    direct = (field_support_weight > 0) & field_valid_mask
    return field_valid_mask & (direct | incident)


def support_local_softmax(logits: Tensor, support_mask: Tensor) -> Tensor:
    """Normalize only over claimed relational support; empty rows return zeros."""
    if logits.shape != support_mask.shape:
        raise ValueError("logits/support_mask shape mismatch")
    if support_mask.dtype != torch.bool:
        raise ValueError("support_mask must be bool")
    if logits.ndim != 2:
        raise ValueError("reference readout expects [B,F]")

    safe = logits.masked_fill(~support_mask, -1e30)
    probs = torch.softmax(safe, dim=-1) * support_mask.to(logits.dtype)
    denom = probs.sum(dim=-1, keepdim=True)
    return torch.where(denom > 0, probs / denom.clamp_min(1e-12), torch.zeros_like(probs))


def qsre_control_state(
    applicability: Tensor,
    has_support: Tensor,
    *,
    low: float = 0.25,
    high: float = 0.75,
) -> Tensor:
    """Return FALLBACK=0, RELATIONAL=1, DEFER=2 for each batch item."""
    if applicability.ndim != 1 or has_support.shape != applicability.shape:
        raise ValueError("applicability/has_support must be [B]")
    if has_support.dtype != torch.bool:
        raise ValueError("has_support must be bool")
    if not (0 <= low < high <= 1):
        raise ValueError("invalid control thresholds")

    out = torch.full_like(applicability, QSRE_FALLBACK, dtype=torch.long)
    middle = (applicability > low) & (applicability < high)
    out[middle] = QSRE_DEFER
    relational = (applicability >= high) & has_support
    out[relational] = QSRE_RELATIONAL
    return out


def support_cardinality(weights: Tensor, *, dim: int = -1) -> Tensor:
    return (weights > 0).sum(dim=dim)
