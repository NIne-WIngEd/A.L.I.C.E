from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor


def _encode_with_offsets(
    tokenizer: Any,
    texts: list[str],
    *,
    max_length: int | None,
) -> dict[str, Tensor]:
    if not texts:
        raise ValueError("evidence target encoding requires at least one text")
    kwargs: dict[str, Any] = {
        "padding": True,
        "return_offsets_mapping": True,
        "return_special_tokens_mask": True,
        "return_tensors": "pt",
    }
    if max_length is not None:
        if int(max_length) <= 0:
            raise ValueError("max_length must be positive when supplied")
        kwargs["truncation"] = True
        kwargs["max_length"] = int(max_length)
    else:
        kwargs["truncation"] = False

    encoded = tokenizer(texts, **kwargs)
    required = {
        "input_ids",
        "attention_mask",
        "offset_mapping",
        "special_tokens_mask",
    }
    missing = required - set(encoded)
    if missing:
        raise ValueError(
            "fast tokenizer evidence alignment requires: "
            + repr(sorted(missing))
        )
    input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
    attention = torch.as_tensor(
        encoded["attention_mask"],
        dtype=torch.bool,
    )
    offsets = torch.as_tensor(
        encoded["offset_mapping"],
        dtype=torch.long,
    )
    special = torch.as_tensor(
        encoded["special_tokens_mask"],
        dtype=torch.bool,
    )
    if input_ids.ndim != 2 or attention.shape != input_ids.shape:
        raise ValueError("tokenizer evidence input geometry drift")
    if offsets.shape != input_ids.shape + (2,):
        raise ValueError("tokenizer offset geometry drift")
    if special.shape != input_ids.shape:
        raise ValueError("tokenizer special-token mask geometry drift")
    content = (
        attention
        & ~special
        & offsets[..., 1].gt(offsets[..., 0])
    )
    if bool((content.sum(dim=-1) == 0).any()):
        raise ValueError("tokenizer produced a text with no content tokens")
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "offset_mapping": offsets,
        "content_mask": content,
    }


def _validated_span(
    text: str,
    span: Mapping[str, Any],
) -> tuple[int, int]:
    start = int(span.get("start", -1))
    end = int(span.get("end", -1))
    expected = str(span.get("text", ""))
    if not (0 <= start < end <= len(text)):
        raise ValueError("evidence character span is outside source text")
    if not expected or text[start:end] != expected:
        raise ValueError("evidence character span text mismatch")
    return start, end


def _span_token_mask(
    *,
    text: str,
    offsets: Tensor,
    content_mask: Tensor,
    span: Mapping[str, Any],
) -> Tensor:
    if offsets.ndim != 2 or offsets.size(-1) != 2:
        raise ValueError("offsets must be [T,2]")
    if content_mask.shape != offsets.shape[:1]:
        raise ValueError("content mask/offset geometry drift")
    start, end = _validated_span(text, span)
    overlap = (
        offsets[:, 1].gt(start)
        & offsets[:, 0].lt(end)
        & content_mask
    )
    if not bool(overlap.any()):
        raise ValueError(
            "evidence span has no surviving tokenizer token; "
            "possible truncation or offset drift"
        )
    return overlap


def _scatter_selected_span(
    *,
    target: Tensor,
    valid: Tensor,
    step: int,
    candidate_index: int,
    text: str,
    offsets: Tensor,
    content_mask: Tensor,
    span: Mapping[str, Any],
) -> None:
    if not (0 <= step < target.size(0)):
        raise ValueError("evidence step outside operator slot axis")
    if not (0 <= candidate_index < target.size(1)):
        raise ValueError("evidence candidate outside runtime bank")
    positive = _span_token_mask(
        text=text,
        offsets=offsets,
        content_mask=content_mask,
        span=span,
    )
    target[step, candidate_index] = positive.to(target.dtype)
    valid[step, candidate_index] = content_mask


def compile_operator_evidence_targets(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    max_length: int | None = None,
) -> dict[str, Any]:
    """Compile governed char spans into exact token-level supervision tensors.

    Only the selected semantic candidate receives a valid evidence mask.
    Distractor candidates are not silently labeled as all-negative evidence.
    STOP/UNKNOWN-only operator slots receive no relation/factor evidence mask.
    If truncation removes a labeled span, compilation fails closed.
    """

    query = str(row.get("query", ""))
    relation_candidates = list(row.get("relation_candidates") or [])
    if not query or not relation_candidates:
        raise ValueError("operator evidence row requires query and relation candidates")
    relation_texts = [str(item.get("text", "")) for item in relation_candidates]
    if any(not text for text in relation_texts):
        raise ValueError("relation candidate text cannot be empty")

    operator_slots = int(row.get("runtime_operator_slots", -1))
    if operator_slots <= 0:
        raise ValueError("runtime_operator_slots must be positive")
    relation_targets = [
        int(value)
        for value in (row.get("relation_sequence_target") or [])
    ]
    query_spans = list(
        row.get("query_relation_evidence_char_spans") or []
    )
    relation_spans = list(
        row.get("relation_schema_evidence_char_spans") or []
    )
    if len(query_spans) != len(relation_targets):
        raise ValueError("query evidence span/program length drift")
    if len(relation_spans) != len(relation_targets):
        raise ValueError("relation schema evidence span/program length drift")
    if len(relation_targets) > operator_slots:
        raise ValueError("relation program exceeds operator slot axis")

    query_encoded = _encode_with_offsets(
        tokenizer,
        [query],
        max_length=max_length,
    )
    relation_encoded = _encode_with_offsets(
        tokenizer,
        relation_texts,
        max_length=max_length,
    )
    q_tokens = query_encoded["input_ids"].size(1)
    r_tokens = relation_encoded["input_ids"].size(1)
    relation_count = len(relation_candidates)

    query_target = torch.zeros(
        operator_slots,
        relation_count,
        q_tokens,
        dtype=torch.float32,
    )
    query_valid = torch.zeros_like(query_target, dtype=torch.bool)
    relation_target = torch.zeros(
        operator_slots,
        relation_count,
        r_tokens,
        dtype=torch.float32,
    )
    relation_valid = torch.zeros_like(relation_target, dtype=torch.bool)

    for step, candidate_index in enumerate(relation_targets):
        q_span = query_spans[step]
        r_span = relation_spans[step]
        if int(q_span.get("step", -1)) != step:
            raise ValueError("query evidence step index drift")
        if int(r_span.get("step", -1)) != step:
            raise ValueError("relation schema evidence step index drift")
        if int(r_span.get("candidate_index", -1)) != candidate_index:
            raise ValueError("relation schema evidence candidate drift")
        _scatter_selected_span(
            target=query_target,
            valid=query_valid,
            step=step,
            candidate_index=candidate_index,
            text=query,
            offsets=query_encoded["offset_mapping"][0],
            content_mask=query_encoded["content_mask"][0],
            span=q_span,
        )
        _scatter_selected_span(
            target=relation_target,
            valid=relation_valid,
            step=step,
            candidate_index=candidate_index,
            text=relation_texts[candidate_index],
            offsets=relation_encoded["offset_mapping"][candidate_index],
            content_mask=relation_encoded["content_mask"][candidate_index],
            span=r_span,
        )

    factor_schemas = dict(row.get("factor_schemas") or {})
    factor_targets = dict(row.get("factor_targets") or {})
    factor_spans = dict(
        row.get("factor_schema_evidence_char_spans") or {}
    )
    if set(factor_schemas) != set(factor_targets):
        raise ValueError("factor schema/target bank mismatch")
    if set(factor_schemas) != set(factor_spans):
        raise ValueError("factor schema/evidence bank mismatch")

    factor_encoded: dict[str, dict[str, Tensor]] = {}
    factor_target_tensors: dict[str, Tensor] = {}
    factor_valid_tensors: dict[str, Tensor] = {}
    for name, bank_raw in factor_schemas.items():
        bank = list(bank_raw)
        texts = [str(item.get("text", "")) for item in bank]
        encoded = _encode_with_offsets(
            tokenizer,
            texts,
            max_length=max_length,
        )
        factor_encoded[str(name)] = encoded
        candidate_index = int(factor_targets[name])
        if not (0 <= candidate_index < len(texts)):
            raise ValueError(f"factor target outside bank: {name}")
        span = factor_spans[name]
        if int(span.get("candidate_index", -1)) != candidate_index:
            raise ValueError(f"factor evidence candidate drift: {name}")
        tokens = encoded["input_ids"].size(1)
        target = torch.zeros(len(texts), tokens, dtype=torch.float32)
        valid = torch.zeros_like(target, dtype=torch.bool)
        positive = _span_token_mask(
            text=texts[candidate_index],
            offsets=encoded["offset_mapping"][candidate_index],
            content_mask=encoded["content_mask"][candidate_index],
            span=span,
        )
        target[candidate_index] = positive.to(target.dtype)
        valid[candidate_index] = encoded["content_mask"][candidate_index]
        factor_target_tensors[str(name)] = target
        factor_valid_tensors[str(name)] = valid

    step_factor_targets = dict(row.get("step_factor_targets") or {})
    step_factor_spans = dict(
        row.get("step_factor_schema_evidence_char_spans") or {}
    )
    if set(step_factor_targets) != set(step_factor_spans):
        raise ValueError("step factor target/evidence bank mismatch")
    if not set(step_factor_targets) <= set(factor_schemas):
        raise ValueError("step factor evidence refers to unknown factor bank")

    step_factor_target_tensors: dict[str, Tensor] = {}
    step_factor_valid_tensors: dict[str, Tensor] = {}
    for name, targets_raw in step_factor_targets.items():
        targets = [int(value) for value in targets_raw]
        spans = list(step_factor_spans[name])
        if len(targets) != len(relation_targets):
            raise ValueError(f"step factor/program length drift: {name}")
        if len(spans) != len(targets):
            raise ValueError(f"step factor evidence length drift: {name}")
        encoded = factor_encoded[name]
        texts = [str(item.get("text", "")) for item in factor_schemas[name]]
        tokens = encoded["input_ids"].size(1)
        target = torch.zeros(
            operator_slots,
            len(texts),
            tokens,
            dtype=torch.float32,
        )
        valid = torch.zeros_like(target, dtype=torch.bool)
        for step, candidate_index in enumerate(targets):
            span = spans[step]
            if int(span.get("step", -1)) != step:
                raise ValueError(f"step factor evidence step drift: {name}")
            if int(span.get("candidate_index", -1)) != candidate_index:
                raise ValueError(f"step factor evidence candidate drift: {name}")
            _scatter_selected_span(
                target=target,
                valid=valid,
                step=step,
                candidate_index=candidate_index,
                text=texts[candidate_index],
                offsets=encoded["offset_mapping"][candidate_index],
                content_mask=encoded["content_mask"][candidate_index],
                span=span,
            )
        step_factor_target_tensors[name] = target
        step_factor_valid_tensors[name] = valid

    return {
        "query": query_encoded,
        "relation_schema": relation_encoded,
        "factor_schemas": factor_encoded,
        "relation_query_evidence_target": query_target,
        "relation_query_evidence_valid_mask": query_valid,
        "relation_schema_evidence_target": relation_target,
        "relation_schema_evidence_valid_mask": relation_valid,
        "factor_schema_evidence_target": factor_target_tensors,
        "factor_schema_evidence_valid_mask": factor_valid_tensors,
        "step_factor_schema_evidence_target": step_factor_target_tensors,
        "step_factor_schema_evidence_valid_mask": step_factor_valid_tensors,
        "operator_slots": operator_slots,
        "relation_steps": len(relation_targets),
    }
