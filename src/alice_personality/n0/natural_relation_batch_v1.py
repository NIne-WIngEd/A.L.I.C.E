from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def _tokenize(tokenizer: Any, texts: Sequence[str]) -> tuple[Tensor, Tensor]:
    encoded=tokenizer(
        list(texts),
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"].bool()


def _query_text(row: Mapping[str,Any]) -> str:
    instruction=str(row["instruction"]).strip()
    sentence=str(row["sentence"]).strip()
    head=str((row.get("head") or {}).get("text","")).strip()
    tail=str((row.get("tail") or {}).get("text","")).strip()
    if not instruction or not sentence or not head or not tail:
        raise ValueError("natural-relation row requires instruction, sentence, head, and tail text")
    return (
        f"{instruction} Natural sentence: {sentence} "
        f"Head entity: {head}. Tail entity: {tail}."
    )


def compile_natural_relation_batch(
    *,
    rows: Sequence[Mapping[str,Any]],
    relation_bank: Mapping[str,Any],
    tokenizer: Any,
) -> dict[str,Any]:
    """Compile FewRel-style public natural relation rows without fake factors.

    Opaque relation keys are used only by the compiler to align labels with
    runtime-supplied semantic descriptions. The registered model receives only
    natural query text, relation-description token states, structural all-valid
    type placeholders, and a per-example candidate mask.
    """
    if not rows:
        raise ValueError("natural-relation batch cannot be empty")
    if relation_bank.get("schema")!="alice.eipm.n0.fewrel-runtime-relation-bank.v1":
        raise ValueError("natural-relation bank schema drift")
    if relation_bank.get("relation_keys_are_metadata_only") is not True:
        raise ValueError("natural-relation keys must remain metadata-only")
    if relation_bank.get("private_identity_data") is not False:
        raise ValueError("natural-relation bank must be public/non-identity")
    bank_relations=dict(relation_bank.get("relations") or {})
    if not bank_relations:
        raise ValueError("natural-relation bank is empty")

    active_keys=sorted({
        str(key)
        for row in rows
        for key in (row.get("candidate_relation_keys") or [])
    })
    if not active_keys:
        raise ValueError("natural-relation rows require candidate relations")
    missing=[key for key in active_keys if key not in bank_relations]
    if missing:
        raise ValueError(f"candidate relations missing from runtime bank: {missing}")

    relation_text=[]
    for key in active_keys:
        value=dict(bank_relations[key])
        text=str(value.get("semantic_text","")).strip()
        if not text:
            raise ValueError(f"natural relation {key!r} lacks semantic_text")
        if key.lower() in text.lower():
            raise ValueError(f"opaque relation key leaked into semantic text: {key}")
        relation_text.append(text)
    relation_index={key:i for i,key in enumerate(active_keys)}

    query_ids,query_mask=_tokenize(
        tokenizer,
        [_query_text(row) for row in rows],
    )
    relation_ids,relation_mask=_tokenize(tokenizer,relation_text)

    batch_size=len(rows)
    candidate_mask=torch.zeros(
        batch_size,len(active_keys),dtype=torch.bool
    )
    target=torch.zeros(batch_size,dtype=torch.long)
    splits=[]
    for b,row in enumerate(rows):
        if row.get("private_identity_data") is not False:
            raise ValueError("natural-relation row contains private identity data")
        split=str(row.get("split",""))
        if split not in {"train","dev"}:
            raise ValueError("natural-relation optimizer batch is TRAIN/DEV only")
        if row.get("final_validation_only") is True:
            raise ValueError("FINAL natural-relation row entered optimizer batch")
        if row.get("training_authorized") is not (split=="train"):
            raise ValueError("natural-relation training authority drift")
        if row.get("model_selection_authorized") is not (split=="dev"):
            raise ValueError("natural-relation model-selection authority drift")

        candidates=[str(x) for x in (row.get("candidate_relation_keys") or [])]
        if not candidates or len(candidates)!=len(set(candidates)):
            raise ValueError("natural-relation candidate set empty or duplicated")
        for key in candidates:
            candidate_mask[b,relation_index[key]]=True

        target_key=str(row.get("target_relation_key",""))
        local_target=int(row.get("target_candidate_index",-1))
        if not 0 <= local_target < len(candidates):
            raise ValueError("natural-relation target candidate index out of range")
        if candidates[local_target]!=target_key:
            raise ValueError("natural-relation target key/index mismatch")
        target[b]=relation_index[target_key]
        if not bool(candidate_mask[b,target[b]]):
            raise RuntimeError("natural-relation target is not active")
        splits.append(split)

    # Natural relation selection does not own runtime type or symmetry labels.
    # The semantic operator's relation matching path does not consume these
    # fields; one all-valid generic argument type retains the relation-schema
    # interface without fabricating query-specific structural supervision.
    relation_domain=torch.ones(len(active_keys),1,dtype=torch.bool)
    relation_range=torch.ones_like(relation_domain)
    relation_symmetric=torch.zeros(len(active_keys),dtype=torch.bool)

    return {
        "batch":{
            "query_input_ids":query_ids,
            "query_attention_mask":query_mask,
            "relation_input_ids":relation_ids,
            "relation_attention_mask":relation_mask,
            "relation_domain_type_mask":relation_domain,
            "relation_range_type_mask":relation_range,
            "relation_symmetric":relation_symmetric,
            "relation_candidate_mask":candidate_mask,
            "max_reasoning_steps":1,
        },
        "target_relation_index":target,
        "metadata":{
            "batch_size":batch_size,
            "runtime_relation_count":len(active_keys),
            "relation_keys_model_visible":False,
            "factor_labels_fabricated":False,
            "downstream_fabric_labels_fabricated":False,
            "private_identity_data":False,
            "final_rows":0,
            "splits":splits,
            "runtime_relation_count_is_capability_ceiling":False,
        },
    }


def natural_relation_semantic_loss(
    *,
    outputs: Mapping[str,Any],
    target_relation_index: Tensor,
) -> Tensor:
    semantic=outputs["semantic_operator"]
    logits=semantic["relation_logits"]
    if logits.ndim!=3 or logits.size(1)!=1:
        raise ValueError("natural-relation task must emit one relation step [B,1,R]")
    target=target_relation_index.to(device=logits.device,dtype=torch.long)
    if target.shape!=(logits.size(0),):
        raise ValueError("natural-relation target geometry drift")
    return F.cross_entropy(logits[:,0,:],target)
