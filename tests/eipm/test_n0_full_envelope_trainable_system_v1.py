from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from alice_personality.n0.n0_full_envelope_trainable_system_v1 import (
    N0FullEnvelopeTrainableSystemConfig,
    N0FullEnvelopeTrainableSystemV1,
)
from alice_personality.n0.full_envelope_training_objective_v1 import (
    DEFAULT_FAMILY_WEIGHTS,
    FullEnvelopeJointTrainingObjectiveV1,
)
from alice_personality.n0.full_envelope_behavioral_batch_v1 import (
    compile_behavioral_batch,
)
from alice_personality.n0.natural_relation_batch_v1 import (
    compile_natural_relation_batch,
    natural_relation_semantic_loss,
)


class TinyBackbone(nn.Module):
    def __init__(self, *, vocab: int = 512, width: int = 24, hidden_states: int = 3) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.layers = nn.ModuleList(
            [nn.Linear(width, width) for _ in range(hidden_states - 1)]
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ):
        x = self.embedding(input_ids)
        states=[x]
        for layer in self.layers:
            x=torch.tanh(layer(x))
            states.append(x)
        mask=attention_mask.to(x.dtype).unsqueeze(-1)
        states=[state*mask for state in states]
        return SimpleNamespace(
            last_hidden_state=states[-1],
            hidden_states=tuple(states) if output_hidden_states else None,
        )


class TinySemanticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._backbone=TinyBackbone()
        self.replay_head=nn.Linear(24,16)
        self.preference=nn.Linear(24,1)

    @property
    def backbone(self) -> nn.Module:
        return self._backbone

    def forward(self, *, task: str, **batch):
        if task == "mlm":
            out=self.backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
            pooled=out.last_hidden_state.mean(dim=1)
            return {"loss":self.replay_head(pooled).square().mean()}
        if task == "teacher":
            out=self.backbone(
                input_ids=batch["candidate_input_ids"],
                attention_mask=batch["candidate_attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
            return {"scores":self.preference(out.last_hidden_state.mean(dim=1)).squeeze(-1)}
        raise ValueError(task)

    def parameter_report(self):
        total=sum(p.numel() for p in self.parameters())
        return {"total_parameters":total,"trainable_parameters":total}


def _tokens(count: int, *, length: int = 10, offset: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    rows=[]
    for i in range(count):
        middle=[offset + ((i*7+j) % 80) for j in range(length-2)]
        rows.append([2,*middle,3])
    ids=torch.tensor(rows,dtype=torch.long)
    return ids, torch.ones_like(ids,dtype=torch.bool)


def _bank(count: int, *, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    return _tokens(count,length=10,offset=offset)


def _full_batch() -> dict:
    q_ids,q_mask=_tokens(1,length=15,offset=10)
    r_ids,r_mask=_tokens(2,length=11,offset=20)

    factor_counts={
        "role":4,
        "traversal":3,
        "direction":3,
        "control":3,
        "reliability":2,
        "recency":2,
        "temporal":2,
        "provenance":2,
        "open_semantic_factor":3,
    }
    factor_input_ids={}
    factor_attention_mask={}
    offset=30
    for name,count in factor_counts.items():
        ids,mask=_bank(count,offset=offset)
        factor_input_ids[name]=ids
        factor_attention_mask[name]=mask
        offset+=7

    descriptor_input_ids={}
    descriptor_attention_mask={}
    for name,count in {"type":3,"provenance":3,"temporal":3}.items():
        ids,mask=_bank(count,offset=offset)
        descriptor_input_ids[name]=ids
        descriptor_attention_mask[name]=mask
        offset+=7

    field_ids=torch.tensor(
        [[
            [2,10,11,12,13,14,15,16,17,3],
            [2,20,21,22,23,24,25,26,27,3],
            [2,30,31,32,33,34,35,36,37,3],
            [0,0,0,0,0,0,0,0,0,0],
        ]],
        dtype=torch.long,
    )
    field_mask=field_ids.ne(0)
    field_valid=torch.tensor([[True,True,True,False]])

    candidate_ids=torch.tensor(
        [[
            [2,40,41,42,43,44,45,46,47,3],
            [2,50,51,52,53,54,55,56,57,3],
            [2,60,61,62,63,64,65,66,67,3],
        ]],
        dtype=torch.long,
    )
    candidate_mask=candidate_ids.ne(0)
    candidate_valid=torch.tensor([[True,True,True]])

    internal_ids,internal_mask=_bank(6,offset=15)

    return {
        "query_input_ids":q_ids,
        "query_attention_mask":q_mask,
        "relation_input_ids":r_ids,
        "relation_attention_mask":r_mask,
        "relation_domain_type_mask":torch.tensor(
            [[True,True,False],[False,True,True]],
            dtype=torch.bool,
        ),
        "relation_range_type_mask":torch.tensor(
            [[False,True,True],[True,False,True]],
            dtype=torch.bool,
        ),
        "relation_symmetric":torch.tensor([False,True]),
        "relation_candidate_mask":torch.tensor([[True,True]]),
        "factor_input_ids":factor_input_ids,
        "factor_attention_mask":factor_attention_mask,
        "factor_candidate_masks":{
            name:torch.ones(1,count,dtype=torch.bool)
            for name,count in factor_counts.items()
        },
        "factor_opcodes":{
            "role":["ROLE_SOURCE","ROLE_TARGET","ROLE_SYMMETRIC","ROLE_NONE"],
            "traversal":["TRAVERSAL_LOCAL","TRAVERSAL_PATH","TRAVERSAL_AGGREGATE"],
            "direction":["DIRECTION_FORWARD","DIRECTION_REVERSE","DIRECTION_BIDIRECTIONAL"],
            "control":["CONTROL_FALLBACK","CONTROL_RELATIONAL","CONTROL_DEFER"],
            "reliability":["MOD_RELIABILITY_OFF","MOD_RELIABILITY_ON"],
            "recency":["MOD_RECENCY_OFF","MOD_RECENCY_ON"],
            "temporal":["MOD_TEMPORAL_OFF","MOD_TEMPORAL_ON"],
            "provenance":["MOD_PROVENANCE_OFF","MOD_PROVENANCE_ON"],
        },
        "field_input_ids":field_ids,
        "field_attention_mask":field_mask,
        "field_valid_mask":field_valid,
        "field_confidence":torch.tensor([[0.9,0.8,0.7,0.0]]),
        "field_missing":torch.tensor([[0.0,0.0,0.0,1.0]]),
        "field_reliability":torch.tensor([[0.95,0.85,0.75,0.0]]),
        "descriptor_input_ids":descriptor_input_ids,
        "descriptor_attention_mask":descriptor_attention_mask,
        "descriptor_indices":{
            "type":torch.tensor([[0,1,2,-1]]),
            "provenance":torch.tensor([[0,1,2,-1]]),
            "temporal":torch.tensor([[0,1,2,-1]]),
        },
        "field_type_index":torch.tensor([[0,1,2,-1]]),
        "field_metadata":torch.tensor(
            [[[0.1,0.2,1.0],[0.3,0.4,1.0],[0.5,0.6,1.0],[0.0,0.0,0.0]]]
        ),
        "edge_index":torch.tensor([[[0,1],[1,2],[0,2]]]),
        "edge_relation_index":torch.tensor([[0,1,0]]),
        "edge_valid_mask":torch.tensor([[True,True,True]]),
        "edge_metadata":torch.tensor(
            [[[0.9,0.8,1.0,1.0],[0.8,0.7,1.0,1.0],[0.7,0.6,1.0,1.0]]]
        ),
        "edge_reliability":torch.tensor([[0.9,0.8,0.7]]),
        "edge_recency":torch.tensor([[0.8,0.7,0.6]]),
        "edge_temporal_match":torch.ones(1,3),
        "edge_provenance_match":torch.ones(1,3),
        "internal_view_descriptor_input_ids":internal_ids,
        "internal_view_descriptor_attention_mask":internal_mask,
        "internal_view_reliability":torch.ones(1,6),
        "candidate_input_ids":candidate_ids,
        "candidate_attention_mask":candidate_mask,
        "candidate_valid_mask":candidate_valid,
        "max_reasoning_steps":3,
        "graph_message_steps":2,
        "fusion_refinement_steps":2,
        "latent_slot_count":4,
        "latent_refinement_steps":2,
    }


def _system() -> N0FullEnvelopeTrainableSystemV1:
    return N0FullEnvelopeTrainableSystemV1(
        semantic_model=TinySemanticModel(),
        config=N0FullEnvelopeTrainableSystemConfig(
            semantic_dim=24,
            model_dim=24,
            num_hidden_states=3,
            num_attention_heads=4,
            structured_layers=1,
            field_metadata_dim=3,
            edge_metadata_dim=4,
            native_window_tokens=8,
            overlap_tokens=2,
            segment_bridge_layers=1,
            segment_query_chunk=2,
            segment_key_chunk=2,
            special_token_ids=(0,1,2,3,4),
            pad_token_id=0,
            dropout=0.0,
        ),
    )


def test_full_trainable_system_runs_end_to_end_with_unified_text_virtualization() -> None:
    torch.manual_seed(281)
    system=_system().eval()
    with torch.no_grad():
        out=system(task="full_envelope",batch=_full_batch())
    assert out["semantic_input_metadata"]["query"]["used_virtualization"] is True
    assert out["semantic_input_metadata"]["query"]["segment_count_max"] >= 2
    assert out["semantic_input_metadata"]["relation_schema"]["used_virtualization"] is True
    assert out["semantic_input_metadata"]["field_text"]["used_virtualization"] is True
    assert out["semantic_input_metadata"]["candidate_text"]["used_virtualization"] is True
    assert any(
        row["used_virtualization"]
        for row in out["semantic_input_metadata"]["factor_schema"].values()
    )
    assert any(
        row["used_virtualization"]
        for row in out["semantic_input_metadata"]["descriptor_text"].values()
    )
    assert out["public_judgment"]["candidate_logits"].shape == (1,3)
    assert out["operator"].relation_distribution.shape == (1,3,2)
    assert out["latent"]["pooled_state"].shape == (1,24)


def test_full_trainable_system_final_judgment_gradient_reaches_shared_backbone_and_long_bridge() -> None:
    torch.manual_seed(282)
    system=_system().train()
    out=system(task="full_envelope",batch=_full_batch())
    loss=(
        out["public_judgment"]["candidate_logits"].square().mean()
        + out["operator"].continuous_state.square().mean()
    )
    loss.backward()
    embedding_grad=system.semantic_model.backbone.embedding.weight.grad
    assert embedding_grad is not None
    assert float(embedding_grad.abs().sum()) > 0.0
    bridge_grads=[
        p.grad for p in system.semantic_input.segment_bridge.parameters()
        if p.requires_grad
    ]
    assert bridge_grads
    assert any(g is not None and float(g.abs().sum()) > 0.0 for g in bridge_grads)


def test_full_trainable_system_encodes_additional_runtime_view_source_text() -> None:
    torch.manual_seed(2821)
    system=_system().eval()
    batch=_full_batch()
    source_ids,source_mask=_tokens(2,length=13,offset=81)
    descriptor_ids,descriptor_mask=_tokens(2,length=12,offset=101)
    batch["additional_view_source_input_ids"]=source_ids.reshape(1,2,-1)
    batch["additional_view_source_attention_mask"]=source_mask.reshape(1,2,-1)
    batch["additional_view_descriptor_input_ids"]=descriptor_ids.reshape(1,2,-1)
    batch["additional_view_descriptor_attention_mask"]=descriptor_mask.reshape(1,2,-1)
    batch["additional_view_available"]=torch.tensor([[True,False]])
    batch["additional_view_reliability"]=torch.tensor([[1.0,1.0]])
    with torch.no_grad():
        out=system(task="full_envelope",batch=batch)
    assert out["source_views"].shape[1] == 8
    assert out["view_available"][0,-2:].tolist() == [True,False]
    assert out["semantic_input_metadata"]["additional_view_source"]["precomputed"] is False
    assert out["semantic_input_metadata"]["additional_view_source"]["used_virtualization"] is True
    assert out["semantic_input_metadata"]["additional_view_descriptor"]["used_virtualization"] is True
    assert torch.equal(
        out["source_views"][0,-1],
        torch.zeros_like(out["source_views"][0,-1]),
    )
    report=system.parameter_report()
    assert report["additional_runtime_view_source_text_adapter"] is True
    assert report["precomputed_additional_runtime_views_still_supported"] is True


def test_full_trainable_system_replay_and_full_envelope_share_same_backbone_parameters() -> None:
    torch.manual_seed(283)
    system=_system().train()
    ids,mask=_tokens(1,length=6,offset=70)
    replay=system(
        task="mlm",
        batch={
            "input_ids":ids,
            "attention_mask":mask,
            "labels":ids.clone(),
        },
    )
    replay["loss"].backward()
    replay_grad=system.semantic_model.backbone.embedding.weight.grad.detach().clone()
    system.zero_grad(set_to_none=True)
    out=system(task="full_envelope",batch=_full_batch())
    out["public_judgment"]["candidate_logits"].square().mean().backward()
    full_grad=system.semantic_model.backbone.embedding.weight.grad
    assert float(replay_grad.abs().sum()) > 0.0
    assert full_grad is not None
    assert float(full_grad.abs().sum()) > 0.0


def test_full_trainable_system_parameter_report_is_exact_and_ceiling_free() -> None:
    system=_system()
    report=system.parameter_report()
    assert report["total_parameters"] == sum(
        p.numel() for p in system.parameters()
    )
    assert report["single_shared_backbone"] is True
    assert report["semantic_replay_and_full_envelope_share_backbone"] is True
    assert report["full_envelope_gradient_path_registered"] is True
    assert report["all_text_surfaces_share_semantic_input"] is True
    assert report["native_window_is_product_ceiling"] is False
    assert report["runtime_relation_ceiling"] is None
    assert report["runtime_factor_ceiling"] is None
    assert report["runtime_field_ceiling"] is None
    assert report["runtime_edge_ceiling"] is None
    assert report["runtime_view_ceiling"] is None
    assert report["runtime_slot_ceiling"] is None
    assert report["runtime_reasoning_step_ceiling"] is None
    assert report["product_context_token_ceiling"] is None


def _training_targets(outputs):
    semantic=outputs["semantic_operator"]
    operator=outputs["operator"]
    relation_logits=semantic["relation_logits"]
    batch,steps,relations=relation_logits.shape
    assert relations >= 2
    relation_targets=torch.zeros(batch,steps,dtype=torch.long)
    counterfactual_relation_targets=torch.ones(batch,steps,dtype=torch.long)
    relation_step_mask=torch.ones(batch,steps,dtype=torch.bool)

    factor_targets={}
    counterfactual_factor_targets={}
    step_factor_targets={}
    for name,logits in semantic["factor_logits"].items():
        assert logits.size(-1) >= 2
        factor_targets[name]=torch.zeros(batch,dtype=torch.long)
        counterfactual_factor_targets[name]=torch.ones(batch,dtype=torch.long)
        step_factor_targets[name]=torch.zeros(batch,steps,dtype=torch.long)

    operator_targets={
        "relation_targets":relation_targets,
        "counterfactual_relation_targets":counterfactual_relation_targets,
        "relation_step_mask":relation_step_mask,
        "factor_targets":factor_targets,
        "counterfactual_factor_targets":counterfactual_factor_targets,
        "step_factor_targets":step_factor_targets,
        "step_factor_mask":torch.ones(batch,steps,dtype=torch.bool),
        "event_targets":torch.zeros(batch,steps,dtype=torch.long),
        "event_mask":torch.ones(batch,steps,dtype=torch.bool),
        "applicability_target":torch.ones(batch),
        "query_evidence_target":torch.zeros_like(
            semantic["relation_query_evidence"]
        ),
        "query_evidence_valid_mask":torch.ones_like(
            semantic["relation_query_evidence"],
            dtype=torch.bool,
        ),
        "relation_schema_evidence_target":torch.zeros_like(
            semantic["relation_schema_evidence"]
        ),
        "relation_schema_evidence_valid_mask":torch.ones_like(
            semantic["relation_schema_evidence"],
            dtype=torch.bool,
        ),
        "factor_schema_evidence_target":{
            name:torch.zeros_like(value)
            for name,value in semantic["factor_schema_evidence"].items()
        },
        "factor_schema_evidence_valid_mask":{
            name:torch.ones_like(value,dtype=torch.bool)
            for name,value in semantic["factor_schema_evidence"].items()
        },
        "step_factor_schema_evidence_target":{
            name:torch.zeros_like(value)
            for name,value in semantic["step_factor_schema_evidence"].items()
        },
        "step_factor_schema_evidence_valid_mask":{
            name:torch.ones_like(value,dtype=torch.bool)
            for name,value in semantic["step_factor_schema_evidence"].items()
        },
        "uncertainty_target":torch.zeros_like(operator.uncertainty),
    }

    support_valid=outputs["binder"]["support_compatible"].detach().clone()
    support_target=torch.zeros_like(
        outputs["binder"]["support_logits"]
    )
    candidate_logits=outputs["public_judgment"]["candidate_logits"]
    fields=outputs["executor"]["source_support_weight"].size(1)
    behavioral_targets={
        "public_target_index":torch.zeros(batch,dtype=torch.long),
        "support_target":support_target,
        "support_valid_mask":support_valid,
        "source_target_index":torch.zeros(batch,dtype=torch.long),
        "target_target_index":torch.full(
            (batch,),
            1 if fields > 1 else 0,
            dtype=torch.long,
        ),
        "endpoint_active_mask":torch.ones(batch,dtype=torch.bool),
        "decisive_view_active_mask":torch.ones(batch,dtype=torch.bool),
        "irrelevant_view_active_mask":torch.ones(batch,dtype=torch.bool),
        "recoverable_view_mask":outputs["view_available"].detach().clone(),
    }
    return operator_targets,behavioral_targets


def _counterfactual_output_views(outputs):
    primary_logits=outputs["public_judgment"]["candidate_logits"]
    decisive=dict(outputs)
    decisive_judgment=dict(outputs["public_judgment"])
    decisive_judgment["candidate_logits"]=primary_logits - torch.nn.functional.one_hot(
        torch.zeros(
            primary_logits.size(0),
            dtype=torch.long,
            device=primary_logits.device,
        ),
        num_classes=primary_logits.size(1),
    ).to(primary_logits.dtype) * 0.5
    decisive["public_judgment"]=decisive_judgment

    irrelevant=dict(outputs)
    irrelevant_judgment=dict(outputs["public_judgment"])
    irrelevant_judgment["candidate_logits"]=primary_logits + 1.0e-3
    irrelevant["public_judgment"]=irrelevant_judgment

    permuted=dict(outputs)
    permuted_latent=dict(outputs["latent"])
    permuted_latent["pooled_state"]=outputs["latent"]["pooled_state"] + 1.0e-3
    permuted["latent"]=permuted_latent
    return decisive,irrelevant,permuted


def _valid_public_replay_loss(outputs) -> torch.Tensor:
    judgment=outputs["public_judgment"]
    return judgment["candidate_logits"].masked_select(
        judgment["candidate_valid_mask"]
    ).square().mean()


def _valid_relation_replay_loss(outputs, batch) -> torch.Tensor:
    logits=outputs["semantic_operator"]["relation_logits"]
    valid=batch["relation_candidate_mask"][:,None,:].expand_as(logits)
    return logits.masked_select(valid).square().mean()


def test_joint_training_objective_wires_every_macro_family_without_learned_task_weights() -> None:
    torch.manual_seed(284)
    system=_system().train()
    batch=_full_batch()
    outputs=system(task="full_envelope",batch=batch)
    operator_targets,behavioral_targets=_training_targets(outputs)
    decisive,irrelevant,permuted=_counterfactual_output_views(outputs)
    objective=FullEnvelopeJointTrainingObjectiveV1()
    result=objective(
        primary_outputs=outputs,
        decisive_ablated_outputs=decisive,
        irrelevant_removed_outputs=irrelevant,
        permuted_outputs=permuted,
        operator_targets=operator_targets,
        behavioral_targets=behavioral_targets,
        broad_semantic_replay_loss=outputs["latent"]["pooled_state"].square().mean(),
        governed_judgment_replay_loss=_valid_public_replay_loss(outputs),
        natural_relation_loss=_valid_relation_replay_loss(outputs,batch),
        update_ema=True,
    )
    assert torch.isfinite(result["loss"])
    assert set(result["families"]) == set(DEFAULT_FAMILY_WEIGHTS)
    report=objective.parameter_report()
    assert report["learned_family_weight_parameters"] == 0
    assert report["test_adaptive_weights"] is False
    assert report["component_double_counting"] is False


def test_joint_training_objective_gradient_reaches_backbone_operator_binder_fusion_and_judgment() -> None:
    torch.manual_seed(285)
    system=_system().train()
    batch=_full_batch()
    outputs=system(task="full_envelope",batch=batch)
    operator_targets,behavioral_targets=_training_targets(outputs)
    decisive,irrelevant,permuted=_counterfactual_output_views(outputs)
    objective=FullEnvelopeJointTrainingObjectiveV1()
    result=objective(
        primary_outputs=outputs,
        decisive_ablated_outputs=decisive,
        irrelevant_removed_outputs=irrelevant,
        permuted_outputs=permuted,
        operator_targets=operator_targets,
        behavioral_targets=behavioral_targets,
        broad_semantic_replay_loss=outputs["source_views"].square().mean(),
        governed_judgment_replay_loss=_valid_public_replay_loss(outputs),
        natural_relation_loss=_valid_relation_replay_loss(outputs,batch),
        update_ema=True,
    )
    result["loss"].backward()

    checks={
        "backbone":system.semantic_model.backbone.embedding.weight.grad,
        "operator":next(system.stack.semantic_operator.parameters()).grad,
        "binder_null_support":next(system.stack.binder.null_support_score.parameters()).grad,
        "fusion_router":next(system.stack.fusion.route_score.parameters()).grad,
        "judgment":next(system.stack.public_judgment_probe.score.parameters()).grad,
        "long_context_bridge":system.semantic_input.segment_bridge.context_scale.grad,
    }
    for name,grad in checks.items():
        assert grad is not None, name
        assert float(grad.abs().sum()) > 0.0, name


def test_joint_training_objective_allows_zero_active_relation_margin_without_nan() -> None:
    torch.manual_seed(286)
    system=_system().train()
    batch=_full_batch()
    outputs=system(task="full_envelope",batch=batch)
    operator_targets,behavioral_targets=_training_targets(outputs)
    operator_targets["relation_step_mask"]=torch.zeros_like(
        operator_targets["relation_step_mask"]
    )
    decisive,irrelevant,permuted=_counterfactual_output_views(outputs)
    objective=FullEnvelopeJointTrainingObjectiveV1()
    result=objective(
        primary_outputs=outputs,
        decisive_ablated_outputs=decisive,
        irrelevant_removed_outputs=irrelevant,
        permuted_outputs=permuted,
        operator_targets=operator_targets,
        behavioral_targets=behavioral_targets,
        broad_semantic_replay_loss=outputs["latent"]["pooled_state"].square().mean(),
        governed_judgment_replay_loss=_valid_public_replay_loss(outputs),
        natural_relation_loss=_valid_relation_replay_loss(outputs,batch),
        update_ema=False,
    )
    assert torch.isfinite(result["loss"])
    assert torch.isfinite(
        result["semantic_operator"]["causal_relation_margin"]
    )


def test_joint_training_objective_allows_absent_counterfactual_sentinel() -> None:
    torch.manual_seed(287)
    system=_system().train()
    batch=_full_batch()
    outputs=system(task="full_envelope",batch=batch)
    operator_targets,behavioral_targets=_training_targets(outputs)
    operator_targets["counterfactual_relation_targets"].fill_(-1)
    first_factor=next(iter(operator_targets["counterfactual_factor_targets"]))
    operator_targets["counterfactual_factor_targets"][first_factor].fill_(-1)
    behavioral_targets["decisive_view_active_mask"].zero_()
    behavioral_targets["irrelevant_view_active_mask"].zero_()
    behavioral_targets["endpoint_active_mask"].zero_()
    behavioral_targets["source_target_index"].fill_(-1)
    behavioral_targets["target_target_index"].fill_(-1)
    decisive,irrelevant,permuted=_counterfactual_output_views(outputs)
    objective=FullEnvelopeJointTrainingObjectiveV1()
    result=objective(
        primary_outputs=outputs,
        decisive_ablated_outputs=decisive,
        irrelevant_removed_outputs=irrelevant,
        permuted_outputs=permuted,
        operator_targets=operator_targets,
        behavioral_targets=behavioral_targets,
        broad_semantic_replay_loss=outputs["latent"]["pooled_state"].square().mean(),
        governed_judgment_replay_loss=_valid_public_replay_loss(outputs),
        natural_relation_loss=_valid_relation_replay_loss(outputs,batch),
        update_ema=False,
    )
    assert torch.isfinite(result["loss"])
    assert float(result["semantic_operator"]["causal_relation_margin"]) == 0.0
    assert float(result["behavioral"]["decisive_view_causality"]) == 0.0
    assert float(result["behavioral"]["irrelevant_view_invariance"]) == 0.0
    assert float(result["behavioral"]["endpoint_roles"]) == 0.0


class _TinyTokenizer:
    pad_token_id=0
    unk_token_id=1
    cls_token_id=2
    sep_token_id=3
    mask_token_id=4

    @staticmethod
    def _word_id(word: str) -> int:
        import hashlib
        value=int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8],16)
        return 5 + (value % 500)

    def __call__(
        self,
        texts,
        *,
        padding=True,
        truncation=False,
        return_tensors="pt",
    ):
        assert truncation is False
        rows=[]
        for text in texts:
            words=str(text).lower().split()
            rows.append(
                [self.cls_token_id]
                + [self._word_id(word) for word in words]
                + [self.sep_token_id]
            )
        width=max(len(row) for row in rows)
        ids=torch.full((len(rows),width),self.pad_token_id,dtype=torch.long)
        mask=torch.zeros(len(rows),width,dtype=torch.long)
        for i,row in enumerate(rows):
            ids[i,:len(row)]=torch.tensor(row,dtype=torch.long)
            mask[i,:len(row)]=1
        return {"input_ids":ids,"attention_mask":mask}


def _behavioral_rows_for_compiler():
    from build_n0_v02_full_envelope_behavioral_curriculum_v1 import (
        materialize_row,
    )
    rows=[]
    for example in (0,1,9,13,14):
        rows.append(
            materialize_row(
                split="train",
                example=example,
                seed=20260922,
                relation_count=(1,2,4,6,8)[example % 5],
                field_count=(4,6,8,12,16)[example % 5],
                answer_count=(2,3,4,6)[example % 4],
            )
        )
    return rows


def test_behavioral_support_targets_do_not_promote_same_relation_distractors() -> None:
    """Binder targets are query-relevant support, not relation-key membership.

    The irrelevant-distractor family deliberately contains two structurally
    valid r_support edges. Only the edge supporting the queried technical claim
    is a Binder-positive target; the unrelated cafeteria support edge must stay
    available as a hard negative.
    """
    from build_n0_v02_full_envelope_behavioral_curriculum_v1 import (
        materialize_row,
    )

    row=materialize_row(
        split="train",
        example=11,
        seed=20260922,
        relation_count=4,
        field_count=4,
        answer_count=4,
    )
    support_relation_edges=[
        i
        for i,item in enumerate(row["edges"])
        if item["relation_key"]=="r_support"
    ]
    assert len(support_relation_edges) >= 2
    positives=set(int(x) for x in row["support_edge_indices"])
    assert support_relation_edges[0] in positives
    assert support_relation_edges[1] not in positives
    assert bool(row["edges"][support_relation_edges[1]]["irrelevant"]) is True



def test_behavioral_candidate_answers_require_context_to_choose_between_identical_options() -> None:
    """Candidate wording alone may not identify the governed public answer.

    This pair must keep the query and candidate-answer set/order identical while
    changing only the evidence state that determines which option is correct.
    The public judgment path therefore has to use the governed context/latent
    state instead of learning that one answer surface simply sounds correct.
    """
    from build_n0_v02_full_envelope_behavioral_curriculum_v1 import (
        materialize_row,
    )

    common=dict(
        split="train",
        seed=20260922,
        relation_count=4,
        field_count=4,
        answer_count=4,
    )
    first=materialize_row(example=15,**common)
    second=materialize_row(example=16,**common)

    assert first["candidate_context_swap_pair_id"] == second[
        "candidate_context_swap_pair_id"
    ]
    assert first["candidate_context_swap_variant"] != second[
        "candidate_context_swap_variant"
    ]
    assert first["query"] == second["query"]
    assert first["candidate_answers"] == second["candidate_answers"]
    assert first["public_target_index"] != second["public_target_index"]
    assert first["relation_sequence_target"] == second["relation_sequence_target"]
    assert first["factor_target_keys"] == second["factor_target_keys"]
    assert first["fields"] != second["fields"]



def test_behavioral_batch_compiler_accepts_internal_view_descriptor_override() -> None:
    from build_n0_v02_full_envelope_behavioral_curriculum_v1 import (
        materialize_row,
    )
    row=materialize_row(
        split="train",
        example=11,
        seed=20260922,
        relation_count=4,
        field_count=4,
        answer_count=4,
    )
    descriptions=[
        "raw semantic field evidence",
        "structured contextual field representation",
        "query-conditioned evidence-specialist view",
        "graph source-endpoint summary",
        "graph target-endpoint summary",
        (
            "neutral descriptor context " * 96
            + "relational executor summary"
        ).strip(),
    ]
    row["internal_view_descriptions"]=descriptions
    compiled=compile_behavioral_batch(
        rows=[row],
        tokenizer=_TinyTokenizer(),
    )
    ids=compiled["primary_batch"]["internal_view_descriptor_input_ids"]
    assert ids.size(0)==6
    assert ids.size(1)>96
    assert int(ids[5].ne(_TinyTokenizer.pad_token_id).sum()) > int(
        ids[0].ne(_TinyTokenizer.pad_token_id).sum()
    )


def test_behavioral_batch_compiler_builds_primary_and_causal_variants_without_key_leak() -> None:
    rows=_behavioral_rows_for_compiler()
    compiled=compile_behavioral_batch(
        rows=rows,
        tokenizer=_TinyTokenizer(),
    )
    primary=compiled["primary_batch"]
    decisive=compiled["decisive_ablated_batch"]
    irrelevant=compiled["irrelevant_removed_batch"]
    permuted=compiled["permuted_batch"]
    targets=compiled["behavioral_targets"]

    assert compiled["metadata"]["batch_size"] == len(rows)
    assert compiled["metadata"]["private_identity_data"] is False
    assert compiled["metadata"]["evidence_targets_owned_by_this_lane"] is False
    assert primary["relation_candidate_mask"].ndim == 2
    assert primary["field_valid_mask"].ndim == 2
    assert primary["edge_valid_mask"].ndim == 2
    assert targets["support_target"].shape == primary["edge_valid_mask"].shape
    assert targets["support_valid_mask"].shape == primary["edge_valid_mask"].shape

    for b,row in enumerate(rows):
        for field_index in row["decisive_field_indices"]:
            assert not bool(decisive["field_valid_mask"][b,field_index])
        for field_index in row["irrelevant_field_indices"]:
            assert not bool(irrelevant["field_valid_mask"][b,field_index])
        for edge_index in row["support_edge_indices"]:
            if row["irrelevant_field_indices"]:
                assert bool(irrelevant["edge_valid_mask"][b,edge_index])
        assert sorted(row["field_permutation"]) == list(range(len(row["fields"])))

    assert not torch.equal(
        primary["field_input_ids"],
        permuted["field_input_ids"],
    )


def test_behavioral_compiler_batches_execute_all_counterfactual_paths_and_joint_objective() -> None:
    torch.manual_seed(288)
    rows=_behavioral_rows_for_compiler()
    compiled=compile_behavioral_batch(
        rows=rows,
        tokenizer=_TinyTokenizer(),
    )
    system=_system().train()
    primary=system(
        task="full_envelope",
        batch=compiled["primary_batch"],
    )
    decisive=system(
        task="full_envelope",
        batch=compiled["decisive_ablated_batch"],
    )
    irrelevant=system(
        task="full_envelope",
        batch=compiled["irrelevant_removed_batch"],
    )
    permuted=system(
        task="full_envelope",
        batch=compiled["permuted_batch"],
    )
    objective=FullEnvelopeJointTrainingObjectiveV1()
    result=objective(
        primary_outputs=primary,
        decisive_ablated_outputs=decisive,
        irrelevant_removed_outputs=irrelevant,
        permuted_outputs=permuted,
        operator_targets=compiled["operator_targets"],
        behavioral_targets=compiled["behavioral_targets"],
        broad_semantic_replay_loss=primary["latent"]["pooled_state"].square().mean(),
        governed_judgment_replay_loss=_valid_public_replay_loss(primary),
        natural_relation_loss=_valid_relation_replay_loss(
            primary,
            compiled["primary_batch"],
        ),
        update_ema=True,
    )
    assert torch.isfinite(result["loss"])
    embedding_parameter=system.semantic_model.backbone.embedding.weight
    nonfinite_component_gradients=[]
    for family_name,components in result["families"].items():
        for component_name,component_loss in components.items():
            gradient=torch.autograd.grad(
                component_loss,
                embedding_parameter,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if gradient is not None and not bool(torch.isfinite(gradient).all()):
                nonfinite_component_gradients.append(
                    f"{family_name}/{component_name}"
                )
    assert not nonfinite_component_gradients, (
        "non-finite component gradients into shared semantic backbone: "
        + repr(nonfinite_component_gradients)
    )
    result["loss"].backward()
    finite_gradient_names=[]
    nonfinite_gradient_names=[]
    for name,parameter in system.named_parameters():
        if parameter.grad is None:
            continue
        if bool(torch.isfinite(parameter.grad).all()):
            finite_gradient_names.append(name)
        else:
            nonfinite_gradient_names.append(name)
    assert not nonfinite_gradient_names, (
        "non-finite full-envelope gradients: "
        + repr(nonfinite_gradient_names)
    )
    assert finite_gradient_names
    embedding_grad=system.semantic_model.backbone.embedding.weight.grad
    assert embedding_grad is not None
    assert float(embedding_grad.abs().sum()) > 0.0
    assert next(system.stack.binder.null_support_score.parameters()).grad is not None
    assert next(system.stack.public_judgment_probe.score.parameters()).grad is not None


def test_behavioral_compiler_support_targets_are_structurally_valid_and_unknown_rows_use_null_support() -> None:
    rows=_behavioral_rows_for_compiler()
    compiled=compile_behavioral_batch(
        rows=rows,
        tokenizer=_TinyTokenizer(),
    )
    support=compiled["behavioral_targets"]["support_target"].bool()
    valid=compiled["behavioral_targets"]["support_valid_mask"]
    assert not bool((support & ~valid).any())
    for b,row in enumerate(rows):
        if row["scenario_family"]=="unknown_defer":
            assert not bool(support[b].any())
            assert compiled["behavioral_targets"]["endpoint_active_mask"][b] == False
            assert int(compiled["behavioral_targets"]["source_target_index"][b]) == -1
            assert int(compiled["behavioral_targets"]["target_target_index"][b]) == -1


def test_registered_semantic_operator_task_uses_shared_backbone_without_fake_downstream_fields() -> None:
    torch.manual_seed(291)
    system=_system().train()
    full=_full_batch()
    semantic_batch={
        "query_input_ids":full["query_input_ids"],
        "query_attention_mask":full["query_attention_mask"],
        "relation_input_ids":full["relation_input_ids"],
        "relation_attention_mask":full["relation_attention_mask"],
        "relation_domain_type_mask":full["relation_domain_type_mask"],
        "relation_range_type_mask":full["relation_range_type_mask"],
        "relation_symmetric":full["relation_symmetric"],
        "relation_candidate_mask":full["relation_candidate_mask"],
        "factor_input_ids":full["factor_input_ids"],
        "factor_attention_mask":full["factor_attention_mask"],
        "factor_candidate_masks":full["factor_candidate_masks"],
        "factor_opcodes":full["factor_opcodes"],
        "max_reasoning_steps":full["max_reasoning_steps"],
    }
    outputs=system(task="semantic_operator",batch=semantic_batch)
    assert "semantic_operator" in outputs
    assert "operator" in outputs
    assert "public_judgment" not in outputs
    assert "binder" not in outputs
    assert outputs["semantic_operator"]["relation_logits"].shape[:2] == (
        1,
        full["max_reasoning_steps"],
    )
    loss=outputs["semantic_operator"]["relation_logits"].square().mean()
    loss.backward()
    backbone_grad=system.semantic_model.backbone.embedding.weight.grad
    operator_grad=next(system.stack.semantic_operator.parameters()).grad
    assert backbone_grad is not None and float(backbone_grad.abs().sum()) > 0.0
    assert operator_grad is not None and float(operator_grad.abs().sum()) > 0.0


def test_joint_objective_accepts_distinct_semantic_operator_and_full_fabric_batches() -> None:
    torch.manual_seed(292)
    system=_system().train()
    full_batch=_full_batch()
    primary=system(task="full_envelope",batch=full_batch)
    operator_targets,behavioral_targets=_training_targets(primary)
    semantic_batch={
        "query_input_ids":full_batch["query_input_ids"],
        "query_attention_mask":full_batch["query_attention_mask"],
        "relation_input_ids":full_batch["relation_input_ids"],
        "relation_attention_mask":full_batch["relation_attention_mask"],
        "relation_domain_type_mask":full_batch["relation_domain_type_mask"],
        "relation_range_type_mask":full_batch["relation_range_type_mask"],
        "relation_symmetric":full_batch["relation_symmetric"],
        "relation_candidate_mask":full_batch["relation_candidate_mask"],
        "factor_input_ids":full_batch["factor_input_ids"],
        "factor_attention_mask":full_batch["factor_attention_mask"],
        "factor_candidate_masks":full_batch["factor_candidate_masks"],
        "factor_opcodes":full_batch["factor_opcodes"],
        "max_reasoning_steps":full_batch["max_reasoning_steps"],
    }
    semantic=system(task="semantic_operator",batch=semantic_batch)
    decisive,irrelevant,permuted=_counterfactual_output_views(primary)
    objective=FullEnvelopeJointTrainingObjectiveV1()
    result=objective(
        primary_outputs=primary,
        decisive_ablated_outputs=decisive,
        irrelevant_removed_outputs=irrelevant,
        permuted_outputs=permuted,
        operator_targets=operator_targets,
        behavioral_targets=behavioral_targets,
        broad_semantic_replay_loss=primary["source_views"].square().mean(),
        governed_judgment_replay_loss=_valid_public_replay_loss(primary),
        natural_relation_loss=_valid_relation_replay_loss(primary,full_batch),
        semantic_operator_outputs=semantic,
        update_ema=False,
    )
    assert torch.isfinite(result["loss"])
    assert result["semantic_operator_source"]=="dedicated_semantic_operator_lane"
    result["loss"].backward()
    assert system.semantic_model.backbone.embedding.weight.grad is not None
    assert next(system.stack.semantic_operator.parameters()).grad is not None
    assert next(system.stack.public_judgment_probe.score.parameters()).grad is not None


def test_natural_relation_lane_uses_shared_operator_without_fake_factor_or_downstream_labels() -> None:
    torch.manual_seed(293)
    rows=[
        {
            "id":"fewrel:train:a",
            "split":"train",
            "instruction":"Select the relation description that matches the ordered head and tail entities.",
            "sentence":"Ada wrote the report for Meridian.",
            "head":{"text":"Ada"},
            "tail":{"text":"the report"},
            "candidate_relation_keys":["r_author","r_location"],
            "target_relation_key":"r_author",
            "target_candidate_index":0,
            "training_authorized":True,
            "model_selection_authorized":False,
            "final_validation_only":False,
            "private_identity_data":False,
        },
        {
            "id":"fewrel:train:b",
            "split":"train",
            "instruction":"Select the relation description that matches the ordered head and tail entities.",
            "sentence":"The sensor is installed in Lab Seven.",
            "head":{"text":"the sensor"},
            "tail":{"text":"Lab Seven"},
            "candidate_relation_keys":["r_location","r_author"],
            "target_relation_key":"r_location",
            "target_candidate_index":0,
            "training_authorized":True,
            "model_selection_authorized":False,
            "final_validation_only":False,
            "private_identity_data":False,
        },
    ]
    bank={
        "schema":"alice.eipm.n0.fewrel-runtime-relation-bank.v1",
        "relations":{
            "r_author":{
                "semantic_text":"Relation meaning: the head entity created or authored the tail work."
            },
            "r_location":{
                "semantic_text":"Relation meaning: the head entity is located in the tail place."
            },
        },
        "relation_keys_are_metadata_only":True,
        "private_identity_data":False,
    }
    compiled=compile_natural_relation_batch(
        rows=rows,
        relation_bank=bank,
        tokenizer=_TinyTokenizer(),
    )
    assert compiled["metadata"]["factor_labels_fabricated"] is False
    assert compiled["metadata"]["downstream_fabric_labels_fabricated"] is False
    assert compiled["metadata"]["relation_keys_model_visible"] is False

    system=_system().train()
    outputs=system(task="natural_relation",batch=compiled["batch"])
    assert "semantic_operator" in outputs
    assert "public_judgment" not in outputs
    assert "binder" not in outputs
    assert outputs["semantic_operator"]["factor_logits"] == {}
    loss=natural_relation_semantic_loss(
        outputs=outputs,
        target_relation_index=compiled["target_relation_index"],
    )
    assert torch.isfinite(loss)
    loss.backward()
    backbone_grad=system.semantic_model.backbone.embedding.weight.grad
    operator_grad=next(system.stack.semantic_operator.parameters()).grad
    assert backbone_grad is not None and float(backbone_grad.abs().sum()) > 0.0
    assert operator_grad is not None and float(operator_grad.abs().sum()) > 0.0
