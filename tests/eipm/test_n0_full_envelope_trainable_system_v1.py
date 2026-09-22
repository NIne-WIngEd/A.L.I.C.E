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


class TinyBackbone(nn.Module):
    def __init__(self, *, vocab: int = 128, width: int = 24, hidden_states: int = 3) -> None:
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


def test_joint_training_objective_wires_every_macro_family_without_learned_task_weights() -> None:
    torch.manual_seed(284)
    system=_system().train()
    outputs=system(task="full_envelope",batch=_full_batch())
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
        governed_judgment_replay_loss=outputs["public_judgment"]["candidate_logits"].square().mean(),
        natural_relation_loss=outputs["semantic_operator"]["relation_logits"].square().mean(),
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
    outputs=system(task="full_envelope",batch=_full_batch())
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
        governed_judgment_replay_loss=outputs["public_judgment"]["candidate_logits"].square().mean(),
        natural_relation_loss=outputs["semantic_operator"]["relation_logits"].square().mean(),
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
    outputs=system(task="full_envelope",batch=_full_batch())
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
        governed_judgment_replay_loss=outputs["public_judgment"]["candidate_logits"].square().mean(),
        natural_relation_loss=outputs["semantic_operator"]["relation_logits"].square().mean(),
        update_ema=False,
    )
    assert torch.isfinite(result["loss"])
    assert torch.isfinite(
        result["semantic_operator"]["causal_relation_margin"]
    )


def test_joint_training_objective_allows_absent_counterfactual_sentinel() -> None:
    torch.manual_seed(287)
    system=_system().train()
    outputs=system(task="full_envelope",batch=_full_batch())
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
        governed_judgment_replay_loss=outputs["public_judgment"]["candidate_logits"].square().mean(),
        natural_relation_loss=outputs["semantic_operator"]["relation_logits"].square().mean(),
        update_ema=False,
    )
    assert torch.isfinite(result["loss"])
    assert float(result["semantic_operator"]["causal_relation_margin"]) == 0.0
    assert float(result["behavioral"]["decisive_view_causality"]) == 0.0
    assert float(result["behavioral"]["irrelevant_view_invariance"]) == 0.0
    assert float(result["behavioral"]["endpoint_roles"]) == 0.0
