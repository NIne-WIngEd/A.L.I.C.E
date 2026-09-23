from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.n0_full_envelope_trainable_system_v1 import (
    N0FullEnvelopeTrainableSystemConfig,
    N0FullEnvelopeTrainableSystemV1,
)
from alice_personality.n0.v02_model import AliceN0V02Model


TOPOLOGY_SCHEMA="alice.eipm.n0.full-envelope-registered-topology.v1"


def sha256_file(path: str | Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_registered_topology(path: str | Path) -> dict[str,Any]:
    value=json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema")!=TOPOLOGY_SCHEMA:
        raise ValueError("registered full-envelope topology schema drift")
    system=dict(value.get("registered_system") or {})
    invariants=dict(value.get("topology_invariants") or {})
    if value.get("status")!="REGISTERED_FULL_PUBLIC_N0_SUCCESSOR_TOPOLOGY":
        raise ValueError("registered full-envelope topology status drift")
    if invariants.get("full_architecture") is not True:
        raise ValueError("registered topology must be the full architecture")
    if invariants.get("reduced_pilot") is not False:
        raise ValueError("reduced pilot topology is forbidden")
    for key in (
        "runtime_relation_ceiling",
        "runtime_factor_ceiling",
        "runtime_field_ceiling",
        "runtime_edge_ceiling",
        "runtime_view_ceiling",
        "runtime_slot_ceiling",
        "runtime_reasoning_step_ceiling",
        "product_context_token_ceiling",
    ):
        if invariants.get(key) is not None:
            raise ValueError(f"registered topology sets capability ceiling: {key}")
    required_positive=(
        "semantic_dim",
        "model_dim",
        "num_hidden_states",
        "num_attention_heads",
        "structured_layers",
        "field_metadata_dim",
        "edge_metadata_dim",
        "native_window_tokens",
        "overlap_tokens",
        "segment_bridge_layers",
        "segment_query_chunk",
        "segment_key_chunk",
    )
    for key in required_positive:
        if int(system.get(key,0))<=0:
            raise ValueError(f"registered topology requires positive {key}")
    if int(system["semantic_dim"])!=int(system["model_dim"]):
        raise ValueError("registered semantic/model width mismatch")
    if int(system["overlap_tokens"])>=int(system["native_window_tokens"]):
        raise ValueError("registered overlap must be below native window")
    return value


def _runtime_values(
    topology: Mapping[str,Any],
    runtime_profile: Mapping[str,Any] | None,
) -> dict[str,Any]:
    values=dict(topology["registered_system"])
    if runtime_profile is None:
        return values
    profile=dict(runtime_profile)
    if profile.get("operating_point_only") is not True:
        raise ValueError(
            "runtime profile override must declare operating_point_only=true"
        )
    if profile.get("product_capability_ceiling") is not False:
        raise ValueError(
            "runtime profile override may not become a product capability ceiling"
        )
    allowed={
        "native_window_tokens",
        "overlap_tokens",
        "segment_query_chunk",
        "segment_key_chunk",
    }
    unknown=set(profile)-allowed-{
        "operating_point_only",
        "product_capability_ceiling",
        "reason",
    }
    if unknown:
        raise ValueError(
            "runtime profile attempted to change learned topology: "
            +repr(sorted(unknown))
        )
    for key in allowed:
        if key in profile:
            values[key]=int(profile[key])
    if int(values["overlap_tokens"])>=int(values["native_window_tokens"]):
        raise ValueError("runtime-profile overlap must be below native window")
    return values


def registered_system_config(
    *,
    topology: Mapping[str,Any],
    semantic_config: Any,
    runtime_profile: Mapping[str,Any] | None = None,
) -> N0FullEnvelopeTrainableSystemConfig:
    values=_runtime_values(topology,runtime_profile)
    return N0FullEnvelopeTrainableSystemConfig(
        semantic_dim=int(values["semantic_dim"]),
        model_dim=int(values["model_dim"]),
        num_hidden_states=int(values["num_hidden_states"]),
        num_attention_heads=int(values["num_attention_heads"]),
        structured_layers=int(values["structured_layers"]),
        field_metadata_dim=int(values["field_metadata_dim"]),
        edge_metadata_dim=int(values["edge_metadata_dim"]),
        native_window_tokens=int(values["native_window_tokens"]),
        overlap_tokens=int(values["overlap_tokens"]),
        segment_bridge_layers=int(values["segment_bridge_layers"]),
        segment_query_chunk=int(values["segment_query_chunk"]),
        segment_key_chunk=int(values["segment_key_chunk"]),
        special_token_ids=(
            int(semantic_config.pad_token_id),
            int(semantic_config.unk_token_id),
            int(semantic_config.cls_token_id),
            int(semantic_config.sep_token_id),
            int(semantic_config.mask_token_id),
        ),
        pad_token_id=int(semantic_config.pad_token_id),
        dropout=float(values["dropout"]),
    )


def load_registered_full_envelope_system(
    *,
    topology_path: str | Path,
    semantic_config_path: str | Path,
    semantic_checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    runtime_profile: Mapping[str,Any] | None = None,
) -> tuple[nn.Module,dict[str,Any]]:
    """Load the one registered full public N0 topology from exact initialization.

    This function does not authorize optimizer or gradient execution. Historical
    checkpoints provide initialization and regression baselines only. Every
    successor module remains instantiated and all serving axes stay dynamic.
    """
    from safetensors.torch import load_file

    topology_path=Path(topology_path).resolve()
    semantic_config_path=Path(semantic_config_path).resolve()
    checkpoint_path=Path(semantic_checkpoint_path).resolve()
    topology=load_registered_topology(topology_path)
    expected=dict(topology["semantic_initialization"])
    if str(semantic_config_path).endswith(str(expected["config"])) is False:
        # Allow relocated repository roots while still requiring the governed
        # config basename/path suffix.
        if semantic_config_path.name!=Path(str(expected["config"])).name:
            raise ValueError("semantic config path does not match topology contract")
    observed_sha=sha256_file(checkpoint_path)
    if observed_sha!=str(expected["checkpoint_sha256"]):
        raise ValueError(
            "semantic initialization checkpoint hash drift "
            f"expected={expected['checkpoint_sha256']} observed={observed_sha}"
        )

    semantic_cfg=load_n0_config(semantic_config_path)
    semantic_model=AliceN0V02Model(semantic_cfg)
    state=load_file(str(checkpoint_path),device="cpu")
    missing,unexpected=semantic_model.load_state_dict(state,strict=False)
    if missing or unexpected:
        raise ValueError(
            "semantic initialization state mismatch "
            f"missing={list(missing)} unexpected={list(unexpected)}"
        )
    semantic_report=semantic_model.parameter_report()
    if int(semantic_report["total_parameters"])!=int(
        expected["exact_model_parameters"]
    ):
        raise ValueError("semantic initialization parameter count drift")

    config=registered_system_config(
        topology=topology,
        semantic_config=semantic_cfg,
        runtime_profile=runtime_profile,
    )
    system=N0FullEnvelopeTrainableSystemV1(
        semantic_model=semantic_model,
        config=config,
    )
    system=system.to(torch.device(device))
    report=system.parameter_report()
    if report.get("single_shared_backbone") is not True:
        raise ValueError("registered system lost the one-shared-backbone invariant")
    for key in (
        "runtime_relation_ceiling",
        "runtime_factor_ceiling",
        "runtime_field_ceiling",
        "runtime_edge_ceiling",
        "runtime_view_ceiling",
        "runtime_slot_ceiling",
        "runtime_reasoning_step_ceiling",
        "product_context_token_ceiling",
    ):
        if report.get(key) is not None:
            raise ValueError(f"registered system reports capability ceiling: {key}")

    receipt={
        "schema":"alice.eipm.n0.registered-topology-load-receipt.v1",
        "topology_sha256":sha256_file(topology_path),
        "semantic_config_sha256":sha256_file(semantic_config_path),
        "semantic_checkpoint_sha256":observed_sha,
        "semantic_parameters":int(semantic_report["total_parameters"]),
        "combined_parameters":int(report["total_parameters"]),
        "registered_system":"N0FullEnvelopeTrainableSystemV1",
        "production_runtime_profile":runtime_profile is None,
        "runtime_profile_override":None if runtime_profile is None else dict(runtime_profile),
        "single_shared_backbone":True,
        "full_architecture":True,
        "reduced_pilot":False,
        "private_identity_data":False,
        "gradient_performed":False,
        "optimizer_performed":False,
        "gpu_training_authorized":False,
        "final_opening_authorized":False,
        "n0_complete":False,
    }
    return system,receipt
