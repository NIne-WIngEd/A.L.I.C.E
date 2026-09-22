#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import unicodedata
from pathlib import Path
from typing import Any

import torch


PASS = "PASS_N0_FULL_ENVELOPE_CPU_RUNTIME_QUALIFICATION_V1"
FAIL = "FAIL_N0_FULL_ENVELOPE_CPU_RUNTIME_QUALIFICATION_V1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rss_mb() -> float:
    # Linux ru_maxrss is KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def tokenize_fixed(tokenizer: Any, texts: list[str], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"].bool()


def summarize_schema(schema: Any) -> torch.Tensor:
    states = schema.token_states.float()
    mask = schema.token_mask[:, None, :, None].to(states.dtype)
    per_layer = (states * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
    return per_layer.mean(dim=1)


def flatten_float_tensors(value: Any, prefix: str = "") -> list[tuple[str, torch.Tensor]]:
    out: list[tuple[str, torch.Tensor]] = []
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            out.append((prefix or "tensor", value))
        return out
    if isinstance(value, dict):
        for key, child in value.items():
            out.extend(flatten_float_tensors(child, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(value, (tuple, list)):
        for i, child in enumerate(value):
            out.extend(flatten_float_tensors(child, f"{prefix}[{i}]"))
    return out


def tokenizer_stress(tokenizer: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    unk = int(getattr(tokenizer, "unk_token_id", 1))
    unknown = 0
    english_ratios: list[float] = []
    english_rows = []
    for text in cfg["english_samples"]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        unknown += sum(int(x == unk) for x in ids)
        chars = len(text.replace(" ", ""))
        ratio = chars / max(len(ids), 1)
        english_ratios.append(ratio)
        english_rows.append({"tokens": len(ids), "chars": chars, "chars_per_token": ratio})

    roundtrip_rows = []
    for text in cfg["unicode_roundtrip_samples"]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        unknown += sum(int(x == unk) for x in ids)
        if len(ids) > int(cfg["max_stress_sequence_tokens"]):
            raise ValueError("unicode tokenizer stress sequence exceeded precommitted token limit")
        decoded = tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        source_nfc = unicodedata.normalize("NFC", text)
        decoded_nfc = unicodedata.normalize("NFC", decoded)
        ok = source_nfc == decoded_nfc
        roundtrip_rows.append({"tokens": len(ids), "roundtrip_nfc_equal": ok})
        if cfg["unicode_nfc_roundtrip_required"] and not ok:
            raise ValueError(f"tokenizer unicode NFC roundtrip failed for {text!r}")

    long_entity = str(cfg["long_entity_sample"])
    long_ids = tokenizer.encode(long_entity, add_special_tokens=False)
    unknown += sum(int(x == unk) for x in long_ids)
    if len(long_ids) > int(cfg["max_stress_sequence_tokens"]):
        raise ValueError("long-entity tokenizer fragmentation exceeded precommitted token limit")

    if unknown > int(cfg["unknown_token_count_max"]):
        raise ValueError(f"tokenizer produced {unknown} unknown tokens")
    mean_ratio = sum(english_ratios) / max(len(english_ratios), 1)
    if mean_ratio < float(cfg["english_mean_characters_per_token_min"]):
        raise ValueError(
            f"English semantic-description fragmentation too high: {mean_ratio:.4f}"
        )
    return {
        "unknown_tokens": unknown,
        "english_mean_characters_per_token": mean_ratio,
        "english_rows": english_rows,
        "unicode_rows": roundtrip_rows,
        "long_entity_tokens": len(long_ids),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--qualification-config", required=True)
    p.add_argument("--semantic-config", required=True)
    p.add_argument("--semantic-checkpoint", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--corpus-dir", required=True)
    p.add_argument("--source-config", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    qualification_path = Path(args.qualification_config).resolve()
    semantic_config_path = Path(args.semantic_config).resolve()
    checkpoint_path = Path(args.semantic_checkpoint).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve()
    corpus_dir = Path(args.corpus_dir).resolve()
    source_config_path = Path(args.source_config).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite {output_path}")

    cfg = json.loads(qualification_path.read_text(encoding="utf-8"))
    if cfg.get("schema") != "alice.eipm.n0.full-envelope-cpu-runtime-qualification.v1":
        raise SystemExit("qualification config schema drift")
    if torch.cuda.is_available():
        raise SystemExit("CPU runtime qualification must not expose CUDA")

    from safetensors.torch import load_file
    from alice_personality.n0.config import load_n0_config
    from alice_personality.n0.curriculum_data import load_tokenizer
    from alice_personality.n0.n0_full_envelope_stack_v1 import (
        N0FullEnvelopeStackConfig,
        N0FullEnvelopeStackV1,
    )
    from alice_personality.n0.semantic_backbone_interface_v1 import (
        FullEnvelopeSemanticBackboneInterfaceV1,
        SemanticBackboneInterfaceConfig,
    )
    from alice_personality.n0.semantic_context_virtualizer_v1 import (
        SemanticContextVirtualizerConfig,
        SemanticContextVirtualizerV1,
    )
    from alice_personality.n0.semantic_segment_context_bridge_v1 import (
        SemanticSegmentContextBridgeConfig,
        SemanticSegmentContextBridgeV1,
    )
    from alice_personality.n0.v02_model import AliceN0V02Model
    from alice_personality.n0.v02_training import (
        verify_public_corpus_v021,
        verify_tokenizer_v021,
    )

    memory = {"start_mb": rss_mb()}
    expected = cfg["semantic_initialization"]
    observed_sha = sha256(checkpoint_path)
    if observed_sha != expected["checkpoint_sha256"]:
        raise SystemExit(
            f"semantic checkpoint SHA drift expected={expected['checkpoint_sha256']} observed={observed_sha}"
        )

    tokenizer_receipt = verify_tokenizer_v021(tokenizer_dir)
    tokenizer = load_tokenizer(tokenizer_dir)
    if len(tokenizer) != int(expected["tokenizer_vocab_size"]):
        raise SystemExit("tokenizer vocabulary size drift")
    tokenizer_result = tokenizer_stress(tokenizer, cfg["tokenizer_stress"])

    corpus_receipt, corpus_paths = verify_public_corpus_v021(
        corpus_dir,
        source_config_path,
    )
    memory["after_corpus_and_tokenizer_verify_mb"] = rss_mb()

    semantic_cfg = load_n0_config(semantic_config_path)
    semantic_model = AliceN0V02Model(semantic_cfg)
    missing, unexpected = semantic_model.load_state_dict(
        load_file(str(checkpoint_path), device="cpu"),
        strict=False,
    )
    if missing or unexpected:
        raise SystemExit(
            f"semantic checkpoint mismatch missing={list(missing)} unexpected={list(unexpected)}"
        )
    semantic_report = semantic_model.parameter_report()
    if int(semantic_report["total_parameters"]) != int(expected["exact_model_parameters"]):
        raise SystemExit("semantic parameter count drift")
    semantic_model.eval()
    memory["after_semantic_model_load_mb"] = rss_mb()

    scfg = cfg["successor"]
    stack = N0FullEnvelopeStackV1(
        N0FullEnvelopeStackConfig(
            semantic_dim=int(scfg["semantic_dim"]),
            model_dim=int(scfg["model_dim"]),
            num_hidden_states=int(scfg["num_hidden_states"]),
            num_attention_heads=int(scfg["num_attention_heads"]),
            structured_layers=int(scfg["structured_layers"]),
            field_metadata_dim=int(scfg["field_metadata_dim"]),
            edge_metadata_dim=int(scfg["edge_metadata_dim"]),
            dropout=0.0,
        )
    ).eval()
    stack_report = stack.parameter_report()
    memory["after_successor_construct_mb"] = rss_mb()

    interface = FullEnvelopeSemanticBackboneInterfaceV1(
        SemanticBackboneInterfaceConfig(
            num_hidden_states=int(scfg["num_hidden_states"]),
            semantic_dim=int(scfg["semantic_dim"]),
            special_token_ids=(
                semantic_cfg.pad_token_id,
                semantic_cfg.unk_token_id,
                semantic_cfg.cls_token_id,
                semantic_cfg.sep_token_id,
                semantic_cfg.mask_token_id,
            ),
        )
    )

    long_cfg = cfg["long_context_bridge_runtime"]
    virtualizer = SemanticContextVirtualizerV1(
        SemanticContextVirtualizerConfig(
            native_window_tokens=int(long_cfg["native_window_tokens"]),
            overlap_tokens=int(long_cfg["overlap_tokens"]),
            pad_token_id=int(semantic_cfg.pad_token_id),
        )
    )
    segment_bridge = SemanticSegmentContextBridgeV1(
        SemanticSegmentContextBridgeConfig(
            semantic_dim=int(scfg["semantic_dim"]),
            num_hidden_states=int(scfg["num_hidden_states"]),
            num_attention_heads=int(long_cfg["bridge_heads"]),
            num_layers=int(long_cfg["bridge_layers"]),
            metadata_dim=3,
            query_chunk_segments=int(long_cfg["query_chunk_segments"]),
            key_chunk_segments=int(long_cfg["key_chunk_segments"]),
            dropout=0.0,
        )
    ).eval()
    bridge_report = segment_bridge.parameter_report()
    memory["after_segment_bridge_construct_mb"] = rss_mb()

    case = cfg["runtime_case"]
    max_tokens = int(case["max_text_tokens"])
    batch = int(case["batch_size"])
    backbone = semantic_model.backbone

    with torch.inference_mode():
        q_ids, q_attn = tokenize_fixed(tokenizer, list(case["queries"]), max_tokens)
        query = interface.encode_batch(
            backbone=backbone,
            input_ids=q_ids,
            attention_mask=q_attn,
        )

        relation_text = [str(x["description"]) for x in case["relations"]]
        r_ids, r_attn = tokenize_fixed(tokenizer, relation_text, max_tokens)
        type_text = [str(x) for x in case["type_descriptions"]]
        type_index = {name: i for i, name in enumerate(type_text)}
        relation_count = len(relation_text)
        type_count = len(type_text)
        domain = torch.zeros(relation_count, type_count, dtype=torch.bool)
        range_mask = torch.zeros_like(domain)
        symmetric = torch.zeros(relation_count, dtype=torch.bool)
        relation_key_index: dict[str, int] = {}
        for i, row in enumerate(case["relations"]):
            relation_key_index[str(row["key"])] = i
            for name in row["domain"]:
                domain[i, type_index[str(name)]] = True
            for name in row["range"]:
                range_mask[i, type_index[str(name)]] = True
            symmetric[i] = bool(row["symmetric"])
        relation_schema = interface.encode_relation_bank(
            backbone=backbone,
            input_ids=r_ids,
            attention_mask=r_attn,
            domain_type_mask=domain,
            range_type_mask=range_mask,
            symmetric=symmetric,
        )

        factor_schemas: dict[str, Any] = {}
        for name, texts in case["factor_banks"].items():
            ids, attn = tokenize_fixed(tokenizer, [str(x) for x in texts], max_tokens)
            factor_schemas[str(name)] = interface.encode_semantic_bank(
                backbone=backbone,
                input_ids=ids,
                attention_mask=attn,
            )

        # Exact 640-wide hierarchical long-context path. Native windows are
        # locally encoded, globally contextualized by the trainable segment
        # bridge, then stitched back to unique token order before semantic
        # operator use.
        long_text = " ".join(str(x) for x in long_cfg["text"])
        long_encoded = tokenizer(
            [long_text],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        long_ids = long_encoded["input_ids"]
        long_attention = long_encoded["attention_mask"].bool()
        segmented = virtualizer.segment(
            input_ids=long_ids,
            attention_mask=long_attention,
        )
        segment_count = int(segmented["segment_valid_mask"].sum().item())
        if segment_count < int(long_cfg["minimum_segments"]):
            raise ValueError(
                f"long-context runtime fixture produced only {segment_count} segments"
            )
        _, segment_slots, window = segmented["segment_input_ids"].shape
        segment_encoded = interface.encode_batch(
            backbone=backbone,
            input_ids=segmented["segment_input_ids"].reshape(
                segment_slots,
                window,
            ),
            attention_mask=segmented["segment_attention_mask"].reshape(
                segment_slots,
                window,
            ),
        )
        segment_hidden = segment_encoded["hidden_states"].reshape(
            1,
            segment_slots,
            int(scfg["num_hidden_states"]),
            window,
            int(scfg["semantic_dim"]),
        )
        bridged = segment_bridge(
            segment_hidden_states=segment_hidden,
            segment_attention_mask=segmented["segment_attention_mask"],
            segment_valid_mask=segmented["segment_valid_mask"],
            segment_metadata=segmented["field_metadata"],
        )
        stitched_long, stitched_long_mask = virtualizer.stitch_owned_content(
            segment_hidden_states=bridged[
                "contextualized_segment_hidden_states"
            ],
            segmented=segmented,
        )
        original_length = int(long_attention.sum().item())
        if stitched_long.size(2) != original_length:
            raise ValueError("stitched long-context token length drift")
        if not bool(stitched_long_mask[:, :original_length].all()):
            raise ValueError("stitched long-context mask lost owned tokens")

        long_content_mask = long_attention.clone()
        for token_id in (
            semantic_cfg.pad_token_id,
            semantic_cfg.unk_token_id,
            semantic_cfg.cls_token_id,
            semantic_cfg.sep_token_id,
            semantic_cfg.mask_token_id,
        ):
            long_content_mask &= long_ids.ne(int(token_id))
        if not bool(long_content_mask.any()):
            raise ValueError("long-context runtime fixture has no content tokens")
        long_operator = stack.semantic_operator(
            query_hidden_states=stitched_long,
            query_token_mask=long_content_mask,
            relation_schema=relation_schema,
            factor_schemas=factor_schemas,
            max_steps=int(long_cfg["max_reasoning_steps"]),
            relation_candidate_mask=torch.tensor(
                [case["relation_candidate_masks"][0]],
                dtype=torch.bool,
            ),
            factor_candidate_masks={
                str(name): torch.tensor(
                    [mask[0]],
                    dtype=torch.bool,
                )
                for name, mask in case["factor_candidate_masks"].items()
            },
        )
        for name, tensor in flatten_float_tensors(
            long_operator,
            "long_operator",
        ):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(
                    f"non-finite cross-window semantic tensor: {name}"
                )
        memory["after_long_context_bridge_mb"] = rss_mb()

        field_texts = [[str(x) for x in row] for row in case["field_texts"]]
        field_count = len(field_texts[0])
        if any(len(row) != field_count for row in field_texts):
            raise ValueError("field row cardinality drift")
        flat_fields = [text for row in field_texts for text in row]
        f_ids, f_attn = tokenize_fixed(tokenizer, flat_fields, max_tokens)
        f_ids = f_ids.reshape(batch, field_count, max_tokens)
        f_attn = f_attn.reshape(batch, field_count, max_tokens)
        field_valid = torch.tensor(case["field_valid_mask"], dtype=torch.bool)
        fields = interface.encode_fields(
            backbone=backbone,
            input_ids=f_ids,
            attention_mask=f_attn,
            field_valid_mask=field_valid,
        )

        descriptor_banks: dict[str, Any] = {}
        descriptor_indices: dict[str, torch.Tensor] = {}
        for name, texts in case["descriptor_banks"].items():
            ids, attn = tokenize_fixed(tokenizer, [str(x) for x in texts], max_tokens)
            descriptor_banks[str(name)] = interface.encode_semantic_bank(
                backbone=backbone,
                input_ids=ids,
                attention_mask=attn,
            )
            descriptor_indices[str(name)] = torch.tensor(
                case["descriptor_indices"][name],
                dtype=torch.long,
            )

        iv_ids, iv_attn = tokenize_fixed(
            tokenizer,
            [str(x) for x in case["internal_view_descriptions"]],
            max_tokens,
        )
        internal_descriptor_schema = interface.encode_semantic_bank(
            backbone=backbone,
            input_ids=iv_ids,
            attention_mask=iv_attn,
        )
        internal_view_descriptors = summarize_schema(
            internal_descriptor_schema
        ).unsqueeze(0).expand(batch, -1, -1).contiguous()

        av_ids, av_attn = tokenize_fixed(
            tokenizer,
            [str(x) for x in case["additional_views"]],
            max_tokens,
        )
        additional_schema = interface.encode_semantic_bank(
            backbone=backbone,
            input_ids=av_ids,
            attention_mask=av_attn,
        )
        additional_summary = summarize_schema(additional_schema)
        additional_source_views = additional_summary.unsqueeze(0).expand(
            batch, -1, -1
        ).contiguous()
        additional_view_descriptors = additional_source_views.clone()
        additional_available = torch.tensor(
            [[True, True], [True, False]],
            dtype=torch.bool,
        )
        additional_reliability = torch.tensor(
            [[0.90, 0.70], [0.85, 0.0]],
            dtype=torch.float32,
        )

        candidate_rows = [[str(x) for x in row] for row in case["candidate_texts"]]
        candidate_count = len(candidate_rows[0])
        flat_candidates = [text for row in candidate_rows for text in row]
        c_ids, c_attn = tokenize_fixed(tokenizer, flat_candidates, max_tokens)
        candidate_encoded = interface.encode_batch(
            backbone=backbone,
            input_ids=c_ids,
            attention_mask=c_attn,
        )
        candidate_hidden = candidate_encoded["hidden_states"].reshape(
            batch,
            candidate_count,
            int(scfg["num_hidden_states"]),
            max_tokens,
            int(scfg["semantic_dim"]),
        )
        candidate_token_mask = candidate_encoded["content_mask"].reshape(
            batch,
            candidate_count,
            max_tokens,
        )
        candidate_valid = torch.tensor(
            case["candidate_valid_mask"],
            dtype=torch.bool,
        )

        edges = case["edges"]
        edge_count = len(edges[0])
        edge_index = torch.zeros(batch, edge_count, 2, dtype=torch.long)
        edge_relation_index = torch.zeros(batch, edge_count, dtype=torch.long)
        edge_valid = torch.zeros(batch, edge_count, dtype=torch.bool)
        for b, rows in enumerate(edges):
            if len(rows) != edge_count:
                raise ValueError("edge cardinality drift")
            for e, row in enumerate(rows):
                edge_index[b, e, 0] = int(row["source"])
                edge_index[b, e, 1] = int(row["target"])
                edge_relation_index[b, e] = relation_key_index[str(row["relation"])]
                edge_valid[b, e] = bool(row.get("valid", True))

        edge_reliability = edge_valid.float() * 0.90
        edge_recency = edge_valid.float() * 0.80
        edge_temporal_match = edge_valid.float()
        edge_provenance_match = edge_valid.float() * 0.95
        edge_metadata = torch.stack(
            [
                edge_reliability,
                edge_recency,
                edge_temporal_match,
                edge_provenance_match,
            ],
            dim=-1,
        )

        relation_candidate_mask = torch.tensor(
            case["relation_candidate_masks"],
            dtype=torch.bool,
        )
        factor_candidate_masks = {
            str(name): torch.tensor(mask, dtype=torch.bool)
            for name, mask in case["factor_candidate_masks"].items()
        }

        memory["after_semantic_encoding_mb"] = rss_mb()

        outputs = stack(
            query_hidden_states=query["hidden_states"],
            query_token_mask=query["content_mask"],
            relation_schema=relation_schema,
            factor_schemas=factor_schemas,
            factor_opcodes={
                str(name): [str(x) for x in values]
                for name, values in case["structural_factor_opcodes"].items()
            },
            relation_candidate_mask=relation_candidate_mask,
            factor_candidate_masks=factor_candidate_masks,
            field_hidden_states=fields["field_hidden_states"],
            field_token_mask=fields["field_token_mask"],
            field_valid_mask=fields["field_valid_mask"],
            field_confidence=torch.tensor(case["field_confidence"], dtype=torch.float32),
            field_missing=torch.tensor(case["field_missing"], dtype=torch.float32),
            field_reliability=torch.tensor(case["field_reliability"], dtype=torch.float32),
            descriptor_banks=descriptor_banks,
            descriptor_indices=descriptor_indices,
            field_type_index=torch.tensor(case["field_type_index"], dtype=torch.long),
            field_metadata=torch.tensor(case["field_metadata"], dtype=torch.float32),
            edge_index=edge_index,
            edge_relation_index=edge_relation_index,
            edge_valid_mask=edge_valid,
            edge_metadata=edge_metadata,
            edge_reliability=edge_reliability,
            edge_recency=edge_recency,
            edge_temporal_match=edge_temporal_match,
            edge_provenance_match=edge_provenance_match,
            internal_view_descriptor_states=internal_view_descriptors,
            internal_view_reliability=torch.tensor(
                [[0.95,0.95,0.90,0.90,0.90,0.90],
                 [0.95,0.95,0.90,0.90,0.90,0.90]],
                dtype=torch.float32,
            ),
            max_reasoning_steps=int(case["max_reasoning_steps"]),
            graph_message_steps=int(case["graph_message_steps"]),
            fusion_refinement_steps=int(case["fusion_refinement_steps"]),
            latent_slot_count=int(case["latent_slot_count"]),
            latent_refinement_steps=int(case["latent_refinement_steps"]),
            additional_source_views=additional_source_views,
            additional_view_descriptor_states=additional_view_descriptors,
            additional_view_available=additional_available,
            additional_view_reliability=additional_reliability,
            candidate_hidden_states=candidate_hidden,
            candidate_token_mask=candidate_token_mask,
            candidate_valid_mask=candidate_valid,
        )
        memory["after_full_forward_mb"] = rss_mb()

    for name, tensor in flatten_float_tensors(outputs):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"non-finite full-envelope output tensor: {name}")

    outputs["operator"].validate(
        relation_count=len(case["relations"]),
        model_dim=int(scfg["model_dim"]),
    )
    if outputs["public_judgment"] is None:
        raise ValueError("public judgment readout missing from full runtime case")
    if outputs["public_judgment"]["candidate_logits"].shape != (
        batch,
        candidate_count,
    ):
        raise ValueError("public judgment candidate geometry drift")
    if outputs["latent"]["pooled_state"].shape != (
        batch,
        int(scfg["model_dim"]),
    ):
        raise ValueError("latent pooled-state geometry drift")
    if outputs["view_available"].size(1) != 8:
        raise ValueError("dynamic additional-view path did not reach fusion")

    ceiling_keys = [
        "runtime_relation_ceiling",
        "runtime_factor_ceiling",
        "runtime_field_ceiling",
        "runtime_edge_ceiling",
        "runtime_view_ceiling",
        "runtime_slot_ceiling",
        "runtime_reasoning_step_ceiling",
    ]
    for key in ceiling_keys:
        if stack_report.get(key) is not None:
            raise ValueError(f"serving-axis ceiling unexpectedly set: {key}")

    if any(parameter.grad is not None for parameter in semantic_model.parameters()):
        raise ValueError("semantic model gradient unexpectedly created")
    if any(parameter.grad is not None for parameter in stack.parameters()):
        raise ValueError("successor gradient unexpectedly created")
    if any(parameter.grad is not None for parameter in segment_bridge.parameters()):
        raise ValueError("segment bridge gradient unexpectedly created")

    memory["peak_rss_mb"] = rss_mb()
    result = {
        "schema": "alice.eipm.n0.full-envelope-cpu-runtime-result.v1",
        "status": PASS,
        "cpu_only": True,
        "inference_mode": True,
        "gradient": False,
        "optimizer": False,
        "model_training": False,
        "private_identity_data": False,
        "final_validation_opened": False,
        "semantic_checkpoint_sha256": observed_sha,
        "semantic_parameters": int(semantic_report["total_parameters"]),
        "successor_parameters": int(stack_report["total_parameters"]),
        "segment_context_bridge_parameters": int(
            bridge_report["total_parameters"]
        ),
        "combined_parameters": int(
            semantic_report["total_parameters"]
            + stack_report["total_parameters"]
            + bridge_report["total_parameters"]
        ),
        "memory_mb": memory,
        "tokenizer_stress": tokenizer_result,
        "corpus_source_count": len(corpus_receipt.get("sources", [])),
        "corpus_shard_count": len(corpus_paths),
        "tokenizer_receipt_model_id": tokenizer_receipt.get("model_id"),
        "long_context_bridge": {
            "native_window_tokens": int(long_cfg["native_window_tokens"]),
            "overlap_tokens": int(long_cfg["overlap_tokens"]),
            "original_tokens": original_length,
            "segments": segment_count,
            "stitched_tokens": int(stitched_long.size(2)),
            "bridge_report": bridge_report,
            "operator_relation_distribution_shape": list(
                long_operator["operator"].relation_distribution.shape
            ),
            "standalone_virtualizer_semantics_complete": virtualizer.parameter_report()[
                "standalone_cross_window_semantics_complete"
            ],
        },
        "runtime_case": {
            "batch_size": batch,
            "relations": len(case["relations"]),
            "factor_banks": len(case["factor_banks"]),
            "fields": field_count,
            "edges": edge_count,
            "views_after_additional": int(outputs["source_views"].size(1)),
            "latent_slots": int(outputs["latent"]["latent_slots"].size(1)),
            "reasoning_steps": int(case["max_reasoning_steps"]),
            "candidate_count": candidate_count,
        },
        "stack_report": stack_report,
        "output_shapes": {
            "latent_pooled_state": list(outputs["latent"]["pooled_state"].shape),
            "candidate_logits": list(outputs["public_judgment"]["candidate_logits"].shape),
            "relation_distribution": list(outputs["operator"].relation_distribution.shape),
            "edge_support_weight": list(outputs["binder"]["edge_support_weight"].shape),
            "relational_probability": list(outputs["executor"]["relational_probability"].shape),
        },
        "gpu_training_authorized": False,
        "n0_complete": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))

    del outputs, long_operator, segment_bridge, stack, semantic_model
    gc.collect()


if __name__ == "__main__":
    main()
