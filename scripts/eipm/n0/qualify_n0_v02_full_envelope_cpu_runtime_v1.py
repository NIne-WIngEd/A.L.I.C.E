#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import subprocess
import time
import unicodedata
from pathlib import Path
from typing import Any

import torch

from alice_personality.n0.source_authority_v1 import require_canonical_source_file


PASS = "PASS_N0_FULL_ENVELOPE_CPU_RUNTIME_QUALIFICATION_V1"
_START_TIME = time.monotonic()


def progress(label: str) -> None:
    elapsed = time.monotonic() - _START_TIME
    print(f"QUAL_PHASE elapsed_s={elapsed:.1f} {label}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def flatten_float_tensors(value: Any, prefix: str = "") -> list[tuple[str, torch.Tensor]]:
    out: list[tuple[str, torch.Tensor]] = []
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            out.append((prefix or "tensor", value))
        return out
    if isinstance(value, dict):
        for key, child in value.items():
            out.extend(
                flatten_float_tensors(
                    child,
                    f"{prefix}.{key}" if prefix else str(key),
                )
            )
        return out
    if isinstance(value, (tuple, list)):
        for i, child in enumerate(value):
            out.extend(flatten_float_tensors(child, f"{prefix}[{i}]"))
    return out


def tokenize_texts(
    tokenizer: Any,
    texts: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"].bool()


def tokenize_nested(
    tokenizer: Any,
    rows: list[list[str]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not rows or not rows[0]:
        raise ValueError("nested text rows must be nonempty")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("nested text row cardinality drift")
    flat = [text for row in rows for text in row]
    ids, mask = tokenize_texts(tokenizer, flat)
    return (
        ids.reshape(len(rows), width, ids.size(1)),
        mask.reshape(len(rows), width, mask.size(1)),
    )


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
        english_rows.append(
            {
                "tokens": len(ids),
                "chars": chars,
                "chars_per_token": ratio,
            }
        )

    roundtrip_rows = []
    for text in cfg["unicode_roundtrip_samples"]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        unknown += sum(int(x == unk) for x in ids)
        if len(ids) > int(cfg["max_stress_sequence_tokens"]):
            raise ValueError(
                "unicode tokenizer stress sequence exceeded precommitted token limit"
            )
        decoded = tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        source_nfc = unicodedata.normalize("NFC", text)
        decoded_nfc = unicodedata.normalize("NFC", decoded)
        ok = source_nfc == decoded_nfc
        roundtrip_rows.append(
            {
                "tokens": len(ids),
                "roundtrip_nfc_equal": ok,
            }
        )
        if cfg["unicode_nfc_roundtrip_required"] and not ok:
            raise ValueError(
                f"tokenizer unicode NFC roundtrip failed for {text!r}"
            )

    long_entity = str(cfg["long_entity_sample"])
    long_ids = tokenizer.encode(long_entity, add_special_tokens=False)
    unknown += sum(int(x == unk) for x in long_ids)
    if len(long_ids) > int(cfg["max_stress_sequence_tokens"]):
        raise ValueError(
            "long-entity tokenizer fragmentation exceeded precommitted token limit"
        )

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
    p.add_argument("--topology-config", required=True)
    p.add_argument("--source-revision", required=True)
    p.add_argument("--semantic-config", required=True)
    p.add_argument("--semantic-checkpoint", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--corpus-dir", required=True)
    p.add_argument("--source-config", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    qualification_path = Path(args.qualification_config).resolve()
    topology_path = Path(args.topology_config).resolve()
    semantic_config_path = Path(args.semantic_config).resolve()
    checkpoint_path = Path(args.semantic_checkpoint).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve()
    corpus_dir = Path(args.corpus_dir).resolve()
    source_config_path = Path(args.source_config).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite {output_path}")
    progress("start")

    status=subprocess.check_output(["git","status","--porcelain"],text=True)
    if status.strip():
        raise SystemExit("CPU runtime qualification requires a clean exact-source worktree")
    require_canonical_source_file(
        qualification_path,
        "configs/eipm/n0/n0_v02_full_envelope_cpu_runtime_qualification_v1.json",
        label="CPU qualification config",
    )
    require_canonical_source_file(
        topology_path,
        "configs/eipm/n0/n0_v02_full_envelope_registered_topology_v1.json",
        label="registered topology config",
    )
    require_canonical_source_file(
        semantic_config_path,
        "configs/eipm/n0/alice_n0_semantic_v0.2.json",
        label="semantic config",
    )
    require_canonical_source_file(
        source_config_path,
        "configs/eipm/n0/public_corpus_v0.2.1.activated.json",
        label="public source config",
    )

    source_revision=str(args.source_revision).strip().lower()
    if len(source_revision)!=40 or any(
        ch not in "0123456789abcdef" for ch in source_revision
    ):
        raise SystemExit("source revision must be exact 40-hex git commit")
    current_revision=subprocess.check_output(
        ["git","rev-parse","HEAD"],text=True
    ).strip().lower()
    if current_revision!=source_revision:
        raise SystemExit("CPU runtime source revision drift")

    cfg = json.loads(qualification_path.read_text(encoding="utf-8"))
    topology=json.loads(topology_path.read_text(encoding="utf-8"))
    if topology.get("schema")!="alice.eipm.n0.full-envelope-registered-topology.v1":
        raise SystemExit("registered topology schema drift")
    if (
        cfg.get("schema")
        != "alice.eipm.n0.full-envelope-cpu-runtime-qualification.v1"
    ):
        raise SystemExit("qualification config schema drift")
    if torch.cuda.is_available():
        raise SystemExit("CPU runtime qualification must not expose CUDA")
    threads=max(1,int(os.environ.get("SLURM_CPUS_PER_TASK","1")))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    progress(f"cpu_threads={threads}")

    from safetensors.torch import load_file

    from alice_personality.n0.config import load_n0_config
    from alice_personality.n0.curriculum_data import load_tokenizer
    from alice_personality.n0.n0_full_envelope_trainable_system_v1 import (
        N0FullEnvelopeTrainableSystemConfig,
        N0FullEnvelopeTrainableSystemV1,
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
            "semantic checkpoint SHA drift "
            f"expected={expected['checkpoint_sha256']} observed={observed_sha}"
        )

    tokenizer_receipt = verify_tokenizer_v021(tokenizer_dir)
    tokenizer = load_tokenizer(tokenizer_dir)
    if len(tokenizer) != int(expected["tokenizer_vocab_size"]):
        raise SystemExit("tokenizer vocabulary size drift")
    tokenizer_result = tokenizer_stress(
        tokenizer,
        cfg["tokenizer_stress"],
    )

    corpus_receipt, corpus_paths = verify_public_corpus_v021(
        corpus_dir,
        source_config_path,
    )
    memory["after_corpus_and_tokenizer_verify_mb"] = rss_mb()
    progress("tokenizer_and_corpus_verified")

    semantic_cfg = load_n0_config(semantic_config_path)
    semantic_model = AliceN0V02Model(semantic_cfg)
    missing, unexpected = semantic_model.load_state_dict(
        load_file(str(checkpoint_path), device="cpu"),
        strict=False,
    )
    if missing or unexpected:
        raise SystemExit(
            "semantic checkpoint mismatch "
            f"missing={list(missing)} unexpected={list(unexpected)}"
        )
    semantic_report = semantic_model.parameter_report()
    if int(semantic_report["total_parameters"]) != int(
        expected["exact_model_parameters"]
    ):
        raise SystemExit("semantic parameter count drift")

    scfg = cfg["successor"]
    registered=dict(topology["registered_system"])
    learned_pairs={
        "semantic_dim":"semantic_dim",
        "model_dim":"model_dim",
        "num_hidden_states":"num_hidden_states",
        "num_attention_heads":"num_attention_heads",
        "structured_layers":"structured_layers",
        "field_metadata_dim":"field_metadata_dim",
        "edge_metadata_dim":"edge_metadata_dim",
    }
    for qualification_key,registered_key in learned_pairs.items():
        if int(scfg[qualification_key])!=int(registered[registered_key]):
            raise SystemExit(
                f"CPU qualification learned-topology drift: {qualification_key}"
            )
    if int(cfg["long_context_bridge_runtime"]["bridge_layers"])!=int(
        registered["segment_bridge_layers"]
    ):
        raise SystemExit("CPU qualification segment-bridge topology drift")
    if topology["semantic_initialization"].get("checkpoint_sha256")!=expected[
        "checkpoint_sha256"
    ]:
        raise SystemExit("CPU qualification semantic initialization contract drift")
    long_cfg = cfg["long_context_bridge_runtime"]
    system = N0FullEnvelopeTrainableSystemV1(
        semantic_model=semantic_model,
        config=N0FullEnvelopeTrainableSystemConfig(
            semantic_dim=int(scfg["semantic_dim"]),
            model_dim=int(scfg["model_dim"]),
            num_hidden_states=int(scfg["num_hidden_states"]),
            num_attention_heads=int(scfg["num_attention_heads"]),
            structured_layers=int(scfg["structured_layers"]),
            field_metadata_dim=int(scfg["field_metadata_dim"]),
            edge_metadata_dim=int(scfg["edge_metadata_dim"]),
            native_window_tokens=int(long_cfg["native_window_tokens"]),
            overlap_tokens=int(long_cfg["overlap_tokens"]),
            segment_bridge_layers=int(long_cfg["bridge_layers"]),
            segment_query_chunk=int(long_cfg["query_chunk_segments"]),
            segment_key_chunk=int(long_cfg["key_chunk_segments"]),
            special_token_ids=(
                int(semantic_cfg.pad_token_id),
                int(semantic_cfg.unk_token_id),
                int(semantic_cfg.cls_token_id),
                int(semantic_cfg.sep_token_id),
                int(semantic_cfg.mask_token_id),
            ),
            pad_token_id=int(semantic_cfg.pad_token_id),
            dropout=0.0,
        ),
    ).eval()
    system_report = system.parameter_report()
    stack_report = system.stack.parameter_report()
    semantic_input_report = system.semantic_input.parameter_report()
    memory["after_registered_system_construct_mb"] = rss_mb()
    progress("registered_system_constructed")

    case = cfg["runtime_case"]
    batch_size = int(case["batch_size"])
    stress = cfg["text_surface_virtualization_stress"]
    stressed_surfaces = set(
        str(x) for x in stress["minimum_virtualized_surfaces"]
    )
    suffix = " " + " ".join(
        [str(stress["suffix_text"])] * int(stress["suffix_repeat_count"])
    )
    native_window = int(stress["native_window_tokens"])
    if native_window != int(long_cfg["native_window_tokens"]):
        raise ValueError("text-surface stress/native-window contract drift")

    def longify(surface: str, text: str) -> str:
        return text + suffix if surface in stressed_surfaces else text

    queries = [str(x) for x in case["queries"]]
    dedicated_long_context_text = "\n\n".join(
        str(x) for x in long_cfg["text"]
    )
    if not dedicated_long_context_text.strip():
        raise ValueError("dedicated long-context fixture is empty")
    queries[0] = longify("query", dedicated_long_context_text)
    q_ids, q_mask = tokenize_texts(tokenizer, queries)

    relation_rows = [dict(x) for x in case["relations"]]
    relation_texts = [str(x["description"]) for x in relation_rows]
    relation_texts[0] = longify("relation_schema", relation_texts[0])
    r_ids, r_mask = tokenize_texts(tokenizer, relation_texts)

    type_texts = [str(x) for x in case["type_descriptions"]]
    type_index = {name: i for i, name in enumerate(type_texts)}
    relation_count = len(relation_texts)
    type_count = len(type_texts)
    relation_domain = torch.zeros(
        relation_count,
        type_count,
        dtype=torch.bool,
    )
    relation_range = torch.zeros_like(relation_domain)
    relation_symmetric = torch.zeros(
        relation_count,
        dtype=torch.bool,
    )
    relation_key_index: dict[str, int] = {}
    for i, row in enumerate(relation_rows):
        relation_key_index[str(row["key"])] = i
        for name in row["domain"]:
            relation_domain[i, type_index[str(name)]] = True
        for name in row["range"]:
            relation_range[i, type_index[str(name)]] = True
        relation_symmetric[i] = bool(row["symmetric"])

    factor_input_ids: dict[str, torch.Tensor] = {}
    factor_attention_mask: dict[str, torch.Tensor] = {}
    for bank_index, (name, texts_raw) in enumerate(
        case["factor_banks"].items()
    ):
        texts = [str(x) for x in texts_raw]
        if bank_index == 0:
            texts[0] = longify("factor_schema", texts[0])
        ids, mask = tokenize_texts(tokenizer, texts)
        factor_input_ids[str(name)] = ids
        factor_attention_mask[str(name)] = mask

    descriptor_input_ids: dict[str, torch.Tensor] = {}
    descriptor_attention_mask: dict[str, torch.Tensor] = {}
    for bank_index, (name, texts_raw) in enumerate(
        case["descriptor_banks"].items()
    ):
        texts = [str(x) for x in texts_raw]
        if bank_index == 0:
            texts[0] = longify("descriptor_text", texts[0])
        ids, mask = tokenize_texts(tokenizer, texts)
        descriptor_input_ids[str(name)] = ids
        descriptor_attention_mask[str(name)] = mask

    field_texts = [
        [str(x) for x in row]
        for row in case["field_texts"]
    ]
    field_count = len(field_texts[0])
    if any(len(row) != field_count for row in field_texts):
        raise ValueError("field row cardinality drift")
    field_texts[0][0] = longify("field_text", field_texts[0][0])
    field_ids, field_attention = tokenize_nested(
        tokenizer,
        field_texts,
    )
    field_valid = torch.tensor(
        case["field_valid_mask"],
        dtype=torch.bool,
    )

    candidate_texts = [
        [str(x) for x in row]
        for row in case["candidate_texts"]
    ]
    candidate_count = len(candidate_texts[0])
    if any(len(row) != candidate_count for row in candidate_texts):
        raise ValueError("candidate row cardinality drift")
    candidate_texts[0][0] = longify(
        "candidate_text",
        candidate_texts[0][0],
    )
    candidate_ids, candidate_attention = tokenize_nested(
        tokenizer,
        candidate_texts,
    )
    candidate_valid = torch.tensor(
        case["candidate_valid_mask"],
        dtype=torch.bool,
    )

    internal_texts = [
        str(x) for x in case["internal_view_descriptions"]
    ]
    internal_texts[0] = longify(
        "internal_view_descriptor",
        internal_texts[0],
    )
    internal_ids, internal_attention = tokenize_texts(
        tokenizer,
        internal_texts,
    )

    additional_descriptor_texts = [
        str(x) for x in case["additional_views"]
    ]
    additional_descriptor_texts[0] = longify(
        "additional_view_descriptor",
        additional_descriptor_texts[0],
    )
    additional_ids_single, additional_attention_single = tokenize_texts(
        tokenizer,
        additional_descriptor_texts,
    )
    additional_source_texts = [
        str(x) for x in case["additional_view_sources"]
    ]
    additional_source_texts[0] = longify(
        "additional_view_source",
        additional_source_texts[0],
    )
    additional_source_ids_single, additional_source_attention_single = tokenize_texts(
        tokenizer,
        additional_source_texts,
    )
    additional_count = len(additional_descriptor_texts)
    if len(additional_source_texts) != additional_count:
        raise ValueError("additional source/descriptor count drift")
    additional_ids = additional_ids_single[None, :, :].expand(
        batch_size,
        -1,
        -1,
    ).contiguous()
    additional_attention = additional_attention_single[
        None,
        :,
        :,
    ].expand(batch_size, -1, -1).contiguous()
    additional_source_ids = additional_source_ids_single[
        None,
        :,
        :,
    ].expand(batch_size, -1, -1).contiguous()
    additional_source_attention = additional_source_attention_single[
        None,
        :,
        :,
    ].expand(batch_size, -1, -1).contiguous()
    additional_available = torch.ones(
        batch_size,
        additional_count,
        dtype=torch.bool,
    )
    if batch_size > 1 and additional_count > 1:
        additional_available[1, -1] = False
    additional_reliability = (
        additional_available.float()
        * torch.linspace(
            0.90,
            0.70,
            additional_count,
        )[None, :]
    )

    edges = case["edges"]
    edge_count = len(edges[0])
    edge_index = torch.zeros(
        batch_size,
        edge_count,
        2,
        dtype=torch.long,
    )
    edge_relation_index = torch.zeros(
        batch_size,
        edge_count,
        dtype=torch.long,
    )
    edge_valid = torch.zeros(
        batch_size,
        edge_count,
        dtype=torch.bool,
    )
    for b, rows in enumerate(edges):
        if len(rows) != edge_count:
            raise ValueError("edge cardinality drift")
        for e, row in enumerate(rows):
            edge_index[b, e, 0] = int(row["source"])
            edge_index[b, e, 1] = int(row["target"])
            edge_relation_index[b, e] = relation_key_index[
                str(row["relation"])
            ]
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
    descriptor_indices = {
        str(name): torch.tensor(values, dtype=torch.long)
        for name, values in case["descriptor_indices"].items()
    }
    progress("runtime_fixture_tokenized")

    with torch.inference_mode():
        runtime_batch = {
            "query_input_ids": q_ids,
            "query_attention_mask": q_mask,
            "relation_input_ids": r_ids,
            "relation_attention_mask": r_mask,
            "relation_domain_type_mask": relation_domain,
            "relation_range_type_mask": relation_range,
            "relation_symmetric": relation_symmetric,
            "relation_candidate_mask": relation_candidate_mask,
            "factor_input_ids": factor_input_ids,
            "factor_attention_mask": factor_attention_mask,
            "factor_candidate_masks": factor_candidate_masks,
            "factor_opcodes": {
                str(name): [str(x) for x in values]
                for name, values in case[
                    "structural_factor_opcodes"
                ].items()
            },
            "field_input_ids": field_ids,
            "field_attention_mask": field_attention,
            "field_valid_mask": field_valid,
            "field_confidence": torch.tensor(
                case["field_confidence"],
                dtype=torch.float32,
            ),
            "field_missing": torch.tensor(
                case["field_missing"],
                dtype=torch.float32,
            ),
            "field_reliability": torch.tensor(
                case["field_reliability"],
                dtype=torch.float32,
            ),
            "descriptor_input_ids": descriptor_input_ids,
            "descriptor_attention_mask": descriptor_attention_mask,
            "descriptor_indices": descriptor_indices,
            "field_type_index": torch.tensor(
                case["field_type_index"],
                dtype=torch.long,
            ),
            "field_metadata": torch.tensor(
                case["field_metadata"],
                dtype=torch.float32,
            ),
            "edge_index": edge_index,
            "edge_relation_index": edge_relation_index,
            "edge_valid_mask": edge_valid,
            "edge_metadata": edge_metadata,
            "edge_reliability": edge_reliability,
            "edge_recency": edge_recency,
            "edge_temporal_match": edge_temporal_match,
            "edge_provenance_match": edge_provenance_match,
            "internal_view_descriptor_input_ids": internal_ids,
            "internal_view_descriptor_attention_mask": internal_attention,
            "internal_view_reliability": torch.tensor(
                [
                    [0.95,0.95,0.90,0.90,0.90,0.90]
                    for _ in range(batch_size)
                ],
                dtype=torch.float32,
            ),
            "candidate_input_ids": candidate_ids,
            "candidate_attention_mask": candidate_attention,
            "candidate_valid_mask": candidate_valid,
            "additional_view_source_input_ids": additional_source_ids,
            "additional_view_source_attention_mask": additional_source_attention,
            "additional_view_descriptor_input_ids": additional_ids,
            "additional_view_descriptor_attention_mask": additional_attention,
            "additional_view_available": additional_available,
            "additional_view_reliability": additional_reliability,
            "max_reasoning_steps": int(case["max_reasoning_steps"]),
            "graph_message_steps": int(case["graph_message_steps"]),
            "fusion_refinement_steps": int(
                case["fusion_refinement_steps"]
            ),
            "latent_slot_count": int(case["latent_slot_count"]),
            "latent_refinement_steps": int(
                case["latent_refinement_steps"]
            ),
        }
        memory["before_full_registered_forward_mb"] = rss_mb()
        progress("full_registered_forward_begin")
        outputs = system(
            task="full_envelope",
            batch=runtime_batch,
        )
        memory["after_full_registered_forward_mb"] = rss_mb()
        progress("full_registered_forward_end")

    for name, tensor in flatten_float_tensors(outputs):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(
                f"non-finite full-envelope output tensor: {name}"
            )

    outputs["operator"].validate(
        relation_count=relation_count,
        model_dim=int(scfg["model_dim"]),
    )
    if outputs["public_judgment"] is None:
        raise ValueError("public judgment readout missing")
    if outputs["public_judgment"]["candidate_logits"].shape != (
        batch_size,
        candidate_count,
    ):
        raise ValueError("public judgment candidate geometry drift")
    if outputs["latent"]["pooled_state"].shape != (
        batch_size,
        int(scfg["model_dim"]),
    ):
        raise ValueError("latent pooled-state geometry drift")
    if outputs["source_views"].size(1) != 6 + additional_count:
        raise ValueError(
            "dynamic additional-view path did not reach fusion"
        )

    meta = outputs["semantic_input_metadata"]
    surface_receipt = {
        "query": bool(meta["query"]["used_virtualization"]),
        "relation_schema": bool(
            meta["relation_schema"]["used_virtualization"]
        ),
        "factor_schema": any(
            bool(row["used_virtualization"])
            for row in meta["factor_schema"].values()
        ),
        "field_text": bool(
            meta["field_text"]["used_virtualization"]
        ),
        "candidate_text": bool(
            meta["candidate_text"]["used_virtualization"]
        ),
        "descriptor_text": any(
            bool(row["used_virtualization"])
            for row in meta["descriptor_text"].values()
        ),
        "internal_view_descriptor": bool(
            meta["internal_view_descriptor"]["used_virtualization"]
        ),
        "additional_view_descriptor": bool(
            meta["additional_view_descriptor"]["used_virtualization"]
        ),
        "additional_view_source": bool(
            meta["additional_view_source"]["used_virtualization"]
        ),
    }
    required_surfaces = set(
        str(x) for x in stress["minimum_virtualized_surfaces"]
    )
    if set(surface_receipt) != required_surfaces:
        raise ValueError(
            "virtualization surface receipt/contract drift "
            f"{sorted(surface_receipt)} != {sorted(required_surfaces)}"
        )
    failed_surfaces = [
        name for name, value in surface_receipt.items() if not value
    ]
    if failed_surfaces:
        raise ValueError(
            "registered full-envelope system did not virtualize required "
            f"surfaces: {failed_surfaces}"
        )

    if int(meta["query"]["segment_count_max"]) < int(
        long_cfg["minimum_segments"]
    ):
        raise ValueError(
            "long query did not exercise minimum cross-window segments"
        )

    ceiling_keys = [
        "runtime_relation_ceiling",
        "runtime_factor_ceiling",
        "runtime_field_ceiling",
        "runtime_edge_ceiling",
        "runtime_view_ceiling",
        "runtime_slot_ceiling",
        "runtime_reasoning_step_ceiling",
        "product_context_token_ceiling",
    ]
    for key in ceiling_keys:
        if system_report.get(key) is not None:
            raise ValueError(
                f"serving-axis ceiling unexpectedly set: {key}"
            )

    if any(
        parameter.grad is not None
        for parameter in system.parameters()
    ):
        raise ValueError(
            "registered trainable system created gradients during CPU "
            "no-gradient qualification"
        )

    memory["peak_rss_mb"] = rss_mb()
    result = {
        "schema": (
            "alice.eipm.n0.full-envelope-cpu-runtime-result.v1"
        ),
        "status": PASS,
        "source_revision":source_revision,
        "qualifier_sha256":sha256(Path(__file__).resolve()),
        "registered_topology_sha256":sha256(topology_path),
        "qualification_config_sha256":sha256(qualification_path),
        "semantic_config_sha256":sha256(semantic_config_path),
        "tokenizer_json_sha256":sha256(tokenizer_dir/"tokenizer.json"),
        "source_config_sha256":sha256(source_config_path),
        "corpus_receipt_sha256":sha256(corpus_dir/"corpus_receipt.json"),
        "cpu_only": True,
        "inference_mode": True,
        "gradient": False,
        "optimizer": False,
        "model_training": False,
        "private_identity_data": False,
        "final_validation_opened": False,
        "semantic_checkpoint_sha256": observed_sha,
        "semantic_parameters": int(
            system_report["semantic_model_parameters"]
        ),
        "semantic_input_parameters": int(
            system_report["semantic_input_parameters"]
        ),
        "successor_parameters": int(
            system_report["successor_stack_parameters"]
        ),
        "segment_context_bridge_parameters": int(
            semantic_input_report["segment_bridge"][
                "total_parameters"
            ]
        ),
        "combined_parameters": int(
            system_report["total_parameters"]
        ),
        "registered_trainable_system": (
            "N0FullEnvelopeTrainableSystemV1"
        ),
        "single_shared_backbone": bool(
            system_report["single_shared_backbone"]
        ),
        "memory_mb": memory,
        "tokenizer_stress": tokenizer_result,
        "corpus_source_count": len(
            corpus_receipt.get("sources", [])
        ),
        "corpus_shard_count": len(corpus_paths),
        "tokenizer_receipt_model_id": tokenizer_receipt.get(
            "model_id"
        ),
        "text_surface_virtualization": {
            "native_window_tokens": native_window,
            "overlap_tokens": int(
                long_cfg["overlap_tokens"]
            ),
            "required_surfaces": sorted(required_surfaces),
            "surface_receipt": surface_receipt,
            "semantic_input_metadata": meta,
        },
        "long_context_bridge": {
            "native_window_tokens": native_window,
            "dedicated_fixture_used": True,
            "overlap_tokens": int(
                long_cfg["overlap_tokens"]
            ),
            "segments": int(
                meta["query"]["segment_count_max"]
            ),
            "bridge_report": semantic_input_report[
                "segment_bridge"
            ],
            "standalone_virtualizer_semantics_complete": (
                system.semantic_input.virtualizer.parameter_report()[
                    "standalone_cross_window_semantics_complete"
                ]
            ),
        },
        "runtime_case": {
            "batch_size": batch_size,
            "relations": relation_count,
            "factor_banks": len(case["factor_banks"]),
            "fields": field_count,
            "edges": edge_count,
            "views_after_additional": int(
                outputs["source_views"].size(1)
            ),
            "latent_slots": int(
                outputs["latent"]["latent_slots"].size(1)
            ),
            "reasoning_steps": int(
                case["max_reasoning_steps"]
            ),
            "candidate_count": candidate_count,
        },
        "system_report": system_report,
        "stack_report": stack_report,
        "semantic_input_report": semantic_input_report,
        "output_shapes": {
            "latent_pooled_state": list(
                outputs["latent"]["pooled_state"].shape
            ),
            "candidate_logits": list(
                outputs["public_judgment"][
                    "candidate_logits"
                ].shape
            ),
            "relation_distribution": list(
                outputs["operator"].relation_distribution.shape
            ),
            "edge_support_weight": list(
                outputs["binder"]["edge_support_weight"].shape
            ),
            "relational_probability": list(
                outputs["executor"][
                    "relational_probability"
                ].shape
            ),
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

    del outputs, system
    gc.collect()


if __name__ == "__main__":
    main()
