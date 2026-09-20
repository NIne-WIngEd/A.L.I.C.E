#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset

from alice_personality.n0.config import load_n0_config
from alice_personality.n0.curriculum_data import load_tokenizer
from alice_personality.n0.evidence_graph_dual_endpoint import (
    DualEndpointEvidenceGraphEncoder,
)
from alice_personality.n0.evidence_graph_query_edge_bridge import (
    QueryEdgeBridgeEvidenceGraphEncoder,
)
from alice_personality.n0.evidence_view_adapter import EvidenceViewAdapter
from alice_personality.n0.query_edge_cross_attention_bridge import (
    QueryEdgeCrossAttentionBridge,
)
from alice_personality.n0.structured_state import StructuredStateEncoder
from alice_personality.n0.v02_model import AliceN0V02Model

import train_n0_v02_relation_repair as rr
import train_n0_v02_relation_conditioned_multilayer_interface_v0_1 as ml


QUAL_SCHEMA = "alice.eipm.n0.query-edge-bridge-runtime-qualification.v0.1"
CURRICULUM_SCHEMA = "alice.eipm.n0.v02-query-edge-binding-curriculum.v0.1"
ANCHOR_SCHEMA = "alice.eipm.n0.query-edge-preservation-train-anchors.v0.1"
CHECKPOINT_SCHEMA = "alice.eipm.n0.query-edge-bridge-training-checkpoint.v0.1"
RESULT_SCHEMA = "alice.eipm.n0.query-edge-bridge-training-result.v0.1"

EXPECTED_QUAL_SHA256 = (
    "b1e073fc6226ac3f3aa88ed10d4ed31059b6e27b0338aa7738c686239f33624f"
)
EXPECTED_CURRICULUM_SHA256 = (
    "c3618a6022519af2a6a140f8b571281c68735fd1412c3ffa904979e9bbea2dcb"
)
EXPECTED_MANIFEST_SHA256 = (
    "0d05f1dcd7d11cc1defdd0a4112c10e3b1d7febfaabf97fe611a854d6905b99a"
)
EXPECTED_ANCHORS_SHA256 = (
    "f5b3f438f776216f6dbd557a8c785aac70af38d256e3f7cf94f0f71a43f53e94"
)
EXPECTED_PARENT_GRAPH_SHA256 = (
    "3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f"
)
EXPECTED_PARENT_ADAPTER_SHA256 = (
    "50eeeea3dfff5b6bddaa8da667b6b6c2f917dc668b1acd0410923822ae5ad2df"
)
EXPECTED_CALIBRATION_SHA256 = (
    "d485fce0e4304d9cbd8af1658ed7cfedfd1bd4d44b0f40cf56a2618d5ad2390b"
)
EXPECTED_POLICY_SHA256 = (
    "fb88b9580e38a648101277815f82c3ba60c19f3bfd16dbe3029ded40b04aea56"
)
EXPECTED_HIDDEN_STATES = 17


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_state_sha256(
    module: torch.nn.Module,
    *,
    exclude_prefixes: tuple[str, ...] = (),
) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if any(name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def compile_query_edge_payload(
    rows: list[dict[str, Any]],
    *,
    semantic: AliceN0V02Model,
    tokenizer: Any,
    structured: StructuredStateEncoder,
    device: torch.device,
    encode_batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    payload = ml.compile_causal_payload(
        rows,
        semantic=semantic,
        tokenizer=tokenizer,
        structured=structured,
        device=device,
        encode_batch_size=encode_batch_size,
        max_length=max_length,
    )
    selected = [
        row for row in rows if str(row["split"]) in {"train", "dev"}
    ]
    if payload["ids"] != [str(row["id"]) for row in selected]:
        raise SystemExit("query-edge causal payload row-order drift")
    max_edges = int(payload["edge_index"].size(1))
    relevant = torch.zeros(len(selected), dtype=torch.long)
    same_mask = torch.zeros(len(selected), max_edges, dtype=torch.bool)
    for i, row in enumerate(selected):
        relevant[i] = int(row["relevant_edge_index"])
        for edge_index in row["same_relation_edge_indices"]:
            same_mask[i, int(edge_index)] = True
        if int(row["relevant_edge_index"]) not in {
            int(x) for x in row["same_relation_edge_indices"]
        }:
            raise SystemExit(f"relevant edge escapes same-relation set: {row['id']}")
    payload["relevant_edge_index"] = relevant
    payload["same_relation_edge_mask"] = same_mask
    return payload


class QueryEdgeQuadDataset(Dataset):
    ORDER = (("A", "source"), ("A", "target"), ("B", "source"), ("B", "target"))

    def __init__(self, payload: dict[str, Any], split: str) -> None:
        grouped: dict[str, dict[tuple[str, str], int]] = defaultdict(dict)
        for index, row_split in enumerate(payload["splits"]):
            if str(row_split) != split:
                continue
            key = (
                str(payload["edge_direction_variants"][index]),
                str(payload["query_roles"][index]),
            )
            grouped[str(payload["quad_ids"][index])][key] = index
        self.payload = payload
        self.quads: list[list[int]] = []
        for quad_id in sorted(grouped):
            members = grouped[quad_id]
            if set(members) != set(self.ORDER):
                raise SystemExit(f"incomplete query-edge quad: {quad_id}")
            self.quads.append([members[key] for key in self.ORDER])

    def __len__(self) -> int:
        return len(self.quads)

    def __getitem__(self, item: int) -> dict[str, Any]:
        indices = self.quads[item]
        out = {
            key: torch.stack([self.payload[key][i] for i in indices], dim=0)
            for key in ml.BASE_TENSOR_KEYS
        }
        out["query_hidden_states"] = torch.stack(
            [self.payload["query_hidden_states"][i] for i in indices], dim=0
        )
        out["query_attention_mask"] = torch.stack(
            [self.payload["query_attention_mask"][i] for i in indices], dim=0
        )
        out["relevant_edge_index"] = torch.stack(
            [self.payload["relevant_edge_index"][i] for i in indices], dim=0
        )
        out["same_relation_edge_mask"] = torch.stack(
            [self.payload["same_relation_edge_mask"][i] for i in indices], dim=0
        )
        out["global_indices"] = torch.tensor(indices, dtype=torch.long)
        return out


def flatten_quad_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if key == "global_indices" or not torch.is_tensor(value):
            continue
        out[key] = value.flatten(0, 1)
    return out


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def candidate_forward(
    graph: QueryEdgeBridgeEvidenceGraphEncoder,
    adapter: EvidenceViewAdapter,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    adapted = ml.adapter_forward(adapter, batch)
    hidden = batch["query_hidden_states"].float()
    hidden_tuple = tuple(hidden[:, layer] for layer in range(hidden.size(1)))
    return graph(
        field_states=adapted["field_states"],
        valid_mask=batch["valid_mask"],
        edge_index=batch["edge_index"],
        edge_type_ids=batch["edge_type_ids"],
        edge_confidence=batch["edge_confidence"],
        edge_valid_mask=batch["edge_valid_mask"],
        query_semantic=batch["query_semantic"],
        base_field_weights=adapted["field_weights"],
        query_hidden_states=hidden_tuple,
        query_attention_mask=batch["query_attention_mask"],
    )


def parent_forward(
    graph: DualEndpointEvidenceGraphEncoder,
    adapter: EvidenceViewAdapter,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return ml.parent_forward(graph, adapter, batch)


def distill_kl(
    student_weights: torch.Tensor,
    teacher_weights: torch.Tensor,
) -> torch.Tensor:
    teacher = teacher_weights.detach().clamp_min(1e-8)
    student_log = student_weights.clamp_min(1e-8).log()
    return F.kl_div(student_log, teacher, reduction="batchmean")


def routing_scores(out: dict[str, torch.Tensor]) -> torch.Tensor:
    language = out["edge_query_language_state"]
    edge_query = out["edge_binding_query"]
    return torch.einsum("bew,bew->be", language, edge_query) / math.sqrt(
        float(language.size(-1))
    )


def routing_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    scores = routing_scores(out)
    mask = batch["same_relation_edge_mask"].bool()
    scores = scores.masked_fill(~mask, -1.0e4)
    return F.cross_entropy(scores, batch["relevant_edge_index"].long())


def irrelevant_gate_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    gate = out["edge_specialist_gate"]
    relevant = torch.zeros_like(gate, dtype=torch.bool)
    relevant.scatter_(1, batch["relevant_edge_index"].long().unsqueeze(1), True)
    irrelevant = batch["edge_valid_mask"].bool() & ~relevant
    if not bool(irrelevant.any().item()):
        return gate.sum() * 0.0
    return gate[irrelevant].pow(2).mean()


def evaluate_causal(
    graph: QueryEdgeBridgeEvidenceGraphEncoder,
    adapter: EvidenceViewAdapter,
    dataset: QueryEdgeQuadDataset,
    payload: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    graph.eval()
    adapter.eval()
    rows = row_ok = quads = quad_ok = 0
    route_total = route_ok = 0
    margins: list[float] = []
    relation_rows: dict[str, list[int]] = defaultdict(list)
    relation_quads: dict[str, list[int]] = defaultdict(list)
    relation_routes: dict[str, list[int]] = defaultdict(list)
    relation_margins: dict[str, list[float]] = defaultdict(list)

    with torch.inference_mode():
        for batch in loader:
            global_indices = batch["global_indices"]
            flat = to_device(flatten_quad_batch(batch), device)
            out = candidate_forward(graph, adapter, flat)
            target = flat["target_distribution"].argmax(dim=-1)
            pred = out["field_weights"].argmax(dim=-1)
            ok = pred.eq(target)
            margin = ml.target_margin(
                out["field_weights"],
                flat["target_distribution"],
                flat["valid_mask"],
            )
            scores = routing_scores(out).masked_fill(
                ~flat["same_relation_edge_mask"].bool(), -1.0e4
            )
            route_pred = scores.argmax(dim=-1)
            route_ok_row = route_pred.eq(flat["relevant_edge_index"].long())

            batch_quads = global_indices.size(0)
            ok4 = ok.reshape(batch_quads, 4)
            margin4 = margin.reshape(batch_quads, 4)
            route4 = route_ok_row.reshape(batch_quads, 4)
            qok = ok4.all(dim=1)

            rows += int(ok.numel())
            row_ok += int(ok.sum().item())
            quads += int(qok.numel())
            quad_ok += int(qok.sum().item())
            route_total += int(route_ok_row.numel())
            route_ok += int(route_ok_row.sum().item())
            margins.extend(float(x) for x in margin.cpu().tolist())

            for local in range(batch_quads):
                first = int(global_indices[local, 0].item())
                relation = str(payload["relations"][first])
                relation_quads[relation].append(int(qok[local].item()))
                relation_rows[relation].extend(
                    int(x) for x in ok4[local].cpu().tolist()
                )
                relation_routes[relation].extend(
                    int(x) for x in route4[local].cpu().tolist()
                )
                relation_margins[relation].extend(
                    float(x) for x in margin4[local].cpu().tolist()
                )

    family_row = {
        key: sum(v) / len(v) for key, v in sorted(relation_rows.items())
    }
    family_quad = {
        key: sum(v) / len(v) for key, v in sorted(relation_quads.items())
    }
    family_route = {
        key: sum(v) / len(v) for key, v in sorted(relation_routes.items())
    }
    family_margin = {
        key: sum(v) / len(v) for key, v in sorted(relation_margins.items())
    }
    return {
        "rows": rows,
        "row_accuracy": row_ok / max(rows, 1),
        "quads": quads,
        "quad_accuracy": quad_ok / max(quads, 1),
        "routing_accuracy": route_ok / max(route_total, 1),
        "family_row_accuracy": family_row,
        "family_min_row_accuracy": min(family_row.values()),
        "family_quad_accuracy": family_quad,
        "family_min_quad_accuracy": min(family_quad.values()),
        "family_routing_accuracy": family_route,
        "family_min_routing_accuracy": min(family_route.values()),
        "mean_target_margin": sum(margins) / max(len(margins), 1),
        "family_mean_target_margin": family_margin,
    }


def evaluate_ordinary(
    graph: QueryEdgeBridgeEvidenceGraphEncoder,
    adapter: EvidenceViewAdapter,
    dataset: Dataset,
    payload: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    graph.eval()
    adapter.eval()
    mass_by_family: dict[str, list[float]] = defaultdict(list)
    top1_by_family: dict[str, list[int]] = defaultdict(list)
    with torch.inference_mode():
        for batch in loader:
            global_index = batch["global_index"]
            batch = to_device(batch, device)
            out = candidate_forward(graph, adapter, batch)
            support = batch["target_distribution"] > 0
            mass = (
                out["field_weights"] * support.to(out["field_weights"].dtype)
            ).sum(dim=-1)
            pred = out["field_weights"].argmax(dim=-1)
            row = torch.arange(pred.size(0), device=device)
            top1 = support[row, pred]
            for local, index in enumerate(global_index.tolist()):
                family = str(payload["families"][int(index)])
                mass_by_family[family].append(float(mass[local].item()))
                top1_by_family[family].append(int(top1[local].item()))
    mass = {k: sum(v) / len(v) for k, v in sorted(mass_by_family.items())}
    top1 = {k: sum(v) / len(v) for k, v in sorted(top1_by_family.items())}
    return {
        "family_macro_target_support_mass": sum(mass.values()) / len(mass),
        "family_min_target_support_mass": min(mass.values()),
        "family_target_support_mass": mass,
        "family_macro_top1_support_accuracy": sum(top1.values()) / len(top1),
        "family_min_top1_support_accuracy": min(top1.values()),
        "family_top1_support_accuracy": top1,
    }


def evaluate_endpoint(
    graph: QueryEdgeBridgeEvidenceGraphEncoder,
    adapter: EvidenceViewAdapter,
    dataset: Dataset,
    payload: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    graph.eval()
    adapter.eval()
    pair_total = pair_ok_total = row_total = row_ok_total = 0
    margins: list[float] = []
    family_ok: dict[str, list[int]] = defaultdict(list)
    with torch.inference_mode():
        for batch in loader:
            a_global = batch["a_global_index"]
            a = {
                key[2:]: value
                for key, value in batch.items()
                if key.startswith("a_") and key != "a_global_index"
            }
            b = {
                key[2:]: value
                for key, value in batch.items()
                if key.startswith("b_") and key != "b_global_index"
            }
            a = to_device(a, device)
            b = to_device(b, device)
            ao = candidate_forward(graph, adapter, a)
            bo = candidate_forward(graph, adapter, b)
            at = a["target_distribution"].argmax(dim=-1)
            bt = b["target_distribution"].argmax(dim=-1)
            aok = ao["field_weights"].argmax(dim=-1).eq(at)
            bok = bo["field_weights"].argmax(dim=-1).eq(bt)
            pok = aok & bok
            am = ml.target_margin(
                ao["field_weights"], a["target_distribution"], a["valid_mask"]
            )
            bm = ml.target_margin(
                bo["field_weights"], b["target_distribution"], b["valid_mask"]
            )
            pm = torch.minimum(am, bm)
            pair_total += int(pok.numel())
            pair_ok_total += int(pok.sum().item())
            row_total += int(aok.numel() + bok.numel())
            row_ok_total += int(aok.sum().item() + bok.sum().item())
            margins.extend(float(x) for x in pm.cpu().tolist())
            for local, index in enumerate(a_global.tolist()):
                family = str(payload["families"][int(index)])
                family_ok[family].append(int(pok[local].item()))
    fam = {k: sum(v) / len(v) for k, v in sorted(family_ok.items())}
    return {
        "pairs": pair_total,
        "pair_accuracy": pair_ok_total / max(pair_total, 1),
        "family_pair_accuracy": fam,
        "family_min_pair_accuracy": min(fam.values()),
        "row_accuracy": row_ok_total / max(row_total, 1),
        "mean_graph_target_margin": sum(margins) / max(len(margins), 1),
    }


def preservation_pass(
    ordinary: dict[str, Any],
    endpoint: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    candidate = {"ordinary": ordinary, "endpoint": endpoint}
    checks: list[dict[str, Any]] = []
    passed = True
    for item in policy["metrics"]:
        current: Any = candidate
        for token in str(item["path"]).split("."):
            current = current[token]
        value = float(current)
        floor = float(item["baseline_mean"]) - float(item["atol"])
        ok = value >= floor
        checks.append({
            "path": str(item["path"]),
            "candidate": value,
            "required_minimum": floor,
            "pass": ok,
        })
        passed = passed and ok
    return passed, checks


def causal_ready(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> tuple[bool, dict[str, bool]]:
    no_family_regression = all(
        float(current["family_row_accuracy"][relation]) + 1e-12
        >= float(baseline["family_row_accuracy"][relation])
        for relation in baseline["family_row_accuracy"]
    )
    detail = {
        "no_relation_family_row_accuracy_regression": no_family_regression,
        "overall_row_accuracy_strictly_improves": (
            float(current["row_accuracy"]) > float(baseline["row_accuracy"]) + 1e-12
        ),
        "overall_quad_accuracy_strictly_improves": (
            float(current["quad_accuracy"]) > float(baseline["quad_accuracy"]) + 1e-12
        ),
        "worst_family_quad_accuracy_not_worse": (
            float(current["family_min_quad_accuracy"]) + 1e-12
            >= float(baseline["family_min_quad_accuracy"])
        ),
        "mean_target_margin_strictly_improves": (
            float(current["mean_target_margin"])
            > float(baseline["mean_target_margin"]) + 1e-12
        ),
        "routing_accuracy_above_half": (
            float(current["routing_accuracy"]) > 0.5
        ),
        "worst_family_routing_at_least_uniform_chance": (
            float(current["family_min_routing_accuracy"]) + 1e-12 >= (1.0 / 3.0)
        ),
        "routing_accuracy_strictly_improves_vs_initial": (
            float(current["routing_accuracy"])
            > float(baseline["routing_accuracy"]) + 1e-12
        ),
    }
    return all(detail.values()), detail


def indices_from_ids(payload: dict[str, Any], ids: list[str], label: str) -> list[int]:
    by_id = {str(value): index for index, value in enumerate(payload["ids"])}
    missing = [value for value in ids if value not in by_id]
    if missing:
        raise SystemExit(f"{label} anchor ids missing from replay cache: {missing[:5]}")
    return [by_id[value] for value in ids]


def next_batch(iterator: Any, loader: DataLoader) -> tuple[Any, Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--qualification-receipt", required=True)
    p.add_argument("--layer-map", required=True)
    p.add_argument("--curriculum", required=True)
    p.add_argument("--curriculum-manifest", required=True)
    p.add_argument("--preservation-anchors", required=True)
    p.add_argument("--calibration-receipt", required=True)
    p.add_argument("--ordinary-source-curriculum", required=True)
    p.add_argument("--endpoint-source-curriculum", required=True)
    p.add_argument("--semantic-config", required=True)
    p.add_argument("--semantic-checkpoint", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--structured-config", required=True)
    p.add_argument("--structured-checkpoint", required=True)
    p.add_argument("--parent-adapter", required=True)
    p.add_argument("--parent-graph", required=True)
    p.add_argument("--ordinary-replay-cache", required=True)
    p.add_argument("--endpoint-replay-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--encode-batch-size", type=int, default=24)
    p.add_argument("--quad-batch-size", type=int, default=4)
    p.add_argument("--preservation-batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=24)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--save-every", type=int, default=40)
    p.add_argument("--learning-rate", type=float, default=1.5e-4)
    p.add_argument("--weight-decay", type=float, default=0.02)
    p.add_argument("--warmup-steps", type=int, default=12)
    p.add_argument("--margin", type=float, default=0.20)
    p.add_argument("--margin-weight", type=float, default=0.25)
    p.add_argument("--routing-weight", type=float, default=0.50)
    p.add_argument("--preservation-weight", type=float, default=1.0)
    p.add_argument("--irrelevant-gate-weight", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20260920)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("query-edge training requires one CUDA device")
    device = torch.device("cuda")
    seed_all(args.seed)

    root = Path(args.repo_root).resolve()
    qual_path = Path(args.qualification_receipt).resolve()
    map_path = Path(args.layer_map).resolve()
    curriculum_path = Path(args.curriculum).resolve()
    manifest_path = Path(args.curriculum_manifest).resolve()
    anchors_path = Path(args.preservation_anchors).resolve()
    cal_path = Path(args.calibration_receipt).resolve()
    ordinary_source = Path(args.ordinary_source_curriculum).resolve()
    endpoint_source = Path(args.endpoint_source_curriculum).resolve()
    semantic_config = Path(args.semantic_config).resolve()
    semantic_checkpoint = Path(args.semantic_checkpoint).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve()
    structured_config = Path(args.structured_config).resolve()
    structured_checkpoint = Path(args.structured_checkpoint).resolve()
    adapter_path = Path(args.parent_adapter).resolve()
    graph_path = Path(args.parent_graph).resolve()
    ordinary_replay_path = Path(args.ordinary_replay_cache).resolve()
    endpoint_replay_path = Path(args.endpoint_replay_cache).resolve()
    output = Path(args.output_dir).resolve()

    for path in (
        qual_path, map_path, curriculum_path, manifest_path, anchors_path, cal_path,
        ordinary_source, endpoint_source, semantic_config,
        semantic_checkpoint / "alice_n0_v02.safetensors",
        tokenizer_dir / "tokenizer.json", structured_config,
        structured_checkpoint / "structured_state.safetensors",
        adapter_path, graph_path, ordinary_replay_path, endpoint_replay_path,
    ):
        if not path.is_file():
            raise SystemExit(f"missing query-edge training input: {path}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite query-edge training output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    exact_hashes = {
        qual_path: EXPECTED_QUAL_SHA256,
        curriculum_path: EXPECTED_CURRICULUM_SHA256,
        manifest_path: EXPECTED_MANIFEST_SHA256,
        anchors_path: EXPECTED_ANCHORS_SHA256,
        graph_path: EXPECTED_PARENT_GRAPH_SHA256,
        adapter_path: EXPECTED_PARENT_ADAPTER_SHA256,
        cal_path: EXPECTED_CALIBRATION_SHA256,
    }
    for path, expected in exact_hashes.items():
        if sha(path) != expected:
            raise SystemExit(f"exact training lineage drift: {path.name}")

    qual = json.loads(qual_path.read_text(encoding="utf-8"))
    if qual.get("schema") != QUAL_SCHEMA:
        raise SystemExit("query-edge qualification schema drift")
    if qual.get("status") != "PASS_QUERY_EDGE_BRIDGE_NO_GRADIENT_RUNTIME_CONTRACT":
        raise SystemExit("query-edge runtime qualification did not pass")
    for key in (
        "parent_parameters_exactly_unchanged",
        "exact_parent_field_weights",
        "exact_parent_pooled_state",
        "zero_initialized_specialist_gate",
        "same_relation_edges_produce_distinct_edge_queries",
        "edge_content_conditions_query_token_attention",
    ):
        if qual.get(key) is not True:
            raise SystemExit(f"qualification invariant failed: {key}")
    if qual.get("forced_antisymmetry") is not False:
        raise SystemExit("forced antisymmetry returned")
    if qual.get("training_authorized") is not False:
        raise SystemExit("qualification receipt must not self-authorize training")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != CURRICULUM_SCHEMA:
        raise SystemExit("query-edge curriculum schema drift")
    if manifest.get("compiled_sha256") != sha(curriculum_path):
        raise SystemExit("query-edge curriculum hash drift")
    if manifest.get("same_relation_edges_per_row") != 3:
        raise SystemExit("same-relation shortcut control drift")
    if manifest.get("relation_global_endpoint_polarity_sufficient") is not False:
        raise SystemExit("relation-global endpoint shortcut reopened")
    if manifest.get("test_split_opening_authorized") is not False:
        raise SystemExit("test split unexpectedly opened")

    anchors = json.loads(anchors_path.read_text(encoding="utf-8"))
    if anchors.get("schema") != ANCHOR_SCHEMA:
        raise SystemExit("preservation anchor schema drift")
    if anchors["gradient_eligibility"]["dev_rows_may_be_used_for_gradient"] is not False:
        raise SystemExit("preservation DEV leakage")
    if anchors["gradient_eligibility"]["heldout_or_test_rows_may_be_used_for_gradient"] is not False:
        raise SystemExit("preservation heldout leakage")

    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    if cal.get("status") != "PASS_PARENT_ONLY_PRESERVATION_CALIBRATION":
        raise SystemExit("frozen parent preservation calibration drift")
    policy = cal["policy"]

    semantic_cfg = load_n0_config(semantic_config)
    semantic = AliceN0V02Model(semantic_cfg)
    state = load_file(
        str(semantic_checkpoint / "alice_n0_v02.safetensors"),
        device="cpu",
    )
    missing, unexpected = semantic.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"semantic checkpoint mismatch missing={list(missing)} unexpected={list(unexpected)}"
        )
    if semantic.parameter_report()["total_parameters"] != rr.EXPECTED_SEMANTIC_PARAMETERS:
        raise SystemExit("semantic parameter count drift")
    for parameter in semantic.parameters():
        parameter.requires_grad = False
    semantic.to(device).eval()
    tokenizer = load_tokenizer(tokenizer_dir)

    structured_cfg = rr.load_structured_config(structured_config)
    structured = StructuredStateEncoder(structured_cfg)
    structured.load_state_dict(
        load_file(
            str(structured_checkpoint / "structured_state.safetensors"),
            device="cpu",
        ),
        strict=True,
    )
    for parameter in structured.parameters():
        parameter.requires_grad = False

    rows = read_jsonl(curriculum_path)
    causal = compile_query_edge_payload(
        rows,
        semantic=semantic,
        tokenizer=tokenizer,
        structured=structured,
        device=device,
        encode_batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )
    cache_path = output / "query_edge_train_dev_hidden_cache.pt"
    torch.save(causal, cache_path)

    ordinary_payload = torch.load(ordinary_replay_path, map_location="cpu")
    endpoint_payload = torch.load(endpoint_replay_path, map_location="cpu")

    ordinary_train_indices = indices_from_ids(
        ordinary_payload,
        [str(x) for x in anchors["ordinary"]["train_anchor_ids"]],
        "ordinary",
    )
    endpoint_train_indices = indices_from_ids(
        endpoint_payload,
        [str(x) for x in anchors["endpoint"]["train_anchor_ids"]],
        "endpoint",
    )
    ordinary_dev_indices = [
        i for i, split in enumerate(ordinary_payload["splits"])
        if str(split) == "dev"
    ]
    endpoint_dev_pairs = rr.pair_indices(endpoint_payload, "dev")
    endpoint_dev_indices = sorted({i for pair in endpoint_dev_pairs for i in pair})

    ordinary_needed = sorted(set(ordinary_train_indices + ordinary_dev_indices))
    endpoint_needed = sorted(set(endpoint_train_indices + endpoint_dev_indices))
    ordinary_hidden = ml.attach_query_hidden_states(
        ordinary_payload,
        read_jsonl(ordinary_source),
        indices=ordinary_needed,
        semantic=semantic,
        tokenizer=tokenizer,
        device=device,
        encode_batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )
    endpoint_hidden = ml.attach_query_hidden_states(
        endpoint_payload,
        read_jsonl(endpoint_source),
        indices=endpoint_needed,
        semantic=semantic,
        tokenizer=tokenizer,
        device=device,
        encode_batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )

    del semantic, state, structured
    torch.cuda.empty_cache()

    adapter = EvidenceViewAdapter(rr.expanded_adapter_config()).to(device)
    adapter.load_state_dict(load_file(str(adapter_path), device="cpu"), strict=True)
    for parameter in adapter.parameters():
        parameter.requires_grad = False
    adapter.eval()

    parent_state = load_file(str(graph_path), device="cpu")
    baseline_graph = DualEndpointEvidenceGraphEncoder(
        rr.expanded_graph_config()
    ).to(device)
    baseline_graph.load_state_dict(parent_state, strict=True)
    for parameter in baseline_graph.parameters():
        parameter.requires_grad = False
    baseline_graph.eval()

    bridge = QueryEdgeCrossAttentionBridge.from_layer_map_file(
        layer_map_path=map_path,
        semantic_size=rr.expanded_graph_config().semantic_size,
        graph_size=rr.expanded_graph_config().graph_size,
        num_relation_types=rr.expanded_graph_config().num_relation_types,
        num_hidden_states=EXPECTED_HIDDEN_STATES,
        interface_size=rr.expanded_graph_config().graph_size,
        dropout=0.0,
    )
    graph = QueryEdgeBridgeEvidenceGraphEncoder(
        config=rr.expanded_graph_config(),
        bridge=bridge,
    ).to(device)
    missing, unexpected = graph.load_state_dict(parent_state, strict=False)
    if any(not name.startswith("query_edge_bridge.") for name in missing) or unexpected:
        raise SystemExit(
            f"candidate parent load drift missing={list(missing)} unexpected={list(unexpected)}"
        )
    trainable = graph.freeze_parent_for_bridge_training()
    parent_hash_before = tensor_state_sha256(
        graph, exclude_prefixes=("query_edge_bridge.",)
    )

    train_quads = QueryEdgeQuadDataset(causal, "train")
    dev_quads = QueryEdgeQuadDataset(causal, "dev")
    if len(train_quads) != 108 or len(dev_quads) != 36:
        raise SystemExit(
            f"query-edge quad split drift train={len(train_quads)} dev={len(dev_quads)}"
        )

    ordinary_train = ml.RowHiddenDataset(
        ordinary_payload, ordinary_train_indices, ordinary_hidden
    )
    endpoint_train = ml.RowHiddenDataset(
        endpoint_payload, endpoint_train_indices, endpoint_hidden
    )
    ordinary_dev = ml.RowHiddenDataset(
        ordinary_payload, ordinary_dev_indices, ordinary_hidden
    )
    endpoint_dev = ml.PairHiddenDataset(
        endpoint_payload, endpoint_dev_pairs, endpoint_hidden
    )

    baseline_dev = evaluate_causal(
        graph, adapter, dev_quads, causal, device, args.eval_batch_size
    )
    initial_ordinary = evaluate_ordinary(
        graph, adapter, ordinary_dev, ordinary_payload, device, args.eval_batch_size
    )
    initial_endpoint = evaluate_endpoint(
        graph, adapter, endpoint_dev, endpoint_payload, device, args.eval_batch_size
    )
    initial_preserved, initial_checks = preservation_pass(
        initial_ordinary, initial_endpoint, policy
    )
    if not initial_preserved:
        raise SystemExit("zero-init query-edge bridge failed frozen preservation policy")

    first_dev = next(iter(DataLoader(dev_quads, batch_size=1, shuffle=False)))
    flat_first = to_device(flatten_quad_batch(first_dev), device)
    with torch.inference_mode():
        parent_out = parent_forward(baseline_graph, adapter, flat_first)
        candidate_out = candidate_forward(graph, adapter, flat_first)
    if not torch.equal(parent_out["field_weights"], candidate_out["field_weights"]):
        raise SystemExit("query-edge zero-init field weights lost exact parent")
    if not torch.equal(parent_out["pooled_state"], candidate_out["pooled_state"]):
        raise SystemExit("query-edge zero-init pooled state lost exact parent")

    print("baseline_query_edge_dev=" + json.dumps(baseline_dev, sort_keys=True), flush=True)
    print("initial_preservation=" + json.dumps(
        {"ordinary": initial_ordinary, "endpoint": initial_endpoint},
        sort_keys=True,
    ), flush=True)

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return max((step + 1) / max(args.warmup_steps, 1), 1e-6)
        progress = (step - args.warmup_steps) / max(
            args.max_steps - args.warmup_steps, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    causal_loader = DataLoader(
        train_quads,
        batch_size=args.quad_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )
    ordinary_loader = DataLoader(
        ordinary_train,
        batch_size=args.preservation_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
        num_workers=0,
    )
    endpoint_loader = DataLoader(
        endpoint_train,
        batch_size=args.preservation_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 2),
        num_workers=0,
    )
    causal_it = iter(causal_loader)
    ordinary_it = iter(ordinary_loader)
    endpoint_it = iter(endpoint_loader)

    receipts: dict[str, dict[str, Any]] = {}

    for step in range(1, args.max_steps + 1):
        causal_batch, causal_it = next_batch(causal_it, causal_loader)
        ordinary_batch, ordinary_it = next_batch(ordinary_it, ordinary_loader)
        endpoint_batch, endpoint_it = next_batch(endpoint_it, endpoint_loader)

        graph.train()
        adapter.eval()
        optimizer.zero_grad(set_to_none=True)

        causal_flat = to_device(flatten_quad_batch(causal_batch), device)
        out = candidate_forward(graph, adapter, causal_flat)
        selection_ce = ml.dist_ce(
            out["field_weights"], causal_flat["target_distribution"]
        )
        margins = ml.target_margin(
            out["field_weights"],
            causal_flat["target_distribution"],
            causal_flat["valid_mask"],
        )
        margin_loss = F.relu(args.margin - margins).mean()
        route = routing_loss(out, causal_flat)
        gate_noop = irrelevant_gate_loss(out, causal_flat)
        causal_total = (
            selection_ce
            + args.margin_weight * margin_loss
            + args.routing_weight * route
            + args.irrelevant_gate_weight * gate_noop
        )
        causal_total.backward()

        preservation_values: list[float] = []
        for preservation_batch in (ordinary_batch, endpoint_batch):
            preserve = to_device(preservation_batch, device)
            with torch.inference_mode():
                teacher = parent_forward(baseline_graph, adapter, preserve)
            student = candidate_forward(graph, adapter, preserve)
            kl = distill_kl(student["field_weights"], teacher["field_weights"])
            weighted = 0.5 * args.preservation_weight * kl
            weighted.backward()
            preservation_values.append(float(kl.detach().cpu()))

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 20 == 0:
            print(json.dumps({
                "step": step,
                "selection_ce": float(selection_ce.detach().cpu()),
                "margin_loss": float(margin_loss.detach().cpu()),
                "routing_loss": float(route.detach().cpu()),
                "irrelevant_gate_loss": float(gate_noop.detach().cpu()),
                "ordinary_parent_kl": preservation_values[0],
                "endpoint_parent_kl": preservation_values[1],
                "lr": scheduler.get_last_lr()[0],
            }, sort_keys=True), flush=True)

        if step % args.save_every == 0 or step == args.max_steps:
            dev = evaluate_causal(
                graph, adapter, dev_quads, causal, device, args.eval_batch_size
            )
            ordinary = evaluate_ordinary(
                graph, adapter, ordinary_dev, ordinary_payload,
                device, args.eval_batch_size
            )
            endpoint = evaluate_endpoint(
                graph, adapter, endpoint_dev, endpoint_payload,
                device, args.eval_batch_size
            )
            preserved, preservation_checks = preservation_pass(
                ordinary, endpoint, policy
            )
            causal_ok, causal_checks = causal_ready(dev, baseline_dev)

            cp = output / f"step-{step:08d}"
            cp.mkdir(parents=True, exist_ok=True)
            candidate_path = cp / "query_edge_bridge.safetensors"
            bridge_state = {
                name: value.detach().cpu().contiguous()
                for name, value in graph.state_dict().items()
                if name.startswith("query_edge_bridge.")
            }
            save_file(bridge_state, str(candidate_path))

            receipt = {
                "schema": CHECKPOINT_SCHEMA,
                "status": "TRAINED_NOT_RATIFIED",
                "step": step,
                "git_revision": git_revision(root),
                "candidate_sha256": sha(candidate_path),
                "qualification_receipt_sha256": sha(qual_path),
                "curriculum_sha256": sha(curriculum_path),
                "curriculum_manifest_sha256": sha(manifest_path),
                "preservation_anchors_sha256": sha(anchors_path),
                "calibration_receipt_sha256": sha(cal_path),
                "parent_graph_sha256": sha(graph_path),
                "parent_adapter_sha256": sha(adapter_path),
                "trainable_scope": "query_edge_bridge_only",
                "trainable_parameters": sum(p.numel() for p in trainable),
                "training_objective": {
                    "causal_field_selection_ce": True,
                    "causal_margin_weight": args.margin_weight,
                    "same_relation_relevant_edge_routing_weight": args.routing_weight,
                    "parent_distillation_weight": args.preservation_weight,
                    "ordinary_and_endpoint_distillation_equal_share": True,
                    "irrelevant_edge_gate_noop_weight": args.irrelevant_gate_weight,
                    "gradient_surgery": False,
                    "parent_or_semantic_parameter_updates": False,
                },
                "parent_parameters_exactly_unchanged": (
                    tensor_state_sha256(
                        graph, exclude_prefixes=("query_edge_bridge.",)
                    )
                    == parent_hash_before
                ),
                "causal_dev_metrics": dev,
                "causal_dev_checks": causal_checks,
                "causal_dev_ready": causal_ok,
                "ordinary_preservation_metrics": ordinary,
                "endpoint_preservation_metrics": endpoint,
                "preservation_checks": preservation_checks,
                "preservation_pass": preserved,
                "eligible_for_heldout_decision": preserved and causal_ok,
                "preservation_train_anchors_used_for_gradient": True,
                "preservation_dev_rows_used_for_gradient": False,
                "causal_test_split_evaluated": False,
                "frozen_challenge_evaluated": False,
                "semantic_parent_mutated": False,
                "structured_parent_mutated": False,
                "adapter_mutated": False,
                "parent_graph_mutated": False,
                "scale_authorized": False,
                "private_identity_gradient": False,
                "production_promotion_authorized": False,
                "n0_complete": False,
            }
            if receipt["parent_parameters_exactly_unchanged"] is not True:
                raise SystemExit("frozen parent parameters changed during query-edge training")
            (cp / "receipt.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            receipts[cp.name] = receipt
            print("query_edge_checkpoint=" + json.dumps(receipt, sort_keys=True), flush=True)

    eligible = {
        key: value for key, value in receipts.items()
        if value["eligible_for_heldout_decision"]
    }

    def score(receipt: dict[str, Any]) -> tuple[float, ...]:
        dev = receipt["causal_dev_metrics"]
        return (
            float(dev["family_min_routing_accuracy"]),
            float(dev["family_min_quad_accuracy"]),
            float(dev["routing_accuracy"]),
            float(dev["quad_accuracy"]),
            float(dev["row_accuracy"]),
            float(dev["mean_target_margin"]),
            -int(receipt["step"]),
        )

    selected = max(eligible, key=lambda key: score(eligible[key])) if eligible else None
    result = {
        "schema": RESULT_SCHEMA,
        "status": (
            "PASS_QUERY_EDGE_CAUSAL_DEV_AND_PRESERVATION_READY_FOR_SEPARATE_HELDOUT_DECISION"
            if selected is not None
            else
            "FAIL_QUERY_EDGE_CAUSAL_DEV_OR_PRESERVATION_STOP_NO_HELDOUT_NO_AUTOMATIC_HOTFIX"
        ),
        "git_revision": git_revision(root),
        "qualification_receipt_sha256": sha(qual_path),
        "curriculum_sha256": sha(curriculum_path),
        "curriculum_manifest_sha256": sha(manifest_path),
        "preservation_anchors_sha256": sha(anchors_path),
        "calibration_receipt_sha256": sha(cal_path),
        "parent_graph_sha256": sha(graph_path),
        "parent_adapter_sha256": sha(adapter_path),
        "causal_train_dev_hidden_cache_sha256": sha(cache_path),
        "baseline_query_edge_dev": baseline_dev,
        "initial_parent_preservation": {
            "ordinary": initial_ordinary,
            "endpoint": initial_endpoint,
            "checks": initial_checks,
        },
        "checkpoints": receipts,
        "eligible_checkpoint_keys": sorted(eligible),
        "selected_checkpoint": selected,
        "selected_candidate_sha256": (
            eligible[selected]["candidate_sha256"] if selected else None
        ),
        "training_objective": {
            "causal_field_selection_ce": True,
            "causal_margin_weight": args.margin_weight,
            "same_relation_relevant_edge_routing_weight": args.routing_weight,
            "parent_distillation_weight": args.preservation_weight,
            "ordinary_and_endpoint_distillation_equal_share": True,
            "irrelevant_edge_gate_noop_weight": args.irrelevant_gate_weight,
            "gradient_surgery": False,
            "rationale": (
                "explicitly supervise edge binding and preserve the frozen parent "
                "on TRAIN-only old-task anchors without changing the qualified architecture"
            ),
        },
        "selection_policy": (
            "preservation_first_then_worst_relation_routing_then_worst_relation_quad_"
            "then_overall_routing_then_overall_quad_then_row_then_margin_then_earlier"
        ),
        "causal_readiness_policy": {
            "no_relation_family_row_accuracy_regression_vs_initial_parent": True,
            "overall_row_accuracy_strictly_improves": True,
            "overall_quad_accuracy_strictly_improves": True,
            "worst_family_quad_accuracy_may_not_regress": True,
            "mean_target_margin_strictly_improves": True,
            "routing_accuracy_must_exceed_half": True,
            "worst_family_routing_accuracy_at_least_uniform_chance": True,
            "routing_accuracy_must_strictly_improve_vs_initial": True,
        },
        "preservation_train_anchors_used_for_gradient": True,
        "preservation_dev_rows_used_for_gradient": False,
        "causal_test_split_evaluated": False,
        "causal_test_split_opened_after_training": False,
        "frozen_challenge_evaluated": False,
        "parent_graph_parameters_exactly_unchanged": (
            tensor_state_sha256(
                graph, exclude_prefixes=("query_edge_bridge.",)
            )
            == parent_hash_before
        ),
        "semantic_parent_mutated": False,
        "structured_parent_mutated": False,
        "adapter_mutated": False,
        "scale_authorized": False,
        "automatic_rerun_or_hotfix_authorized": False,
        "heldout_opening_authorized": False,
        "frozen_challenge_rerun_authorized": False,
        "private_identity_data": False,
        "private_identity_gradient": False,
        "production_promotion_authorized": False,
        "n0_complete": False,
        "next_action": (
            "make_one_explicit_heldout_opening_decision_without_changing_candidate"
            if selected is not None
            else
            "stop_and_reassess_query_edge_training_objective_or_boundary_from_first_principles"
        ),
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
