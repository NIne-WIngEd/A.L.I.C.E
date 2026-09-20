from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

CONTROL_RELATIONAL = 1
NONE_RELATION = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def labels(row: dict) -> dict[str, str]:
    target = row["operator_target"]
    seq = [int(x) for x in target.get("relation_sequence_id", [])][:2]
    seq += [NONE_RELATION] * (2 - len(seq))
    control = int(target["control_id"])
    operation = int(target["operation_id"]) if control == CONTROL_RELATIONAL else -1
    role = int(target["role_id"])
    return {
        "relation_first": str(seq[0]),
        "relation_sequence": ",".join(str(x) for x in seq),
        "role": str(role),
        "operation": str(operation),
        "control": str(control),
        "signature": f"{seq[0]},{seq[1]}|{role}|{operation}|{control}",
    }


def special_token_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, special_ids: set[int]) -> torch.Tensor:
    mask = attention_mask.bool().clone()
    for token_id in special_ids:
        mask &= input_ids.ne(int(token_id))
    empty = mask.sum(dim=-1).eq(0)
    if bool(empty.any()):
        mask[empty] = attention_mask[empty].bool()
    return mask


@torch.inference_mode()
def encode_rows(
    *,
    model,
    tokenizer,
    rows: list[dict],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> list[torch.Tensor]:
    special_ids = set(int(x) for x in (getattr(tokenizer, "all_special_ids", []) or []))
    layer_chunks: list[list[torch.Tensor]] | None = None

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        text = [str(row["query_text"]) for row in batch_rows]
        encoded = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ids = encoded["input_ids"].to(device)
        attention = encoded["attention_mask"].to(device)
        outputs = model.backbone(
            input_ids=ids,
            attention_mask=attention,
            output_hidden_states=True,
            return_dict=True,
        )
        states = outputs.hidden_states
        if states is None or len(states) != 17:
            raise RuntimeError(f"semantic hidden-state depth drift: {0 if states is None else len(states)}")

        mask = special_token_mask(
            encoded["input_ids"],
            encoded["attention_mask"],
            special_ids,
        ).to(device)
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1).to(torch.float32)

        if layer_chunks is None:
            layer_chunks = [[] for _ in range(len(states))]

        for layer_index, state in enumerate(states):
            pooled = (
                state.float() * mask.unsqueeze(-1).to(state.dtype)
            ).sum(dim=1) / denom
            layer_chunks[layer_index].append(pooled.cpu())

    assert layer_chunks is not None
    return [torch.cat(chunks, dim=0) for chunks in layer_chunks]


def centroid_predict(train_x: torch.Tensor, train_y: list[str], eval_x: torch.Tensor) -> tuple[list[str], set[str]]:
    classes = sorted(set(train_y))
    centroids = []
    for label in classes:
        idx = torch.tensor([i for i, y in enumerate(train_y) if y == label], dtype=torch.long)
        centroid = train_x.index_select(0, idx).mean(dim=0)
        centroids.append(centroid)
    centers = F.normalize(torch.stack(centroids, dim=0).float(), dim=-1)
    q = F.normalize(eval_x.float(), dim=-1)
    pred_idx = (q @ centers.T).argmax(dim=-1).tolist()
    return [classes[i] for i in pred_idx], set(classes)


def accuracy(pred: list[str], gold: list[str]) -> float:
    return sum(int(a == b) for a, b in zip(pred, gold)) / max(len(gold), 1)


def majority_accuracy(train_y: list[str], eval_y: list[str]) -> float:
    counts = defaultdict(int)
    for y in train_y:
        counts[y] += 1
    winner = max(counts, key=counts.get)
    return sum(int(y == winner) for y in eval_y) / max(len(eval_y), 1)


def normalized_lift(acc: float, majority: float) -> float:
    if majority >= 1.0:
        return 0.0
    return (acc - majority) / max(1.0 - majority, 1e-9)


def evaluate_head(
    train_rows: list[dict],
    eval_rows: list[dict],
    train_layers: list[torch.Tensor],
    eval_layers: list[torch.Tensor],
    head: str,
    *,
    relational_only: bool,
) -> dict:
    train_indices = [
        i for i, row in enumerate(train_rows)
        if (not relational_only or int(row["operator_target"]["control_id"]) == CONTROL_RELATIONAL)
    ]
    eval_indices = [
        i for i, row in enumerate(eval_rows)
        if (not relational_only or int(row["operator_target"]["control_id"]) == CONTROL_RELATIONAL)
    ]
    train_y = [labels(train_rows[i])[head] for i in train_indices]
    eval_y = [labels(eval_rows[i])[head] for i in eval_indices]
    majority = majority_accuracy(train_y, eval_y)

    by_layer = {}
    best = None
    for layer, (train_x_all, eval_x_all) in enumerate(zip(train_layers, eval_layers)):
        train_x = train_x_all[train_indices]
        eval_x = eval_x_all[eval_indices]
        pred, classes = centroid_predict(train_x, train_y, eval_x)
        seen = [gold in classes for gold in eval_y]
        seen_count = sum(seen)
        seen_acc = (
            sum(int(p == g) for p, g, ok in zip(pred, eval_y, seen) if ok) / max(seen_count, 1)
        )
        overall = accuracy(pred, eval_y)
        item = {
            "accuracy": overall,
            "seen_class_accuracy": seen_acc,
            "seen_class_fraction": seen_count / max(len(eval_y), 1),
        }
        by_layer[str(layer)] = item
        if best is None or item["accuracy"] > best["accuracy"]:
            best = {"layer": layer, **item}

    assert best is not None
    best["majority_accuracy"] = majority
    best["normalized_lift_over_majority"] = normalized_lift(best["accuracy"], majority)
    return {
        "rows": len(eval_y),
        "best": best,
        "by_layer": by_layer,
    }


def pair_cosines(rows: list[dict], layers: list[torch.Tensor]) -> dict:
    def grouped(key: str) -> list[list[int]]:
        groups: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(rows):
            value = row.get(key)
            if value:
                groups[str(value)].append(i)
        return [idx for idx in groups.values() if len(idx) >= 2]

    para = grouped("paraphrase_group")
    contrast = grouped("contrast_group")
    out = {}
    for layer_index, x in enumerate(layers):
        z = F.normalize(x.float(), dim=-1)

        def avg(groups: list[list[int]], require_same: bool | None) -> float | None:
            vals = []
            for indices in groups:
                for a_pos in range(len(indices)):
                    for b_pos in range(a_pos + 1, len(indices)):
                        a, b = indices[a_pos], indices[b_pos]
                        same = labels(rows[a])["signature"] == labels(rows[b])["signature"]
                        if require_same is not None and same != require_same:
                            continue
                        vals.append(float((z[a] * z[b]).sum().item()))
            return None if not vals else sum(vals) / len(vals)

        same = avg(para, True)
        changed = avg(contrast, False)
        out[str(layer_index)] = {
            "same_operator_paraphrase_cosine": same,
            "counterfactual_operator_change_cosine": changed,
            "separation": None if same is None or changed is None else same - changed,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-config", required=True)
    parser.add_argument("--semantic-checkpoint", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--t2-curriculum", required=True)
    parser.add_argument("--reality-corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=96)
    args = parser.parse_args()

    from safetensors.torch import load_file
    from alice_personality.n0.config import load_n0_config
    from alice_personality.n0.curriculum_data import load_tokenizer
    from alice_personality.n0.v02_model import AliceN0V02Model

    t2_rows = read_jsonl(Path(args.t2_curriculum))
    train_rows = [r for r in t2_rows if r.get("split") == "train"]
    dev_rows = [r for r in t2_rows if r.get("split") == "dev"]
    reality_rows = read_jsonl(Path(args.reality_corpus))

    if len(train_rows) != 720 or len(dev_rows) != 288:
        raise SystemExit("T2 curriculum count drift")
    if any(r.get("private_identity_data") is not False for r in reality_rows):
        raise SystemExit("private identity data present in reality corpus")

    config_path = Path(args.semantic_config)
    checkpoint = Path(args.semantic_checkpoint)
    tokenizer_dir = Path(args.tokenizer_dir)
    cfg = load_n0_config(config_path)
    model = AliceN0V02Model(cfg)
    missing, unexpected = model.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=False)
    if missing or unexpected:
        raise SystemExit(f"semantic checkpoint mismatch missing={list(missing)} unexpected={list(unexpected)}")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    tokenizer = load_tokenizer(tokenizer_dir)

    train_layers = encode_rows(
        model=model,
        tokenizer=tokenizer,
        rows=train_rows,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    dev_layers = encode_rows(
        model=model,
        tokenizer=tokenizer,
        rows=dev_rows,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    reality_layers = encode_rows(
        model=model,
        tokenizer=tokenizer,
        rows=reality_rows,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    heads = {
        "relation_first": True,
        "relation_sequence": True,
        "role": True,
        "operation": True,
        "control": False,
        "signature": False,
    }

    t2_dev = {}
    reality = {}
    for head, relational_only in heads.items():
        t2_dev[head] = evaluate_head(
            train_rows, dev_rows, train_layers, dev_layers, head, relational_only=relational_only
        )
        reality[head] = evaluate_head(
            train_rows, reality_rows, train_layers, reality_layers, head, relational_only=relational_only
        )

    critical = ("relation_first", "role", "operation", "control")
    critical_lifts = [reality[h]["best"]["normalized_lift_over_majority"] for h in critical]
    payload = {
        "schema": "alice.eipm.n0.qsre-semantic-reality-probe.v0.1",
        "status": "PASS_QSRE_FROZEN_SEMANTIC_REALITY_PROBE",
        "semantic_checkpoint_sha256": sha256(checkpoint),
        "semantic_backbone_gradient": False,
        "optimizer": False,
        "gradient": False,
        "gpu": False,
        "private_identity_data": False,
        "train_rows": len(train_rows),
        "t2_dev_rows": len(dev_rows),
        "reality_rows": len(reality_rows),
        "t2_dev": t2_dev,
        "reality": reality,
        "reality_pair_cosines": pair_cosines(reality_rows, reality_layers),
        "summary": {
            "critical_heads": list(critical),
            "mean_best_layer_normalized_lift": sum(critical_lifts) / len(critical_lifts),
            "min_best_layer_normalized_lift": min(critical_lifts),
            "best_layers": {h: reality[h]["best"]["layer"] for h in critical},
            "best_accuracy": {h: reality[h]["best"]["accuracy"] for h in critical},
            "majority_accuracy": {h: reality[h]["best"]["majority_accuracy"] for h in critical},
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))

    del model
    gc.collect()


if __name__ == "__main__":
    main()
