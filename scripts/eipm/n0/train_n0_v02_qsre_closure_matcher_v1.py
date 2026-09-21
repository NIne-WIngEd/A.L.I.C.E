from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from alice_personality.n0.qsre_production_core import CONTROL_RELATIONAL
from alice_personality.n0.qsre_schema_matcher import QSRESchemaMatcher
from prepare_n0_v02_qsre_production_cache_v1 import encode_texts
from qsre_production_runtime import (
    load_frozen_semantic_model,
    load_tokenizer,
    sha256,
)


RESULT_SCHEMA = "alice.eipm.n0.qsre-closure-matcher-result.v1"
CHECKPOINT_SCHEMA = "alice.eipm.n0.qsre-closure-matcher-checkpoint.v1"
FACTOR_CACHE_SCHEMA = "alice.eipm.n0.qsre-closure-factor-schema-cache.v1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def description_text(key: str, description: str) -> str:
    return (
        f"Schema key: {key}. "
        f"Meaning: {description}"
    )


def relation_examples(
    entries: list[dict],
    *,
    phrase_field: str,
    seed: int,
    examples_per_relation: int,
    view_two: bool = True,
) -> tuple[list[str], list[str], torch.Tensor]:
    rng = random.Random(seed)
    names = [
        "Aster", "Beryl", "Cinder", "Dorian", "Elio", "Fenn",
        "Galen", "Hera", "Ivo", "Juno", "Kora", "Lyra",
        "Miro", "Nola", "Orin", "Pia", "Quill", "Rhea",
    ]
    view0: list[str] = []
    view1: list[str] = []
    targets: list[int] = []
    for target, entry in enumerate(entries):
        phrases = list(entry[phrase_field])
        if not phrases:
            raise RuntimeError(f"{entry['key']}: empty phrase bank {phrase_field}")
        for example in range(examples_per_relation):
            left = names[(target * 3 + example) % len(names)]
            right = names[(target * 7 + example + 5) % len(names)]
            if left == right:
                right = names[(names.index(right) + 1) % len(names)]
            phrase = phrases[example % len(phrases)]
            view0.append(
                f"{left} {phrase} {right}. "
                "Which supplied schema meaning best matches the relationship expressed here?"
            )
            if view_two:
                alternate = phrases[(example + 1) % len(phrases)]
                view1.append(
                    f"Identify the schema relation conveyed when {left} {alternate} {right}. "
                    "Choose by meaning rather than by a memorized relation label."
                )
            else:
                view1.append(view0[-1])
            targets.append(target)
    order = list(range(len(targets)))
    rng.shuffle(order)
    return (
        [view0[index] for index in order],
        [view1[index] for index in order],
        torch.tensor([targets[index] for index in order], dtype=torch.long),
    )


def factor_holdout_examples(meta: dict) -> dict[str, tuple[list[str], torch.Tensor]]:
    result: dict[str, tuple[list[str], torch.Tensor]] = {}
    for category in ("role", "traversal", "direction", "control"):
        texts: list[str] = []
        targets: list[int] = []
        for index, row in enumerate(meta["factors"][category]):
            for phrase in row["heldout_phrases"]:
                texts.append(
                    f"{phrase}. Which supplied {category} meaning best matches this instruction?"
                )
                targets.append(index)
        result[category] = (texts, torch.tensor(targets, dtype=torch.long))

    modifier_texts: list[str] = []
    modifier_targets: list[tuple[int, int]] = []
    for modifier_index, row in enumerate(meta["factors"]["modifiers"]):
        for phrase in row["heldout_off"]:
            modifier_texts.append(
                f"{phrase}. Decide whether the {row['key'].lower()} modifier is active."
            )
            modifier_targets.append((modifier_index, 0))
        for phrase in row["heldout_on"]:
            modifier_texts.append(
                f"{phrase}. Decide whether the {row['key'].lower()} modifier is active."
            )
            modifier_targets.append((modifier_index, 1))
    result["modifiers"] = (
        modifier_texts,
        torch.tensor(modifier_targets, dtype=torch.long),
    )
    return result


def encode_factor_schemas(
    *,
    meta: dict,
    model,
    tokenizer,
    device: torch.device,
) -> dict:
    cache: dict[str, object] = {
        "schema": FACTOR_CACHE_SCHEMA,
        "private_identity_data": False,
    }
    for category in ("role", "traversal", "direction", "control"):
        rows = list(meta["factors"][category])
        texts = [
            description_text(str(row["key"]), str(row["description"]))
            for row in rows
        ]
        states, mask = encode_texts(
            texts=texts,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=96,
            batch_size=16,
            all_hidden_states=True,
        )
        cache[category] = {
            "keys": [str(row["key"]) for row in rows],
            "token_states": states,
            "token_mask": mask,
        }

    modifier_rows = list(meta["factors"]["modifiers"])
    modifier_texts: list[str] = []
    for row in modifier_rows:
        modifier_texts.extend(
            [
                description_text(
                    f"{row['key']}_OFF",
                    str(row["off_description"]),
                ),
                description_text(
                    f"{row['key']}_ON",
                    str(row["on_description"]),
                ),
            ]
        )
    states, mask = encode_texts(
        texts=modifier_texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=96,
        batch_size=16,
        all_hidden_states=True,
    )
    cache["modifiers"] = {
        "keys": [str(row["key"]) for row in modifier_rows],
        "token_states": states.reshape(
            len(modifier_rows),
            2,
            *states.shape[1:],
        ),
        "token_mask": mask.reshape(
            len(modifier_rows),
            2,
            mask.size(-1),
        ),
    }
    return cache


def encode_factor_holdout_queries(
    *,
    meta: dict,
    model,
    tokenizer,
    device: torch.device,
) -> dict:
    source = factor_holdout_examples(meta)
    result: dict[str, dict] = {}
    for category, (texts, targets) in source.items():
        states, mask = encode_texts(
            texts=texts,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=96,
            batch_size=16,
            all_hidden_states=True,
        )
        result[category] = {
            "states": states,
            "mask": mask,
            "targets": targets,
        }
    return result


def cyclic_batch(
    values: torch.Tensor,
    *,
    step: int,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + step)
    order = values[torch.randperm(values.numel(), generator=generator)]
    if order.numel() >= batch_size:
        return order[:batch_size]
    repeats = (batch_size + order.numel() - 1) // order.numel()
    return order.repeat(repeats)[:batch_size]


def relation_loss(
    matcher: QSRESchemaMatcher,
    *,
    query: torch.Tensor,
    query_mask: torch.Tensor,
    schema_states: torch.Tensor,
    schema_mask: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = matcher(
        query_hidden_states=query,
        query_token_mask=query_mask,
        schema_hidden_states=schema_states,
        schema_token_mask=schema_mask,
    )
    loss = F.cross_entropy(output["logits"], target)
    return loss, output["logits"]


def factor_losses(
    matcher: QSRESchemaMatcher,
    *,
    query: torch.Tensor,
    query_mask: torch.Tensor,
    factor_cache: dict,
    targets: dict[str, torch.Tensor],
    relational_mask: torch.Tensor,
    weights: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    total = query.sum() * 0.0
    parts: dict[str, float] = {}
    for category in ("role", "traversal", "direction"):
        cache = factor_cache[category]
        output = matcher(
            query_hidden_states=query,
            query_token_mask=query_mask,
            schema_hidden_states=cache["token_states"].to(
                query.device, dtype=torch.float32
            ),
            schema_token_mask=cache["token_mask"].to(query.device).bool(),
        )
        if bool(relational_mask.any()):
            loss = F.cross_entropy(
                output["logits"][relational_mask],
                targets[category][relational_mask],
            )
        else:
            loss = output["logits"].sum() * 0.0
        total = total + float(weights[category]) * loss
        parts[category] = float(loss.detach().item())

    cache = factor_cache["control"]
    output = matcher(
        query_hidden_states=query,
        query_token_mask=query_mask,
        schema_hidden_states=cache["token_states"].to(
            query.device, dtype=torch.float32
        ),
        schema_token_mask=cache["token_mask"].to(query.device).bool(),
    )
    control_loss = F.cross_entropy(output["logits"], targets["control"])
    total = total + float(weights["control"]) * control_loss
    parts["control"] = float(control_loss.detach().item())

    modifier_cache = factor_cache["modifiers"]
    modifier_losses = []
    for modifier in range(modifier_cache["token_states"].size(0)):
        output = matcher(
            query_hidden_states=query,
            query_token_mask=query_mask,
            schema_hidden_states=modifier_cache["token_states"][modifier].to(
                query.device, dtype=torch.float32
            ),
            schema_token_mask=modifier_cache["token_mask"][modifier].to(
                query.device
            ).bool(),
        )
        if bool(relational_mask.any()):
            modifier_losses.append(
                F.cross_entropy(
                    output["logits"][relational_mask],
                    targets["modifiers"][relational_mask, modifier],
                )
            )
    modifier_loss = (
        torch.stack(modifier_losses).mean()
        if modifier_losses
        else query.sum() * 0.0
    )
    total = total + float(weights["modifiers"]) * modifier_loss
    parts["modifiers"] = float(modifier_loss.detach().item())
    return total, parts


@torch.no_grad()
def batched_accuracy(
    matcher: QSRESchemaMatcher,
    *,
    query_states: torch.Tensor,
    query_mask: torch.Tensor,
    schema_states: torch.Tensor,
    schema_mask: torch.Tensor,
    target: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    correct = 0
    count = int(target.numel())
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        output = matcher(
            query_hidden_states=query_states[start:stop].to(
                device, dtype=torch.float32
            ),
            query_token_mask=query_mask[start:stop].to(device).bool(),
            schema_hidden_states=schema_states.to(
                device, dtype=torch.float32
            ),
            schema_token_mask=schema_mask.to(device).bool(),
        )
        prediction = output["logits"].argmax(dim=-1).cpu()
        correct += int(prediction.eq(target[start:stop]).sum().item())
    return correct / max(count, 1)


@torch.no_grad()
def evaluate(
    matcher: QSRESchemaMatcher,
    *,
    aux_train_eval: dict,
    aux_holdout: dict,
    aux_train_schema: tuple[torch.Tensor, torch.Tensor],
    aux_full_schema: tuple[torch.Tensor, torch.Tensor],
    production_train: dict,
    production_core_schema: tuple[torch.Tensor, torch.Tensor],
    core_single_indices: torch.Tensor,
    factor_cache: dict,
    factor_holdout: dict,
    batch_size: int,
    device: torch.device,
) -> dict:
    matcher.eval()
    aux_seen_views = []
    for view in ("view0", "view1"):
        aux_seen_views.append(
            batched_accuracy(
                matcher,
                query_states=aux_train_eval[view],
                query_mask=aux_train_eval[f"mask{0 if view == 'view0' else 1}"],
                schema_states=aux_train_schema[0],
                schema_mask=aux_train_schema[1],
                target=aux_train_eval["target"],
                batch_size=batch_size,
                device=device,
            )
        )

    holdout_target = aux_holdout["target"] + aux_train_schema[0].size(0)
    aux_holdout_views = []
    for view in ("view0", "view1"):
        aux_holdout_views.append(
            batched_accuracy(
                matcher,
                query_states=aux_holdout[view],
                query_mask=aux_holdout[f"mask{0 if view == 'view0' else 1}"],
                schema_states=aux_full_schema[0],
                schema_mask=aux_full_schema[1],
                target=holdout_target,
                batch_size=batch_size,
                device=device,
            )
        )

    core_view_scores = []
    core_target = production_train["relation_target"][
        core_single_indices, 0
    ].long()
    for view in (0, 1):
        core_view_scores.append(
            batched_accuracy(
                matcher,
                query_states=production_train["query_hidden_states"][
                    core_single_indices, view
                ],
                query_mask=production_train["query_token_mask"][
                    core_single_indices, view
                ],
                schema_states=production_core_schema[0],
                schema_mask=production_core_schema[1],
                target=core_target,
                batch_size=batch_size,
                device=device,
            )
        )

    factor_scores: dict[str, float] = {}
    for category in ("role", "traversal", "direction", "control"):
        item = factor_holdout[category]
        cache = factor_cache[category]
        factor_scores[category] = batched_accuracy(
            matcher,
            query_states=item["states"],
            query_mask=item["mask"],
            schema_states=cache["token_states"],
            schema_mask=cache["token_mask"],
            target=item["targets"],
            batch_size=batch_size,
            device=device,
        )

    modifier_item = factor_holdout["modifiers"]
    mod_correct = 0
    mod_count = int(modifier_item["targets"].size(0))
    for row in range(mod_count):
        modifier_index = int(modifier_item["targets"][row, 0].item())
        target = modifier_item["targets"][row, 1:2]
        cache = factor_cache["modifiers"]
        score = batched_accuracy(
            matcher,
            query_states=modifier_item["states"][row : row + 1],
            query_mask=modifier_item["mask"][row : row + 1],
            schema_states=cache["token_states"][modifier_index],
            schema_mask=cache["token_mask"][modifier_index],
            target=target,
            batch_size=1,
            device=device,
        )
        mod_correct += int(score == 1.0)
    factor_scores["modifiers"] = mod_correct / max(mod_count, 1)

    return {
        "auxiliary_seen_relation_top1": min(aux_seen_views),
        "auxiliary_holdout_relation_top1": min(aux_holdout_views),
        "production_core_single_relation_top1": min(core_view_scores),
        "factor_accuracy": factor_scores,
        "heldout_factor_macro_accuracy": sum(factor_scores.values())
        / max(len(factor_scores), 1),
    }


def eligible(metrics: dict, threshold: dict) -> bool:
    return (
        metrics["auxiliary_seen_relation_top1"]
        >= threshold["auxiliary_seen_relation_top1"]
        and metrics["auxiliary_holdout_relation_top1"]
        >= threshold["auxiliary_holdout_relation_top1"]
        and metrics["production_core_single_relation_top1"]
        >= threshold["production_core_single_relation_top1"]
        and metrics["heldout_factor_macro_accuracy"]
        >= threshold["heldout_factor_macro_accuracy"]
    )


def score_tuple(metrics: dict) -> tuple[float, ...]:
    return (
        float(metrics["auxiliary_holdout_relation_top1"]),
        float(metrics["production_core_single_relation_top1"]),
        float(metrics["heldout_factor_macro_accuracy"]),
        float(metrics["auxiliary_seen_relation_top1"]),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--meta-config", required=True)
    p.add_argument("--semantic-config", required=True)
    p.add_argument("--semantic-checkpoint", required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--production-cache", required=True)
    p.add_argument("--production-schema-cache", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("closure matcher training requires CUDA")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise SystemExit("refusing to overwrite closure matcher evidence")
    output_dir.mkdir(parents=True)

    plan_path = Path(args.plan)
    meta_path = Path(args.meta_config)
    semantic_config_path = Path(args.semantic_config)
    semantic_checkpoint = Path(args.semantic_checkpoint)
    tokenizer_dir = Path(args.tokenizer_dir)
    production_cache_path = Path(args.production_cache)
    production_schema_path = Path(args.production_schema_cache)

    plan = read_json(plan_path)
    if plan.get("schema") != "alice.eipm.n0.qsre-closure-matcher-plan.v1":
        raise SystemExit("closure matcher plan schema drift")
    meta = read_json(meta_path)
    if meta.get("schema") != "alice.eipm.n0.qsre-closure-schema-meta.v1":
        raise SystemExit("closure schema-meta version drift")
    if meta.get("governance", {}).get("private_identity_data") is not False:
        raise SystemExit("private identity data entered closure schema-meta")

    production = torch.load(production_cache_path, map_location="cpu")
    if production.get("private_identity_data") is not False:
        raise SystemExit("private identity data entered Production cache")
    if production.get("test_present") is not False:
        raise SystemExit("TEST entered Production cache")
    production_schema = torch.load(production_schema_path, map_location="cpu")
    if production_schema.get("private_identity_data") is not False:
        raise SystemExit("private identity data entered Production schema cache")
    if production_schema.get("semantic_checkpoint_sha256") != sha256(
        semantic_checkpoint
    ):
        raise SystemExit("Production schema cache semantic checkpoint drift")

    device = torch.device("cuda")
    semantic_model = load_frozen_semantic_model(
        semantic_config_path=semantic_config_path,
        semantic_checkpoint=semantic_checkpoint,
        device=device,
    )
    tokenizer = load_tokenizer(tokenizer_dir)

    relation_train = list(meta["relation_train"])
    relation_holdout = list(meta["relation_holdout"])
    train_schema_text = [
        description_text(str(row["key"]), str(row["description"]))
        for row in relation_train
    ]
    full_schema_text = train_schema_text + [
        description_text(str(row["key"]), str(row["description"]))
        for row in relation_holdout
    ]
    full_aux_states, full_aux_mask = encode_texts(
        texts=full_schema_text,
        model=semantic_model,
        tokenizer=tokenizer,
        device=device,
        max_length=96,
        batch_size=16,
        all_hidden_states=True,
    )
    train_aux_states = full_aux_states[: len(relation_train)]
    train_aux_mask = full_aux_mask[: len(relation_train)]

    generation = meta["generation"]
    train_v0, train_v1, train_targets = relation_examples(
        relation_train,
        phrase_field="train_phrases",
        seed=int(generation["seed"]),
        examples_per_relation=int(generation["train_examples_per_relation"]),
    )
    seen_v0, seen_v1, seen_targets = relation_examples(
        relation_train,
        phrase_field="heldout_phrases",
        seed=int(generation["seed"]) + 1,
        examples_per_relation=max(
            4,
            int(generation["holdout_examples_per_relation"]) // 2,
        ),
    )
    hold_v0, hold_v1, hold_targets = relation_examples(
        relation_holdout,
        phrase_field="phrases",
        seed=int(generation["seed"]) + 2,
        examples_per_relation=int(generation["holdout_examples_per_relation"]),
    )

    train_q0, train_qmask = encode_texts(
        texts=train_v0,
        model=semantic_model,
        tokenizer=tokenizer,
        device=device,
        max_length=96,
        batch_size=16,
        all_hidden_states=True,
    )
    train_q1, train_qmask1 = encode_texts(
        texts=train_v1,
        model=semantic_model,
        tokenizer=tokenizer,
        device=device,
        max_length=96,
        batch_size=16,
        all_hidden_states=True,
    )
    if not torch.equal(train_qmask, train_qmask1):
        # Token lengths may differ between views; masks need not be identical.
        pass
    seen_q0, seen_mask0 = encode_texts(
        texts=seen_v0,
        model=semantic_model,
        tokenizer=tokenizer,
        device=device,
        max_length=96,
        batch_size=16,
        all_hidden_states=True,
    )
    seen_q1, seen_mask1 = encode_texts(
        texts=seen_v1,
        model=semantic_model,
        tokenizer=tokenizer,
        device=device,
        max_length=96,
        batch_size=16,
        all_hidden_states=True,
    )
    hold_q0, hold_mask0 = encode_texts(
        texts=hold_v0,
        model=semantic_model,
        tokenizer=tokenizer,
        device=device,
        max_length=96,
        batch_size=16,
        all_hidden_states=True,
    )
    hold_q1, hold_mask1 = encode_texts(
        texts=hold_v1,
        model=semantic_model,
        tokenizer=tokenizer,
        device=device,
        max_length=96,
        batch_size=16,
        all_hidden_states=True,
    )

    factor_cache = encode_factor_schemas(
        meta=meta,
        model=semantic_model,
        tokenizer=tokenizer,
        device=device,
    )
    factor_holdout = encode_factor_holdout_queries(
        meta=meta,
        model=semantic_model,
        tokenizer=tokenizer,
        device=device,
    )

    factor_cache_path = output_dir / "factor_schema_cache.pt"
    torch.save(factor_cache, factor_cache_path)

    del semantic_model
    torch.cuda.empty_cache()

    semantic_dim = int(production["train"]["query_hidden_states"].size(-1))
    hidden_states = int(production["train"]["query_hidden_states"].size(2))
    matcher = QSRESchemaMatcher(
        semantic_dim=semantic_dim,
        model_dim=semantic_dim,
        num_hidden_states=hidden_states,
    ).to(device)

    stage = plan["training"]
    seed = int(stage["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    optimizer = torch.optim.AdamW(
        matcher.parameters(),
        lr=float(stage["learning_rate"]),
        weight_decay=float(stage["weight_decay"]),
    )

    train_split = production["train"]
    relation_mask = train_split["relation_target_mask"].bool()
    control = train_split["control_target"].long()
    core_single = relation_mask.sum(dim=-1).eq(1) & control.eq(CONTROL_RELATIONAL)
    core_single_indices = core_single.nonzero(as_tuple=False).flatten()
    if core_single_indices.numel() == 0:
        raise SystemExit("no single-relation core Production rows for matcher training")
    factor_indices = torch.arange(len(train_split["ids"]))

    core_count = len(production_schema["core_train_relation_keys"])
    if production_schema["relation_keys"][:core_count] != production_schema["core_train_relation_keys"]:
        raise SystemExit("Production core relation prefix drift")
    core_schema_states = production_schema["token_states"][:core_count]
    core_schema_mask = production_schema["token_mask"][:core_count]

    aux_indices = torch.arange(train_targets.numel())
    history: list[dict] = []
    selected = None
    best = None
    best_score = None
    batch_size = int(stage["batch_size"])
    max_steps = int(stage["max_steps"])
    eval_every = int(stage["eval_every"])

    for step in range(1, max_steps + 1):
        matcher.train()
        optimizer.zero_grad(set_to_none=True)

        aux_idx = cyclic_batch(
            aux_indices,
            step=step,
            batch_size=batch_size,
            seed=seed,
        )
        core_idx = cyclic_batch(
            core_single_indices,
            step=step,
            batch_size=batch_size,
            seed=seed + 1000,
        )
        factor_idx = cyclic_batch(
            factor_indices,
            step=step,
            batch_size=batch_size,
            seed=seed + 2000,
        )
        view = step % 2

        aux_query_bank = train_q0 if view == 0 else train_q1
        aux_mask_bank = train_qmask if view == 0 else train_qmask1
        aux_loss, aux_logits = relation_loss(
            matcher,
            query=aux_query_bank[aux_idx].to(device, dtype=torch.float32),
            query_mask=aux_mask_bank[aux_idx].to(device).bool(),
            schema_states=train_aux_states.to(device, dtype=torch.float32),
            schema_mask=train_aux_mask.to(device).bool(),
            target=train_targets[aux_idx].to(device),
        )

        core_query = train_split["query_hidden_states"][
            core_idx, view
        ].to(device, dtype=torch.float32)
        core_query_mask = train_split["query_token_mask"][
            core_idx, view
        ].to(device).bool()
        core_target = train_split["relation_target"][core_idx, 0].to(
            device
        ).long()
        core_loss, core_logits = relation_loss(
            matcher,
            query=core_query,
            query_mask=core_query_mask,
            schema_states=core_schema_states.to(device, dtype=torch.float32),
            schema_mask=core_schema_mask.to(device).bool(),
            target=core_target,
        )

        factor_query = train_split["query_hidden_states"][
            factor_idx, view
        ].to(device, dtype=torch.float32)
        factor_query_mask = train_split["query_token_mask"][
            factor_idx, view
        ].to(device).bool()
        factor_target = {
            "role": train_split["role_target"][factor_idx].to(device).long(),
            "traversal": train_split["traversal_target"][factor_idx].to(device).long(),
            "direction": train_split["direction_target"][factor_idx].to(device).long(),
            "control": train_split["control_target"][factor_idx].to(device).long(),
            "modifiers": train_split["modifier_target"][factor_idx].to(device).long(),
        }
        relational = train_split["control_target"][factor_idx].to(device).eq(
            CONTROL_RELATIONAL
        )
        factor_total, factor_parts = factor_losses(
            matcher,
            query=factor_query,
            query_mask=factor_query_mask,
            factor_cache=factor_cache,
            targets=factor_target,
            relational_mask=relational,
            weights={
                "role": stage["role_weight"],
                "traversal": stage["traversal_weight"],
                "direction": stage["direction_weight"],
                "control": stage["control_weight"],
                "modifiers": stage["modifier_weight"],
            },
        )

        other_view = 1 - view
        pair_output = matcher(
            query_hidden_states=train_split["query_hidden_states"][
                core_idx, other_view
            ].to(device, dtype=torch.float32),
            query_token_mask=train_split["query_token_mask"][
                core_idx, other_view
            ].to(device).bool(),
            schema_hidden_states=core_schema_states.to(
                device, dtype=torch.float32
            ),
            schema_token_mask=core_schema_mask.to(device).bool(),
        )
        p = F.log_softmax(core_logits, dim=-1)
        q = F.softmax(pair_output["logits"], dim=-1)
        pair_loss = F.kl_div(p, q, reduction="batchmean")

        loss = (
            float(stage["auxiliary_relation_weight"]) * aux_loss
            + float(stage["core_relation_weight"]) * core_loss
            + factor_total
            + float(stage["pair_consistency_weight"]) * pair_loss
        )
        if not torch.isfinite(loss):
            raise RuntimeError("closure matcher nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            matcher.parameters(),
            float(stage["gradient_clip"]),
        )
        optimizer.step()

        if step % eval_every == 0 or step == max_steps:
            metrics = evaluate(
                matcher,
                aux_train_eval={
                    "view0": seen_q0,
                    "view1": seen_q1,
                    # Views can tokenize to different widths, so evaluate stores
                    # their own masks below.
                    "mask0": seen_mask0,
                    "mask1": seen_mask1,
                    "target": seen_targets,
                },
                aux_holdout={
                    "view0": hold_q0,
                    "view1": hold_q1,
                    "mask0": hold_mask0,
                    "mask1": hold_mask1,
                    "target": hold_targets,
                },
                aux_train_schema=(train_aux_states, train_aux_mask),
                aux_full_schema=(full_aux_states, full_aux_mask),
                production_train=train_split,
                production_core_schema=(core_schema_states, core_schema_mask),
                core_single_indices=core_single_indices,
                factor_cache=factor_cache,
                factor_holdout=factor_holdout,
                batch_size=batch_size,
                device=device,
            )
            passed = eligible(metrics, plan["eligibility"])
            record = {
                "step": step,
                "train_loss": float(loss.detach().item()),
                "auxiliary_relation_loss": float(aux_loss.detach().item()),
                "core_relation_loss": float(core_loss.detach().item()),
                "pair_consistency_loss": float(pair_loss.detach().item()),
                "factor_loss": factor_parts,
                "metrics": metrics,
                "eligible": passed,
            }
            history.append(record)
            print(
                "CLOSURE_MATCHER_EVAL=" + json.dumps(record, sort_keys=True),
                flush=True,
            )

            checkpoint_path = (
                output_dir
                / f"step-{step:08d}"
                / "qsre_closure_matcher.pt"
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema": CHECKPOINT_SCHEMA,
                    "step": step,
                    "matcher": matcher.state_dict(),
                    "matcher_report": matcher.parameter_report(),
                    "plan_sha256": sha256(plan_path),
                    "meta_config_sha256": sha256(meta_path),
                    "semantic_checkpoint_sha256": sha256(semantic_checkpoint),
                    "production_cache_sha256": sha256(production_cache_path),
                    "production_schema_cache_sha256": sha256(production_schema_path),
                    "factor_schema_cache_sha256": sha256(factor_cache_path),
                    "private_identity_gradient": False,
                },
                checkpoint_path,
            )
            checkpoint_sha = sha256(checkpoint_path)
            score = score_tuple(metrics)
            if best_score is None or score > best_score:
                best_score = score
                best = {
                    "step": step,
                    "checkpoint_sha256": checkpoint_sha,
                    "metrics": metrics,
                }
            if passed:
                selected = {
                    "step": step,
                    "checkpoint_sha256": checkpoint_sha,
                    "metrics": metrics,
                }
                break

    result = {
        "schema": RESULT_SCHEMA,
        "status": (
            "PASS_QSRE_CLOSURE_SCHEMA_MATCHER"
            if selected
            else "FAIL_QSRE_CLOSURE_SCHEMA_MATCHER"
        ),
        "selected": selected,
        "best_observed": best,
        "history": history,
        "matcher_report": matcher.parameter_report(),
        "plan_sha256": sha256(plan_path),
        "meta_config_sha256": sha256(meta_path),
        "semantic_checkpoint_sha256": sha256(semantic_checkpoint),
        "production_cache_sha256": sha256(production_cache_path),
        "production_schema_cache_sha256": sha256(production_schema_path),
        "factor_schema_cache_sha256": sha256(factor_cache_path),
        "production_open_schema_query_labels_used": False,
        "production_open_schema_descriptions_used_in_gradient": False,
        "final_only_relation_descriptions_used_in_gradient": False,
        "auxiliary_holdout_relation_descriptions_used_in_gradient": False,
        "private_identity_gradient": False,
        "automatic_rerun": False,
        "p2_authorized": bool(selected),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "CLOSURE_MATCHER_RESULT=" + json.dumps(result, sort_keys=True),
        flush=True,
    )
    if not selected:
        raise SystemExit(41)


if __name__ == "__main__":
    main()
