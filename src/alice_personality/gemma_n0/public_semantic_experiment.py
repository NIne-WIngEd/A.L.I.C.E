"""Bounded public semantic readout experiment, always unqualified.

Plans are written before forwards/training. Only admitted FewRel TRAIN/DEV
payloads are read; final data, class IDs as inputs and publisher gradients are
excluded. A small operating subset is not a personality capability ceiling.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import gc
import json
import math
import os
from pathlib import Path
import sys
import time


PLAN_SCHEMA = "alice-personality-gemma-n0-public-semantic-plan-v1"
RECEIPT_SCHEMA = "alice-personality-gemma-n0-public-semantic-experiment-v1"
CACHE_SCHEMA = "alice-personality-gemma-n0-public-feature-cache-v1"
CHECKPOINT_SCHEMA = "alice-personality-gemma-n0-semantic-checkpoint-v1"
EXPORT_SCHEMA = "alice-personality-gemma-n0-semantic-export-v1"
STYLE_PREFIXES = {
    "helpful": "Please provide a helpful answer.\n",
    "flattering": "Your answer will be wonderful and impressive.\n",
}
REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_PATHS = {
    "semantic_readout.py": Path(__file__).with_name("semantic_readout.py"),
    "public_semantic_experiment.py": Path(__file__),
    "run_public_semantic_experiment.py": REPO_ROOT / "scripts/eipm/gemma_n0/run_public_semantic_experiment.py",
    "public_fewrel.py": Path(__file__).with_name("public_fewrel.py"),
    "backbone.py": Path(__file__).with_name("backbone.py"),
    "preparation.py": Path(__file__).with_name("preparation.py"),
}


class ExperimentError(ValueError):
    """Experiment evidence failed public-input, state or source binding."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExperimentError(message)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(path: Path) -> str:
    before = path.stat()
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    after = path.stat()
    _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
             (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "bound file changed while hashing")
    return digest.hexdigest()


def _file(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    _require(path.is_absolute() and path.is_file() and not path.is_symlink()
             and path == path.resolve(strict=True), "bound file must be absolute, regular and nonsymlink")
    return path


def _read(path: Path) -> dict:
    from .public_fewrel import _json
    value = _json(path.read_bytes())
    _require(isinstance(value, dict), "bound JSON must be an object")
    return value


def _code() -> list[dict]:
    return [{"name": name, "sha256": _digest(_file(path))} for name, path in sorted(CODE_PATHS.items())]


def _verify_source(path: Path) -> dict:
    from .public_fewrel import verify_fewrel_admission
    return verify_fewrel_admission(path)


def _verify_preparation(path: Path) -> dict:
    from .preparation import verify_prepared
    return verify_prepared(path)


def _seal(value: dict) -> dict:
    value = json.loads(_canonical(value))
    value["receipt_sha256"] = sha256(_canonical(value)).hexdigest()
    return value


def _new_path(raw: str | Path, forbidden: list[Path]) -> Path:
    path = Path(raw).expanduser()
    _require(path.is_absolute() and not path.is_symlink() and not path.exists()
             and path.parent.is_dir() and path.parent == path.parent.resolve(strict=True),
             "output must use a fresh absolute path with an existing nonsymlink parent")
    _require(all(root != path and root not in path.parents for root in forbidden),
             "experiment evidence must remain outside source/model artifacts")
    return path


def _write_json(path: Path, value: dict) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_canonical(value) + b"\n")


def _verify_seal(value: dict) -> dict:
    record = dict(value)
    digest = record.pop("receipt_sha256", None)
    _require(isinstance(digest, str) and sha256(_canonical(record)).hexdigest() == digest,
             "experiment/plan receipt digest differs")
    return record


def _recipe(value: dict) -> dict:
    required = {"train_rows_per_family", "dev_rows_per_family", "style_rows_per_dev_family",
                "width", "steps", "seed", "learning_rate", "weight_decay"}
    _require(isinstance(value, dict) and set(value) == required, "recipe fields differ")
    for key in ("train_rows_per_family", "dev_rows_per_family", "width", "steps"):
        _require(type(value[key]) is int and value[key] > 0, f"recipe {key} must be positive")
    _require(value["steps"] >= 2, "at least two steps are required to execute a post-checkpoint resume update")
    for key in ("style_rows_per_dev_family", "seed"):
        _require(type(value[key]) is int and value[key] >= 0, f"recipe {key} must be nonnegative")
    _require(value["style_rows_per_dev_family"] <= value["dev_rows_per_family"], "style subset exceeds DEV subset")
    for key in ("learning_rate", "weight_decay"):
        _require(type(value[key]) in {int, float} and math.isfinite(value[key])
                 and (value[key] > 0 if key == "learning_rate" else value[key] >= 0), f"invalid {key}")
    return dict(value)


def _source_input(row: dict, *, prefix: str = "") -> dict:
    # Sentence is copied complete; entity IDs/types, relation IDs and targets
    # are metadata and never used to form an input string.
    sentence = row["sentence"]
    head, tail = row["head"]["text"], row["tail"]["text"]
    before_head = prefix + sentence + "\nHead entity: "
    before_tail = before_head + head + "\nTail entity: "
    text = before_tail + tail + "\nQuestion: How does the head entity relate to the tail entity?"
    return {"text": text, "spans": {"source_mask": [len(prefix), len(prefix) + len(sentence)],
            "head_mask": [len(before_head), len(before_head) + len(head)],
            "tail_mask": [len(before_tail), len(before_tail) + len(tail)],
            "query_mask": [len(prefix) + len(sentence), len(text)]}}


def _selection(source: dict, recipe: dict) -> list[dict]:
    files = {row["kind"]: _file(row["path"]) for row in source["files"]}
    bank = _read(files["bank"])["relations"]
    groups = defaultdict(list)
    all_train_sentence_digests = set()
    digest = sha256()
    with files["rows"].open("rb") as stream:
        for payload in stream:
            digest.update(payload)
            from .public_fewrel import _json
            row = _json(payload)
            _require(isinstance(row, dict) and row.get("split") in {"train", "dev"}, "selection encountered forbidden split")
            if row["split"] == "train":
                from .public_fewrel import _normalized
                all_train_sentence_digests.add(sha256(_normalized(row["sentence"]).encode()).hexdigest())
            group = groups[(row["split"], row["target_relation_key"])]
            group.append((sha256(("personality-gemma-public-subset-v1:" + row["id"]).encode()).hexdigest(), row))
            group.sort(key=lambda item: item[0])
            del group[recipe[f'{row["split"]}_rows_per_family']:]
    expected_digest = next(row["sha256"] for row in source["files"] if row["kind"] == "rows")
    _require(digest.hexdigest() == expected_digest, "admitted rows changed during deterministic subset selection")
    examples = []
    for (split, family), rows in sorted(groups.items()):
        _require(len(rows) == recipe[f"{split}_rows_per_family"], "requested subset exceeds available family records")
        for rank, (_, row) in enumerate(rows):
            descriptions = [bank[key]["semantic_text"] for key in row["candidate_relation_keys"]]
            source_input = _source_input(row)
            forbidden = [row["id"], *row["candidate_relation_keys"], row["head"]["type"], row["tail"]["type"]]
            # IDs may exist as literal natural text only by coincidence, but
            # this first experiment rejects that ambiguity before execution.
            import re
            _require(not any(value and re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", source_input["text"])
                             for value in forbidden), "forbidden target/opaque metadata occurs in model input")
            _require(all(not re.search(r"\bP\d+\b", text) for text in descriptions), "description exposes opaque relation ID")
            variants = ({name: _source_input(row, prefix=prefix) for name, prefix in STYLE_PREFIXES.items()}
                        if split == "dev" and rank < recipe["style_rows_per_dev_family"] else {})
            from .public_fewrel import _normalized
            sentence_unseen = (sha256(_normalized(row["sentence"]).encode()).hexdigest()
                               not in all_train_sentence_digests)
            examples.append({"id_metadata": row["id"], "split": split, "family_metadata": family,
                             "source": source_input, "descriptions": descriptions,
                             "target_index_metadata": row["target_candidate_index"], "style_variants": variants,
                             "row_sha256": sha256(_canonical(row)).hexdigest(),
                             "sentence_unseen_in_all_train": sentence_unseen})
    counts = Counter(row["split"] for row in examples)
    _require(counts["train"] > 0 and counts["dev"] > 0, "experiment requires public TRAIN and DEV")
    return examples


def create_plan(*, source_admission: str | Path, preparation_receipt: str | Path,
                output_plan: str | Path, recipe: dict) -> dict:
    recipe = _recipe(recipe)
    source_path, prepared_path = _file(source_admission), _file(preparation_receipt)
    source_raw, prepared_raw, code = _digest(source_path), _digest(prepared_path), _code()
    source, prepared = _verify_source(source_path), _verify_preparation(prepared_path)
    _require(prepared["runtime"]["dtype"] == "bfloat16", "this experiment requires frozen BF16 source states")
    source_dir = Path(source["files"][0]["path"]).parent
    output = _new_path(output_plan, [source_dir, Path(prepared["snapshot_path"])])
    examples = _selection(source, recipe)
    inputs = {_canonical(example["source"]) for example in examples}
    for example in examples:
        inputs.update(_canonical({"text": description, "spans": {}}) for description in example["descriptions"])
        inputs.update(_canonical(value) for value in example["style_variants"].values())
    _require(source_raw == _digest(source_path) and prepared_raw == _digest(prepared_path) and code == _code(),
             "plan inputs/code changed during selection")
    plan = _seal({"schema": PLAN_SCHEMA, "state": "PRECOMMITTED_PUBLIC_READOUT_UNQUALIFIED",
                  "source_admission_path": str(source_path), "source_admission_file_sha256": source_raw,
                  "source_admission_sha256": source["receipt_sha256"],
                  "preparation_path": str(prepared_path), "preparation_file_sha256": prepared_raw,
                  "preparation_sha256": prepared["receipt_sha256"], "model_geometry": prepared["model_geometry"],
                  "runtime": prepared["runtime"], "code": code, "recipe": recipe, "examples": examples,
                  "unique_complete_feature_inputs": len(inputs), "source_counts_define_capability_ceiling": False,
                  "original_source_sentence_only_overlap": source["statistics"]["normalized_sentence_only_overlap"],
                  "dev_sentence_unseen_metric_rule": "NFKC/casefold/whitespace sentence absent from ALL original TRAIN sentences; report additionally without dropping source rows",
                  "selection_rule": "SHA256 personality-gemma-public-subset-v1:row_id ascending per TRAIN/DEV family",
                  "final_payload_opened": False, "private_identity_data": False, "n0_approved": False,
                  "qualification": "unqualified; operating experiment only"})
    _write_json(output, plan)
    return plan


def _load_plan(path: Path) -> tuple[dict, dict, dict]:
    plan = _read(path)
    _verify_seal(plan)
    _require(plan.get("schema") == PLAN_SCHEMA and plan.get("code") == _code(), "precommitted plan/code differs")
    _require(plan.get("state") == "PRECOMMITTED_PUBLIC_READOUT_UNQUALIFIED"
             and plan.get("final_payload_opened") is False and plan.get("private_identity_data") is False
             and plan.get("n0_approved") is False and plan.get("source_counts_define_capability_ceiling") is False,
             "plan public/unqualified boundary differs")
    _recipe(plan["recipe"])
    source_path, prepared_path = _file(plan["source_admission_path"]), _file(plan["preparation_path"])
    _require(_digest(source_path) == plan["source_admission_file_sha256"]
             and _digest(prepared_path) == plan["preparation_file_sha256"], "precommitted receipt file changed")
    source, prepared = _verify_source(source_path), _verify_preparation(prepared_path)
    _require(source["receipt_sha256"] == plan["source_admission_sha256"]
             and prepared["receipt_sha256"] == plan["preparation_sha256"]
             and prepared["model_geometry"] == plan["model_geometry"]
             and prepared["runtime"] == plan["runtime"], "precommitted source/model/runtime differs")
    _require(type(plan["original_source_sentence_only_overlap"]) is int
             and plan["original_source_sentence_only_overlap"] == source["statistics"]["normalized_sentence_only_overlap"]
             and plan["dev_sentence_unseen_metric_rule"] ==
             "NFKC/casefold/whitespace sentence absent from ALL original TRAIN sentences; report additionally without dropping source rows",
             "precommitted sentence independence metric differs")
    _require(_canonical(_selection(source, plan["recipe"])) == _canonical(plan["examples"]),
             "precommitted complete inputs/subset/targets differ")
    return plan, source, prepared


def _tokenize(tokenizer, item: dict, torch) -> tuple[dict, dict]:
    encoded = tokenizer(item["text"], truncation=False, padding=False, add_special_tokens=True,
                        return_tensors="pt", return_attention_mask=True,
                        return_token_type_ids=False, return_offsets_mapping=True)
    _require(set(encoded) == {"input_ids", "attention_mask", "offset_mapping"}, "raw tokenizer outputs differ")
    ids, mask, offsets = encoded["input_ids"], encoded["attention_mask"], encoded["offset_mapping"]
    _require(isinstance(ids, torch.Tensor) and ids.dtype == torch.int64 and ids.ndim == 2
             and ids.shape[0] == 1 and ids.shape[1] > 0 and mask.shape == ids.shape
             and bool(((mask == 0) | (mask == 1)).all()) and bool(mask.bool().any())
             and offsets.shape == (*ids.shape, 2), "raw tokenization/masks/offsets differ")
    duplicate = tokenizer(item["text"], truncation=False, padding=False, add_special_tokens=True,
                          return_tensors="pt", return_attention_mask=True, return_token_type_ids=False)
    _require(set(duplicate) == {"input_ids", "attention_mask"}
             and torch.equal(ids, duplicate["input_ids"]) and torch.equal(mask, duplicate["attention_mask"]),
             "tokenizer failed exact complete-source retokenization")
    role_masks = {}
    for name, (start, stop) in item["spans"].items():
        _require(name in {"source_mask", "query_mask", "head_mask", "tail_mask"}
                 and type(start) is int and type(stop) is int and 0 <= start < stop <= len(item["text"]),
                 "role/source character spans differ")
        selected = ((offsets[0, :, 1] > start) & (offsets[0, :, 0] < stop)
                    & (offsets[0, :, 1] > offsets[0, :, 0]) & mask[0].bool())
        _require(bool(selected.any()), "role/source span has no preserved source tokens")
        role_masks[name] = selected
    return {"input_ids": ids.cpu(), "attention_mask": mask.cpu()}, role_masks


def _token_binding(encoded: dict, masks: dict) -> dict:
    return {"input_ids": encoded["input_ids"][0].tolist(),
            "attention_mask": encoded["attention_mask"][0].bool().tolist(),
            "role_masks": {key: value.tolist() for key, value in sorted(masks.items())}}


def _cache_bank(*, item: dict, cache_root: Path, binding: dict, tokenizer,
                torch, geometry: dict, provider_factory, provider_holder: list,
                expected_token_sha256: str) -> tuple[object, dict]:
    from .semantic_readout import LayerTokenBank, validate_bank
    encoded, masks = _tokenize(tokenizer, item, torch)
    _require(encoded["input_ids"].shape[1] <= geometry["max_position_embeddings"],
             "complete input exceeds actual source context; truncation is forbidden")
    token_digest = sha256(_canonical(_token_binding(encoded, masks))).hexdigest()
    _require(token_digest == expected_token_sha256, "raw tokens differ from pre-forward feature budget")
    raw_binding = {"experiment_feature_binding": binding, "complete_public_input_sha256": sha256(_canonical(item)).hexdigest(),
                   "raw_token_binding_sha256": token_digest}
    key = sha256(_canonical(raw_binding)).hexdigest()
    metadata_path, tensor_path = cache_root / f"{key}.json", cache_root / f"{key}.pt"
    created = False
    if metadata_path.exists():
        metadata = _read(_file(metadata_path))
        _verify_seal(metadata)
        _require(metadata.get("schema") == CACHE_SCHEMA and metadata.get("binding") == raw_binding
                 and metadata.get("tensor_sha256") == _digest(_file(tensor_path)), "immutable public cache binding differs")
        payload = torch.load(tensor_path, weights_only=True, map_location="cpu")
    else:
        _require(not tensor_path.exists() and not tensor_path.is_symlink(), "preserve partial public cache evidence")
        if not provider_holder:
            provider_holder.append(provider_factory())
        start = time.monotonic()
        features = provider_holder[0].extract_features(**encoded)
        _require(torch.equal(features.attention_mask.cpu(), encoded["attention_mask"].bool()), "provider changed source mask")
        states = features.all_hidden_states
        _require(isinstance(states, tuple) and len(states) == geometry["hidden_state_count"]
                 and all(state.shape == (*encoded["input_ids"].shape, geometry["hidden_size"])
                         and state.dtype == torch.bfloat16 and not state.requires_grad and state.grad_fn is None
                         and bool(torch.isfinite(state).all()) for state in states)
                 and torch.equal(features.hidden_states, states[-1]), "provider lacks complete actual frozen state chain")
        payload = {"layers": torch.stack([state[0].detach().cpu() for state in states]),
                   "input_ids": encoded["input_ids"][0], "attention_mask": encoded["attention_mask"][0].bool(), **masks}
        descriptor = os.open(tensor_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
        metadata = _seal({"schema": CACHE_SCHEMA, "binding": raw_binding, "tensor_sha256": _digest(tensor_path),
                          "state_count": geometry["hidden_state_count"], "hidden_size": geometry["hidden_size"],
                          "source_tokens": encoded["input_ids"].shape[1], "dtype": "bfloat16",
                          "complete_input": True, "public_only": True, "upstream_frozen": True,
                          "forward_seconds": time.monotonic() - start,
                          "logical_feature_bytes": payload["layers"].numel() * payload["layers"].element_size()})
        _write_json(metadata_path, metadata)
        created = True
    _require(isinstance(payload, dict) and set(payload) == {"layers", "input_ids", "attention_mask", *masks},
             "cached tensor fields differ")
    bank = LayerTokenBank(**payload)
    validate_bank(bank, state_count=geometry["hidden_state_count"], hidden_size=geometry["hidden_size"],
                  source=bool(item["spans"]))
    _require(_canonical(_token_binding({"input_ids": bank.input_ids[None], "attention_mask": bank.attention_mask[None]},
                                     {name: getattr(bank, name) for name in masks})) ==
             _canonical(_token_binding(encoded, masks)), "cached IDs/masks differ from exact raw retokenization")
    return bank, {"key": key, "metadata_path": str(metadata_path), "metadata_sha256": _digest(metadata_path),
                  "tensor_path": str(tensor_path), "tensor_sha256": metadata["tensor_sha256"], "created": created,
                  "forward_seconds": metadata["forward_seconds"], "logical_feature_bytes": metadata["logical_feature_bytes"]}


def _equal(left, right, torch) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(_equal(left[key], right[key], torch) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_equal(a, b, torch) for a, b in zip(left, right))
    return left == right


def _save_torch(path: Path, value: dict, torch) -> str:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        torch.save(value, stream)
    return _digest(path)


def _checkpoint(model, optimizer, sampler, step: int, binding: dict, torch) -> dict:
    return {"schema": CHECKPOINT_SCHEMA, "binding": binding, "geometry": model.geometry,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
            "torch_cpu_rng": torch.get_rng_state(), "sampler_rng": sampler.get_state()}


def restore_checkpoint(path: Path, *, model, optimizer, sampler, binding: dict,
                       expected_file_sha256: str, torch) -> int:
    path = _file(path)
    _require(_digest(path) == expected_file_sha256, "checkpoint bytes differ from the external pin")
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    _require(isinstance(checkpoint, dict) and set(checkpoint) ==
             {"schema", "binding", "geometry", "model", "optimizer", "step", "torch_cpu_rng", "sampler_rng"}
             and checkpoint["schema"] == CHECKPOINT_SCHEMA and checkpoint["binding"] == binding
             and checkpoint["geometry"] == model.geometry and type(checkpoint["step"]) is int
             and checkpoint["step"] >= 0, "checkpoint source/code/plan/recipe binding differs")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_cpu_rng"])
    sampler.set_state(checkpoint["sampler_rng"])
    return checkpoint["step"]


def load_semantic_export(path: Path, *, binding: dict, expected_file_sha256: str, torch):
    """Reload only an externally pinned first-party export for this protocol."""
    from .semantic_readout import SemanticReadout
    path = _file(path)
    _require(_digest(path) == expected_file_sha256, "export bytes differ from the external pin")
    exported = torch.load(path, weights_only=True, map_location="cpu")
    _require(isinstance(exported, dict) and set(exported) == {"schema", "binding", "geometry", "model"}
             and exported["schema"] == EXPORT_SCHEMA and exported["binding"] == binding
             and isinstance(exported["geometry"], dict)
             and set(exported["geometry"]) == {"state_count", "hidden_size", "width"}, "export binding differs")
    model = SemanticReadout(**exported["geometry"])
    _require(isinstance(exported["model"], dict)
             and all(isinstance(value, torch.Tensor) and value.dtype == torch.float32
                     and bool(torch.isfinite(value).all()) for value in exported["model"].values()),
             "export contains nonfinite or non-FP32 learned parameters")
    model.load_state_dict(exported["model"], strict=True)
    return model.eval()


def _train_steps(model, optimizer, sampler, examples: list, *, steps: int, torch) -> list[float]:
    losses = []
    model.train()
    for _ in range(steps):
        example = examples[int(torch.randint(len(examples), (), generator=sampler))]
        _require(example["split"] == "train", "DEV cannot enter optimizer updates")
        optimizer.zero_grad(set_to_none=True)
        logits = model(example["source"], example["candidates"])
        loss = torch.nn.functional.cross_entropy(logits[None], torch.tensor([example["target"]]))
        _require(bool(torch.isfinite(loss)), "training loss is nonfinite")
        loss.backward()
        _require(all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                     for parameter in model.parameters()), "readout gradients are missing/nonfinite")
        optimizer.step()
        _require(all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()), "readout parameters became nonfinite")
        losses.append(float(loss.detach()))
    return losses


def _evaluate(scorer, examples: list, torch) -> dict:
    if not examples:
        return {"rows": 0, "mean_cross_entropy": None, "accuracy": None,
                "macro_family_accuracy": None, "family_count": 0}
    losses, correct, families = [], [], defaultdict(list)
    with torch.no_grad():
        for example in examples:
            scores = scorer(example["source"], example["candidates"])
            loss = float(torch.nn.functional.cross_entropy(scores[None], torch.tensor([example["target"]])))
            _require(bool(torch.isfinite(scores).all()) and math.isfinite(loss), "evaluation loss/scores are nonfinite")
            hit = int(int(scores.argmax()) == example["target"])
            losses.append(loss)
            correct.append(hit)
            families[example["family"]].append(hit)
    return {"rows": len(examples), "mean_cross_entropy": sum(losses) / len(losses),
            "accuracy": sum(correct) / len(correct),
            "macro_family_accuracy": sum(sum(hits) / len(hits) for hits in families.values()) / len(families),
            "family_count": len(families)}


def _permutations(scorer, examples: list, torch) -> dict:
    largest = 0.0
    with torch.no_grad():
        for example in examples:
            baseline = scorer(example["source"], example["candidates"])
            permuted = scorer(example["source"], list(reversed(example["candidates"]))).flip(0)
            difference = float((baseline - permuted).abs().max())
            largest = max(largest, difference)
            _require(torch.allclose(baseline, permuted, atol=1e-6, rtol=1e-6), "candidate order changed semantic readout scores")
    return {"rows_checked": len(examples), "max_aligned_logit_difference": largest,
            "candidate_order_is_identity_input": False, "tolerance": {"atol": 1e-6, "rtol": 1e-6}}


def _style_diagnostics(scorer, examples: list, torch) -> dict:
    results = {}
    with torch.no_grad():
        for style in STYLE_PREFIXES:
            flips, shifts, baseline_hits, changed_hits = [], [], [], []
            for example in examples:
                if style not in example["styles"]:
                    continue
                original = scorer(example["source"], example["candidates"])
                altered = scorer(example["styles"][style], example["candidates"])
                p, q = original.softmax(0), altered.softmax(0)
                mixture = (p + q) / 2
                js = ((p * (p.clamp_min(1e-12).log() - mixture.clamp_min(1e-12).log())).sum()
                      + (q * (q.clamp_min(1e-12).log() - mixture.clamp_min(1e-12).log())).sum()) / 2
                flips.append(int(int(original.argmax()) != int(altered.argmax())))
                shifts.append(float(js))
                baseline_hits.append(int(int(original.argmax()) == example["target"]))
                changed_hits.append(int(int(altered.argmax()) == example["target"]))
            results[style] = {"rows": len(flips), "top_candidate_flips": sum(flips),
                              "mean_js_divergence": sum(shifts) / len(shifts) if shifts else None,
                              "baseline_correct": sum(baseline_hits), "style_correct": sum(changed_hits)}
    return {"variants": results, "target_annotation_unchanged": True,
            "meaning": "public irrelevant-helper-wording sensitivity; no neutral-personality claim"}


def _runtime(prepared: dict):
    import torch
    import transformers
    from .backbone import load_prepared_gemma_n0
    expected = prepared["runtime"]
    _require(expected == {"backend": "transformers", "dtype": "bfloat16",
                          "torch_version": str(torch.__version__), "transformers_version": str(transformers.__version__)},
             "actual runtime differs from preparation")
    processor = transformers.AutoProcessor.from_pretrained(prepared["snapshot_path"],
                  local_files_only=True, trust_remote_code=False)
    return torch, processor.tokenizer, lambda: load_prepared_gemma_n0(prepared["receipt_path"], device="cpu", dtype="bfloat16")


def run_experiment(*, plan_path: str | Path, output_directory: str | Path,
                   cache_directory: str | Path) -> dict:
    start = time.monotonic()
    plan_file = _file(plan_path)
    plan_digest = _digest(plan_file)
    plan, source, prepared = _load_plan(plan_file)
    prepared = {**prepared, "receipt_path": plan["preparation_path"]}
    forbidden = [Path(source["files"][0]["path"]).parent, Path(prepared["snapshot_path"])]
    output = _new_path(output_directory, forbidden)
    cache = Path(cache_directory).expanduser()
    _require(cache.is_absolute() and not cache.is_symlink() and cache.parent.is_dir()
             and cache.parent == cache.parent.resolve(strict=True)
             and all(root != cache and root not in cache.parents for root in forbidden), "cache must remain outside source/model artifacts")
    _require(cache != output and cache not in output.parents and output not in cache.parents,
             "cache and experiment output must use separate directories")
    if not cache.exists():
        cache.mkdir(mode=0o700)
    _require(cache.is_dir() and cache == cache.resolve(strict=True), "cache directory differs")
    output.mkdir(mode=0o700)
    torch, tokenizer, provider_factory = _runtime(prepared)
    from .semantic_readout import SemanticReadout, fixed_semantic_scores
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    geometry = plan["model_geometry"]
    feature_binding = {"preparation_sha256": plan["preparation_sha256"],
                       "source_admission_sha256": plan["source_admission_sha256"], "code": plan["code"],
                       "runtime": plan["runtime"], "model_geometry": geometry, "provider_device": "cpu"}
    unique_inputs = {}
    for selected in plan["examples"]:
        values = [selected["source"], *selected["style_variants"].values(),
                  *[{"text": text, "spans": {}} for text in selected["descriptions"]]]
        for item in values:
            unique_inputs[sha256(_canonical(item)).hexdigest()] = item
    token_budgets = {}
    for key, item in sorted(unique_inputs.items()):
        encoded, masks = _tokenize(tokenizer, item, torch)
        length = encoded["input_ids"].shape[1]
        _require(length <= geometry["max_position_embeddings"],
                 "complete input exceeds actual source context; truncation is forbidden")
        token_budgets[key] = {"complete_public_input_sha256": key, "source_tokens": length,
                             "raw_token_binding_sha256": sha256(_canonical(_token_binding(encoded, masks))).hexdigest(),
                             "logical_all_state_bf16_bytes": geometry["hidden_state_count"] * length * geometry["hidden_size"] * 2}
    _require(len(token_budgets) == plan["unique_complete_feature_inputs"], "pre-forward complete input count differs")
    feature_budget = _seal({"plan_sha256": plan_digest, "feature_binding": feature_binding,
                            "complete_unique_inputs": len(token_budgets), "inputs": list(token_budgets.values()),
                            "estimated_logical_feature_cache_bytes": sum(row["logical_all_state_bf16_bytes"] for row in token_budgets.values()),
                            "maximum_complete_input_tokens": max(row["source_tokens"] for row in token_budgets.values()),
                            "measurement": "Exact tokenizer lengths times all actual BF16 states; excludes tensors for IDs/masks, serialization, parameters, temporary allocations and training",
                            "cpu_peak_memory_bytes": None, "cpu_peak_memory_measurement": "unmeasured before forwards",
                            "model_loaded": False, "qualification": "unqualified resource estimate"})
    _write_json(output / "feature-budget.json", feature_budget)
    print(json.dumps({"stage": "complete_input_budget_before_model_load", "unique_inputs": len(token_budgets),
                      "logical_feature_cache_bytes": feature_budget["estimated_logical_feature_cache_bytes"],
                      "maximum_tokens": feature_budget["maximum_complete_input_tokens"]}, sort_keys=True), flush=True)
    provider_holder, bank_memo, cache_records = [], {}, []
    def feature(item):
        key = sha256(_canonical(item)).hexdigest()
        if key not in bank_memo:
            bank, record = _cache_bank(item=item, cache_root=cache, binding=feature_binding, tokenizer=tokenizer,
                                     torch=torch, geometry=geometry, provider_factory=provider_factory,
                                     provider_holder=provider_holder,
                                     expected_token_sha256=token_budgets[key]["raw_token_binding_sha256"])
            bank_memo[key] = bank
            cache_records.append(record)
            print(json.dumps({"stage": "public_feature_cache", "completed": len(cache_records),
                              "planned": len(token_budgets), "created": record["created"],
                              "forward_seconds": record["forward_seconds"]}, sort_keys=True), flush=True)
        return bank_memo[key]
    examples = []
    for selected in plan["examples"]:
        examples.append({"id": selected["id_metadata"], "split": selected["split"],
                         "family": selected["family_metadata"], "target": selected["target_index_metadata"],
                         "sentence_unseen_in_all_train": selected["sentence_unseen_in_all_train"],
                         "source": feature(selected["source"]),
                         "candidates": [feature({"text": text, "spans": {}}) for text in selected["descriptions"]],
                         "styles": {name: feature(value) for name, value in selected["style_variants"].items()}})
    _require(len(cache_records) == plan["unique_complete_feature_inputs"], "complete planned cache coverage differs")
    provider_holder.clear()
    gc.collect()
    feature_seconds = time.monotonic() - start
    train, dev = ([example for example in examples if example["split"] == split] for split in ("train", "dev"))
    sentence_unseen_dev = [example for example in dev if example["sentence_unseen_in_all_train"]]
    recipe = plan["recipe"]
    torch.manual_seed(recipe["seed"])
    model = SemanticReadout(state_count=geometry["hidden_state_count"], hidden_size=geometry["hidden_size"], width=recipe["width"])
    model.eval()
    baselines = {"untrained_readout": {"train": _evaluate(model, train, torch), "dev": _evaluate(model, dev, torch),
                                       "sentence_unseen_dev": _evaluate(model, sentence_unseen_dev, torch)},
                 "fixed_semantic_control": {"train": _evaluate(fixed_semantic_scores, train, torch),
                                            "dev": _evaluate(fixed_semantic_scores, dev, torch),
                                            "sentence_unseen_dev": _evaluate(fixed_semantic_scores, sentence_unseen_dev, torch)}}
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    sampler = torch.Generator().manual_seed(recipe["seed"] + 1)
    binding = {"plan_sha256": plan_digest, "plan_content_sha256": plan["receipt_sha256"],
               "feature_binding": feature_binding, "recipe": recipe,
               "cache_tensor_bindings": [{"key": record["key"], "tensor_sha256": record["tensor_sha256"]}
                                         for record in sorted(cache_records, key=lambda row: row["key"])]}
    split_step = max(1, recipe["steps"] // 2)
    history = _train_steps(model, optimizer, sampler, train, steps=split_step, torch=torch)
    checkpoint = output / "midpoint.pt"
    checkpoint_digest = _save_torch(checkpoint, _checkpoint(model, optimizer, sampler, split_step, binding, torch), torch)
    history.extend(_train_steps(model, optimizer, sampler, train, steps=recipe["steps"] - split_step, torch=torch))
    final = _checkpoint(model, optimizer, sampler, recipe["steps"], binding, torch)
    resumed = SemanticReadout(**model.geometry)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
    resumed_sampler = torch.Generator()
    _require(_digest(checkpoint) == checkpoint_digest, "midpoint checkpoint changed before resume")
    resume_step = restore_checkpoint(checkpoint, model=resumed, optimizer=resumed_optimizer,
                                     sampler=resumed_sampler, binding=binding,
                                     expected_file_sha256=checkpoint_digest, torch=torch)
    resumed_history = _train_steps(resumed, resumed_optimizer, resumed_sampler, train,
                                  steps=recipe["steps"] - resume_step, torch=torch)
    resume_final = _checkpoint(resumed, resumed_optimizer, resumed_sampler, recipe["steps"], binding, torch)
    _require(_equal(final, resume_final, torch) and history[resume_step:] == resumed_history,
             "optimizer/RNG resume does not match uninterrupted readout training")
    model.eval()
    learned = {"train": _evaluate(model, train, torch), "dev": _evaluate(model, dev, torch),
               "sentence_unseen_dev": _evaluate(model, sentence_unseen_dev, torch)}
    permutations = {"learned": _permutations(model, dev, torch),
                    "fixed_control": _permutations(fixed_semantic_scores, dev, torch)}
    styles = {"learned": _style_diagnostics(model, dev, torch),
              "fixed_control": _style_diagnostics(fixed_semantic_scores, dev, torch)}
    export_path = output / "semantic-readout.pt"
    export_digest = _save_torch(export_path, {"schema": EXPORT_SCHEMA, "binding": binding,
                               "geometry": model.geometry, "model": model.state_dict()}, torch)
    reloaded = load_semantic_export(export_path, binding=binding, expected_file_sha256=export_digest, torch=torch)
    _require(_equal(model.state_dict(), reloaded.state_dict(), torch)
             and _evaluate(reloaded, dev, torch) == learned["dev"], "export/reload differs")
    plan_after, _, _ = _load_plan(plan_file)
    _require(_digest(plan_file) == plan_digest and _canonical(plan_after) == _canonical(plan), "plan changed during experiment")
    for record in cache_records:
        _require(_digest(_file(record["tensor_path"])) == record["tensor_sha256"]
                 and _digest(_file(record["metadata_path"])) == record["metadata_sha256"], "public frozen cache changed during training")
    peak = None
    if sys.platform.startswith("linux"):
        import resource
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    receipt = _seal({"schema": RECEIPT_SCHEMA, "state": "PUBLIC_SEMANTIC_EXPERIMENT_UNQUALIFIED",
                     "binding": binding, "baselines": baselines, "learned": learned,
                     "pre_forward_feature_budget": {"path": str(output / "feature-budget.json"),
                                                    "sha256": _digest(output / "feature-budget.json"),
                                                    "estimated_logical_feature_cache_bytes": feature_budget["estimated_logical_feature_cache_bytes"],
                                                    "maximum_complete_input_tokens": feature_budget["maximum_complete_input_tokens"]},
                     "original_source_sentence_only_overlap": plan["original_source_sentence_only_overlap"],
                     "dev_sentence_unseen_metric_rule": plan["dev_sentence_unseen_metric_rule"],
                     "training_loss_history": history, "candidate_permutation": permutations,
                     "inherited_helper_wording_sensitivity": styles, "cache": cache_records,
                     "feature_stage_seconds_including_admission_and_model_load": feature_seconds,
                     "elapsed_seconds": time.monotonic() - start,
                     "process_peak_rss_bytes": peak,
                     "peak_measurement": "Linux process peak RSS; not GPU allocation or Slurm job peak" if peak is not None else "unmeasured",
                     "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_digest,
                                    "step": resume_step, "optimizer_rng_resume_exact": True,
                                    "rng_domains": ["torch_cpu_global", "torch_cpu_sampler_generator"]},
                     "export": {"path": str(export_path), "sha256": export_digest, "reload_exact": True},
                     "readout_dtype": "float32", "provider_dtype": "bfloat16", "provider_device": "cpu",
                     "observed_provider_state_count": geometry["hidden_state_count"],
                     "learned_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                     "upstream_gradient": False, "upstream_tensor_mutation": False,
                     "private_identity_data": False, "final_payload_opened": False,
                     "source_counts_define_capability_ceiling": False, "n0_approved": False,
                     "personality_qualified": False, "qualification": "unqualified; public semantic experiment only",
                     "limits": ["DEV is model-selection evidence, not unopened FINAL",
                                "Relation-family holdout is not full lexical independence; shared sentences are reported and sentence-unseen DEV measured separately",
                                "Human annotation provenance does not guarantee every label is correct",
                                "Helper-wording diagnostics do not establish removed pretrained personality priors",
                                "This first-party semantic readout is not N1/N2/N3 identity qualification"]})
    _write_json(output / "experiment.json", receipt)
    return receipt
