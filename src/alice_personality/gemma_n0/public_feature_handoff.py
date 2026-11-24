"""Portable, externally pinned PUBLIC frozen-feature custody; never N0 approval.

The exporter freshly verifies a completed producer experiment and its source.
The importer verifies that closed evidence and exact feature bytes, without a
publisher checkpoint, Transformers or a network call. Original receipts are
copied byte-for-byte; their historical absolute paths are never resealed.
External pins must come from trusted producer custody, not the imported package.
Hashes do not authenticate an author or establish honest historical execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil

MANIFEST_SCHEMA = "alice-personality-closed-public-feature-inventory-v1"
CLOSURE_SCHEMA = "alice-personality-closed-public-feature-session-v1"
IMPORT_SCHEMA = "alice-personality-closed-public-feature-import-v1"
MODEL = "google/gemma-4-12B"
REVISION = "023679ed352de9bb66cc873c9009ce3482585c08"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_CACHE_NAME = re.compile(r"features/[0-9a-f]{64}\.(?:json|pt)\Z")
_ROLE_NAMES = {"source_mask", "head_mask", "tail_mask", "query_mask"}
_PRODUCER_NAMES = {"semantic_readout.py", "public_semantic_experiment.py",
    "run_public_semantic_experiment.py", "public_fewrel.py", "backbone.py", "preparation.py"}
_FALSE_FLAGS = {"private_identity_data", "final_payload_opened", "upstream_gradient",
    "upstream_tensor_mutation", "n0_approved", "personality_qualified",
    "source_acceptance_authority", "source_counts_define_capability_ceiling"}
# Parser admission limit for one JSON document, not a corpus/context ceiling.
_MAX_JSON_BYTES = 32 * 1024 * 1024
_ROOT = Path(__file__).resolve().parents[3]
_OWN_CODE = {"public_feature_handoff.py": Path(__file__),
    "handoff_public_features.py": _ROOT / "scripts/eipm/gemma_n0/handoff_public_features.py"}


class HandoffError(ValueError):
    """Public bytes, provenance, complete feature coverage or runtime differs."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise HandoffError(reason)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")


def _pin(value: object) -> str:
    _require(type(value) is str and _DIGEST.fullmatch(value) is not None,
        "an exact lowercase external SHA256 pin is required")
    return value


def _integer(value: object, *, minimum: int = 1) -> int:
    _require(type(value) is int and value >= minimum, "invalid actual integer count")
    return value


def _number(value: object) -> float:
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
        "invalid finite nonnegative measurement")
    return value


def _regular(raw: str | Path, *, directory: bool = False) -> Path:
    path = Path(raw)
    _require(path.is_absolute() and not any(p.is_symlink() or
        (hasattr(p, "is_junction") and p.is_junction()) for p in (path, *path.parents)),
        "paths must be absolute and cannot traverse links")
    _require((path.is_dir() if directory else path.is_file()) and
        path == path.resolve(strict=True), "paths must resolve to exact regular files/directories")
    return path


def _hash(path: Path) -> str:
    before = path.stat()
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    after = path.stat()
    identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    _require(identity(before) == identity(after), "file changed while hashing")
    return digest.hexdigest()


def _read(path: Path, expected: str | None = None) -> dict:
    path = _regular(path)
    if expected is not None:
        _require(_hash(path) == _pin(expected), "file bytes differ from external pin")
    _require(0 < path.stat().st_size <= _MAX_JSON_BYTES, "JSON parser admission size exceeded")
    before = _hash(path)
    with path.open("rb") as stream:
        raw = stream.read(_MAX_JSON_BYTES + 1)
    _require(len(raw) <= _MAX_JSON_BYTES and sha256(raw).hexdigest() == before == _hash(path)
        and (expected is None or before == expected), "JSON changed while reading or differs from pin")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(HandoffError("nonfinite JSON")))
    except (ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, HandoffError):
            raise
        raise HandoffError("invalid JSON") from None
    _require(type(value) is dict, "receipt must be a JSON object")
    # JSON's exponent parser can also produce infinity, without parse_constant.
    try:
        _canonical(value)
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise HandoffError("nonfinite or unsupported JSON value") from None
    return value


def _seal(value: dict) -> dict:
    value = json.loads(_canonical(value))
    value["receipt_sha256"] = sha256(_canonical(value)).hexdigest()
    return value


def _verify_seal(value: dict) -> None:
    unsigned = dict(value)
    supplied = _pin(unsigned.pop("receipt_sha256", None))
    _require(sha256(_canonical(unsigned)).hexdigest() == supplied, "canonical receipt digest differs")


def _write(path: Path, value: dict) -> None:
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(_canonical(value) + b"\n")


def _own_code() -> dict:
    return {name: _hash(_regular(path)) for name, path in sorted(_OWN_CODE.items())}


def _recipe(value: dict) -> dict:
    fields = {"train_rows_per_family", "dev_rows_per_family", "style_rows_per_dev_family",
        "width", "steps", "seed", "learning_rate", "weight_decay"}
    _require(type(value) is dict and set(value) == fields, "operating recipe fields differ")
    for key in ("train_rows_per_family", "dev_rows_per_family", "width", "steps"):
        _integer(value[key])
    _integer(value["seed"], minimum=0)
    _integer(value["style_rows_per_dev_family"], minimum=0)
    _require(value["steps"] >= 2 and value["style_rows_per_dev_family"] <= value["dev_rows_per_family"],
        "recipe cannot execute resumable training/style controls")
    _require(_number(value["learning_rate"]) > 0, "positive learning rate required")
    _number(value["weight_decay"])
    return value


def _geometry(value: dict, mechanics: bool) -> dict:
    _require(type(value) is dict, "actual source geometry is required")
    for key in ("hidden_state_count", "hidden_size", "num_hidden_layers", "max_position_embeddings"):
        _integer(value.get(key))
    _require(value["hidden_state_count"] == value["num_hidden_layers"] + 1,
        "all embedding and decoder states are required")
    if not mechanics:
        _require(value["hidden_state_count"] == 49 and value["hidden_size"] == 3840
            and value.get("model_type") == "gemma4_unified"
            and value.get("text_model_type") == "gemma4_unified_text", "wrong pinned Gemma geometry")
    return value


def _input(item: dict) -> str:
    _require(type(item) is dict and set(item) == {"text", "spans"}
        and type(item["text"]) is str and bool(item["text"].strip())
        and type(item["spans"]) is dict and set(item["spans"]) in (set(), _ROLE_NAMES),
        "complete raw source/description input fields differ")
    for span in item["spans"].values():
        _require(type(span) is list and len(span) == 2
            and all(type(i) is int for i in span)
            and 0 <= span[0] < span[1] <= len(item["text"]), "invalid complete role text spans")
    return sha256(_canonical(item)).hexdigest()


def _inputs(plan: dict) -> dict[str, dict]:
    _require(type(plan.get("examples")) is list and bool(plan["examples"]), "selected public examples missing")
    result, ids, counts = {}, set(), {"train": 0, "dev": 0}
    fields = {"id_metadata", "split", "family_metadata", "source", "descriptions",
        "target_index_metadata", "style_variants", "row_sha256", "sentence_unseen_in_all_train"}
    for row in plan["examples"]:
        _require(type(row) is dict and set(row) == fields and row["split"] in counts,
            "unsupported public row or non-TRAIN/DEV split")
        _require(type(row["id_metadata"]) is str and row["id_metadata"] not in ids
            and type(row["family_metadata"]) is str and bool(row["family_metadata"])
            and type(row["sentence_unseen_in_all_train"]) is bool, "invalid public row metadata")
        ids.add(row["id_metadata"])
        counts[row["split"]] += 1
        _pin(row["row_sha256"])
        descriptions = row["descriptions"]
        _require(type(descriptions) is list and len(descriptions) > 1
            and all(type(text) is str and text.strip() and not re.search(r"\bP\d+\b", text)
                for text in descriptions), "dynamic semantic descriptions required")
        target = _integer(row["target_index_metadata"], minimum=0)
        _require(target < len(descriptions), "target index is outside candidate descriptions")
        _require(type(row["style_variants"]) is dict and
            set(row["style_variants"]) in (set(), {"helpful", "flattering"})
            and (not row["style_variants"] or row["split"] == "dev"), "style controls changed")
        _input(row["source"])
        _require(set(row["source"]["spans"]) == _ROLE_NAMES, "complete source roles required")
        for item in [row["source"], *row["style_variants"].values(),
            *[{"text": text, "spans": {}} for text in descriptions]]:
            key = _input(item)
            result[key] = item
    _require(all(counts.values()) and len(result) == _integer(plan.get("unique_complete_feature_inputs")),
        "complete planned public input coverage differs")
    return dict(sorted(result.items()))


def _protocol(plan: dict, experiment: dict, budget: dict, prepared: dict,
              source: dict, mechanics: bool) -> tuple[dict, dict]:
    for value in (plan, experiment, budget, prepared, source):
        _verify_seal(value)
    _require(plan.get("schema") == "alice-personality-gemma-n0-public-semantic-plan-v1"
        and plan.get("state") == "PRECOMMITTED_PUBLIC_READOUT_UNQUALIFIED"
        and experiment.get("schema") == "alice-personality-gemma-n0-public-semantic-experiment-v1"
        and experiment.get("state") == "PUBLIC_SEMANTIC_EXPERIMENT_UNQUALIFIED", "completed original protocol required")
    for key in _FALSE_FLAGS - {"source_acceptance_authority"}:
        _require(experiment.get(key) is False, "experiment crossed public frozen/unqualified boundary")
    for key in ("private_identity_data", "final_payload_opened", "n0_approved", "source_counts_define_capability_ceiling"):
        _require(plan.get(key) is False, "plan crossed public/unqualified boundary")
    geometry = _geometry(plan.get("model_geometry"), mechanics)
    runtime = plan.get("runtime")
    _require(type(runtime) is dict and set(runtime) == {"backend", "dtype", "torch_version", "transformers_version"}
        and runtime["backend"] == "transformers" and runtime["dtype"] == "bfloat16"
        and all(type(runtime[key]) is str and runtime[key].strip() == runtime[key] and runtime[key]
            for key in ("torch_version", "transformers_version")), "exact producer BF16 runtime required")
    _require(prepared.get("model_geometry") == geometry and prepared.get("runtime") == runtime
        and prepared.get("receipt_sha256") == plan["preparation_sha256"]
        and source.get("receipt_sha256") == plan["source_admission_sha256"], "original source/preparation links differ")
    _require(source.get("schema") == "alice-personality-gemma-n0-public-fewrel-admission-v1"
        and source.get("state") == "ELIGIBLE_PUBLIC_SOURCE_UNQUALIFIED"
        and source.get("public_train_dev_declared") is True
        and source.get("private_identity_data") is False and source.get("n0_approved") is False
        and source.get("target_keys_indices_and_ids_are_model_input") is False,
        "source eligibility must preserve public human metadata boundary")
    _require(prepared.get("schema") == "alice-personality-gemma-n0-preparation-v2"
        and prepared.get("state") == "PREPARED_UNQUALIFIED" and prepared.get("role") == "personality"
        and prepared.get("behavior_qualification") is None
        and prepared.get("repository") == MODEL and prepared.get("revision") == REVISION,
        "original personality source preparation pin differs")
    code = plan.get("code")
    _require(type(code) is list and len(code) == len(_PRODUCER_NAMES)
        and all(type(row) is dict and set(row) == {"name", "sha256"} for row in code)
        and {row["name"] for row in code} == _PRODUCER_NAMES, "producer code inventory differs")
    for row in code:
        _pin(row["sha256"])
    recipe = _recipe(plan["recipe"])
    feature_binding = {"preparation_sha256": plan["preparation_sha256"],
        "source_admission_sha256": plan["source_admission_sha256"], "code": code,
        "runtime": runtime, "model_geometry": geometry, "provider_device": "cpu"}
    binding = experiment.get("binding")
    _require(type(binding) is dict and set(binding) == {"plan_sha256", "plan_content_sha256", "feature_binding", "recipe", "cache_tensor_bindings"}
        and binding["plan_content_sha256"] == plan["receipt_sha256"]
        and binding["feature_binding"] == feature_binding and binding["recipe"] == recipe,
        "completed experiment plan/source/code/recipe binding differs")
    _pin(binding["plan_sha256"])
    losses = experiment.get("training_loss_history")
    _require(type(losses) is list and len(losses) == recipe["steps"], "completed actual training history required")
    for value in losses:
        _number(value)
    _require(experiment.get("checkpoint", {}).get("optimizer_rng_resume_exact") is True
        and experiment.get("export", {}).get("reload_exact") is True
        and experiment.get("provider_device") == "cpu" and experiment.get("provider_dtype") == "bfloat16"
        and experiment.get("readout_dtype") == "float32"
        and experiment.get("observed_provider_state_count") == geometry["hidden_state_count"],
        "completed frozen producer training/resume/export evidence required")
    _require(budget.get("feature_binding") == feature_binding and budget.get("plan_sha256") == binding["plan_sha256"]
        and budget.get("model_loaded") is False and type(budget.get("inputs")) is list,
        "original pre-forward budget binding differs")
    _require(experiment.get("original_source_sentence_only_overlap") == plan.get("original_source_sentence_only_overlap")
        == source.get("statistics", {}).get("normalized_sentence_only_overlap")
        and type(plan.get("original_source_sentence_only_overlap")) is int
        and plan["original_source_sentence_only_overlap"] >= 0
        and experiment.get("dev_sentence_unseen_metric_rule") == plan.get("dev_sentence_unseen_metric_rule"),
        "original sentence overlap/DEV metric rule changed")
    return feature_binding, _inputs(plan)


def _token_digest(bank, role_names: set[str]) -> str:
    value = {"input_ids": bank.input_ids.cpu().tolist(), "attention_mask": bank.attention_mask.cpu().tolist(),
        "role_masks": {name: getattr(bank, name).cpu().tolist() for name in sorted(role_names)}}
    return sha256(_canonical(value)).hexdigest()


def _bank(tensor_path: Path, metadata: dict, *, item: dict, input_sha: str,
          feature_binding: dict, token_budget: dict, geometry: dict):
    import torch
    from .semantic_readout import LayerTokenBank, validate_bank
    _verify_seal(metadata)
    _require(set(metadata) == {"schema", "binding", "tensor_sha256", "state_count", "hidden_size",
        "source_tokens", "dtype", "complete_input", "public_only", "upstream_frozen", "forward_seconds",
        "logical_feature_bytes", "receipt_sha256"}, "cache metadata contains unsupported fields")
    expected = {"experiment_feature_binding": feature_binding,
        "complete_public_input_sha256": input_sha,
        "raw_token_binding_sha256": _pin(token_budget["raw_token_binding_sha256"])}
    _require(metadata.get("schema") == "alice-personality-gemma-n0-public-feature-cache-v1"
        and _canonical(metadata.get("binding")) == _canonical(expected), "cache exact input/token/source/code binding differs")
    key = sha256(_canonical(expected)).hexdigest()
    _require(tensor_path.name == key + ".pt", "cache filename/key differs")
    _require(_hash(tensor_path) == _pin(metadata.get("tensor_sha256")), "cached tensor bytes differ")
    for field in ("complete_input", "public_only", "upstream_frozen"):
        _require(metadata.get(field) is True, "cache public/frozen/complete flags differ")
    length = _integer(token_budget["source_tokens"])
    _require(length <= geometry["max_position_embeddings"] and metadata.get("source_tokens") == length
        and type(metadata.get("source_tokens")) is int
        and metadata.get("dtype") == "bfloat16" and type(metadata.get("state_count")) is int
        and metadata["state_count"] == geometry["hidden_state_count"]
        and type(metadata.get("hidden_size")) is int and metadata["hidden_size"] == geometry["hidden_size"],
        "cache geometry/context/dtype differs")
    _number(metadata.get("forward_seconds"))
    before = _hash(tensor_path)
    payload = torch.load(tensor_path, weights_only=True, map_location="cpu")
    _require(before == _hash(tensor_path), "feature bytes changed during safe tensor loading")
    roles = set(item["spans"])
    _require(type(payload) is dict and set(payload) == {"layers", "input_ids", "attention_mask", *roles},
        "feature tensor payload includes unsupported fields")
    bank = LayerTokenBank(**payload)
    validate_bank(bank, state_count=geometry["hidden_state_count"], hidden_size=geometry["hidden_size"], source=bool(roles))
    _require(bank.layers.shape[1] == length and _token_digest(bank, roles) == expected["raw_token_binding_sha256"],
        "cached token IDs/masks/roles differ from producer token evidence")
    logical = geometry["hidden_state_count"] * length * geometry["hidden_size"] * 2
    _require(type(metadata.get("logical_feature_bytes")) is int and metadata["logical_feature_bytes"] == logical
        and type(token_budget.get("logical_all_state_bf16_bytes")) is int
        and token_budget["logical_all_state_bf16_bytes"] == logical, "logical all-state bytes differ")
    return key, bank


def _budgets(budget: dict, inputs: dict, geometry: dict) -> dict:
    rows = budget["inputs"]
    _require(len(rows) == len(inputs) and type(budget.get("complete_unique_inputs")) is int
        and budget["complete_unique_inputs"] == len(inputs), "feature budget coverage differs")
    result = {}
    for row in rows:
        _require(type(row) is dict and set(row) == {"complete_public_input_sha256", "source_tokens",
            "raw_token_binding_sha256", "logical_all_state_bf16_bytes"}, "budget fields differ")
        digest = _pin(row["complete_public_input_sha256"])
        _require(digest in inputs and digest not in result, "duplicate/extra/missing budget input")
        _pin(row["raw_token_binding_sha256"])
        length = _integer(row["source_tokens"])
        _require(length <= geometry["max_position_embeddings"] and type(row["logical_all_state_bf16_bytes"]) is int
            and row["logical_all_state_bf16_bytes"] == geometry["hidden_state_count"] * geometry["hidden_size"] * length * 2,
            "budget complete token geometry differs")
        result[digest] = row
    _require(set(result) == set(inputs) and type(budget.get("estimated_logical_feature_cache_bytes")) is int
        and budget["estimated_logical_feature_cache_bytes"] == sum(row["logical_all_state_bf16_bytes"] for row in rows)
        and type(budget.get("maximum_complete_input_tokens")) is int
        and budget["maximum_complete_input_tokens"] == max(row["source_tokens"] for row in rows), "budget totals differ")
    return result


def export_public_features(*, experiment_path: str | Path, expected_experiment_sha256: str,
    plan_path: str | Path, cache_directory: str | Path, output_directory: str | Path,
    mechanics_only: bool = False) -> dict:
    """Create-only expanded exact inventory after a completed source recheck.

    mechanics_only is for explicit tiny public test stand-ins. Its receipts are
    PUBLIC_MECHANICS_ONLY and production import rejects them by default.
    """
    _require(type(mechanics_only) is bool, "mechanics scope must be an explicit bool")
    from . import public_semantic_experiment as producer
    exp_path, plan_file, cache = _regular(experiment_path), _regular(plan_path), _regular(cache_directory, directory=True)
    experiment = _read(exp_path, expected_experiment_sha256)
    _verify_seal(experiment)
    _require(_hash(plan_file) == _pin(experiment.get("binding", {}).get("plan_sha256")), "original plan byte pin differs")
    plan, source, prepared = producer._load_plan(plan_file)
    prepared_file, source_file = _regular(plan["preparation_path"]), _regular(plan["source_admission_path"])
    budget_file = _regular(experiment["pre_forward_feature_budget"]["path"])
    budget = _read(budget_file, experiment["pre_forward_feature_budget"]["sha256"])
    feature_binding, inputs = _protocol(plan, experiment, budget, prepared, source, mechanics_only)
    geometry, budgets = plan["model_geometry"], _budgets(budget, inputs, plan["model_geometry"])
    records = experiment.get("cache")
    _require(type(records) is list and len(records) == len(inputs), "completed cache coverage required")
    indexed = {}
    for row in records:
        _require(type(row) is dict and set(row) == {"key", "metadata_path", "metadata_sha256", "tensor_path",
            "tensor_sha256", "created", "forward_seconds", "logical_feature_bytes"}, "experiment cache record fields differ")
        key = _pin(row["key"])
        _require(key not in indexed and type(row["created"]) is bool, "duplicate cache record or invalid creation flag")
        _number(row["forward_seconds"])
        _integer(row["logical_feature_bytes"])
        for field, suffix in (("metadata", ".json"), ("tensor", ".pt")):
            file = _regular(row[field + "_path"])
            _require(file.parent == cache and file.name == key + suffix
                and _hash(file) == _pin(row[field + "_sha256"]), "cache escaped explicit root or changed bytes")
        indexed[key] = row
    _require(experiment["binding"]["cache_tensor_bindings"] == [
        {"key": key, "tensor_sha256": indexed[key]["tensor_sha256"]} for key in sorted(indexed)],
        "checkpoint/experiment cache bindings differ")
    # Processor only; no provider is constructed or invoked during closure.
    torch, tokenizer, _unused_provider_factory = producer._runtime(prepared)
    mapping = []
    for digest, item in inputs.items():
        encoded, roles = producer._tokenize(tokenizer, item, torch)
        _require(sha256(_canonical(producer._token_binding(encoded, roles))).hexdigest()
            == budgets[digest]["raw_token_binding_sha256"], "fresh raw retokenization differs")
        key = sha256(_canonical({"experiment_feature_binding": feature_binding,
            "complete_public_input_sha256": digest,
            "raw_token_binding_sha256": budgets[digest]["raw_token_binding_sha256"]})).hexdigest()
        _require(key in indexed, "complete input is missing its original feature bank")
        record = indexed[key]
        metadata = _read(_regular(record["metadata_path"]), record["metadata_sha256"])
        checked, bank = _bank(_regular(record["tensor_path"]), metadata, item=item, input_sha=digest,
            feature_binding=feature_binding, token_budget=budgets[digest], geometry=geometry)
        _require(checked == key and metadata["tensor_sha256"] == record["tensor_sha256"]
            and metadata["logical_feature_bytes"] == record["logical_feature_bytes"]
            and metadata["forward_seconds"] == record["forward_seconds"], "cache record measurements differ")
        del bank
        mapping.append({"input_sha256": digest, "cache_key": key,
            "token_binding_sha256": budgets[digest]["raw_token_binding_sha256"]})
    _require(len({row["cache_key"] for row in mapping}) == len(indexed), "cache coverage is not exact")
    original_files = {"experiment": exp_path, "plan": plan_file, "preparation": prepared_file,
        "source_admission": source_file, "feature_budget": budget_file,
        "clone": _regular(prepared["clone_receipt_path"])}
    clone = _read(original_files["clone"])
    _verify_seal(clone)
    _require(clone["receipt_sha256"] == prepared["clone_receipt_sha256"], "clone ancestry digest differs")
    output = Path(output_directory)
    _require(output.is_absolute() and not output.exists() and not output.is_symlink(), "export requires a fresh absolute output")
    _regular(output.parent, directory=True)
    forbidden = [cache, Path(prepared["snapshot_path"]), *[Path(row["path"]).parent for row in source["files"]]]
    _require(all(root != output and root not in output.parents and output not in root.parents for root in forbidden),
        "portable output must stay outside source/model/cache artifacts")
    source_pins = {name: _hash(path) for name, path in original_files.items()}
    implementation = _own_code()
    candidates = {f"original/{name}.json": path for name, path in original_files.items()}
    for name, path in producer.CODE_PATHS.items():
        candidates[f"producer-code/{name}"] = _regular(path)
    for name, path in _OWN_CODE.items():
        candidates[f"producer-code/{name}"] = _regular(path)
    for key, row in indexed.items():
        candidates[f"features/{key}.json"] = _regular(row["metadata_path"])
        candidates[f"features/{key}.pt"] = _regular(row["tensor_path"])
    # Copies are evidence only, original checkpoint/data paths are never rewritten.
    output.mkdir(mode=0o700)
    inventory = []
    for member, path in sorted(candidates.items()):
        destination = output / member
        destination.parent.mkdir(mode=0o700, exist_ok=True)
        digest = _hash(path)
        with path.open("rb") as src, os.fdopen(os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as dst:
            shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
        _require(_hash(path) == digest == _hash(destination), "source/copy changed while exporting")
        inventory.append({"path": member, "size": destination.stat().st_size, "sha256": digest})
    after, after_source, after_prepared = producer._load_plan(plan_file)
    _require(_canonical(after) == _canonical(plan) and _canonical(after_source) == _canonical(source)
        and _canonical(after_prepared) == _canonical(prepared) and implementation == _own_code()
        and all(_hash(path) == source_pins[name] for name, path in original_files.items()),
        "producer source/code/original receipt changed before closure")
    for entry in inventory:
        _require(_hash(candidates[entry["path"]]) == entry["sha256"] == _hash(output / entry["path"]),
            "feature/code/original bytes changed before closure")
    closure = _seal({"schema": CLOSURE_SCHEMA,
        "status": "PUBLIC_MECHANICS_ONLY" if mechanics_only else "CLOSED_SOURCE_VERIFIED",
        "state": "CLOSED_PUBLIC_FEATURE_UNQUALIFIED", "qualification": "UNQUALIFIED",
        "source_class": "public_mechanical_fixture" if mechanics_only else "externally_closed_public_frozen_semantic_features",
        "source_recheck_passed": not mechanics_only, "mechanics_only": mechanics_only,
        "repository": MODEL, "revision": REVISION, "producer_runtime": plan["runtime"],
        "model_geometry": geometry, "producer_code": plan["code"], "handoff_code": implementation,
        "completed_experiment_file_sha256": _pin(expected_experiment_sha256),
        "original_file_sha256": source_pins, "complete_input_mapping": mapping,
        "complete_unique_inputs": len(inputs), "inventory": inventory,
        "original_source_sentence_only_overlap": plan["original_source_sentence_only_overlap"],
        "dev_sentence_unseen_metric_rule": plan["dev_sentence_unseen_metric_rule"],
        "producer_retokenization_passed": True, "archive_extraction": False,
        **{key: False for key in _FALSE_FLAGS}})
    _write(output / "closure.json", closure)
    inventory = [*inventory, {"path": "closure.json", "size": (output / "closure.json").stat().st_size,
        "sha256": _hash(output / "closure.json")}]
    manifest = _seal({"schema": MANIFEST_SCHEMA, "state": "PORTABLE_PUBLIC_FEATURES_UNQUALIFIED",
        "qualification": "UNQUALIFIED", "files": sorted(inventory, key=lambda row: row["path"]),
        "closure_file_sha256": _hash(output / "closure.json"), "archive_extraction": False})
    _write(output / "manifest.json", manifest)
    return {"manifest_sha256": _hash(output / "manifest.json"), "closure_sha256": _hash(output / "closure.json"),
        "status": closure["status"], "complete_unique_inputs": len(inputs), "n0_approved": False}


def _member(root: Path, raw: str) -> Path:
    _require(type(raw) is str and raw and "\\" not in raw and ":" not in raw
        and not PurePosixPath(raw).is_absolute() and all(part not in ("", ".", "..") for part in raw.split("/")),
        "inventory member must be a canonical relative path")
    path = _regular(root.joinpath(*raw.split("/")))
    _require(root in path.parents, "member escaped package root")
    return path


def _inventory(root: Path, entries: list) -> dict[str, Path]:
    _require(type(entries) is list and bool(entries), "exact portable inventory required")
    paths, aliases = {}, set()
    allowed = {"closure.json", *[f"original/{name}.json" for name in
        ("experiment", "plan", "preparation", "source_admission", "feature_budget", "clone")],
        *[f"producer-code/{name}" for name in _PRODUCER_NAMES | set(_OWN_CODE)]}
    for row in entries:
        _require(type(row) is dict and set(row) == {"path", "size", "sha256"}, "inventory entry fields differ")
        raw = row["path"]
        _require(type(raw) is str and raw not in paths and raw.casefold() not in aliases
            and (raw in allowed or _CACHE_NAME.fullmatch(raw) is not None), "extra/duplicate/aliased inventory member")
        path = _member(root, raw)
        _require(path.stat().st_size == _integer(row["size"])
            and _hash(path) == _pin(row["sha256"]), "package member size/hash differs")
        paths[raw] = path
        aliases.add(raw.casefold())
    _require(allowed <= set(paths), "portable inventory is missing original/producer evidence")
    return paths


@dataclass(frozen=True)
class ImportedPublicFeatures:
    plan: dict
    examples: list[dict]
    binding: dict
    geometry: dict
    package_directory: Path


def import_public_features(package_directory: str | Path, *, expected_manifest_sha256: str,
    expected_closure_sha256: str, allow_mechanics_only: bool = False) -> ImportedPublicFeatures:
    """Admit only an externally pinned expanded package; no upstream loading.

    Complete target/split metadata is returned for a separately bound trainer.
    This import does not itself authorize private/identity gradients or qualify N0.
    """
    _require(type(allow_mechanics_only) is bool, "mechanics scope must be an explicit bool")
    consumer_code = _own_code()
    root = _regular(package_directory, directory=True)
    manifest = _read(root / "manifest.json", expected_manifest_sha256)
    _verify_seal(manifest)
    _require(set(manifest) == {"schema", "state", "qualification", "files", "closure_file_sha256", "archive_extraction", "receipt_sha256"}
        and manifest["schema"] == MANIFEST_SCHEMA and manifest["state"] == "PORTABLE_PUBLIC_FEATURES_UNQUALIFIED"
        and manifest["qualification"] == "UNQUALIFIED" and manifest["archive_extraction"] is False
        and manifest["closure_file_sha256"] == _pin(expected_closure_sha256), "portable manifest boundary/pin differs")
    paths = _inventory(root, manifest["files"])
    # Inventory the explicit package ONLY; no sibling or unrelated input scan.
    actual_files, actual_directories = set(), set()
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in directories:
            path = _regular(Path(parent) / name, directory=True)
            actual_directories.add(path.relative_to(root).as_posix())
        for name in files:
            path = _regular(Path(parent) / name)
            actual_files.add(path.relative_to(root).as_posix())
    _require(actual_files == set(paths) | {"manifest.json"}, "expanded package contains extra or missing files")
    expected_directories = {parent.as_posix() for name in paths
        for parent in PurePosixPath(name).parents if parent.as_posix() != "."}
    _require(actual_directories == expected_directories, "expanded package contains unsupported directories")
    closure = _read(paths["closure.json"], expected_closure_sha256)
    _verify_seal(closure)
    _require(set(closure) == {"schema", "status", "state", "qualification", "source_class",
        "source_recheck_passed", "mechanics_only", "repository", "revision", "producer_runtime",
        "model_geometry", "producer_code", "handoff_code", "completed_experiment_file_sha256",
        "original_file_sha256", "complete_input_mapping", "complete_unique_inputs", "inventory",
        "original_source_sentence_only_overlap", "dev_sentence_unseen_metric_rule",
        "producer_retokenization_passed", "archive_extraction", "receipt_sha256", *_FALSE_FLAGS},
        "closed public receipt contains unsupported fields")
    mechanics = closure.get("mechanics_only")
    _require(type(mechanics) is bool and (not mechanics or allow_mechanics_only), "mechanics-only evidence cannot enter production import")
    _require(closure.get("schema") == CLOSURE_SCHEMA and closure.get("state") == "CLOSED_PUBLIC_FEATURE_UNQUALIFIED"
        and closure.get("qualification") == "UNQUALIFIED" and closure.get("archive_extraction") is False
        and closure.get("status") == ("PUBLIC_MECHANICS_ONLY" if mechanics else "CLOSED_SOURCE_VERIFIED")
        and closure.get("source_recheck_passed") is (not mechanics)
        and closure.get("source_class") == ("public_mechanical_fixture" if mechanics else "externally_closed_public_frozen_semantic_features")
        and closure.get("producer_retokenization_passed") is True
        and closure.get("repository") == MODEL and closure.get("revision") == REVISION,
        "closed public producer status/source pin differs")
    for key in _FALSE_FLAGS:
        _require(closure.get(key) is False, "closure crossed public frozen/unqualified boundary")
    _require(closure["inventory"] == [row for row in manifest["files"] if row["path"] != "closure.json"],
        "closed inventory differs from portable manifest")
    _require(type(closure.get("original_file_sha256")) is dict
        and set(closure["original_file_sha256"]) == {"experiment", "plan", "preparation", "source_admission", "feature_budget", "clone"},
        "original receipt bindings differ")
    originals = {name: _read(paths[f"original/{name}.json"], digest)
        for name, digest in closure["original_file_sha256"].items()}
    _require(set(originals) == {"experiment", "plan", "preparation", "source_admission", "feature_budget", "clone"},
        "original unchanged receipt inventory differs")
    experiment, plan, prepared, source, budget, clone = (originals[name] for name in
        ("experiment", "plan", "preparation", "source_admission", "feature_budget", "clone"))
    feature_binding, inputs = _protocol(plan, experiment, budget, prepared, source, mechanics)
    geometry = _geometry(closure["model_geometry"], mechanics)
    _require(geometry == plan["model_geometry"] and closure["producer_runtime"] == plan["runtime"]
        and closure["producer_code"] == plan["code"] and closure["complete_unique_inputs"] == len(inputs)
        and type(closure["complete_unique_inputs"]) is int
        and closure["completed_experiment_file_sha256"] == closure["original_file_sha256"]["experiment"]
        and _hash(paths["original/plan.json"]) == experiment["binding"]["plan_sha256"]
        and _hash(paths["original/preparation.json"]) == plan["preparation_file_sha256"]
        and _hash(paths["original/source_admission.json"]) == plan["source_admission_file_sha256"]
        and _hash(paths["original/feature_budget.json"]) == experiment["pre_forward_feature_budget"]["sha256"],
        "portable original/source/code/budget links differ")
    _verify_seal(clone)
    _require(clone["receipt_sha256"] == prepared["clone_receipt_sha256"]
        and clone.get("schema") == "alice-gemma4-v1-clone-v1" and clone.get("role") == "personality"
        and clone.get("repository") == MODEL and clone.get("revision") == REVISION
        and clone.get("files") == prepared.get("files"), "portable unchanged clone ancestry differs")
    _require(closure["original_source_sentence_only_overlap"] == plan["original_source_sentence_only_overlap"]
        and closure["dev_sentence_unseen_metric_rule"] == plan["dev_sentence_unseen_metric_rule"], "heldout overlap rule differs")
    for code in plan["code"]:
        _require(_hash(paths["producer-code/" + code["name"]]) == code["sha256"], "original producer code bytes differ")
    _require(type(closure["handoff_code"]) is dict and set(closure["handoff_code"]) == set(_OWN_CODE), "producer handoff code differs")
    for name, digest in closure["handoff_code"].items():
        _require(_hash(paths["producer-code/" + name]) == _pin(digest), "producer closure code bytes differ")
    consumer_readout = _regular(Path(__file__).with_name("semantic_readout.py"))
    consumer_readout_sha = _hash(consumer_readout)
    _require(consumer_readout_sha ==
        next(row["sha256"] for row in plan["code"] if row["name"] == "semantic_readout.py"),
        "consumer readout implementation differs from frozen producer protocol")
    budgets = _budgets(budget, inputs, geometry)
    records = experiment.get("cache")
    _require(type(records) is list and len(records) == len(inputs), "completed portable cache inventory differs")
    recorded_keys = set()
    for row in records:
        _require(type(row) is dict and set(row) == {"key", "metadata_path", "metadata_sha256", "tensor_path",
            "tensor_sha256", "created", "forward_seconds", "logical_feature_bytes"}, "original cache record fields differ")
        key = _pin(row["key"])
        _require(key not in recorded_keys and type(row["created"]) is bool, "duplicate/invalid original cache record")
        recorded_keys.add(key)
        _number(row["forward_seconds"])
        _integer(row["logical_feature_bytes"])
        for kind, suffix in (("metadata", ".json"), ("tensor", ".pt")):
            member = f"features/{key}{suffix}"
            _require(member in paths and type(row[kind + "_path"]) is str
                and Path(row[kind + "_path"]).name == key + suffix
                and _hash(paths[member]) == _pin(row[kind + "_sha256"]),
                "portable cache bytes differ from completed original experiment")
    mapping, banks, cache_bindings = [], {}, []
    for digest, item in inputs.items():
        raw = {"experiment_feature_binding": feature_binding, "complete_public_input_sha256": digest,
            "raw_token_binding_sha256": budgets[digest]["raw_token_binding_sha256"]}
        key = sha256(_canonical(raw)).hexdigest()
        _require(f"features/{key}.json" in paths and f"features/{key}.pt" in paths,
            "complete planned feature bank is missing")
        metadata = _read(paths[f"features/{key}.json"])
        _, bank = _bank(paths[f"features/{key}.pt"], metadata, item=item, input_sha=digest,
            feature_binding=feature_binding, token_budget=budgets[digest], geometry=geometry)
        banks[digest] = bank
        mapping.append({"input_sha256": digest, "cache_key": key,
            "token_binding_sha256": budgets[digest]["raw_token_binding_sha256"]})
        cache_bindings.append({"key": key, "tensor_sha256": metadata["tensor_sha256"]})
    _require(mapping == closure["complete_input_mapping"] and sorted(cache_bindings, key=lambda row: row["key"])
        == experiment["binding"]["cache_tensor_bindings"], "closed complete bank coverage/input mapping differs")
    expected_members = {"closure.json", *[f"original/{name}.json" for name in originals],
        *[f"producer-code/{name}" for name in _PRODUCER_NAMES | set(_OWN_CODE)],
        *[f"features/{row['cache_key']}{suffix}" for row in mapping for suffix in (".pt", ".json")]}
    _require(set(paths) == expected_members, "closed package includes extra banks or evidence")
    examples = []
    for row in plan["examples"]:
        examples.append({"id": row["id_metadata"], "split": row["split"], "family": row["family_metadata"],
            "target": row["target_index_metadata"], "sentence_unseen_in_all_train": row["sentence_unseen_in_all_train"],
            "source": banks[_input(row["source"])],
            "candidates": [banks[_input({"text": text, "spans": {}})] for text in row["descriptions"]],
            "styles": {name: banks[_input(item)] for name, item in row["style_variants"].items()}})
    _inventory(root, manifest["files"])
    _require(_hash(root / "manifest.json") == expected_manifest_sha256 and consumer_code == _own_code()
        and _hash(consumer_readout) == consumer_readout_sha,
        "portable bytes changed during import")
    import torch
    binding = _seal({"schema": IMPORT_SCHEMA, "state": "PUBLIC_FEATURE_IMPORT_UNQUALIFIED",
        "source_class": closure["source_class"], "mechanics_only": mechanics,
        "manifest_file_sha256": expected_manifest_sha256, "closure_file_sha256": expected_closure_sha256,
        "producer_experiment_file_sha256": closure["completed_experiment_file_sha256"],
        "producer_runtime": plan["runtime"], "consumer_code": consumer_code,
        "consumer_semantic_readout_sha256": consumer_readout_sha,
        "consumer_runtime": {"torch_version": str(torch.__version__), "storage_device": "cpu", "storage_dtype": "bfloat16"},
        "consumer_retokenized": False, "publisher_checkpoint_loaded": False,
        "qualification": "UNQUALIFIED", **{key: False for key in _FALSE_FLAGS}})
    return ImportedPublicFeatures(plan, examples, binding, geometry, root)


def write_import_receipt(imported: ImportedPublicFeatures, output_path: str | Path) -> None:
    """Persist create-only eligibility outside the unchanged imported package."""
    path = Path(output_path)
    _require(path.is_absolute() and not path.exists() and not path.is_symlink(), "import receipt requires a fresh absolute path")
    _regular(path.parent, directory=True)
    _require(path != imported.package_directory and imported.package_directory not in path.parents,
        "import receipt must remain outside source package")
    _write(path, imported.binding)
