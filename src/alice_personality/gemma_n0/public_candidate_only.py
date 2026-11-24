"""Matched TRAIN-fit description-only diagnostic on closed PUBLIC features.

The source is an explicitly artificial two-token zero bank, never a Gemma
observation. The unchanged readout can learn global query biases and description
priors. A matched budget is not adequate-optimization proof, an architecture
choice, a persona attribution or N0 qualification. No publisher model is loaded.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import json
import time

from . import public_feature_handoff as handoff
from . import public_feature_diagnostics as diagnostic

SCHEMA = "alice-personality-closed-public-candidate-only-fit-v1"
PLAN_SCHEMA = "alice-personality-closed-public-candidate-only-plan-v1"
EXPORT_SCHEMA = "alice-personality-artificial-source-candidate-only-export-v1"
PRODUCTION_RECIPE = {"train_rows_per_family": 1, "dev_rows_per_family": 2,
    "style_rows_per_dev_family": 1, "width": 64, "steps": 256,
    "seed": 20261002, "learning_rate": 0.0003, "weight_decay": 0.01}
ORIGINAL_MIDPOINT_SHA256 = "fcaec4867816532b7b8d139f1b33460606b2bd71d75f621da6aacc961e48f4f4"
_ROOT = Path(__file__).resolve().parents[3]
_CODE = {"public_candidate_only.py": Path(__file__),
    "fit_public_candidate_only.py": _ROOT / "scripts/eipm/gemma_n0/fit_public_candidate_only.py",
    "magnolia_fit_public_candidate_only.sbatch": _ROOT / "scripts/eipm/gemma_n0/magnolia_fit_public_candidate_only.sbatch",
    "public_feature_diagnostics.py": Path(diagnostic.__file__),
    "diagnose_public_features.py": _ROOT / "scripts/eipm/gemma_n0/diagnose_public_features.py"}


class CandidateOnlyError(ValueError):
    """A public source, recipe, sampler proof or immutable boundary differs."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise CandidateOnlyError(reason)


def _own_code() -> dict:
    return {name: handoff._hash(handoff._regular(path)) for name, path in sorted(_CODE.items())}


def artificial_source(geometry: dict, torch):
    """Fixed synthetic masks/IDs; no actual source tokens, spans or features."""
    from .semantic_readout import LayerTokenBank, validate_bank
    source = LayerTokenBank(
        layers=torch.zeros((geometry["state_count"], 2, geometry["hidden_size"]), dtype=torch.bfloat16),
        input_ids=torch.zeros(2, dtype=torch.int64), attention_mask=torch.ones(2, dtype=torch.bool),
        source_mask=torch.ones(2, dtype=torch.bool), query_mask=torch.ones(2, dtype=torch.bool),
        head_mask=torch.tensor([True, False]), tail_mask=torch.tensor([False, True]))
    validate_bank(source, state_count=geometry["state_count"], hidden_size=geometry["hidden_size"], source=True)
    return source


def _state_sha(model, torch) -> dict:
    return {name: diagnostic._tensor_sha(value.detach(), torch) for name, value in model.state_dict().items()}


def _check_tensor(value, reference, torch, reason: str) -> None:
    _require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
        and value.dtype == reference.dtype and value.shape == reference.shape
        and not value.requires_grad and value.grad_fn is None
        and bool(torch.isfinite(value).all()), reason)


def _same_typed_value(left, right) -> bool:
    if type(left) is not type(right):
        return False
    if type(right) is dict:
        return set(left) == set(right) and all(_same_typed_value(left[key], right[key]) for key in right)
    if type(right) in (list, tuple):
        return len(left) == len(right) and all(_same_typed_value(a, b) for a, b in zip(left, right))
    return left == right


def _validate_midpoint(value: dict, *, binding: dict, model, optimizer, step: int,
                       initial_global_rng, torch) -> None:
    from . import public_semantic_experiment as producer
    _require(type(value) is dict and set(value) == {"schema", "binding", "geometry", "model", "optimizer",
        "step", "torch_cpu_rng", "sampler_rng"} and value["schema"] == producer.CHECKPOINT_SCHEMA
        and _same_typed_value(value["binding"], binding) and _same_typed_value(value["geometry"], model.geometry)
        and type(value["step"]) is int and value["step"] == step,
        "original midpoint schema, binding, geometry or step differs")
    expected = model.state_dict()
    _require(isinstance(value["model"], dict) and set(value["model"]) == set(expected),
        "midpoint readout parameter names differ")
    for name, reference in expected.items():
        _check_tensor(value["model"][name], reference.detach(), torch, "midpoint parameters differ or are nonfinite")
    for key in ("torch_cpu_rng", "sampler_rng"):
        _check_tensor(value[key], initial_global_rng, torch, "midpoint RNG state has an invalid tensor type/shape")
    # The old producer evaluates deterministic readouts and uses a separate
    # sampler Generator; nothing consumes global RNG after model initialization.
    _require(torch.equal(value["torch_cpu_rng"], initial_global_rng),
        "midpoint global RNG does not match exact seeded readout initialization")
    actual, template = value["optimizer"], optimizer.state_dict()
    _require(type(actual) is dict and set(actual) == {"state", "param_groups"}
        and type(actual["state"]) is dict and type(actual["param_groups"]) is list
        and len(actual["param_groups"]) == len(template["param_groups"]) == 1,
        "midpoint optimizer structure differs")
    group = actual["param_groups"][0]
    _require(type(group) is dict and set(group) == set(template["param_groups"][0])
        and _same_typed_value(group, template["param_groups"][0]), "midpoint AdamW recipe/defaults or parameter ordering differs")
    ordered = list(model.parameters())
    _require(set(actual["state"]) == set(range(len(ordered)))
        and all(type(key) is int for key in actual["state"]), "midpoint optimizer parameter state differs")
    for index, parameter in enumerate(ordered):
        state = actual["state"][index]
        _require(type(state) is dict and set(state) == {"step", "exp_avg", "exp_avg_sq"},
            "midpoint AdamW state fields differ")
        _check_tensor(state["step"], torch.zeros((), dtype=torch.float32), torch, "midpoint optimizer step is invalid")
        _require(float(state["step"]) == step, "midpoint optimizer step differs")
        for name in ("exp_avg", "exp_avg_sq"):
            _check_tensor(state[name], parameter.detach(), torch, "midpoint optimizer moments differ or are nonfinite")
        _require(not bool((state["exp_avg_sq"] < 0).any()), "midpoint optimizer squared moments are negative")


def _sampler_proof(checkpoint: dict, *, recipe: dict, train_count: int, torch) -> dict:
    sampler = torch.Generator().manual_seed(recipe["seed"] + 1)
    initial = diagnostic._tensor_sha(sampler.get_state(), torch)
    indices, midpoint = [], None
    for step in range(1, recipe["steps"] + 1):
        # Same scalar randint, range, independent Generator and call order as
        # original _train_steps. No batched sampling or seed substitution.
        indices.append(int(torch.randint(train_count, (), generator=sampler)))
        if step == checkpoint["step"]:
            observed = sampler.get_state()
            _require(torch.equal(observed, checkpoint["sampler_rng"]),
                "sample sequence does not match externally pinned original midpoint RNG")
            midpoint = diagnostic._tensor_sha(observed, torch)
    _require(midpoint is not None, "original midpoint is outside the update sequence")
    return {"algorithm": "scalar torch.randint(train_count, (), generator=Generator(seed+1)) per update",
        "train_count": train_count, "seed": recipe["seed"] + 1, "update_indices_metadata": indices,
        "initial_sampler_rng_sha256": initial, "midpoint_sampler_rng_sha256": midpoint,
        "final_sampler_rng_sha256": diagnostic._tensor_sha(sampler.get_state(), torch),
        "observed_original_midpoint_step": checkpoint["step"], "midpoint_rng_matches_original_checkpoint": True,
        "post_midpoint_continuation": "deterministic replay under exact original Torch runtime; no original final RNG artifact",
        "indices_are_model_input": False}


def _exposures(rows: list[dict], indices: list[int]) -> dict:
    """Receipt-only public description hashes; never scorer inputs."""
    train = [row for row in rows if row["split"] == "train"]
    digests = sorted({diagnostic._digest({"text": text, "spans": {}}) for row in rows for text in row["descriptions"]})
    def counts(selected):
        values = {digest: {"positive_occurrences": 0, "negative_occurrences": 0} for digest in digests}
        for row in selected:
            for index, text in enumerate(row["descriptions"]):
                digest = diagnostic._digest({"text": text, "spans": {}})
                values[digest]["positive_occurrences" if index == row["target_index_metadata"] else "negative_occurrences"] += 1
        return {"rows_or_updates": len(selected), "positive_slots": sum(value["positive_occurrences"] for value in values.values()),
            "negative_slots": sum(value["negative_occurrences"] for value in values.values()), "by_description_input_sha256": values}
    return {"static_train_pool_each_row_once": counts(train),
        "actual_sampler_selected_optimizer_updates": counts([train[index] for index in indices]),
        "candidate_pool_and_targets_unchanged": True, "metadata_is_model_input": False}


def _train(model, source, optimizer, train: list, indices: list[int], torch) -> tuple[list, dict]:
    model.train()
    losses = []
    statistics = {name: {"gradient_observed_steps": 0, "nonzero_gradient_steps": 0,
        "sum_gradient_l2_norm": 0.0, "maximum_gradient_l2_norm": 0.0}
        for name, _ in model.named_parameters()}
    print(json.dumps({"stage": "candidate_only_fit_started", "optimizer_steps": len(indices)}, sort_keys=True), flush=True)
    for step, index in enumerate(indices, start=1):
        row = train[index]
        _require(row["split"] == "train", "DEV cannot enter candidate-only optimizer updates")
        optimizer.zero_grad(set_to_none=True)
        # Only this synthetic constant and unchanged description banks enter
        # forward; the row's actual source, family, target/slot/ID do not.
        scores = model(source, row["candidates"])
        loss = torch.nn.functional.cross_entropy(scores[None], torch.tensor([row["target"]]))
        _require(bool(torch.isfinite(loss)), "candidate-only loss became nonfinite")
        loss.backward()
        for name, parameter in model.named_parameters():
            gradient = parameter.grad
            _require(gradient is not None and bool(torch.isfinite(gradient).all()),
                "candidate-only gradients are missing/nonfinite")
            norm = float(gradient.detach().norm())
            _require(math.isfinite(norm), "candidate-only gradient norm became nonfinite")
            stats = statistics[name]
            stats["gradient_observed_steps"] += 1
            stats["nonzero_gradient_steps"] += int(bool((gradient != 0).any()))
            stats["sum_gradient_l2_norm"] += norm
            stats["maximum_gradient_l2_norm"] = max(stats["maximum_gradient_l2_norm"], norm)
        optimizer.step()
        _require(all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()),
            "candidate-only parameters became nonfinite")
        losses.append(float(loss.detach()))
        if step in {max(1, len(indices) // 2), len(indices)}:
            print(json.dumps({"stage": "candidate_only_fit_progress", "completed_optimizer_steps": step,
                "planned_optimizer_steps": len(indices)}, sort_keys=True), flush=True)
    return losses, statistics


def _score(scorer, row: dict, torch) -> dict:
    value = diagnostic._score(scorer, row["source"], row["candidates"], row["target"], torch)
    scores = torch.tensor(value["logits"], dtype=torch.float32)
    probabilities = scores.softmax(0)
    target = torch.zeros_like(probabilities)
    target[row["target"]] = 1
    value["probabilities"] = probabilities.tolist()
    value["multiclass_brier_sum"] = float(((probabilities - target) ** 2).sum())
    return value


def _metrics(records: list, scorer: str) -> dict:
    values = [row["results"][scorer] for row in records]
    families = defaultdict(list)
    for row, value in zip(records, values):
        families[row["family_metadata"]].append(value["correct_against_retained_reference_target"])
    mean = lambda key: sum(value[key] for value in values) / len(values) if values else None
    return {"rows": len(values), "family_count": len(families), "accuracy": mean("correct_against_retained_reference_target"),
        "macro_family_accuracy": sum(sum(group) / len(group) for group in families.values()) / len(families) if families else None,
        "mean_cross_entropy": mean("cross_entropy"), "mean_multiclass_brier_sum": mean("multiclass_brier_sum"),
        "mean_target_rank_strict": mean("target_rank_strict"), "mean_target_logit_margin": mean("target_logit_margin"),
        "predicted_train_positive_description_count": sum(row["candidate_descriptions_seen_positive_in_train"]
            [value["predicted_candidate_index_metadata"]] for row, value in zip(records, values)),
        "uniform_random_expected_accuracy": sum(1 / row["candidate_count"] for row in records) / len(records) if records else None,
        "uniform_logit_mean_cross_entropy": sum(math.log(row["candidate_count"]) for row in records) / len(records) if records else None}


def _aggregates(records: list) -> dict:
    result = {}
    for split in ("train", "dev", "sentence_unseen_dev"):
        rows = [row for row in records if row["split"] == ("dev" if split == "sentence_unseen_dev" else split)
            and (split != "sentence_unseen_dev" or row["sentence_unseen_in_all_train"])]
        result[split] = {}
        for scorer in ("original_learned", "fixed_semantic_control", "candidate_only"):
            strata = {}
            for field in ("candidate_count", "target_candidate_index_metadata", "target_description_seen_positive_in_train"):
                keys = sorted({str(row[field]) for row in rows})
                strata[field] = {key: _metrics([row for row in rows if str(row[field]) == key], scorer) for key in keys}
            result[split][scorer] = {"metrics": _metrics(rows, scorer), "strata": strata}
        pairs = {}
        for baseline in ("original_learned", "fixed_semantic_control"):
            left = [row["results"]["candidate_only"] for row in rows]
            right = [row["results"][baseline] for row in rows]
            pairs[baseline] = {"rows": len(rows), "candidate_only_correct_baseline_correct": sum(a["correct_against_retained_reference_target"] and b["correct_against_retained_reference_target"] for a, b in zip(left, right)),
                "candidate_only_correct_baseline_wrong": sum(a["correct_against_retained_reference_target"] and not b["correct_against_retained_reference_target"] for a, b in zip(left, right)),
                "candidate_only_wrong_baseline_correct": sum(not a["correct_against_retained_reference_target"] and b["correct_against_retained_reference_target"] for a, b in zip(left, right)),
                "both_wrong": sum(not a["correct_against_retained_reference_target"] and not b["correct_against_retained_reference_target"] for a, b in zip(left, right)),
                "winner_changes": sum(a["predicted_candidate_index_metadata"] != b["predicted_candidate_index_metadata"] for a, b in zip(left, right)),
                "mean_cross_entropy_delta_candidate_minus_baseline": sum(a["cross_entropy"] - b["cross_entropy"] for a, b in zip(left, right)) / len(rows) if rows else None,
                "mean_brier_sum_delta_candidate_minus_baseline": sum(a["multiclass_brier_sum"] - b["multiclass_brier_sum"] for a, b in zip(left, right)) / len(rows) if rows else None}
        result[split]["paired_comparisons"] = pairs
    return result


def _reload_export(path: Path, *, binding: dict, geometry: dict, expected_sha256: str, torch):
    from .semantic_readout import SemanticReadout
    _require(handoff._hash(handoff._regular(path)) == expected_sha256, "candidate-only export pin differs")
    value = torch.load(path, weights_only=True, map_location="cpu")
    _require(type(value) is dict and set(value) == {"schema", "binding", "geometry", "model"}
        and value["schema"] == EXPORT_SCHEMA and value["binding"] == binding and value["geometry"] == geometry,
        "candidate-only export schema/binding/geometry differs")
    model = SemanticReadout(**geometry)
    _require(isinstance(value["model"], dict) and set(value["model"]) == set(model.state_dict()), "candidate-only export state differs")
    for name, reference in model.state_dict().items():
        _check_tensor(value["model"][name], reference, torch, "candidate-only export tensors differ")
    model.load_state_dict(value["model"], strict=True)
    return model.eval()


def fit_public_candidate_only(package_directory: str | Path, *, expected_manifest_sha256: str,
    expected_closure_sha256: str, semantic_export_path: str | Path, expected_export_sha256: str,
    original_midpoint_path: str | Path, expected_midpoint_sha256: str,
    output_directory: str | Path, cpu_threads: int = 4, allow_mechanics_only: bool = False) -> dict:
    """Fit only first-party description priors, with immutable original controls.

    Mechanics fixtures require explicit Python-only admission; the CLI has no
    fixture switch, hyperparameter override, data/model-source or retokenizer.
    """
    _require(type(cpu_threads) is int and cpu_threads > 0 and type(allow_mechanics_only) is bool,
        "CPU threads and explicit fixture admission types differ")
    own_code = _own_code()
    imported = handoff.import_public_features(package_directory,
        expected_manifest_sha256=expected_manifest_sha256, expected_closure_sha256=expected_closure_sha256,
        allow_mechanics_only=allow_mechanics_only)
    package, output = imported.package_directory, Path(output_directory)
    _require(output.is_absolute() and not output.exists() and not output.is_symlink()
        and output != package and package not in output.parents, "output must be fresh and outside the immutable package")
    handoff._regular(output.parent, directory=True)
    export_path, midpoint_path = (handoff._regular(path) for path in (semantic_export_path, original_midpoint_path))
    _require(handoff._hash(export_path) == handoff._pin(expected_export_sha256), "original export external pin differs")
    _require(handoff._hash(midpoint_path) == handoff._pin(expected_midpoint_sha256), "original midpoint external pin differs")
    experiment = handoff._read(package / "original/experiment.json", imported.binding["producer_experiment_file_sha256"])
    handoff._verify_seal(experiment)
    _require(experiment["export"]["sha256"] == expected_export_sha256 and experiment["export"]["reload_exact"] is True
        and experiment["checkpoint"]["sha256"] == expected_midpoint_sha256
        and experiment["checkpoint"]["optimizer_rng_resume_exact"] is True,
        "export/midpoint differs from original completed experiment")
    from . import public_semantic_experiment as producer
    import torch
    from .semantic_readout import SemanticReadout, fixed_semantic_scores
    recipe = handoff._recipe(imported.plan["recipe"])
    _require(recipe == experiment["binding"]["recipe"] and producer._code() == imported.plan["code"],
        "original producer code or recipe differs")
    _require(str(torch.__version__) == imported.plan["runtime"]["torch_version"], "exact original Torch runtime required")
    mechanics = imported.binding["mechanics_only"]
    if not mechanics:
        _require(recipe == PRODUCTION_RECIPE and str(torch.__version__) == "2.7.1+cu118" and cpu_threads == 4
            and expected_midpoint_sha256 == ORIGINAL_MIDPOINT_SHA256,
            "production requires the reviewed original recipe/runtime/midpoint and four CPU threads")
    geometry = {"state_count": imported.geometry["hidden_state_count"], "hidden_size": imported.geometry["hidden_size"], "width": recipe["width"]}
    train = [row for row in imported.examples if row["split"] == "train"]
    dev = [row for row in imported.examples if row["split"] == "dev"]
    _require(bool(train) and bool(dev), "original TRAIN and DEV must both be present")
    old_threads, old_rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(cpu_threads)
    start = time.monotonic()
    try:
        original = producer.load_semantic_export(export_path, binding=experiment["binding"],
            expected_file_sha256=expected_export_sha256, torch=torch).requires_grad_(False).eval()
        _require(original.geometry == geometry, "original readout geometry differs")
        original_before = _state_sha(original, torch)
        banks = diagnostic._banks(imported)
        bank_before = {key: diagnostic._bank_sha(bank, torch) for key, bank in banks.items()}
        source = artificial_source(geometry, torch)
        source_sha = diagnostic._bank_sha(source, torch)
        # Reconstruct initialization before inspecting checkpoint state. The
        # sampler is entirely independent of model construction and evaluation.
        torch.manual_seed(recipe["seed"])
        model = SemanticReadout(**geometry)
        initial_rng = torch.get_rng_state().clone()
        initial_state = _state_sha(model, torch)
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
        midpoint = torch.load(midpoint_path, weights_only=True, map_location="cpu")
        _require(handoff._hash(handoff._regular(midpoint_path)) == expected_midpoint_sha256,
            "original midpoint changed while loading")
        step = max(1, recipe["steps"] // 2)
        _require(type(experiment["checkpoint"]["step"]) is int and experiment["checkpoint"]["step"] == step,
            "original checkpoint step differs from original split-step recipe")
        _validate_midpoint(midpoint, binding=experiment["binding"], model=model, optimizer=optimizer,
            step=step, initial_global_rng=initial_rng, torch=torch)
        sampler_proof = _sampler_proof(midpoint, recipe=recipe, train_count=len(train), torch=torch)
        del midpoint
        exposures = _exposures(imported.plan["examples"], sampler_proof["update_indices_metadata"])
        source_spec = {"artificial": True, "provider_produced": False, "real_source_tokens_or_features_used": False,
            "construction": "two identical zero BF16 token vectors at every state; zero synthetic IDs; full attention/source/query masks; head first, tail second",
            "shape": list(source.layers.shape), "bank_sha256": source_sha,
            "variable_model_inputs": "unchanged candidate description banks only",
            "global_query_biases_learned": True, "same_parameter_count_is_same_effective_capacity": False,
            "source_length_matches_original": False, "counterfactual_truth": False}
        plan = handoff._seal({"schema": PLAN_SCHEMA, "state": "PREDECLARED_PUBLIC_CANDIDATE_ONLY_UNQUALIFIED",
            "import": imported.binding, "original_binding": experiment["binding"], "code": own_code,
            "original_export_file_sha256": expected_export_sha256, "original_midpoint_file_sha256": expected_midpoint_sha256,
            "recipe": recipe, "geometry": geometry, "artificial_source": source_spec, "sampler_proof": sampler_proof,
            "optimizer": {"type": "torch.optim.AdamW", "initial_parameter_groups": optimizer.state_dict()["param_groups"]},
            "description_exposure": exposures,
            "ordered_train_examples_metadata": [{"id_metadata": row["id"], "family_metadata": row["family"],
                "target_index_metadata": row["target"], "candidate_count": len(row["candidates"])} for row in train],
            "initial_model_state_tensor_sha256": initial_state,
            "initial_global_rng_sha256": diagnostic._tensor_sha(initial_rng, torch),
            "runtime": {"torch_version": str(torch.__version__), "device": "cpu", "cpu_threads": cpu_threads,
                "feature_dtype": "bfloat16", "readout_dtype": "float32"}, "mechanics_only": mechanics,
            "metrics": {"brier_convention": "sum_k (softmax(logit)_k - one_hot(target)_k)^2; no division by candidate count",
                "rank": "one plus number of strictly higher candidate logits; argmax first-index ties reported",
                "margin": "target logit minus maximum other-candidate logit",
                "dev_use": "already observed exploratory evidence; no new acceptance threshold"},
            "n0_approved": False, "personality_qualified": False, "repair_selected": False})
        output.mkdir(mode=0o700)
        handoff._write(output / "candidate-only-plan.json", plan)
        plan_sha = handoff._hash(output / "candidate-only-plan.json")
        subsets = {"train": train, "dev": dev, "sentence_unseen_dev": [row for row in dev if row["sentence_unseen_in_all_train"]]}
        reproduced = {}
        for name, scorer in (("learned", original), ("fixed_semantic_control", fixed_semantic_scores), ("untrained_readout", model)):
            actual = {split: producer._evaluate(scorer, rows, torch) for split, rows in subsets.items()}
            expected = experiment["learned"] if name == "learned" else experiment["baselines"][name]
            _require(actual == expected, "original learned/fixed/seeded-untrained metrics did not reproduce exactly")
            reproduced[name] = actual
        _require(_state_sha(model, torch) == initial_state and torch.equal(torch.get_rng_state(), initial_rng),
            "pre-fit initialization/global RNG changed during original verification")
        print(json.dumps({"stage": "original_controls_and_midpoint_verified", "train_rows": len(train), "dev_rows": len(dev)}, sort_keys=True), flush=True)
        candidate = lambda ignored_source, candidates: model(source, candidates)
        untrained_candidate_metrics = {split: producer._evaluate(candidate, rows, torch) for split, rows in subsets.items()}
        initial_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        losses, gradients = _train(model, source, optimizer, train, sampler_proof["update_indices_metadata"], torch)
        model.eval()
        final_state = _state_sha(model, torch)
        _require(final_state != initial_state, "no first-party candidate-only parameters changed")
        for name, stats in gradients.items():
            stats["parameter_changed"] = final_state[name] != initial_state[name]
            update = float((dict(model.named_parameters())[name].detach() - initial_parameters[name]).norm())
            _require(math.isfinite(update), "candidate-only parameter update norm became nonfinite")
            stats["initial_to_final_parameter_update_l2_norm"] = update
        del initial_parameters
        permutation = producer._permutations(candidate, imported.examples, torch)
        train_positive = {diagnostic._digest({"text": row["descriptions"][row["target_index_metadata"]], "spans": {}})
            for row in imported.plan["examples"] if row["split"] == "train"}
        records = []
        with torch.no_grad():
            for row, example in zip(imported.plan["examples"], imported.examples, strict=True):
                seen = [diagnostic._digest({"text": text, "spans": {}}) in train_positive for text in row["descriptions"]]
                results = {name: _score(scorer, example, torch) for name, scorer in
                    (("original_learned", original), ("fixed_semantic_control", fixed_semantic_scores), ("candidate_only", candidate))}
                # Exact equality under an entirely absent source argument and
                # every available style verifies this forward path ignores it.
                expected_scores = torch.tensor(results["candidate_only"]["logits"])
                for ignored in (None, *example["styles"].values()):
                    _require(torch.equal(candidate(ignored, example["candidates"]), expected_scores), "candidate-only source invariance failed")
                records.append({"id_metadata": example["id"], "family_metadata": example["family"], "split": example["split"],
                    "sentence_unseen_in_all_train": example["sentence_unseen_in_all_train"], "candidate_count": len(example["candidates"]),
                    "target_candidate_index_metadata": example["target"], "row_sha256": row["row_sha256"],
                    "candidate_input_sha256": [diagnostic._digest({"text": text, "spans": {}}) for text in row["descriptions"]],
                    "target_description_seen_positive_in_train": seen[example["target"]],
                    "candidate_descriptions_seen_positive_in_train": seen, "results": results})
        export_binding = {"plan_file_sha256": plan_sha, "plan_content_sha256": plan["receipt_sha256"],
            "original_binding": experiment["binding"], "import": imported.binding,
            "artificial_source": source_spec, "code": own_code, "qualification": "UNQUALIFIED"}
        fitted_export = output / "candidate-only-readout.pt"
        fitted_sha = producer._save_torch(fitted_export, {"schema": EXPORT_SCHEMA, "binding": export_binding,
            "geometry": geometry, "model": model.state_dict()}, torch)
        reloaded = _reload_export(fitted_export, binding=export_binding, geometry=geometry, expected_sha256=fitted_sha, torch=torch)
        reloaded_scorer = lambda ignored_source, candidates: reloaded(source, candidates)
        _require(_state_sha(reloaded, torch) == final_state and all(
            producer._evaluate(candidate, rows, torch) == producer._evaluate(reloaded_scorer, rows, torch) for rows in subsets.values()),
            "candidate-only export/reload differs")
        _require(source_sha == diagnostic._bank_sha(source, torch)
            and bank_before == {key: diagnostic._bank_sha(bank, torch) for key, bank in banks.items()}
            and all(bank.layers.grad is None and not bank.layers.requires_grad for bank in [source, *banks.values()]),
            "in-memory frozen or artificial features changed/received gradients")
        _require(original_before == _state_sha(original, torch), "original readout parameters changed")
        fresh = handoff.import_public_features(package, expected_manifest_sha256=expected_manifest_sha256,
            expected_closure_sha256=expected_closure_sha256, allow_mechanics_only=allow_mechanics_only)
        _require(fresh.binding == imported.binding and handoff._canonical(fresh.plan) == handoff._canonical(imported.plan),
            "closed public package changed during fitting")
        del fresh
        _require(handoff._hash(handoff._regular(export_path)) == expected_export_sha256
            and handoff._hash(handoff._regular(midpoint_path)) == expected_midpoint_sha256
            and producer._code() == imported.plan["code"] and own_code == _own_code()
            and str(torch.__version__) == imported.plan["runtime"]["torch_version"]
            and torch.get_num_threads() == cpu_threads
            and handoff._hash(output / "candidate-only-plan.json") == plan_sha,
            "code, original artifacts, runtime or predeclared plan changed during fitting")
        peak = None
        import sys
        if sys.platform.startswith("linux"):
            import resource
            peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        receipt = handoff._seal({"schema": SCHEMA, "state": "PUBLIC_CANDIDATE_ONLY_FIT_UNQUALIFIED",
            "status": "PUBLIC_MECHANICS_ONLY" if mechanics else "CLOSED_PUBLIC_CANDIDATE_ONLY_DIAGNOSTIC_COMPLETE",
            "binding": export_binding, "runtime": plan["runtime"], "recipe": recipe, "artificial_source": source_spec,
            "sampler_proof": sampler_proof, "description_exposure": exposures, "original_metrics_reproduced_exact": reproduced,
            "untrained_candidate_only_metrics": untrained_candidate_metrics, "training_loss_history": losses,
            "gradient_statistics": gradients, "optimizer": plan["optimizer"], "initial_model_state_tensor_sha256": initial_state,
            "final_model_state_tensor_sha256": final_state, "original_model_state_tensor_sha256": original_before,
            "all_in_memory_feature_bank_sha256": bank_before, "candidate_permutation": permutation,
            "candidate_only_ignores_actual_sources_and_styles_exact": True,
            "aggregates": _aggregates(records), "per_example": records,
            "export": {"path": str(fitted_export), "sha256": fitted_sha, "reload_exact": True, "schema": EXPORT_SCHEMA},
            "learned_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "original_features_reverified": True, "original_export_and_midpoint_reverified": True,
            "elapsed_seconds": time.monotonic() - start, "process_peak_rss_bytes": peak,
            "peak_measurement": "Linux process peak RSS; not GPU allocation or Slurm job peak" if peak is not None else "unmeasured",
            "first_party_public_training_performed": True, "optimizer_steps": len(losses),
            "publisher_checkpoint_loaded": False, "publisher_forward_performed": False,
            "upstream_gradient": False, "upstream_tensor_mutation": False, "private_identity_data": False,
            "final_payload_opened": False, "n0_approved": False, "personality_qualified": False,
            "repair_selected": False, "gemma_neutrality_established": False,
            "source_counts_define_capability_ceiling": False, "qualification": "UNQUALIFIED",
            "limits": ["Artificial source is not an actual Gemma feature or counterfactual observation",
                "Same architecture/init/update budget is not equal effective capacity or adequate optimization",
                "Learned global query biases remain; this is a description/pool-only partial-input diagnostic",
                "DEV has already been observed and is exploratory; no FINAL or new threshold",
                "Candidate-only success or failure cannot establish all full-model shortcuts or causal persona influence",
                "FewRel results do not qualify full Alice N0, N1 identity, N2 judgment or N3 calibration",
                "Historical update indices are reconstructed from pinned midpoint state and exact deterministic recipe, not a recorded original index log"]})
        handoff._write(output / "candidate-only-fit.json", receipt)
        return receipt
    finally:
        torch.set_rng_state(old_rng)
        torch.set_num_threads(old_threads)
