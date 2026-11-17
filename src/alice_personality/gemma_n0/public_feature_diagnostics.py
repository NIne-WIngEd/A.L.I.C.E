"""Reproduce and audit one CLOSED PUBLIC semantic cache; no fitting or forwards.

Source substitutions are destructive source-dependence ablations. Their retained
targets are reference annotations, never new counterfactual or inverse-role gold.
All outputs remain unqualified; relation retention is not personality authority.
"""
from __future__ import annotations

from hashlib import sha256
import math
from pathlib import Path

from . import public_feature_handoff as handoff

SCHEMA = "alice-personality-closed-public-feature-diagnostics-v1"
PLAN_SCHEMA = "alice-personality-closed-public-feature-diagnostic-plan-v1"
_ROOT = Path(__file__).resolve().parents[3]
_CODE = {"public_feature_diagnostics.py": Path(__file__),
         "diagnose_public_features.py": _ROOT / "scripts/eipm/gemma_n0/diagnose_public_features.py"}


class DiagnosticError(ValueError):
    """A public cache, runtime, model, reproduction or diagnostic binding differs."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise DiagnosticError(reason)


def _own_code() -> dict:
    return {name: handoff._hash(handoff._regular(path)) for name, path in sorted(_CODE.items())}


def _digest(value: object) -> str:
    return sha256(handoff._canonical(value)).hexdigest()


def _source_key(row: dict) -> str:
    # Choice is deterministic public metadata, independent of target or scores.
    return _digest({"id_metadata": row["id_metadata"], "row_sha256": row["row_sha256"],
                    "source_input_sha256": _digest(row["source"])})


def _controls(rows: list[dict]) -> dict:
    train = sorted((row for row in rows if row["split"] == "train"), key=_source_key)
    _require(bool(train), "a constant source requires an actual TRAIN bank")
    constant = train[0]["id_metadata"]
    cyclic, eligible = {}, {}
    for split in ("train", "dev"):
        ordered = sorted((row for row in rows if row["split"] == split), key=_source_key)
        _require(bool(ordered), "both original TRAIN and DEV are required")
        eligible[split] = len(ordered) > 1
        for index, row in enumerate(ordered):
            cyclic[row["id_metadata"]] = ordered[(index + 1) % len(ordered)]["id_metadata"]
    return {"constant_train_source": {"source_id_metadata": constant,
                "selection": "minimum SHA256(canonical id_metadata,row_sha256,source_input_sha256) among TRAIN"},
            "within_split_cyclic_source": {"source_by_example_id": cyclic,
                "eligible_splits": eligible,
                "selection": "sort by the same target-independent source metadata SHA256 within each split; next source, wrapping"},
            "targets_and_candidates_unchanged": True, "counterfactual_truth": False,
            "entity_role_reversal": False,
            "meaning": "destructive unrelated-source substitutions; retained reference targets are not new gold"}


def _tensor_sha(value, torch) -> str:
    _require(value.device.type == "cpu" and not value.requires_grad and value.grad_fn is None,
             "diagnostic tensors must remain detached CPU values")
    header = handoff._canonical({"dtype": str(value.dtype), "shape": list(value.shape)})
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
    digest = sha256(header)
    digest.update(memoryview(raw))
    return digest.hexdigest()


def _bank_sha(bank, torch) -> str:
    names = ("layers", "input_ids", "attention_mask", "source_mask", "query_mask", "head_mask", "tail_mask")
    return _digest({name: None if getattr(bank, name) is None else _tensor_sha(getattr(bank, name), torch)
                    for name in names})


def _banks(imported) -> dict:
    banks = {}
    for row, example in zip(imported.plan["examples"], imported.examples, strict=True):
        _require(row["id_metadata"] == example["id"], "original example ordering differs")
        pairs = [(row["source"], example["source"]),
                 *[({"text": text, "spans": {}}, bank)
                   for text, bank in zip(row["descriptions"], example["candidates"], strict=True)],
                 *[(item, example["styles"][name]) for name, item in row["style_variants"].items()]]
        for item, bank in pairs:
            digest = _digest(item)
            _require(digest not in banks or banks[digest] is bank, "same public input maps to different in-memory banks")
            banks[digest] = bank
    return banks


def _score(scorer, source, candidates, target: int, torch) -> dict:
    # IDs, relation metadata, target and ordering are absent from scorer inputs.
    scores = scorer(source, candidates)
    _require(scores.ndim == 1 and len(scores) == len(candidates)
             and bool(torch.isfinite(scores).all()), "diagnostic candidate scores differ or are nonfinite")
    prediction = int(scores.argmax())
    target_score = scores[target]
    others = torch.cat((scores[:target], scores[target + 1:]))
    return {"logits": scores.tolist(), "predicted_candidate_index_metadata": prediction,
            "correct_against_retained_reference_target": int(prediction == target),
            "cross_entropy": float(torch.nn.functional.cross_entropy(scores[None], torch.tensor([target]))),
            "target_rank_strict": 1 + int((scores > target_score).sum()),
            "target_logit_margin": float(target_score - others.max()) if len(others) else None}


def _metric(records: list[dict], condition: str, scorer: str) -> dict:
    if not records:
        return {"rows": 0, "mean_cross_entropy": None, "accuracy": None,
                "macro_family_accuracy": None, "family_count": 0}
    families = {}
    outcomes = [row["results"][condition][scorer] for row in records]
    for row, outcome in zip(records, outcomes, strict=True):
        families.setdefault(row["family_metadata"], []).append(outcome["correct_against_retained_reference_target"])
    return {"rows": len(records), "mean_cross_entropy": sum(row["cross_entropy"] for row in outcomes) / len(records),
            "accuracy": sum(row["correct_against_retained_reference_target"] for row in outcomes) / len(records),
            "macro_family_accuracy": sum(sum(hits) / len(hits) for hits in families.values()) / len(families),
            "family_count": len(families)}


def _strata(records: list[dict], condition: str, scorer: str) -> dict:
    result = {}
    for field in ("candidate_count", "target_candidate_index_metadata", "target_description_seen_positive_in_train"):
        groups = {}
        for row in records:
            groups.setdefault(str(row[field]), []).append(row)
        result[field] = {key: _metric(rows, condition, scorer) for key, rows in sorted(groups.items())}
    result["uniform_random_expected_accuracy"] = sum(1 / row["candidate_count"] for row in records) / len(records) if records else None
    result["uniform_logit_mean_cross_entropy"] = sum(math.log(row["candidate_count"]) for row in records) / len(records) if records else None
    result["predicted_train_positive_description_count"] = sum(
        row["candidate_descriptions_seen_positive_in_train"][row["results"][condition][scorer]["predicted_candidate_index_metadata"]]
        for row in records)
    return result


def _dependence(records: list[dict], condition: str, scorer: str, torch) -> dict:
    rows = [row for row in records if row["source_substitutions"][condition]["source_input_sha256"]
            != row["source_feature"]["input_sha256"]]
    changes, js, differences = [], [], []
    for row in rows:
        original, altered = (row["results"][name][scorer] for name in ("original", condition))
        changes.append(int(original["predicted_candidate_index_metadata"] != altered["predicted_candidate_index_metadata"]))
        left, right = (torch.tensor(value["logits"], dtype=torch.float32) for value in (original, altered))
        p, q = left.softmax(0), right.softmax(0)
        mixture = (p + q) / 2
        value = ((p * (p.clamp_min(1e-12).log() - mixture.clamp_min(1e-12).log())).sum()
                 + (q * (q.clamp_min(1e-12).log() - mixture.clamp_min(1e-12).log())).sum()) / 2
        js.append(float(value))
        differences.append(float((left - right).abs().max()))
    return {"actually_replaced_rows": len(rows), "top_candidate_changes": sum(changes),
            "mean_js_divergence": sum(js) / len(js) if js else None,
            "max_absolute_logit_change": max(differences) if differences else None,
            "retained_target_is_substitution_gold": False}


def diagnose_public_features(package_directory: str | Path, *, expected_manifest_sha256: str,
    expected_closure_sha256: str, semantic_export_path: str | Path, expected_export_sha256: str,
    output_directory: str | Path, cpu_threads: int = 4, allow_mechanics_only: bool = False) -> dict:
    """Reproduce actual CPU metrics, then predeclared immutable-bank ablations.

    Fixture admission is explicit, marked PUBLIC_MECHANICS_ONLY and unavailable
    in the production CLI. No source snapshot, tokenizer, model or optimizer is
    constructed. Different CUDA/runtime reproduction is a separate experiment.
    """
    _require(type(cpu_threads) is int and cpu_threads > 0, "positive CPU thread count required")
    own_code = _own_code()
    imported = handoff.import_public_features(package_directory,
        expected_manifest_sha256=expected_manifest_sha256, expected_closure_sha256=expected_closure_sha256,
        allow_mechanics_only=allow_mechanics_only)
    package = imported.package_directory
    output = Path(output_directory)
    _require(output.is_absolute() and not output.exists() and not output.is_symlink()
             and package != output and package not in output.parents,
             "diagnostic output must be fresh and outside the immutable package")
    handoff._regular(output.parent, directory=True)
    export_path = handoff._regular(semantic_export_path)
    _require(handoff._hash(export_path) == handoff._pin(expected_export_sha256), "semantic export external pin differs")
    experiment = handoff._read(package / "original/experiment.json", imported.binding["producer_experiment_file_sha256"])
    _require(experiment["export"]["sha256"] == expected_export_sha256 and experiment["export"]["reload_exact"] is True,
             "export differs from the original completed experiment")
    from . import public_semantic_experiment as producer
    _require(producer._code() == imported.plan["code"], "diagnostic producer implementation differs from original code")
    import torch
    from .semantic_readout import fixed_semantic_scores
    _require(str(torch.__version__) == imported.plan["runtime"]["torch_version"],
             "exact original Torch runtime required for CPU reproduction")
    old_threads, old_rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(cpu_threads)
    try:
        rows = imported.plan["examples"]
        control = _controls(rows)
        _require(len({row["id_metadata"] for row in rows}) == len(rows), "original public row IDs must be unique")
        closure = handoff._read(package / "closure.json", expected_closure_sha256)
        mapping = {row["input_sha256"]: row for row in closure["complete_input_mapping"]}
        cache = {row["key"]: row for row in experiment["cache"]}
        def feature_record(item):
            digest = _digest(item)
            link = mapping[digest]
            record = cache[link["cache_key"]]
            return {**link, "tensor_sha256": record["tensor_sha256"],
                    "metadata_sha256": record["metadata_sha256"]}
        bank_map = _banks(imported)
        before_banks = {key: _bank_sha(bank, torch) for key, bank in bank_map.items()}
        plan = handoff._seal({"schema": PLAN_SCHEMA, "state": "PREDECLARED_PUBLIC_DIAGNOSTICS_UNQUALIFIED",
            "manifest_file_sha256": expected_manifest_sha256, "closure_file_sha256": expected_closure_sha256,
            "original_experiment_file_sha256": imported.binding["producer_experiment_file_sha256"],
            "original_plan_file_sha256": experiment["binding"]["plan_sha256"],
            "original_plan_content_sha256": imported.plan["receipt_sha256"],
            "semantic_export_file_sha256": expected_export_sha256,
            "original_binding": experiment["binding"], "diagnostic_code": own_code,
            "controls": control, "runtime": {"torch_version": str(torch.__version__), "device": "cpu",
            "readout_dtype": "float32", "feature_dtype": "bfloat16", "cpu_threads": cpu_threads},
            "mechanics_only": imported.binding["mechanics_only"], "n0_approved": False,
            "personality_qualified": False, "training_performed": False, "publisher_forward_performed": False})
        output.mkdir(mode=0o700)
        handoff._write(output / "diagnostic-plan.json", plan)
        plan_file_sha = handoff._hash(output / "diagnostic-plan.json")
        model = producer.load_semantic_export(export_path, binding=experiment["binding"],
                                             expected_file_sha256=expected_export_sha256, torch=torch)
        _require(model.geometry == {"state_count": imported.geometry["hidden_state_count"],
            "hidden_size": imported.geometry["hidden_size"], "width": imported.plan["recipe"]["width"]},
            "readout geometry differs from original plan")
        model.requires_grad_(False).eval()
        model_before = {name: _tensor_sha(value, torch) for name, value in model.state_dict().items()}
        train, dev = ([row for row in imported.examples if row["split"] == split] for split in ("train", "dev"))
        subsets = {"train": train, "dev": dev,
                   "sentence_unseen_dev": [row for row in dev if row["sentence_unseen_in_all_train"]]}
        # First reproduce every original learned/fixed metric before ablations.
        reproduced = {}
        for name, scorer in (("learned", model), ("fixed_semantic_control", fixed_semantic_scores)):
            actual = {split: producer._evaluate(scorer, selected, torch) for split, selected in subsets.items()}
            expected = experiment["learned"] if name == "learned" else experiment["baselines"][name]
            _require(actual == expected, "original learned/fixed CPU metrics did not reproduce exactly")
            reproduced[name] = actual
        examples = {row["id"]: row for row in imported.examples}
        plan_rows = {row["id_metadata"]: row for row in rows}
        train_positive = {_digest({"text": row["descriptions"][row["target_index_metadata"]], "spans": {}})
                          for row in rows if row["split"] == "train"}
        records = []
        with torch.no_grad():
            for row, example in zip(rows, imported.examples, strict=True):
                candidate_features = [feature_record({"text": text, "spans": {}}) for text in row["descriptions"]]
                replacements = {"constant_train_source": control["constant_train_source"]["source_id_metadata"],
                                "within_split_cyclic_source": control["within_split_cyclic_source"]["source_by_example_id"][example["id"]]}
                substituted = {name: {"source_id_metadata": rid, "source_input_sha256": _digest(plan_rows[rid]["source"]),
                                     "source_feature": feature_record(plan_rows[rid]["source"])} for name, rid in replacements.items()}
                outcomes = {}
                for condition, source in [("original", example["source"]),
                        *[(name, examples[rid]["source"]) for name, rid in replacements.items()]]:
                    outcomes[condition] = {name: _score(scorer, source, example["candidates"], example["target"], torch)
                        for name, scorer in (("learned", model), ("fixed_semantic_control", fixed_semantic_scores))}
                seen = [value["input_sha256"] in train_positive for value in candidate_features]
                records.append({"id_metadata": example["id"], "row_sha256": row["row_sha256"],
                    "family_metadata": example["family"], "split": example["split"],
                    "sentence_unseen_in_all_train": example["sentence_unseen_in_all_train"],
                    "candidate_count": len(example["candidates"]), "target_candidate_index_metadata": example["target"],
                    "target_description_seen_positive_in_train": seen[example["target"]],
                    "candidate_descriptions_seen_positive_in_train": seen,
                    "source_feature": feature_record(row["source"]), "candidate_features": candidate_features,
                    "source_substitutions": substituted, "results": outcomes})
        aggregates = {}
        for condition in ("original", "constant_train_source", "within_split_cyclic_source"):
            aggregates[condition] = {}
            for scorer in ("learned", "fixed_semantic_control"):
                aggregates[condition][scorer] = {}
                for split in ("train", "dev", "sentence_unseen_dev"):
                    selected = [row for row in records if row["split"] == ("dev" if split == "sentence_unseen_dev" else split)
                                and (split != "sentence_unseen_dev" or row["sentence_unseen_in_all_train"])]
                    value = {"metrics_against_retained_reference_targets": _metric(selected, condition, scorer),
                             "strata": _strata(selected, condition, scorer)}
                    if condition != "original":
                        value["source_dependence"] = _dependence(selected, condition, scorer, torch)
                    aggregates[condition][scorer][split] = value
        for name in reproduced:
            _require({split: aggregates["original"][name][split]["metrics_against_retained_reference_targets"]
                      for split in subsets} == reproduced[name], "per-example reproduction evidence differs")
        weights = model.layer_logits.detach().softmax(0)
        layer_stats = {"global_query_independent": True, "state_count": len(weights), "softmax_weights": weights.tolist(),
            "entropy": float(-(weights * weights.clamp_min(1e-30).log()).sum()),
            "maximum_uniform_entropy": math.log(len(weights)), "effective_states": float(torch.exp(-(weights * weights.clamp_min(1e-30).log()).sum())),
            "embedding_weight": float(weights[0]), "final_normalized_state_weight": float(weights[-1])}
        _require(before_banks == {key: _bank_sha(bank, torch) for key, bank in bank_map.items()},
                 "in-memory frozen features changed during diagnostics")
        _require(model_before == {name: _tensor_sha(value, torch) for name, value in model.state_dict().items()},
                 "first-party semantic export parameters changed during diagnostics")
        # Fresh complete inventory/tensor admission, without original source or model.
        fresh = handoff.import_public_features(package, expected_manifest_sha256=expected_manifest_sha256,
            expected_closure_sha256=expected_closure_sha256, allow_mechanics_only=allow_mechanics_only)
        _require(fresh.binding == imported.binding and handoff._canonical(fresh.plan) == handoff._canonical(imported.plan),
                 "closed public package changed during diagnostics")
        del fresh
        _require(handoff._hash(handoff._regular(export_path)) == expected_export_sha256
            and producer._code() == imported.plan["code"] and own_code == _own_code()
            and handoff._hash(output / "diagnostic-plan.json") == plan_file_sha,
            "diagnostic code, export or declared plan changed during diagnostics")
        receipt = handoff._seal({"schema": SCHEMA, "state": "PUBLIC_FEATURE_DIAGNOSTICS_UNQUALIFIED",
            "status": "PUBLIC_MECHANICS_ONLY" if imported.binding["mechanics_only"] else "CLOSED_PUBLIC_DIAGNOSTICS_COMPLETE",
            "binding": {"diagnostic_plan_file_sha256": plan_file_sha, "diagnostic_plan_content_sha256": plan["receipt_sha256"],
                        "import": imported.binding, "original_binding": experiment["binding"],
                        "semantic_export_file_sha256": expected_export_sha256, "diagnostic_code": own_code},
            "runtime": plan["runtime"], "original_metrics_reproduced_exact": reproduced,
            "original_semantic_export_reloaded": True, "controls": control,
            "aggregates": aggregates, "per_example": records, "learned_layer_mixture": layer_stats,
            "all_in_memory_feature_bank_sha256": before_banks,
            "semantic_model_state_tensor_sha256": model_before,
            "original_source_sentence_only_overlap": imported.plan["original_source_sentence_only_overlap"],
            "dev_sentence_unseen_metric_rule": imported.plan["dev_sentence_unseen_metric_rule"],
            "original_features_reverified": True, "training_performed": False,
            "publisher_checkpoint_loaded": False, "publisher_forward_performed": False,
            "upstream_gradient": False, "upstream_tensor_mutation": False,
            "private_identity_data": False, "final_payload_opened": False,
            "n0_approved": False, "personality_qualified": False, "gemma_neutrality_established": False,
            "source_counts_define_capability_ceiling": False, "qualification": "UNQUALIFIED",
            "limits": ["Source substitutions are ablations with retained reference targets, not counterfactual gold or entity-role reversal",
                       "Familiar-description preference and dependence are diagnostic observations, not causal persona identification",
                       "FewRel relation results do not qualify N0 or Alice personality",
                       "DEV is observed model-selection evidence; no FINAL payload is opened",
                       "Exact CPU reproduction does not establish CUDA determinism or runtime interchangeability"]})
        handoff._write(output / "diagnostics.json", receipt)
        return receipt
    finally:
        torch.set_rng_state(old_rng)
        torch.set_num_threads(old_threads)
